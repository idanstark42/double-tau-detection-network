import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split
import numpy as np

from utils import long_operation, seconds_to_time
from visualization import ModelVisualizer
from settings import EPOCHS, BATCH_SIZE, TRAINING_PERCENTAGE, VALIDATION_PERCENTAGE, TEST_ARROWS_PERCENTAGE

def train_module(dataset, model, output_folder, options={}):
  start_time = time.time()
  train_loader, validation_loader, test_loader = init_dataloaders(dataset)
  print(f'training over {len(train_loader.dataset)} samples, validating over {len(validation_loader.dataset)} samples, testing over {len(test_loader.dataset)} samples')

  if options.get('cache') == 'false':
    dataset.use_cache = False
    print(' -- cache disabled')

  optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
  criterion = nn.MSELoss()

  use_cuda = torch.cuda.is_available()
  if use_cuda:
    model = model.cuda()
    print(f'using device {torch.cuda.get_device_name(0)}')
  else:
    print('using device cpu')

  # Train the model
  print()
  print('1. Training')
  best_validation_loss = float('inf')
  best_model = None
  losses = []
  epoch_start_times = []
  for epoch in range(EPOCHS):
    epoch_start_times.append(time.time())
    training_loss = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
    validation_loss = validate(validation_loader, model, criterion, epoch, use_cuda)
    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      best_model = model.state_dict()
    losses.append((training_loss, validation_loss))

  # Load the best model
  model.load_state_dict(best_model)

  # make a directory for the model if it doesn't exist
  os.makedirs(output_folder, exist_ok=True)

  # Test the best model
  test_start_time = time.time()
  if len(test_loader) > 0:
    print('2. Testing')
    test(test_loader, model, criterion, output_folder, use_cuda)
  else:
    print(' -- skipping testing')

  # Save the model
  torch.save(model.state_dict(), output_folder + '\\model.pth')

  # print summary
  print()
  print('Done')
  print()
  print(f'Time: {seconds_to_time(time.time() - start_time)}')
  print(f'(trainig: {seconds_to_time(sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]))}, testing: {seconds_to_time(time.time() - test_start_time)})')
  print(f'Average Epoch Time: {seconds_to_time(sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]) / len(epoch_start_times))}')
  print(f'Best Validation Loss: {best_validation_loss}')

  # Plot the losses as a function of epoch
  ModelVisualizer(model).show_losses(losses, output_folder + '\\losses.png')

def init_dataloaders (dataset):
  train_size = int(len(dataset) * TRAINING_PERCENTAGE)
  validation_size = int(len(dataset) * VALIDATION_PERCENTAGE)
  test_size = len(dataset) - train_size - validation_size

  train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
  
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
  
  return train_loader, validation_loader, test_loader

# train the model
def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
  model.train()
  
  def run (next):
    total_loss = 0
    for batch_idx, (inputs, target) in enumerate(train_loader):
      optimizer.zero_grad()
      if use_cuda:
        inputs = inputs.to('cuda')
        target = target.to('cuda')
      output = model(*inputs)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      next(BATCH_SIZE)
      total_loss += loss.item()
    return total_loss

  total_loss = long_operation(run, max=len(train_loader) * BATCH_SIZE, message=f'Epoch {epoch+1} ')
  return total_loss / len(train_loader)

# validate the model
def validate(val_loader, model, criterion, epoch, use_cuda):
  model.eval()

  with torch.no_grad():
    def run (next):
      total_loss = 0
      for batch_idx, (inputs, target) in enumerate(val_loader):
        if use_cuda:
          inputs = inputs.to('cuda')
          target = target.to('cuda')
        output = model(*inputs)
        loss = criterion(output, target)
        next(BATCH_SIZE)
        total_loss += loss.item()
      return total_loss
  
    total_loss = long_operation(run, max=len(val_loader) * BATCH_SIZE, message=f'Epoch {epoch+1} ')
  return total_loss / len(val_loader)

# test the model
def test(test_loader, model, criterion, output_folder, use_cuda):
  model.eval()
  outputs, targets = [], []
  random_indeces = np.random.choice(len(test_loader.dataset), int(TEST_ARROWS_PERCENTAGE * len(test_loader.dataset)), replace=False)

  with torch.no_grad():
    def run (next):
      total_loss = 0
      for batch_idx, (inputs, target) in enumerate(test_loader):
        if use_cuda:
          inputs = inputs.to('cuda')
          target = target.to('cuda')
        output = model(*inputs)
        loss = criterion(output, target)
        next(BATCH_SIZE)
        for index, (output, target) in enumerate(zip(output, target)):
          if batch_idx * BATCH_SIZE + index in random_indeces:
            outputs.append(output)
            targets.append(target)
        total_loss += loss.item()
      return total_loss
    
    total_loss = long_operation(run, max=len(test_loader) * BATCH_SIZE, message='Testing ')
  print(f'\nTest set average loss: {total_loss / len(test_loader):.4f}\n')
  ModelVisualizer(model).plot_results(outputs, targets, output_folder + '\\testing.png')
