import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data.dataloader import default_collate
import numpy as np

from utils import long_operation, seconds_to_time
from visualization import ModelVisualizer
from model.cylindrical_loss import CylindricalLoss
from settings import EPOCHS, BATCH_SIZE, TRAINING_PERCENTAGE, VALIDATION_PERCENTAGE

def train_module(dataset, model, output_folder, options={}):
  start_time = time.time()

  if options.get('cache') == 'false':
    dataset.use_cache = False
    print(' -- cache disabled')
  
  split = int(options.get('split')) if options.get('split') else 1

  optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
  criterion = CylindricalLoss()

  use_cuda = torch.cuda.is_available()
  if use_cuda:
    model = model.cuda()
    print(f'Using Device:                     {torch.cuda.get_device_name(0)}')
  else:
    print('Using Device:                     cpu')

  device = torch.device('cuda' if use_cuda else 'cpu')
  train_loaders, validation_loaders, test_loader = init_dataloaders(dataset, device, split)
  print(f'training set size:                {sum([len(loader.dataset) for loader in train_loaders])}')
  print(f'validation set size:              {sum([len(loader.dataset) for loader in validation_loaders])}')
  print(f'test set size:                    {len(test_loader.dataset)}')
  print(f'split:                            {split}')

  # Train the model
  print()
  print('1. Training')
  best_validation_loss = float('inf')
  best_model = None
  losses = []
  epoch_start_times = []
  for i in range(split):
    train_loader, validation_loader = train_loaders[i], validation_loaders[i]
    if split > 1:
      print(f'Split {i + 1}/{split}')
    for epoch in range(EPOCHS):
      epoch_start_times.append(time.time())
      training_loss = train(train_loader, model, criterion, optimizer, epoch)
      validation_loss = validate(validation_loader, model, criterion, epoch)
      if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model = model.state_dict()
      losses.append((training_loss, validation_loss))
    dataset.clear_cache()

  # Load the best model
  model.load_state_dict(best_model)

  # make a directory for the model if it doesn't exist
  os.makedirs(output_folder, exist_ok=True)

  # Test the best model
  test_start_time = time.time()
  if len(test_loader.dataset) > 0:
    print('2. Testing')
    test(test_loader, model, criterion, output_folder, dataset, use_cuda)
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

def init_dataloaders (dataset, device, split):
  split_dataset_size = int(len(dataset) / split)
  train_size = int(split_dataset_size * TRAINING_PERCENTAGE)
  validation_size = int(split_dataset_size * VALIDATION_PERCENTAGE)
  test_size = len(dataset) - (train_size + validation_size) * split

  split_sizes = [train_size, validation_size] * split + [test_size]
  datasets = random_split(dataset, split_sizes)

  train_loaders, validation_loaders = [], []
  for i in range(split):
    train_loaders.append(DataLoader(datasets[i * 2], batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))), num_worker=24)
    validation_loaders.append(DataLoader(datasets[i * 2 + 1], batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))), num_worker=24)
  test_loader = DataLoader(datasets[-1], batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), num_worker=24)
  
  return train_loaders, validation_loaders, test_loader

# train the model
def train(train_loader, model, criterion, optimizer, epoch):
  model.train()
  
  def run (next):
    total_loss = 0
    for batch_idx, (input, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output, loss = calc(model, input, target, criterion)
      loss.backward()
      optimizer.step()
      next(BATCH_SIZE)
      total_loss += loss.item()
    return total_loss

  total_loss = long_operation(run, max=len(train_loader) * BATCH_SIZE, message=f'Epoch {epoch+1} training')
  return total_loss / len(train_loader)

# validate the model
def validate(val_loader, model, criterion, epoch):
  model.eval()

  with torch.no_grad():
    def run (next):
      total_loss = 0
      for batch_idx, (input, target) in enumerate(val_loader):
        output, loss = calc(model, input, target, criterion)
        next(BATCH_SIZE)
        total_loss += loss.item()
      return total_loss
  
    total_loss = long_operation(run, max=len(val_loader) * BATCH_SIZE, message=f'Epoch {epoch+1} validation')
  return total_loss / len(val_loader)

# test the model
def test(test_loader, model, criterion, output_folder, dataset, use_cuda=False):
  model.eval()
  outputs, targets = [], []

  with torch.no_grad():
    def run (next):
      total_loss = 0
      for batch_idx, (input, target) in enumerate(test_loader):
        output, loss = calc(model, input, target, criterion)
        next(BATCH_SIZE)
        for index, (output, target) in enumerate(zip(output, target)):
          outputs.append(output)
          targets.append(target)
        total_loss += loss.item()
      return total_loss
    total_loss = long_operation(run, max=len(test_loader) * BATCH_SIZE, message='Testing ')
  print(f'\nTest set average loss: {total_loss / len(test_loader):.4f}\n')

  if use_cuda:
    outputs = [output.cpu() for output in outputs]
    targets = [target.cpu() for target in targets]
  ModelVisualizer(model).plot_results(outputs, targets, test_loader, dataset, output_folder + '\\testing.png')

def calc (model, input, target, criterion):
  output = model(input)
  loss = criterion(output, target)
  return output, loss
