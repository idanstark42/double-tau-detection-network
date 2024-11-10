import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data.dataloader import default_collate

from utils import long_operation, seconds_to_time
from visualization import ModelVisualizer
from model.cylindrical_loss import CylindricalLoss
from settings import EPOCHS, BATCH_SIZE, TRAINING_PERCENTAGE, VALIDATION_PERCENTAGE

def train_module(dataset, model, output_folder, options={}):
  trainer = Trainer(dataset, model, output_folder, options)
  trainer.train_module()

class Trainer:
  def __init__(self, dataset, model, output_folder, options):
    self.dataset = dataset
    self.model = model
    self.output_folder = output_folder
    self.options = options

  def train_module(self):
    start_time = time.time()

    cache_type = self.options.get('cache', 'events')
    self.dataset.cache_type = cache_type
    
    preload_type = self.options.get('preload', 'none')
    split = int(self.options.get('split')) if self.options.get('split') else 1
    limit = int(self.options.get('limit')) if self.options.get('limit') else None

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = CylindricalLoss()
    
    epochs = int(self.options.get('epochs', EPOCHS))
    midsave = self.options.get('midsave', 'false') == 'true'
    batch_size = int(self.options.get('batch_size', BATCH_SIZE))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
      # spawen start method to avoid error
      torch.multiprocessing.set_start_method('spawn')
      model = model.cuda()
      print(f'Using Device:                     {torch.cuda.get_device_name(0)}')
    else:
      print('Using Device:                     CPU')

    device = torch.device('cuda' if use_cuda else 'cpu')
    self.device = device
    train_loaders, validation_loaders, test_loader = self.init_dataloaders(self.dataset, device, split, self.options)
    
    using_multiprocessing = int(self.options.get('num_workers', 0)) > 0
    print(f'training set size:                {sum([len(loader.dataset) for loader in train_loaders])}')
    print(f'validation set size:              {sum([len(loader.dataset) for loader in validation_loaders])}')
    print(f'test set size:                    {len(test_loader.dataset)}')
    print(f'split:                            {split}')
    print(f'limit:                            {limit if limit else "none"}')
    print('Using Multiprocessing:            ' + ('yes' if using_multiprocessing else 'no'))
    print(f'Batch Size:                       {batch_size}')
    print(f'Epochs:                           {epochs}')
    print(f'Preload Type:                     {preload_type}')
    print(f'Midsave:                          {midsave}')
    print(f'Cache:                            {cache_type}')
    print(f'Output Folder:                    {self.output_folder}')
    print()

    if preload_type == 'full':
      self.dataset.full_preload()

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
      if preload_type == 'partial':
        preload_start_time = time.time()
        self.dataset.start_partial_preloading()
        self.partial_preload(train_loader, 'Preloading Training')
        self.partial_preload(validation_loader, 'Preloading Validation')
        self.dataset.finish_partial_preloading()
        print(f'Preloading time: {seconds_to_time(time.time() - preload_start_time)}')
      for epoch in range(epochs):
        traintime_start = time.time()
        epoch_start_times.append(traintime_start)
        training_loss = self.train(train_loader, model, criterion, optimizer, epoch, batch_size)
        valtime_start = time.time()
        validation_loss = self.validate(validation_loader, model, criterion, epoch, batch_size)
        print(f'Training time: {seconds_to_time(valtime_start - traintime_start)}, Validation time: {seconds_to_time(time.time() - valtime_start)}')
        print('Training Loss: {:.6f}, Validation Loss: {:.6f}'.format(training_loss, validation_loss))
        if validation_loss < best_validation_loss:
          best_validation_loss = validation_loss
          best_model = model.state_dict()
        losses.append((training_loss, validation_loss))
      self.dataset.clear_cache()
      if limit and i == limit - 1:
        break
      if midsave:
        torch.save(model.state_dict(), self.output_folder + f'\\model_{i}_{epoch}.pth')

    # Load the best model
    model.load_state_dict(best_model)

    # make a directory for the model if it doesn't exist
    os.makedirs(self.output_folder, exist_ok=True)

    # Test the best model
    test_start_time = time.time()
    if len(test_loader.dataset) > 0:
      print('2. Testing')
      self.test(test_loader, model, criterion, self.output_folder, self.dataset, batch_size, use_xla=False, use_cuda=use_cuda)
    else:
      print(' -- skipping testing')

    # Save the model
    torch.save(model.state_dict(), self.output_folder + '\\model.pth')

    # print summary
    print()
    print('Done')
    print()
    print(f'Time: {seconds_to_time(time.time() - start_time)}')
    print(f'(trainig: {seconds_to_time(sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]))}, testing: {seconds_to_time(time.time() - test_start_time)})')
    print(f'Average Epoch Time: {seconds_to_time(sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]) / len(epoch_start_times))}')
    print(f'Best Validation Loss: {best_validation_loss}')

    # Plot the losses as a function of epoch
    ModelVisualizer(model).show_losses(losses, self.output_folder + '\\losses.png')

  def init_dataloaders (self, dataset, device, split, options):
    split_dataset_size = int(len(dataset) / split)
    train_size = int(split_dataset_size * TRAINING_PERCENTAGE)
    validation_size = int(split_dataset_size * VALIDATION_PERCENTAGE)
    test_size = len(dataset) - (train_size + validation_size) * split

    split_sizes = [train_size, validation_size] * split + [test_size]
    datasets = random_split(dataset, split_sizes)

    train_loaders, validation_loaders = [], []
    for i in range(split):
      train_loaders.append(self.generate_dataloader(datasets[i * 2], device, options))
      validation_loaders.append(self.generate_dataloader(datasets[i * 2 + 1], device, options))
    test_loader = self.generate_dataloader(datasets[-1], device, options)
    
    return train_loaders, validation_loaders, test_loader

  def generate_dataloader (self, dataset, options):
    num_workers = int(options.get('num_workers', 0))
    pin_memory = num_workers > 0
    batch_size = int(options.get('batch_size', BATCH_SIZE))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn(), num_workers=num_workers, pin_memory=pin_memory)

  def collate_fn (self):
    return lambda x: tuple(x_.to  (self.device, non_blocking=True) for x_ in default_collate(x))

  def partial_preload (self, loader, message='Preloading'):
    dataset = loader.dataset
    def run (next):
      for index in range(len(dataset)):
        dataset[index]
        next(1)
    long_operation(run, max=len(dataset), message=message)

  # train the model
  def train(self, train_loader, model, criterion, optimizer, epoch, batch_size):
    model.train()
    def run (next):
      total_loss = 0
      for batch_idx, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output, loss = self.calc(model, input, target, criterion)
        loss.backward()
        optimizer.step()
        next(batch_size)
        total_loss += loss.item()
      return total_loss

    total_loss = long_operation(run, max=len(train_loader) * batch_size, message=f'Epoch {epoch+1} training', ending_message=lambda l: f'loss: {l / len(train_loader):.4f}')
    return total_loss / len(train_loader)

  # validate the model
  def validate(self, val_loader, model, criterion, epoch, batch_size):
    model.eval()

    with torch.no_grad():
      def run (next):
        total_loss = 0
        for batch_idx, (input, target) in enumerate(val_loader):
          output, loss = self.calc(model, input, target, criterion)
          next(batch_size)
          total_loss += loss.item()
        return total_loss
    
      total_loss = long_operation(run, max=len(val_loader) * batch_size, message=f'Epoch {epoch+1} validation', ending_message=lambda l: f'loss: {l / len(val_loader):.4f}')
    return total_loss / len(val_loader)

  # test the model
  def test(self, test_loader, model, criterion, output_folder, dataset, batch_size, use_xla=False, use_cuda=False):
    model.eval()
    outputs, targets = [], []

    with torch.no_grad():
      def run (next):
        total_loss = 0
        for batch_idx, (input, target) in enumerate(test_loader):
          output, loss = self.calc(model, input, target, criterion)
          next(batch_size)
          for index, (output, target) in enumerate(zip(output, target)):
            outputs.append(output)
            targets.append(target)
          total_loss += loss.item()
        return total_loss
      total_loss = long_operation(run, max=len(test_loader) * batch_size, message='Testing ')
    print(f'\nTest set average loss: {total_loss / len(test_loader):.4f}\n')

    if use_xla or use_cuda:
      outputs = [output.cpu() for output in outputs]
      targets = [target.cpu() for target in targets]
    ModelVisualizer(model).plot_results(outputs, targets, test_loader, dataset, output_folder + '\\testing.png')

  def calc (self, model, input, target, criterion):
    output = model(input)
    loss = criterion(output, target)
    return output, loss
