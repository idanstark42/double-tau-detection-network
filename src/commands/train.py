import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split

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

    self.cache_type = self.options.get('cache', 'events')
    self.dataset.cache_type = self.cache_type
    
    self.preload_type = self.options.get('preload', 'none')
    self.split = int(self.options.get('split')) if self.options.get('split') else 1
    self.limit = int(self.options.get('limit')) if self.options.get('limit') else None
    
    self.epochs = int(self.options.get('epochs', EPOCHS))
    self.midsave = self.options.get('midsave', 'false') == 'true'
    self.batch_size = int(self.options.get('batch_size', BATCH_SIZE))
    self.use_xla = self.options.get('use_xla', 'false') == 'true'
    self.persistent_workers = self.options.get('persistent_workers', 'false') == 'true'
    self.learning_rate = float(self.options.get('learning_rate', 0.001))
    self.weight_decay = float(self.options.get('weight_decay', 0.0001))

  def train_module(self):
    start_time = time.time()

    self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    self.criterion = CylindricalLoss()

    self.use_cuda = torch.cuda.is_available()
    if self.use_cuda:
      # spawen start method to avoid error
      torch.multiprocessing.set_start_method('spawn')
      torch.multiprocessing.set_sharing_strategy('file_system')
      self.model = self.model.cuda()
      print(f'Using Device:                     {torch.cuda.get_device_name(0)}')
    else:
      print('Using Device:                     CPU')

    self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    self.init_dataloaders()
    
    using_multiprocessing = int(self.options.get('num_workers', 0)) > 0
    print(f'training set size:                {sum([len(loader.dataset) for loader in self.train_loaders])}')
    print(f'validation set size:              {sum([len(loader.dataset) for loader in self.validation_loaders])}')
    print(f'test set size:                    {len(self.test_loader.dataset)}')
    print(f'split:                            {self.split}')
    print(f'limit:                            {self.limit if self.limit else "none"}')
    print('Using Multiprocessing:            ' + ('yes' if using_multiprocessing else 'no'))
    if using_multiprocessing:
      print(f'Number of Workers:                {int(self.options.get("num_workers", 0))}')
      print(f'Persistent Workers:               ' + ('yes' if self.persistent_workers else 'no'))
    print(f'Batch Size:                       {self.batch_size}')
    print(f'Epochs:                           {self.epochs}')
    print(f'Preload Type:                     {self.preload_type}')
    print(f'Midsave:                          {self.midsave}')
    print(f'Cache:                            {self.cache_type}')
    print(f'Output Folder:                    {self.output_folder}')
    print(f'Learning Rate:                    {self.learning_rate}')
    print(f'Weight Decay:                     {self.weight_decay}')
    print()

    if self.preload_type == 'full':
      self.dataset.full_preload()

    # Train the model
    print()
    print('1. Training')
    best_validation_loss = float('inf')
    best_model = None
    losses = []
    epoch_start_times = []
    for i in range(self.split):
      train_loader, validation_loader = self.train_loaders[i], self.validation_loaders[i]
      if self.split > 1:
        print(f'Split {i + 1}/{self.split}')
      if self.preload_type == 'partial':
        preload_start_time = time.time()
        self.dataset.start_partial_preloading()
        self.partial_preload(train_loader, 'Preloading Training')
        self.partial_preload(validation_loader, 'Preloading Validation')
        self.dataset.finish_partial_preloading()
        print(f'Preloading time: {seconds_to_time(time.time() - preload_start_time)}')
      for epoch in range(self.epochs):
        traintime_start = time.time()
        epoch_start_times.append(traintime_start)
        training_loss = self.train(train_loader, epoch)
        valtime_start = time.time()
        validation_loss = self.validate(validation_loader, epoch)
        print(f'Training time: {seconds_to_time(valtime_start - traintime_start)}, Validation time: {seconds_to_time(time.time() - valtime_start)}')
        print('Training Loss: {:.6f}, Validation Loss: {:.6f}'.format(training_loss, validation_loss))
        if validation_loss < best_validation_loss:
          best_validation_loss = validation_loss
          best_model = self.model.state_dict()
        losses.append((training_loss, validation_loss))
        torch.cuda.empty_cache()
      self.dataset.clear_cache()
      if self.limit and i == self.limit - 1:
        break
      if self.midsave:
        torch.save(self.model.state_dict(), self.output_folder + f'\\model_{i}_{epoch}.pth')
      self.train_loaders[i] = None
      self.validation_loaders[i] = None
      del train_loader
      del validation_loader

    # Load the best model
    self.model.load_state_dict(best_model)

    # make a directory for the model if it doesn't exist
    os.makedirs(self.output_folder, exist_ok=True)

    # Test the best model
    test_start_time = time.time()
    if len(self.test_loader.dataset) > 0:
      print('2. Testing')
      self.test(self.test_loader, use_xla=False)
    else:
      print(' -- skipping testing')

    # Save the model
    torch.save(self.model.state_dict(), self.output_folder + '\\model.pth')

    # print summary
    print()
    print('Done')
    print()
    print(f'Time: {seconds_to_time(time.time() - start_time)}')
    print(f'(trainig: {seconds_to_time(sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]))}, testing: {seconds_to_time(time.time() - test_start_time)})')
    print(f'Average Epoch Time: {seconds_to_time(sum([epoch_start_times[i + 1] - epoch_start_times[i] for i in range(len(epoch_start_times) - 1)]) / len(epoch_start_times))}')
    print(f'Best Validation Loss: {best_validation_loss}')

    # Plot the losses as a function of epoch
    ModelVisualizer(self.model).show_losses(losses, self.output_folder + '\\losses.png')

    # data loaders initialization

  def init_dataloaders (self):
    split_dataset_size = int(len(self.dataset) / self.split)
    train_size = int(split_dataset_size * TRAINING_PERCENTAGE)
    validation_size = int(split_dataset_size * VALIDATION_PERCENTAGE)
    test_size = len(self.dataset) - (train_size + validation_size) * self.split

    split_sizes = [train_size, validation_size] * self.split + [test_size]
    datasets = random_split(self.dataset, split_sizes)

    self.train_loaders, self.validation_loaders = [], []
    for i in range(self.split):
      self.train_loaders.append(self.generate_dataloader(datasets[i * 2]))
      self.validation_loaders.append(self.generate_dataloader(datasets[i * 2 + 1]))
    self.test_loader = self.generate_dataloader(datasets[-1])

  def generate_dataloader (self, dataset):
    num_workers = int(self.options.get('num_workers', 0))
    pin_memory = num_workers > 0 and self.device.type == 'cpu'
    batch_size = int(self.options.get('batch_size', BATCH_SIZE))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=self.persistent_workers)

  # preloading

  def partial_preload (self, loader, message='Preloading'):
    dataset = loader.dataset
    def run (next):
      for index in range(len(dataset)):
        dataset[index]
        next(1)
    long_operation(run, max=len(dataset), message=message)

  # preocedures

  def train(self,training_loader, epoch):
    self.model.train()

    def run (next):
      total_loss = 0
      for batch_idx, (input, target) in enumerate(training_loader):
        self.optimizer.zero_grad()
        output, loss = self.calc(input, target)
        loss.backward()
        self.optimizer.step()
        next(self.batch_size)
        total_loss += loss.item()
      return total_loss

    total_loss = long_operation(run, max=len(training_loader) * self.batch_size, message=f'Epoch {epoch+1} training', ending_message=lambda l: f'loss: {l / len(training_loader):.4f}')
    return total_loss / len(training_loader)

  def validate(self, validation_loader, epoch):
    self.model.eval()

    with torch.no_grad():
      def run (next):
        total_loss = 0
        for batch_idx, (input, target) in enumerate(validation_loader):
          output, loss = self.calc(input, target)
          next(self.batch_size)
          total_loss += loss.item()
        return total_loss
    
      total_loss = long_operation(run, max=len(validation_loader) * self.batch_size, message=f'Epoch {epoch+1} validation', ending_message=lambda l: f'loss: {l / len(validation_loader):.4f}')
    return total_loss / len(validation_loader)

  def test(self):
    self.model.eval()
    outputs, targets = [], []

    with torch.no_grad():
      def run (next):
        total_loss = 0
        for batch_idx, (input, target) in enumerate(self.test_loader):
          output, loss = self.calc(input, target)
          next(self.batch_size)
          for index, (output, target) in enumerate(zip(output, target)):
            outputs.append(output)
            targets.append(target)
          total_loss += loss.item()
        return total_loss
      total_loss = long_operation(run, max=len(self.test_loader) * self.batch_size, message='Testing ')
    print(f'\nTest set average loss: {total_loss / len(self.test_loader):.4f}\n')

    if self.use_xla or self.use_cuda:
      outputs = [output.cpu() for output in outputs]
      targets = [target.cpu() for target in targets]
    ModelVisualizer(self.model).plot_results(outputs, targets, self.test_loader, self.dataset, self.output_folder + '\\testing.png')

  def calc (self, input, target):
    input = input.to(self.device, non_blocking=True)
    target = target.to(self.device, non_blocking=True)
    output = self.model(input)
    loss = self.criterion(output, target)
    return output, loss
