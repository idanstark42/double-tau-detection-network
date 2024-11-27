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

def train(dataset, model, model_folder, options={}):
  if 'checkpoint' in options:
    if 'backup_folder' in options:
      model_folder_ending = model_folder.split('/')[-1]
      os.makedirs(model_folder, exist_ok=True)
      os.system(f'cp -r {options["backup_folder"]}/{model_folder_ending}/* {model_folder}')
    checkpoint = Trainer.last_checkpoint(model_folder) if options['checkpoint'] == 'true' else os.path.join(model_folder, 'checkpoints', options['checkpoint'])
    trainer = Trainer.from_checkpoint(dataset, model, checkpoint)
  else:
    trainer = Trainer(dataset, model, model_folder, options)
  trainer.train_model()

class Trainer:

  # initializers

  def __init__(self, dataset, model, model_folder, options):
    self.dataset = dataset
    self.model = model
    self.model_folder = model_folder
    self.options = options
    self.load_options()
    self.load_initial_state()

  def load_options(self):
    self.backup_folder = self.options.get('backup_folder', '')

    self.loading_type = self.options.get('preload', 'none')
    self.preloading_output = self.options.get('preload_output', 'none')
    self.saving_mode = self.options.get('saving_mode', 'none')
    self.cache_type = self.options.get('cache', 'events')
    self.checkpoint = False

    self.split = int(self.options.get('split', '1'))
    self.limit = int(self.options.get('limit')) if self.options.get('limit') and self.split != 1 else None
    self.epochs = int(self.options.get('epochs', EPOCHS))
    self.batch_size = int(self.options.get('batch_size', BATCH_SIZE))

    self.use_xla = self.options.get('use_xla', 'false') == 'true'
    self.persistent_workers = self.options.get('persistent_workers', 'false') == 'true'
    self.num_workers = int(self.options.get('num_workers', 0))
    self.using_multiprocessing = self.num_workers > 0

    self.learning_rate = float(self.options.get('learning_rate', 0.001))
    self.weight_decay = float(self.options.get('weight_decay', 0.0001))
    self.initial_weights = self.options.get('start_from', None)

  def load_initial_state(self):
    self.position = { 'split': 0, 'epoch': 0 }
    self.best_validation_loss = float('inf')
    self.best_model = None
    self.losses = []
    self.epoch_start_times = []
    self.pretraining_over = False

  def train_model(self):
    self.pretraining()
    print('Training')
    splits = range(self.position['split'] + 1, self.split) if self.checkpoint else range(self.split)
    for split in splits:
      self.run_split_training(split)
      if self.limit and split == self.limit - 1:
        break
    self.posttraining()

  # main procedure

  def pretraining(self):
    if self.pretraining_over:
      return

    self.start_time = time.time()

    self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    self.criterion = CylindricalLoss()
    self.init_device()
    self.init_dataloaders()

    if self.initial_weights:
      models_folder = '/'.join(self.model_folder.split('/')[:-1])
      self.model.load_state_dict(torch.load(os.path.join(models_folder, self.initial_weights)))

    self.print_starting_log()

    if self.loading_type == 'full':
      self.dataset.full_preload()

    self.pretraining_over = True

  def run_split_training(self, split):
    split_start_time = time.time()
    if self.split > 1:
      print(f'Split {split + 1}/{self.limit if self.limit else self.split}')

    if self.loading_type == 'partial':
      self.partial_preload(self.train_loaders[split], self.validation_loaders[split])

    if self.checkpoint and self.position['split'] == split and self.saving_mode.startswith('epoch-'):
      epochs = range(self.position['epoch'] + 1, self.epochs)
    else:
      epochs = range(self.epochs)
    for epoch in epochs:
      self.run_epoch_training(split, epoch)

    self.train_loaders[split] = None
    self.validation_loaders[split] = None
    self.dataset.clear_cache()

    if self.split > 1:
      print(f'Split time: {seconds_to_time(time.time() - split_start_time)}/{seconds_to_time((time.time() - self.start_time) * (self.limit if self.limit else self.split) / (split + 1))} estimated')

    if self.saving_mode == 'split':
      self.save_checkpoint(f'split-{split + 1}')

  def run_epoch_training(self, split, epoch):
    self.epoch_start_times.append(time.time())

    training_loss = self.train(self.train_loaders[split], epoch)
    validation_loss = self.validate(self.validation_loaders[split], epoch)

    self.save_if_best(validation_loss)
    self.losses.append((training_loss, validation_loss))

    torch.cuda.empty_cache()
    self.position = { 'split': split, 'epoch:' : epoch }

    if self.saving_mode.startswith('epoch-') and (epoch + 1) % int(self.saving_mode[6:]) == 0:
      self.save_checkpoint(f'epoch-{epoch + 1}')

  def posttraining(self):
    self.model.load_state_dict(self.best_model)

    self.test_start_time = time.time()
    if len(self.test_loader.dataset) > 0:
      print('Testing')
      self.test()
    else:
      print(' -- skipping testing')

    self.save_model()
    self.print_summary()

  # training building blocks

  def train(self, training_loader, epoch):
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

    total_loss = long_operation(run, max=len(training_loader) * self.batch_size, message=f'Epoch {epoch+1} training  ', ending_message=lambda l, t: f'loss: {l / len(training_loader):.6f} [{seconds_to_time(t)}]')
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

      total_loss = long_operation(run, max=len(validation_loader) * self.batch_size, message=f'Epoch {epoch+1} validation', ending_message=lambda l, t: f'loss: {l / len(validation_loader):.6f} [{seconds_to_time(t)}]')
    return total_loss / len(validation_loader)

  def test(self):
    if self.loading_type == 'partial':
      self.dataset.start_partial_preloading()
      self.preload_loader(self.test_loader, 'Preloading test set')
      self.dataset.finish_partial_preloading()

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

    self.print_test_summary(outputs, targets, total_loss)

  # proess initialization

  def init_device (self):
    self.use_cuda = torch.cuda.is_available()
    if self.use_cuda:
      # spawen start method to avoid error
      torch.multiprocessing.set_start_method('spawn')
      torch.multiprocessing.set_sharing_strategy('file_system')
      self.model = self.model.cuda()
    self.device = torch.device('cuda' if self.use_cuda else 'cpu')

  def init_dataloaders (self):
    split_dataset_size = int(len(self.dataset) / self.split)
    self.training_loader_size = int(split_dataset_size * TRAINING_PERCENTAGE)
    self.validation_loader_size = int(split_dataset_size * VALIDATION_PERCENTAGE)
    test_size = len(self.dataset) - (self.training_loader_size + self.validation_loader_size) * self.split
    if self.limit:
      test_size = (test_size * self.limit) // self.split

    leftover = len(self.dataset) - (self.training_loader_size + self.validation_loader_size) * self.split - test_size

    datasets = random_split(self.dataset, [self.training_loader_size, self.validation_loader_size] * self.split + [test_size, leftover])

    self.train_loaders, self.validation_loaders = [], []
    for i in range(self.split):
      self.train_loaders.append(self.generate_dataloader(datasets[i * 2]))
      self.validation_loaders.append(self.generate_dataloader(datasets[i * 2 + 1]))
    self.test_loader = self.generate_dataloader(datasets[-2])
    self.unused_indices = datasets[-1].indices

  def generate_dataloader (self, dataset):
    num_workers = int(self.options.get('num_workers', 0))
    pin_memory = num_workers > 0 and self.device.type == 'cpu'

    return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=self.persistent_workers)

  # preloading

  def partial_preload (self, training_loader, validation_loader):
    preload_start_time = time.time()
    self.dataset.start_partial_preloading()
    self.preload_loader(training_loader, 'Preloading training set')
    self.preload_loader(validation_loader, 'Preloading validation set')
    self.dataset.finish_partial_preloading()
    print(f'Preloading time: {seconds_to_time(time.time() - preload_start_time)}')

  def preload_loader (self, loader, message):
    dataset = loader.dataset
    if self.preloading_output == 'mass':
      masses = { f'{20 + 10 * i} GeV': 0 for i in range(5) }
    def run (next):
      for index in range(len(dataset)):
        if self.preloading_output == 'mass':
          mass = self.dataset.get_event(dataset.indices[index]).mass_by_channel_number()
          if mass in masses:
            masses[mass] += 1
        dataset[index]
        next(1)
    long_operation(run, max=len(dataset), message=message)

    if self.preloading_output == 'mass':
      print('Mass distribution:')
      for mass, count in masses.items():
        print(f'{mass}: {count}')

  # helper functions

  def calc (self, input, target):
    input = input.to(self.device, non_blocking=True)
    target = target.to(self.device, non_blocking=True)
    output = self.model(input)
    loss = self.criterion(output, target)
    return output, loss

  def save_if_best(self, validation_loss):
    if validation_loss < self.best_validation_loss:
      self.best_validation_loss = validation_loss
      self.best_model = self.model.state_dict()

  # logging & visualization

  def print_starting_log (self):
    print()
    print(f'Training set size:                {sum([len(loader.dataset) if loader else self.training_loader_size for loader in self.train_loaders])}')
    print(f'Validation set size:              {sum([len(loader.dataset) if loader else self.validation_loader_size for loader in self.validation_loaders])}')
    print(f'Test set size:                    {len(self.test_loader.dataset)}')
    print(f'Split:                            {self.split}')
    print('Limit:                            ' + (f'{self.limit} [{int(100 * self.limit / self.split)}%]' if self.limit else 'none'))
    print(f'Batch Size:                       {self.batch_size}')
    print(f'Epochs:                           {self.epochs}')
    print()
    print(f'Saving Mode:                      {self.saving_mode}')
    if self.checkpoint:
      print(f'From checkpoint:                  {self.checkpoint}')
      print(f'Starting from:                   {self.position["split"] + 2}/{self.split} split, {(self.position["epoch"]) + 2}/{self.epochs} epoch')
    print(f'Preload Type:                     {self.loading_type}')
    print(f'Cache:                            {self.cache_type}')
    print(f'Model Folder:                     {self.model_folder}')
    print(f'Backup Folder:                    {self.backup_folder}')
    print()
    print('Using Multiprocessing:            ' + ('yes' if self.using_multiprocessing else 'no'))
    if self.using_multiprocessing:
      print(f'Number of Workers:                {int(self.num_workers)}')
      print(f'Persistent Workers:               ' + ('yes' if self.persistent_workers else 'no'))
    print('Using Device:                     ' + (torch.cuda.get_device_name(0) if self.use_cuda else 'CPU'))
    print()
    print(f'Learning Rate:                    {self.learning_rate}')
    print(f'Weight Decay:                     {self.weight_decay}')
    if self.initial_weights:
      print(f'Starting from:                    {self.initial_weights}')
    print()

  def print_test_summary (self, outputs, targets, total_loss):
    print(f'\nTest set average loss: {total_loss / len(self.test_loader):.4f}\n')
    if self.use_xla or self.use_cuda:
      outputs = [output.cpu() for output in outputs]
      targets = [target.cpu() for target in targets]
    events = [self.dataset.get_event(index) for index in self.test_loader.dataset.indices]
    ModelVisualizer(self.model).plot_results(outputs, targets, events, os.path.join(self.model_folder, 'graphs.png'))

  def print_summary (self):
    print()
    print('Done')
    print()
    print(f'Time: {seconds_to_time(time.time() - self.start_time)}')
    print(f'(trainig: {seconds_to_time(sum([self.epoch_start_times[i + 1] - self.epoch_start_times[i] for i in range(len(self.epoch_start_times) - 1)]))}, testing: {seconds_to_time(time.time() - self.test_start_time)})')
    print(f'Average Epoch Time: {seconds_to_time(sum([self.epoch_start_times[i + 1] - self.epoch_start_times[i] for i in range(len(self.epoch_start_times) - 1)]) / len(self.epoch_start_times))}')
    print(f'Best Validation Loss: {self.best_validation_loss}')

    # Plot the losses as a function of epoch
    ModelVisualizer(self.model).show_losses(self.losses, os.path.join(self.model_folder, 'losses.png'))

  # io operations

  def save_model (self):
    os.makedirs(self.model_folder, exist_ok=True)
    torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'model.pth'))

  def save_checkpoint(self, name):
    checkpoint = {
      'options': self.options,
      'model_folder': self.model_folder,
      'model': self.model.state_dict(),
      'position': self.position,
      'optimizer': self.optimizer.state_dict(),
      'training_loaders': [(loader.dataset.indices if loader else None) for loader in self.train_loaders],
      'validation_loaders': [(loader.dataset.indices if loader else None) for loader in self.validation_loaders],
      'test_loader': self.test_loader.dataset.indices,
      'name': name if name else f'checkpoint-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
      'epoch_start_times': self.epoch_start_times,
      'losses': self.losses,
      'best_validation_loss': self.best_validation_loss,
      'best_model': self.best_model,
      'backup_folder': self.backup_folder,
    }
    os.makedirs(self.model_folder, exist_ok=True)
    os.makedirs(os.path.join(self.model_folder, 'checkpoints'), exist_ok=True)
    torch.save(checkpoint, os.path.join(self.model_folder, 'checkpoints', f'{name}.pth'))

    if self.backup_folder:
      model_folder_ending = self.model_folder.split('/')[-1]
      os.makedirs(os.path.join(self.backup_folder, model_folder_ending), exist_ok=True)
      os.system(f'cp -r {self.model_folder}/* {os.path.join(self.backup_folder, model_folder_ending)}')

  @staticmethod
  def from_checkpoint(dataset, model, checkpoint_file, options={}):
    checkpoint = torch.load(checkpoint_file)
    model_folder = checkpoint['model_folder'] if 'model_folder' in checkpoint else checkpoint['output_folder'] # backward compatibility
    trainer = Trainer(dataset, model, model_folder, { **checkpoint['options'], **options })
    trainer.model.load_state_dict(checkpoint['model'])
    trainer.position = checkpoint['position']
    if 'epoch:' in trainer.position: # backward compatibility
      trainer.position['epoch'] = trainer.position.pop('epoch:')
    trainer.epoch_start_times = checkpoint['epoch_start_times']
    trainer.losses = checkpoint['losses']
    trainer.best_validation_loss = checkpoint['best_validation_loss']
    trainer.best_model = checkpoint['best_model']
    trainer.checkpoint = checkpoint['name']

    # doing the part of the pretraining that is different from the normal pretraining
    trainer.pretraining()
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.train_loaders = [None if indices is None else trainer.generate_dataloader(torch.utils.data.Subset(dataset, indices)) for indices in checkpoint['training_loaders']]
    trainer.validation_loaders = [None if indices is None else trainer.generate_dataloader(torch.utils.data.Subset(dataset, indices)) for indices in checkpoint['validation_loaders']]
    trainer.test_loader = trainer.generate_dataloader(torch.utils.data.Subset(dataset, checkpoint['test_loader']))

    return trainer

  @staticmethod
  def last_checkpoint(model_folder):
    return max([os.path.join(model_folder, 'checkpoints', file) for file in os.listdir(os.path.join(model_folder, 'checkpoints'))], key=os.path.getctime)
