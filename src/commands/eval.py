import torch
import numpy as np
import os

from visualization.model_visualizer import ModelVisualizer
from utils import long_operation
from settings import BATCH_SIZE

def evaluate (dataset, module, model_file, output_folder, params):
  module.load_state_dict(torch.load(model_file))
  n = int(params.get('n', '1000'))
  batch_size = int(params.get('batch_size', BATCH_SIZE))
  module.eval()
  print()
  print('1. Loading data')
  events, inputs, targets = load_data(dataset, n)
  print('2. Running model')
  outputs = run_model(module, inputs, batch_size)
  print('3. Visualizing results')
  visualizer = ModelVisualizer(module, show=False)
  visualizer.show_reconstruction_rate_stats(outputs, targets, events, output_folder)
  sample_event_index = int(params.get('sample_event', np.random.randint(0, len(events))))
  visualizer.sample_event_plot(events[sample_event_index], targets[sample_event_index], outputs[sample_event_index], output_file=os.path.join(output_folder, 'sample_event.png'))
  print('4. Done')

def load_data (dataset, n):
  random_indices = torch.randperm(len(dataset))[:n]
  
  def load (next):
    events = []
    inputs = []
    targets = []
    for i in random_indices:
      events.append(dataset.get_event(i))
      inputs.append(dataset[i][0])
      targets.append(dataset[i][1])
      next()
    return events, inputs, targets

  events, inputs, targets = long_operation(load, max=n, message='Loading data')
  inputs = torch.stack(inputs)
  targets = torch.stack(targets)
  return events, inputs, targets

def run_model (module, inputs, batch_size):
  def run (next):
    outputs = []
    with torch.no_grad():
      for i in range(0, len(inputs), batch_size):
        output = module(inputs[i:i + batch_size])
        outputs.append(output)
        next(batch_size)
      return torch.cat(outputs)
  return long_operation(run, max=len(inputs), message='Running model')