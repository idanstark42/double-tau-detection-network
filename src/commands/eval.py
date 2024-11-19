import torch

from visualization.model_visualizer import ModelVisualizer
from utils import long_operation

def evaluate (dataset, module, model_file, output_folder, params):
  module.load_state_dict(torch.load(model_file))
  n = int(params.get('n', '1000'))
  module.eval()
  print()
  with torch.no_grad():
    print('1. Loading data')
    events, inputs, targets = load_data(dataset, n)
    print('2. Running model')
    outputs = module(inputs)
    print('3. Visualizing results')
    visualizer = ModelVisualizer(module)
    visualizer.show_reconstruction_rate_stats(outputs, targets, events, output_folder)
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