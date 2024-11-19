import torch

from visualization.model_visualizer import ModelVisualizer

def evaluate (dataset, module, model_file, output_folder, params):
  module.load_state_dict(torch.load(model_file))
  n = int(params.get('n', '1000'))
  random_indices = torch.randperm(len(dataset))[:n]
  module.eval()
  print()
  print('Calculating outputs...')
  with torch.no_grad():
    inputs, targets = zip(*[dataset[i] for i in random_indices])
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    outputs = module(inputs)

    visualizer = ModelVisualizer()
    visualizer.show_reconstruction_rate_stats(outputs, targets, dataset, output_folder)