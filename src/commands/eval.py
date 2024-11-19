import torch

from visualization.model_visualizer import ModelVisualizer

def evaluate (dataset, module, model_file, output_folder, params):
  module.load_state_dict(torch.load(model_file))
  n = int(params.get('n', '1000'))
  random_indices = torch.randperm(len(dataset))[:n]
  module.eval()
  print()
  with torch.no_grad():
    print('1. Preparing data')
    events = [dataset.get_event(i) for i in random_indices]
    inputs, targets = zip(*[dataset[i] for i in random_indices])
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    print('2. Running model')
    outputs = module(inputs)
    print('3. Visualizing results')
    visualizer = ModelVisualizer(module)
    visualizer.show_reconstruction_rate_stats(outputs, targets, events, output_folder)
    print('4. Done')