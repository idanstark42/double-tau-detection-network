# This command evaluates the model it was given as a parameter.
# First, it loads the model.
# Then, running it a random set of n events from the give dataset, it the percentage of events for which the distance between the output and the taget is less then 0.2.

import torch
from data.position import Position

def evaluate (dataset, module, model_file, params):
  module.load_state_dict(torch.load(model_file))
  n = params.get('n', 1000)
  cutoff = params.get('cutoff', 0.2)
  random_indices = torch.randperm(len(dataset))[:n]
  module.eval()
  print()
  print('Calculating outputs...')
  with torch.no_grad():
    correct = 0
    input_target_touples = [dataset[i] for i in random_indices]
    input = torch.stack([input for input, _ in input_target_touples])
    target = torch.stack([target for _, target in input_target_touples])
    output = module(input)
    print('Evaluating model accuracy...')
    for i in range(n):
      current_target = target[i]
      current_output = output[i]

      target_positions = [Position(tau[0], tau[1]) for tau in current_target.view(-1, 2)]
      output_positions = [Position(tau[0], tau[1]) for tau in current_output.view(-1, 2)]

      distances = torch.tensor([target_positions[i].distance(output_positions[i]) for i in range(len(target_positions))])
      if torch.all(distances < cutoff):
        correct += 1

    print('Done.')
    print()
    print(f'Accuracy (< {cutoff}): {100 * correct/n:.2f}%')