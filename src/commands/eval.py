# This command evaluates the model it was given as a parameter.
# First, it loads the model.
# Then, running it a random set of n events from the give dataset, it the percentage of events for which the distance between the output and the taget is less then 0.2.

import torch
from data.position import Position

def evaluate (model, dataset, n=1000, cutoff=0.2):
  random_indices = torch.randperm(len(dataset))[:n]
  model.eval()
  with torch.no_grad():
    correct = 0
    for i in random_indices:
      input, target = dataset[i]
      output = model(input)

      target_positions = [Position(tau[0], tau[1]) for tau in target.view(-1, 2)]
      output_positions = [Position(tau[0], tau[1]) for tau in output.view(-1, 2)]
      distances = torch.tensor([target_position.distance(output_position) for target_position, output_position in zip(target_positions, output_positions)])
      if torch.all(distances < cutoff):
        correct += 1
    print(f'Accuracy: {100 * correct/n:.2f}%')