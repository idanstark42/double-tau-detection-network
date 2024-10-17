# an MSE loss with one of the dimensions is a circle and the other is a line

import torch

from data.position import Position

class CylindricalLoss (torch.nn.Module):
  def __init__(self):
    super(CylindricalLoss, self).__init__()

  def forward(self, output, target):
    target_positions = [Position(tau[0], tau[1]) for tau in target.view(-1, 2)]
    output_positions = [Position(tau[0], tau[1]) for tau in output.view(-1, 2)]
    distances = [target_position.distance(output_position) for target_position, output_position in zip(target_positions, output_positions)]
    distances = torch.tensor(distances, requires_grad=True)
    return torch.mean(distances)