import torch

class CylindricalLoss (torch.nn.Module):
  def __init__(self):
    super(CylindricalLoss, self).__init__()

  def forward(self, output, target):
    output = output.view(-1, 2)
    target = target.view(-1, 2)

    linear_diff = torch.abs(output[:, 0] - target[:, 0])
    circular_diff = torch.remainder(output[:, 1] - target[:, 1] + torch.pi, 2 * torch.pi) - torch.pi

    distances = torch.sqrt(circular_diff ** 2 + linear_diff ** 2)

    return torch.mean(distances)
