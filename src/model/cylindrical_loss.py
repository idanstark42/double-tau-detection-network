import torch

class CylindricalLoss (torch.nn.Module):
  def __init__(self):
    super(CylindricalLoss, self).__init__()

  def forward(self, output, target):
    output = output.view(-1, 2)
    target = target.view(-1, 2)

    eta_diff = torch.abs(output[:, 0] - target[:, 0])
    phi_diff = torch.abs(output[:, 1] - target[:, 1])
    if phi_diff > torch.pi:
      phi_diff = 2 * torch.pi - phi_diff

    distances = torch.sqrt(torch.pow(eta_diff, 2) + torch.pow(phi_diff, 2))

    return torch.mean(distances)
