import torch.nn as nn
import torch.nn.functional as F

def circular_pad_2d(x, padding):
  """
  Apply circular padding to a 4D tensor (batch_size, channels, height, width)
  :param x: input tensor
  :param padding: tuple of (padding_height, padding_width)
  :return: padded tensor
  """
  pad_h, pad_w = padding
  x_padded = F.pad(x, (pad_w, pad_w, 0, 0), mode='circular')
  x_padded = F.pad(x_padded, (0, 0, pad_h, pad_h), mode='constant', value=0)
  return x_padded

class CylindricalConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    super(CylindricalConv2d, self).__init__()
    self.padding = padding
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

  def forward(self, x):
    if self.padding > 0:
      x = circular_pad_2d(x, (self.padding, self.padding))
    return self.conv(x)
