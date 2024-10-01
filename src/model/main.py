import torch
from torch import nn
from dropblock import DropBlock2D

from model.cylinrical_conv import CylindricalConv2d
from settings import RESOLUTION

class MainModel (nn.Module):
  def __init__(self, input_channels=2, dropout_probability=0.15, post_processing=False):
    super(MainModel, self).__init__()
    self.dropout_probability = dropout_probability
    self.input_channels = input_channels
    self.post_processing = post_processing

    def conv_block(input_channels, output_channels, kernel_size, padding, stride, bias, drop=False):
      layers = (
        CylindricalConv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
        nn.BatchNorm2d(output_channels),
      )
      if drop:
        layers = layers + (DropBlock2D(block_size=3, drop_prob=0.1),)
      layers = layers + (nn.PReLU(),)

      return nn.Sequential(*layers)
    
    self.conv_layers = nn.ModuleList([
      conv_block(self.input_channels, 32, kernel_size=3, padding=1, stride=1, bias=False, drop=True),
      nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
      conv_block(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
      conv_block(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
      conv_block(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
      nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
      conv_block(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
      conv_block(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
      conv_block(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
      nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
      conv_block(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
      conv_block(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
      nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
      conv_block(512, 1024, kernel_size=3, padding=1, stride=1, bias=False),
      conv_block(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False),
    ])

    self.linear_layers = nn.ModuleList([
      nn.Linear(1024 * 9 ** 2, 4)
    ])

    # count and print the number of parameters in the network
    convulational_params = sum(p.numel() for p in self.conv_layers.parameters() if p.requires_grad)
    linear_params = sum(p.numel() for p in self.linear_layers.parameters() if p.requires_grad)
    print(f"Initialized NN.")
    print(f"Convolutional layers parameters:  {convulational_params}")
    print(f"Linear layers parameters:         {linear_params}")
    print(f"Total parameters:                 {convulational_params + linear_params}")

  def forward(self, x):
    for layer in self.conv_layers:
      x = layer(x)
    
    x = x.view(-1, 1024 * 9 ** 2)
    
    for layer in self.linear_layers:
      x = layer(x)

    if self.post_processing != False:
      # if the tensor has more then one dimensions, it is a batch of tensors, iterate over the batch
      if len(x.size()) > 1:
        for i in range(x.size()[0]):
          x[i] = self.post_processing(x[i])
      else:
        x = self.post_processing(x)
    
    return x
