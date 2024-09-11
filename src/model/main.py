import torch
from torch import nn

from model.cylinrical_conv import CylindricalConv2d
from settings import RESOLUTION

class MainModel (nn.Module):
  def __init__(self, input_channels=10, dropout_probability=0.15, post_processing=False):
    super(MainModel, self).__init__()
    self.dropout_probability = dropout_probability
    self.input_channels = input_channels
    self.post_processing = post_processing

    def conv_block(input_channels, output_channels, kernel_size, padding):
      return nn.Sequential(
        CylindricalConv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding),
        nn.PReLU(),
        nn.Dropout2d(self.dropout_probability)
      )
    
    self.conv_layers = nn.ModuleList([
      conv_block(self.input_channels, 16, kernel_size=3, padding=1),
      nn.AvgPool2d(2),
      conv_block(16, 32, kernel_size=3, padding=1),
      nn.AvgPool2d(2),
      conv_block(32, 32, kernel_size=3, padding=1),
      conv_block(32, 32, kernel_size=3, padding=1),
      nn.AvgPool2d(5),
      conv_block(32, 16, kernel_size=3, padding=1),
    ])

    self.linear_layers = nn.ModuleList([
      nn.Linear(16 * (RESOLUTION // 20) * (RESOLUTION // 20), 64),
      nn.PReLU(),
      nn.Dropout(self.dropout_probability),
      nn.Linear(64, 4)
    ])

    # count and print the number of parameters in the network
    convulational_params = sum(p.numel() for p in self.conv_layers.parameters() if p.requires_grad)
    linear_params = sum(p.numel() for p in self.linear_layers.parameters() if p.requires_grad)
    print(f"Initialized NN.")
    print(f"Convolutional layers parameters:  {convulational_params}")
    print(f"Linear layers parameters:         {linear_params}")

  def forward(self, x):
    for layer in self.conv_layers:
      x = layer(x)
    
    x = x.view(-1, 16 * (RESOLUTION // 20) * (RESOLUTION // 20))
    
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
