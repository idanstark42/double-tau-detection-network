import torch
from torch import nn
from dropblock import DropBlock2D

from model.cylinrical_conv import CylindricalConv2d
from model.attention import AttentionLayer
from settings import RESOLUTION

class MainModel (nn.Module):
  def __init__(self, input_channels=10, dropout_probability=0.15, post_processing=False, model='small'):
    super(MainModel, self).__init__()
    self.dropout_probability = dropout_probability
    self.input_channels = input_channels
    self.post_processing = post_processing
    self.model_name = model

    model = self.model(model)
    self.conv_layers = model['conv_layers']
    self.connection_size = model['connection_size']
    self.linear_layers = model['linear_layers']

    self.print_parameters()
  
  def forward(self, x):
    for layer in self.conv_layers:
      x = layer(x)
    
    x = x.view(-1, self.connection_size)
    
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

  def print_parameters(self):
    convulational_params = sum(p.numel() for p in self.conv_layers.parameters() if p.requires_grad)
    linear_params = sum(p.numel() for p in self.linear_layers.parameters() if p.requires_grad)
    print(f"Initialized NN.")
    print(f"Model:                            {self.model_name}")
    print(f"Dropout probability:              {self.dropout_probability}")
    print(f"Post processing:                  {self.post_processing}")
    print(f"Convolutional layers parameters:  {convulational_params}")
    print(f"Linear layers parameters:         {linear_params}")
    print(f"Total parameters:                 {convulational_params + linear_params}")

  def mini_model(self):
    return {
      'conv_layers': nn.ModuleList([
        self.conv_block(self.input_channels, 16, kernel_size=3, padding=1, drop=True),
        nn.AvgPool2d(2),
        self.conv_block(16, 32, kernel_size=3, padding=1),
        AttentionLayer(),
        nn.AvgPool2d(2),
        self.conv_block(32, 32, kernel_size=3, padding=1),
        self.conv_block(32, 32, kernel_size=3, padding=1),
        AttentionLayer(),
        nn.AvgPool2d(5),
        self.conv_block(32, 16, kernel_size=3, padding=1),
        AttentionLayer()
      ]),
      'connection_size': 16 * (RESOLUTION // 20) * (RESOLUTION // 20),
      'linear_layers': nn.ModuleList([
        nn.Linear(16 * (RESOLUTION // 20) * (RESOLUTION // 20), 64),
        nn.PReLU(),
        nn.Dropout(self.dropout_probability),
        nn.Linear(64, 4)
      ])
    }
  
  def jet_ssd_model(self):
    return {
      'conv_layers': nn.ModuleList([
        self.conv_block(self.input_channels, 32, kernel_size=3, padding=1, stride=1, bias=False, drop=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        self.conv_block(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        self.conv_block(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        self.conv_block(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        self.conv_block(512, 1024, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False),
      ]),
      'connection_size': 1024 * 9 ** 2,
      'linear_layers': nn.ModuleList([
        nn.Linear(1024 * 9 ** 2, 4)
      ])
    }
  
  def jet_ssd_min_model(self):
    return {
      'conv_layers': nn.ModuleList([
        self.conv_block(self.input_channels, 32, kernel_size=3, padding=1, stride=1, bias=False, drop=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        self.conv_block(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        self.conv_block(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
        self.conv_block(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        self.conv_block(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
      ]),
      'connection_size': 256 * 17 ** 2,
      'linear_layers': nn.ModuleList([
        nn.Linear(256 * 17 ** 2, 4)
      ])
    }
  
  def conv_block(self, input_channels, output_channels, kernel_size, padding, bias=False, drop=False, stride=1):
    layers = (
      CylindricalConv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
      nn.BatchNorm2d(output_channels),
    )
    if drop:
      layers = layers + (DropBlock2D(block_size=3, drop_prob=0.1),)
    layers = layers + (nn.PReLU(),)

    return nn.Sequential(*layers)
  
  def model (self, name):
    return {
      'small': self.mini_model,
      'jet_ssd': self.jet_ssd_model,
      'jet_ssd_min': self.jet_ssd_min_model
    }[name]()