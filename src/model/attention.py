import torch
from torch import nn

class AttentionLayer(nn.Module):

    def __init__(self):
        super(AttentionLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        weights = self.avg_pool(input)
        weights = self.conv1d(weights.squeeze(-1).transpose(-1, -2))
        weights = weights.transpose(-1, -2).unsqueeze(-1)
        weights = self.sigmoid(weights)

        return input * weights.expand_as(input)