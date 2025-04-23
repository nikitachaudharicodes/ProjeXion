import torch 
import torch.nn as nn

class CNNBlock(torch.nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super().__init__()
      self.layers = torch.nn.Sequential(
         torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
         torch.nn.BatchNorm2d(out_channels),
         torch.nn.PReLU(),
      )
   def forward(self, X):
      return self.layers(X)