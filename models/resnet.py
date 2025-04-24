import torch
from sublayers import CNNBlock

class ResNet6(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.model = torch.nn.Sequential(
         # Initial block: 3 --> 64 channels
         ResNetBlock(3, 16, kernel_size=3, stride=1),
         ResNetBlock(16, 16, kernel_size=3, stride=1),
         ResNetBlock(16, 16, kernel_size=3, stride=1),
         torch.nn.Conv2d(16, 1, kernel_size=1, stride=1),
      )
   def forward(self, X):
      X = X[:, 0, :, :, :]
      Z = self.model(X)
      return Z
   
class ResNetBlock(torch.nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, stride):
      super().__init__()
      padding = kernel_size // 2
      self.stride = stride
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.cnn_1 = CNNBlock(in_channels, out_channels, kernel_size, stride, padding)
      self.cnn_2 = CNNBlock(out_channels, out_channels, kernel_size, 1, padding)
      self.linear = None
      if in_channels != out_channels:
         self.linear = torch.nn.Linear(in_channels, out_channels)
      
   def forward(self, X):
      # X: (N,  C_in,  H_in,  W_in)
      # Z: (N, C_out, H_out, W_out)
      Z = self.cnn_1(X)
      Z = self.cnn_2(Z)
      # If C_in != C_out, we need to apply a linear transform to C
      if self.linear:
         X = torch.transpose(X, 1, 3)
         X = self.linear(X) # Move channels to the end
         X = torch.transpose(X, 3, 1) # Bring channels to the second dim
      # If (H_in, W_in) != (H_out, W_out) we need to downsample the result
      if self.stride > 1:
         X = X[:, :, ::self.stride, ::self.stride] # (B, C, H, W)
      return Z + X
   
