import torch

class MaskedMSELoss(torch.nn.Module):
   def __init__(self):
      super().__init__()
   def forward(self, input, target, mask):
      # assert input.shape == target.shape == mask.shape, f"Shapes of input {input.shape}, target {target.shape} and mask {mask.shape} must match"
      mean_squared_error = (((input - target) * mask) ** 2).sum() / mask.sum()
      return mean_squared_error
