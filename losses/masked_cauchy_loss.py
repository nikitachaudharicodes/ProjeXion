import torch

class MaskedCauchyLoss(torch.nn.Module):
   def __init__(self, c: float):
      super().__init__()
      self.c = c
   def forward(self, input, target, mask):
      cauchy_loss = (
         self.c ** 2 / 2 * torch.log(1 + ((input - target) / self.c) ** 2 * mask)
      ).sum() / mask.sum()
      return cauchy_loss
