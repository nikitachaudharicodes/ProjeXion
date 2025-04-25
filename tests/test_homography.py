import torch
from models.layers.homografy import Homography

def test_homography():
   depths = torch.arange(0, 2, 1)
   homography = Homography(depths)
   B, T, C, H, W  = 5, 3, 32, 125, 150
   X = torch.rand((B, T, C, H, W))
   intrinsic = torch.rand((B, T, 3, 3))
   extrinsic = torch.rand((B, T, 4, 4))
   X = homography(X, intrinsic, extrinsic)
   assert X.shape == (B, T, C, 2, H, W)