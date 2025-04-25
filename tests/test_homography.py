import torch
from models.homografy import Homography

def test_homography():
   homography = Homography(0, 2, 1)
   B, T, C, H, W  = 5, 3, 32, 500, 600
   X = torch.rand((B, T, C, H, W))
   intrinsic = torch.rand((B, T, 3, 3))
   extrinsic = torch.rand((B, T, 4, 4))
   X = homography(X, intrinsic, extrinsic)
   assert X.shape == (B, T, 2, C, H, W)