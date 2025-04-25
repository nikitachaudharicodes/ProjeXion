from pathlib import Path
from typing import Tuple
import torch

def parse_cam(cam_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
   """
   :return K: Camera intrinsic matrix (3x3)
   :return M: Camera extrinsic matrix in homogeneous form (4x4)
   """
   with open(cam_path, 'r') as cam_file:
      # Extrinsic
      assert cam_file.readline().strip() == 'extrinsic'
      extrinsic = [
         [float(x) for x in cam_file.readline().strip().split()]
         for i in range(4)
      ]
      extrinsic = torch.tensor(extrinsic)

      # Empty line
      assert cam_file.readline().strip() == ''

      # Intrinsic
      assert cam_file.readline().strip() == 'intrinsic'
      intrinsic = [
         [float(x) for x in cam_file.readline().strip().split()]
         for i in range(3)
      ]
      intrinsic = torch.tensor(intrinsic)
   return intrinsic, extrinsic
