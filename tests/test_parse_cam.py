from utils import parse_cam
import torch

def test_parse_cam():
   cam_path = 'data_sample/BlendedMVS/dataset_low_res/5a3ca9cb270f0e3f14d0eddb/cams/00000000_cam.txt'
   intrinsic, extrinsic = parse_cam(cam_path=cam_path)
   reference_intrinsic = torch.tensor([
      [565.665, 0.0, 380.88],
      [0.0, 565.665, 287.361],
      [0.0, 0.0, 1.0],
   ])
   reference_extrinsic = torch.tensor([
      [0.443983, -0.0398572, 0.895148, -0.0228101],
      [0.237827, 0.96842, -0.0748397, -0.317713],
      [-0.863897, 0.246118, 0.439441, 0.550928],
      [0.0, 0.0, 0.0, 1.0],
   ])
   assert torch.all(intrinsic == reference_intrinsic)
   assert torch.all(extrinsic == reference_extrinsic)
