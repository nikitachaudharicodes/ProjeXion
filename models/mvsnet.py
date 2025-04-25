import torch
from .sublayers import CNNBlock
from .layers.feature_extraction import ImageEncoder
from .layers.homografy import Homography
from .layers.variance import VarianceLayer
from .layers.cost_regularizer import CostRegularizer
from .layers.soft_argmin import SoftArgmin
from .layers.refinement import Refine

class MVSNet(torch.nn.Module):
   def __init__(self, depth_start, depth_end, depth_step):
      super().__init__()
      self.depths = torch.arange(depth_start, depth_end, depth_step)
      self.feature_extractor = ImageEncoder()
      self.homography = Homography(depths=self.depths)
      self.variance = VarianceLayer()
      self.cost_regularizer = CostRegularizer(in_channels=32)
      self.soft_argmin = SoftArgmin()
      self.refinement = Refine()

   def forward(self, images, intrinsics, extrinsics):
      """
      :param images: (N, T, C, H, W)
      """
      image_features = self.feature_extractor(images) # (N, T, 32, H, W)
      voxels = self.homography(image_features, intrinsics, extrinsics) # (N, T, C, D, H, W)
      voxels = self.variance(voxels) # (N, C, D, H, W)
      voxels = self.cost_regularizer(voxels) # (N, 1, D, H, W)
      initial_depth_map = self.soft_argmin(voxels, self.depths)
      refined_depth_map = self.refinement(images, initial_depth_map)
      return initial_depth_map, refined_depth_map
   