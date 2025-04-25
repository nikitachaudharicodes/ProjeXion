import torch
from .sublayers import CNNBlock
from .layers.feature_extraction import ImageEncoder
from .layers.homografy import Homography
from .layers.variance import VarianceLayer
from .layers.cost_regularizer import CostRegularizer
from .layers.soft_argmin import SoftArgmin
from .layers.refinement import Refine

class MVSNet(torch.nn.Module):
   def __init__(self, n_depths):
      super().__init__()
      self.depths = torch.arange(0, 1.0001, 1 / n_depths)
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
      N, T, C, H, W = images.shape
      image_features = self.feature_extractor(images.reshape((N*T, C, H, W))) # (N*T, 32, H, W)
      image_features = image_features.reshape(((N, T, 32, H // 4, W // 4)))
      voxels = self.homography(image_features, intrinsics, extrinsics) # (N, T, C, D, H, W)
      voxels = self.variance(voxels) # (N, C, D, H, W)
      voxels = self.cost_regularizer(voxels) # (N, 1, D, H, W)
      initial_depth_map = self.soft_argmin(voxels, self.depths) # (N, 1, H, W)
      refined_depth_map = self.refinement(images[:, 0], initial_depth_map)
      return initial_depth_map, refined_depth_map
   