import torch 
import torch.nn as nn
from torchvision.transforms.v2 import Resize
from torchvision.transforms import InterpolationMode

class Refine(nn.Module):
    """
    Depth refinement module.

    Takes an inital depth map and the reference image as input and refines as the depth map 
    using a residual learning strategy.
    """

    def __init__(self):
        super(Refine, self).__init__()
        self.refine_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, ref_img, init_depth):
        """
        Forward pass of the refinement module.

        Args:
            ref_img (torch.Tensor): Reference image of shape (N, 3, H, W).
            init_depth (torch.Tensor): Initial depth map of shape (N, C, H, W).

        Returns:
            torch.Tensor: Refined depth map of shape (N, 1, H, W).
        """
        #  Resize reference image
        N, C, H, W = init_depth.shape
        resize_transform = Resize((H, W), InterpolationMode.BILINEAR)
        resized_reference_image = resize_transform(ref_img)
        min = init_depth.view(N, C, H*W).min(dim=-1)[0] # (N, C)
        max = init_depth.view(N, C, H*W).max(dim=-1)[0] # (N, C)
        scale = max - min
        scale[scale == 0] = (1e-7)
        normalized_depth_map = (init_depth - min.view(N, C, 1, 1)) / scale.view(N, C, 1, 1)
            
        x = torch.concat((resized_reference_image, normalized_depth_map), dim=1)  # Concatenate along channel dimension
        delta_depth = self.refine_net(x)  # Pass through the refinement network

        normalized_refined_depth = normalized_depth_map + delta_depth
        refined_depth = normalized_refined_depth * scale.view(N, C, 1, 1) + min.view(N, C, 1, 1)
        return refined_depth
    





