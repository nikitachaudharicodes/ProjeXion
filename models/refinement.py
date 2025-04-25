import torch 
import torch.nn as nn

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
            ref_img (torch.Tensor): Reference image of shape (B, 3, H, W).
            init_depth (torch.Tensor): Initial depth map of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Refined depth map of shape (B, 1, H, W).
        """
        x = torch.cat((ref_img, init_depth), dim=1)  # Concatenate along channel dimension
        delta_depth = self.refine_net(x)  # Pass through the refinement network
        refined_depth = init_depth + delta_depth
        return refined_depth
    





