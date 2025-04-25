import torch
import torch.nn as nn

class VarianceLayer(nn.Module):
    """
    Calculates the element-wise variance along the first dimension of an input tensor.

    This layer takes a tensor typically representing features from multiple views
    (e.g., from an MVSNet setup) and computes the variance across those views
    to produce a consolidated feature map.

    Input shape: (N, H, W, D, C) where N is the number of views, H, W, D are spatial/depth dimensions, and C is the channel count.
    Output shape: (H, W, D, C)
    """
    def __init__(self):
        super(VarianceLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VarianceLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, D, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, D, C, H, W) representing the element-wise variance along dim 0.
        """
        # Calculate the variance along the first dimension (N)
        # unbiased=False ensures division by N (population variance) as per the formula
        # keepdim=False is the default and removes the first dimension
        variance = torch.var(x, dim=1, correction=0)
        return variance
