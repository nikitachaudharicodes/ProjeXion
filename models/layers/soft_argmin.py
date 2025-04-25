"""
Create a soft_argmin.py module inside the models directory that implements 
the class SoftArgmin. This class should inherit from torch.nn.Module and 
should take as input a HxWxD embedding and generate an output embedding of 
size HxW by first calculating the softmax over the depth dimension (dim=2, zero-indexed) 
and then calculating the expected value of the depth using those probabilities and the depth values.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmin(nn.Module):
    
    def __init__(self):
        super(SoftArgmin, self).__init__()

    def forward(self, cost_volume, depth_values):
        """
        Forward pass of the soft armin operation.
        Args:
            cost_volume: Tensor of shape [N, 1, D, H, W] or [N, D, H, W] representing
            the cost regularized cost volume
            depth_values: Tensor of shape [D] representing the dpeth hypotheses for
            each sample 
        Returns:
            depth_map: Tensor of shape [N, H, W] representing the continuous depth map
        """

        if cost_volume.dim() == 5:
            cost_volume = cost_volume.squeeze(1) # [B, D, H, W]
        
        batch_size, depth_num, height, width = cost_volume.shape

        # apply softmax along the depth dimension to get probabilities (with negative sign for argmin)
        # the negative sign turns the minimum cost into maximum probability
        probability_volume = F.softmax(-cost_volume, dim=-3)

        # prep the depth_values for correct broadcasting with probability_volume 
        depth_values = depth_values.view(depth_num, 1, 1).to(cost_volume.device)

        # comute the expectation (weighted average) along the depth dimension
        depth_map = torch.sum(probability_volume * depth_values, dim=-3, keepdim=True)

        return depth_map

