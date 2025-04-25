# Create a feature_extraction.py module inside the models directory that implements 
# the class ImageEncoder. This class should inherit from torch.nn.Module and should 
# take as input a HxWx3 image and generate an output embedding of size H/4xW/4x32 
# using 8 2D-CNN layers:

# An eight-layer 2D CNN is applied, where the strides of layer 3 and 6 are set to two to 
# divide the feature towers into three scales. Within each scale, two convolutional layers 
# are applied to extract the higher-level image representation. Each convolutional layer is 
# followed by a batch-normalization (BN) layer and a rectified linear unit (ReLU) except for the last layer. 
# Also, similar to common matching tasks, parameters are shared among all feature towers for eficient learning. 
# The outputs of the 2D network are N 32-channel feature maps downsized by four in each dimension compared with input images.

"""
Feature extraction network for MVSNet.
"""

import torch 
import torch.nn as nn

class ImageEncoder(nn.Module):
    """
    Feature extraction network for MVSNet.
    An eight-layer 2D CNN that takes an HxWX3 image and generates H/4xw/4x32 feature maps.
    The strides of layer 3 and 6 are set to two to divide the feature towers into three scales.
    Within each scale, two convolutional layers are applied to extract the higher-level image representation.
    Each convolutional layer is followed by a batch-normalization (BN) layer and a rectified linear unit (ReLU) except for the last layer.
    """

    def __init__(self):
        super().__init__()
        base_filter = 8 
        
        # Scale 0: Original resolution
        self.conv0_0 = nn.Sequential(
            nn.Conv2d(3, base_filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filter),
            nn.ReLU(inplace=True),
        )

        self.conv0_1 = nn.Sequential(
            nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filter),
            nn.ReLU(inplace=True),
        )

        # Scale 2: First downsampling
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(base_filter, base_filter * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base_filter * 2),
            nn.ReLU(inplace=True),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(base_filter * 2, base_filter * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filter * 2),
            nn.ReLU(inplace=True),
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(base_filter * 2, base_filter * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filter * 2),
            nn.ReLU(inplace=True),
        )

        # Scale 3: Second downsampling (layer 6) - stride 2
        self.conv2_0 = nn.Sequential(
            nn.Conv2d(base_filter * 2, base_filter * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base_filter * 4),
            nn.ReLU(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filter * 4),
            nn.ReLU(inplace=True)
        )

        self.conv2_2 = nn.Conv2d(base_filter *4, base_filter * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass of the feature extraction network 
        Args:
            x : Input image tensor of shape [B, 3, H, W]

        Returns:
            Feature maps of shape [B, 32, H/4, W/4]
        """

        # Scale 1:
        conv0_0 = self.conv0_0(x) # [B, 8, H, W]
        conv0_1 = self.conv0_1(conv0_0)

        # Scale 2:
        conv1_0 = self.conv1_0(conv0_1)
        conv1_1 = self.conv1_1(conv1_0)
        conv1_2 = self.conv1_2(conv1_1)

        # Scale 3:
        conv2_0 = self.conv2_0(conv1_2)
        conv2_1 = self.conv2_1(conv2_0)
        conv2_2 = self.conv2_2(conv2_1)

        return conv2_2 # [B, 32, H/4, W/4]
    
