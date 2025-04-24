# models/cost_regularization.py

#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Modified for PyTorch implementation.
MVSNet Cost Volume Regularization Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU3D(nn.Module):
    """Basic Conv3D + BatchNorm3D + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, **kwargs):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=pad, bias=False, **kwargs)
        # Note: Batch norm parameters (center/scale) are enabled by default in PyTorch unlike the TF wrapper
        self.bn = nn.BatchNorm3d(out_channels)
        # Using inplace ReLU can save memory
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DeconvBnReLU3D(nn.Module):
    """Basic ConvTranspose3D + BatchNorm3D + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, pad=1, output_pad=1, **kwargs):
        super(DeconvBnReLU3D, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=pad, output_padding=output_pad,
                                       bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CostRegularizer(nn.Module):
    """
    Network for regularizing the 3D cost volume using a 3D U-Net architecture.

    Takes an input cost volume of shape (B, 32, D, H, W) and outputs a
    regularized volume of shape (B, 1, D, H, W) representing logits before
    softmax probability normalization.
    """
    def __init__(self, in_channels=32):
        super(CostRegularizer, self).__init__()
        print('Cost Regularizer: 3D UNet with 4 scales and 2 convs per scale.')
        print(f'Input channels: {in_channels}')

        # Initial convolution block reducing channels from 32 to 8
        # Scale 0
        self.conv0_0 = ConvBnReLU3D(in_channels, 8, kernel_size=3, stride=1, pad=1)
        self.conv0_1 = ConvBnReLU3D(8, 8, kernel_size=3, stride=1, pad=1)

        # Encoder Path
        # Scale 1
        self.conv1_0 = ConvBnReLU3D(8, 16, kernel_size=3, stride=2, pad=1)
        self.conv1_1 = ConvBnReLU3D(16, 16, kernel_size=3, stride=1, pad=1)

        # Scale 2
        self.conv2_0 = ConvBnReLU3D(16, 32, kernel_size=3, stride=2, pad=1)
        self.conv2_1 = ConvBnReLU3D(32, 32, kernel_size=3, stride=1, pad=1)

        # Scale 3 (Bottleneck)
        self.conv3_0 = ConvBnReLU3D(32, 64, kernel_size=3, stride=2, pad=1)
        self.conv3_1 = ConvBnReLU3D(64, 64, kernel_size=3, stride=1, pad=1)

        # Decoder Path
        # Scale 2
        self.deconv2 = DeconvBnReLU3D(64, 32, kernel_size=3, stride=2, pad=1, output_pad=1)
        # After adding skip connection
        self.deconv2_1 = ConvBnReLU3D(32, 32, kernel_size=3, stride=1, pad=1)

        # Scale 1
        self.deconv1 = DeconvBnReLU3D(32, 16, kernel_size=3, stride=2, pad=1, output_pad=1)
        # After adding skip connection
        self.deconv1_1 = ConvBnReLU3D(16, 16, kernel_size=3, stride=1, pad=1)

        # Scale 0
        self.deconv0 = DeconvBnReLU3D(16, 8, kernel_size=3, stride=2, pad=1, output_pad=1)
        # After adding skip connection
        self.deconv0_1 = ConvBnReLU3D(8, 8, kernel_size=3, stride=1, pad=1)

        # Final 1x1x1 convolution to get 1 channel output (logits)
        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        """
        Args:
            x: Input cost volume, shape (B, C, D, H, W). C=32 typically.
        Returns:
            Output volume, shape (B, 1, D, H, W). Represents logits before softmax.
        """
        # Encoder
        conv0_0_out = self.conv0_0(x)
        conv0_1_out = self.conv0_1(conv0_0_out) # Skip connection 0: (B, 8, D, H, W)

        conv1_0_out = self.conv1_0(conv0_1_out)
        conv1_1_out = self.conv1_1(conv1_0_out) # Skip connection 1: (B, 16, D, H/2, W/2)

        conv2_0_out = self.conv2_0(conv1_1_out)
        conv2_1_out = self.conv2_1(conv2_0_out) # Skip connection 2: (B, 32, D, H/4, W/4)

        conv3_0_out = self.conv3_0(conv2_1_out)
        conv3_1_out = self.conv3_1(conv3_0_out) # Bottleneck: (B, 64, D, H/8, W/8)

        # Decoder
        deconv2_out = self.deconv2(conv3_1_out) # Upsampled from bottleneck: (B, 32, D, H/4, W_deconv2)
        # Crop deconv2_out to match conv2_1_out spatial size before adding skip connection
        cropped_deconv2 = self._crop_like(deconv2_out, conv2_1_out)
        add2 = cropped_deconv2 + conv2_1_out
        deconv2_1_out = self.deconv2_1(add2)

        deconv1_out = self.deconv1(deconv2_1_out) # Upsampled from scale 2: (B, 16, D, H/2, W_deconv1)
        # Crop deconv1_out to match conv1_1_out spatial size before adding skip connection
        cropped_deconv1 = self._crop_like(deconv1_out, conv1_1_out)
        add1 = cropped_deconv1 + conv1_1_out
        deconv1_1_out = self.deconv1_1(add1)

        deconv0_out = self.deconv0(deconv1_1_out) # Upsampled from scale 1: (B, 8, D, H, W_deconv0)
        # Crop deconv0_out to match conv0_1_out spatial size before adding skip connection
        cropped_deconv0 = self._crop_like(deconv0_out, conv0_1_out)
        add0 = cropped_deconv0 + conv0_1_out
        deconv0_1_out = self.deconv0_1(add0)

        # Final convolution
        out_logits = self.final_conv(deconv0_1_out)

        return out_logits
    
    def _crop_like(self, tensor_to_crop, target_tensor):
        """Center crops tensor_to_crop to match target_tensor's spatial dims."""
        target_shape = target_tensor.shape[2:] # D, H, W
        current_shape = tensor_to_crop.shape[2:] # D, H, W

        # If shapes already match, return original tensor
        if target_shape == current_shape:
            return tensor_to_crop

        # Calculate cropping amounts (center crop)
        diff_d = current_shape[0] - target_shape[0]
        diff_h = current_shape[1] - target_shape[1]
        diff_w = current_shape[2] - target_shape[2]

        # Ensure we are not trying to pad (crop amount should be >= 0)
        if diff_d < 0 or diff_h < 0 or diff_w < 0:
             raise ValueError("Target shape must be smaller or equal for cropping.")

        crop_d_start = diff_d // 2
        crop_d_end = crop_d_start + target_shape[0]
        crop_h_start = diff_h // 2
        crop_h_end = crop_h_start + target_shape[1]
        crop_w_start = diff_w // 2
        crop_w_end = crop_w_start + target_shape[2]

        return tensor_to_crop[:, :, crop_d_start:crop_d_end, crop_h_start:crop_h_end, crop_w_start:crop_w_end]


# Example Usage (for testing the module)
if __name__ == '__main__':
    # Assume input Batch=2, Depth=48, Height=64, Width=80
    B, D, H, W = 2, 48, 64, 80
    C_in = 32 # Input channels specified by MVSNet feature extraction
    C_out = 1 # Output channels (probability logits)

    # Create dummy input tensor
    # PyTorch uses (B, C, D, H, W) format for Conv3D
    dummy_cost_volume = torch.randn(B, C_in, D, H, W)

    # Instantiate the network
    cost_regularizer = CostRegularizer(in_channels=C_in)

    # Pass the dummy data through the network
    output_volume = cost_regularizer(dummy_cost_volume)

    # Check output shape
    print(f"Input shape: {dummy_cost_volume.shape}")
    print(f"Output shape: {output_volume.shape}")

    # Verify the output shape matches expectation (B, C_out, D, H, W)
    assert output_volume.shape == (B, C_out, D, H, W)

    print("CostRegularizer module created and test forward pass successful!")

    # You would typically apply Softmax after this outside the module:
    # probability_volume = F.softmax(output_volume, dim=2) # Softmax along depth (D) dimension
    # print(f"Shape after Softmax: {probability_volume.shape}")