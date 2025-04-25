from torch import nn

class Conv3dBlock(nn.Module):
    """Basic Conv3D + BatchNorm3D + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, **kwargs):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=pad, bias=False, **kwargs)
        # Note: Batch norm parameters (center/scale) are enabled by default in PyTorch unlike the TF wrapper
        self.bn = nn.BatchNorm3d(out_channels)
        # Using inplace ReLU can save memory
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Deconv3dBlock(nn.Module):
    """Basic ConvTranspose3D + BatchNorm3D + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, pad=1, output_pad=1, **kwargs):
        super(Deconv3dBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=pad, output_padding=output_pad,
                                       bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))