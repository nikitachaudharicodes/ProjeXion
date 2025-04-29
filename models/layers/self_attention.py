import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    """
    Applies self-attention across views and returns the output for the reference image.
    
    This layer takes a tensor representing features from multiple views
    (e.g., from an MVSNet setup) and applies self-attention across those views.
    It then returns the output corresponding to the reference image only.
    
    Input shape: (N, T, D, C, H, W) where N is batch size, T is number of views, 
    D is depth, C is channels, H and W are spatial dimensions.
    Output shape: (N, D, C, H, W)
    """
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1)
        self.value_conv = nn.Identity()  # Identity to preserve the original values
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SelfAttentionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, D, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, D, C, H, W) representing the 
                         attended features for the reference image.
        """
        N, T, D, C, H, W = x.shape
        
        # Reference image is at index 0
        reference_img = x[:, 0]  # Shape: (N, D, C, H, W)
        
        # Reshape for attention computation
        # We'll treat each channel separately for simplicity
        batch_size = N * D * C
        
        # Reshape to (N*D*C, T, H*W)
        x_reshaped = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, T, H*W)
        
        # Create query from reference image
        # Shape: (N*D*C, 1, H*W)
        query = x_reshaped[:, 0:1, :]
        
        # Compute attention weights (scaled dot product attention)
        # Shape: (N*D*C, 1, T)
        attention = torch.bmm(query, x_reshaped.transpose(1, 2))
        attention = F.softmax(attention / (H*W)**0.5, dim=2)
        
        # Apply attention to values
        # Shape: (N*D*C, 1, H*W)
        out = torch.bmm(attention, x_reshaped)
        
        # Reshape back to original dimensions but without the T dimension
        # Shape: (N, D, C, H, W)
        out = out.view(N, D, C, H, W)
        
        # Apply learnable scaling and residual connection for the reference image
        return self.gamma * out + reference_img