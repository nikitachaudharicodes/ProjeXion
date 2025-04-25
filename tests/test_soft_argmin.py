# test_soft_argmin.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.layers.soft_argmin import SoftArgmin

def test_soft_argmin():
    """Test the SoftArgmin module to verify correct behavior."""
    print("Testing SoftArgmin module...")
    
    # Create dummy input tensors
    B, D, H, W = 2, 48, 64, 80
    
    # Create example cost volume (output from CostRegularizer)
    cost_volume = torch.randn(B, 1, D, H, W)
    
    # Create example depth values
    depth_start = 0.5
    depth_end = 5.0
    depth_values = torch.linspace(depth_start, depth_end, D)
    
    # Instantiate the network
    soft_argmin = SoftArgmin()
    
    # Pass the dummy data through the network
    depth_map = soft_argmin(cost_volume, depth_values)
    
    # Check output shape
    print(f"Input cost volume shape: {cost_volume.shape}")
    print(f"Input depth values shape: {depth_values.shape}")
    print(f"Output depth map shape: {depth_map.shape}")
    
    # Verify the output shape matches expectation (B, H, W)
    assert depth_map.shape == (B, 1, H, W)
    
    # Test with 4D input (channel dimension already squeezed)
    cost_volume_4d = cost_volume.squeeze(1)  # [B, D, H, W]
    depth_map_4d = soft_argmin(cost_volume_4d, depth_values)
    assert depth_map_4d.shape == (B, H, W)
    print(f"4D input test passed, output shape: {depth_map_4d.shape}")
    
    # Verify values are in expected range
    min_depth = torch.min(depth_map).item()
    max_depth = torch.max(depth_map).item()
    print(f"Output depth range: [{min_depth:.4f}, {max_depth:.4f}]")
    assert min_depth >= depth_start and max_depth <= depth_end, \
        f"Depth values outside expected range [{depth_start}, {depth_end}]"
    
    # Test numerical stability with extreme values
    extreme_cost = torch.ones(B, 1, D, H, W) * -1000  # Very negative costs
    extreme_cost[:, :, D//2, :, :] = 0  # Make middle depth most probable
    depth_map_extreme = soft_argmin(extreme_cost, depth_values)
    expected_depth = depth_values[D//2].item()
    mean_depth = torch.mean(depth_map_extreme).item()
    print(f"Extreme test - Expected depth: {expected_depth:.4f}, Mean result: {mean_depth:.4f}")
    assert abs(mean_depth - expected_depth) < 0.1, "Failed numerical stability test"
    
    print("All tests passed! SoftArgmin module is working correctly.")

if __name__ == "__main__":
    test_soft_argmin()