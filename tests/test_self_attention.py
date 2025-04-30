# tests/test_self_attention.py
import unittest
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.layers.self_attention import SelfAttentionLayer

class TestSelfAttentionLayer(unittest.TestCase):
    
    def test_output_shape(self):
        """Tests if the output shape is correct."""
        N, T, D, C, H, W = 4, 3, 8, 16, 32, 32
        input_tensor = torch.randn(N, T, D, C, H, W)
        attention_layer = SelfAttentionLayer()
        output_tensor = attention_layer(input_tensor)
        
        # Output should be (N, D, C, H, W) - without the T dimension
        self.assertEqual(output_tensor.shape, (N, D, C, H, W))
    
    def test_reference_image_preservation(self):
        """Tests if the reference image is preserved when gamma is zero."""
        N, T, D, C, H, W = 2, 3, 4, 8, 16, 16
        input_tensor = torch.randn(N, T, D, C, H, W)
        
        # Store the reference image
        reference_img = input_tensor[:, 0].clone()
        
        # Create layer with gamma initialized to zero (default)
        attention_layer = SelfAttentionLayer()
        # Ensure gamma is zero
        attention_layer.gamma.data.zero_()
        
        output_tensor = attention_layer(input_tensor)
        
        # When gamma is zero, output should equal reference image
        self.assertTrue(torch.allclose(output_tensor, reference_img))
    
    def test_attention_mechanism(self):
        """Tests the attention mechanism with a simple known case."""
        N, T, D, C, H, W = 1, 2, 1, 1, 2, 2
        
        # Create a simple input where the first view is the reference image
        # and the second view has higher values
        view1 = torch.ones((N, D, C, H, W))  # Reference image with all 1s
        view2 = torch.full((N, D, C, H, W), 2.0)  # Second view with all 2s
        input_tensor = torch.stack((view1, view2), dim=1)  # Shape (N, T, D, C, H, W)
        
        attention_layer = SelfAttentionLayer()
        # Set gamma to 1 for full attention effect
        attention_layer.gamma.data.fill_(1.0)
        
        output_tensor = attention_layer(input_tensor)
        
        # With attention active and gamma=1, output should be influenced by view2
        # But still have residual connection to view1
        # The exact value depends on the attention implementation, but should be > 1
        self.assertTrue(torch.all(output_tensor > view1))
        self.assertEqual(output_tensor.shape, (N, D, C, H, W))
    
    def test_with_single_view(self):
        """Tests behavior when only the reference view is provided."""
        N, D, C, H, W = 2, 4, 8, 16, 16
        # Only one view (the reference)
        input_tensor = torch.randn(N, 1, D, C, H, W)
        reference_img = input_tensor[:, 0].clone()
        
        attention_layer = SelfAttentionLayer()
        output_tensor = attention_layer(input_tensor)
        
        # With only one view, output should be equal to input reference
        self.assertTrue(torch.allclose(output_tensor, reference_img))
        self.assertEqual(output_tensor.shape, (N, D, C, H, W))
    
    def test_different_dimensions(self):
        """Tests with different valid dimensions."""
        N, T, D, C, H, W = 5, 4, 16, 32, 24, 24
        input_tensor = torch.randn(N, T, D, C, H, W)
        attention_layer = SelfAttentionLayer()
        output_tensor = attention_layer(input_tensor)
        
        self.assertEqual(output_tensor.shape, (N, D, C, H, W))
        # Check if attention calculation runs without error
        self.assertTrue(output_tensor is not None)

if __name__ == '__main__':
    unittest.main()