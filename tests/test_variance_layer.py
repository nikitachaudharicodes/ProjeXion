import unittest
import torch
from models.layers.variance import VarianceLayer

class TestVarianceLayer(unittest.TestCase):

    def test_output_shape(self):
        """Tests if the output shape is correct."""
        N, T, C, D, H, W = 4, 3, 16, 16, 8, 32
        input_tensor = torch.randn(N, T, C, D, H, W)
        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)
        self.assertEqual(output_tensor.shape, (N, C, D, H, W))

    def test_variance_calculation(self):
        """Tests the variance calculation with a simple known case."""
        # Create a simple tensor where variance is easy to calculate manually
        # Values: View 1: all 1s, View 2: all 3s
        # Mean = (1 + 3) / 2 = 2
        # Variance = [(1-2)^2 + (3-2)^2] / 2 = [(-1)^2 + 1^2] / 2 = (1 + 1) / 2 = 1
        N, C, D, H, W = 2, 3, 2, 2, 2
        view1 = torch.ones((N, C, D, H, W))
        view2 = torch.full((N, C, D, H, W), 3.0)
        input_tensor = torch.stack((view1, view2), dim=1) # Shape (N, 2, C, D, H, W)

        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)

        expected_variance = torch.ones((N, C, D, H, W))
        self.assertTrue(torch.allclose(output_tensor, expected_variance))
        self.assertEqual(output_tensor.shape, ((N, C, D, H, W)))

    def test_variance_with_single_view(self):
        """Tests if variance is zero when only one view (N=1) is provided."""
        N, T, C, D, H, W = 1, 1, 16, 16, 8, 32
        input_tensor = torch.randn(N, T, C, D, H, W)
        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)

        expected_variance = torch.zeros(N, C, D, H, W)
        self.assertTrue(torch.allclose(output_tensor, expected_variance, atol=1e-8)) # Use atol for float comparison
        self.assertEqual(output_tensor.shape, (N, C, D, H, W))

    def test_different_dimensions(self):
        """Tests with different valid dimensions."""
        N, T, C, D, H, W = 5, 25, 8, 10, 4, 16
        input_tensor = torch.randn(N, T, C, D, H, W)
        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)
        self.assertEqual(output_tensor.shape, (N, C, D, H, W))
        # Check if variance calculation runs without error and gives non-negative values
        self.assertTrue(torch.all(output_tensor >= 0))


if __name__ == '__main__':
    unittest.main()