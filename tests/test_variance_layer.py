import unittest
import torch
from models.variance import VarianceLayer

class TestVarianceLayer(unittest.TestCase):

    def test_output_shape(self):
        """Tests if the output shape is correct."""
        N, H, W, D, C = 4, 16, 16, 8, 32
        input_tensor = torch.randn(N, H, W, D, C)
        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)
        self.assertEqual(output_tensor.shape, (H, W, D, C))

    def test_variance_calculation(self):
        """Tests the variance calculation with a simple known case."""
        # Create a simple tensor where variance is easy to calculate manually
        # Values: View 1: all 1s, View 2: all 3s
        # Mean = (1 + 3) / 2 = 2
        # Variance = [(1-2)^2 + (3-2)^2] / 2 = [(-1)^2 + 1^2] / 2 = (1 + 1) / 2 = 1
        H, W, D, C = 2, 2, 2, 1
        view1 = torch.ones(1, H, W, D, C)
        view2 = torch.full((1, H, W, D, C), 3.0)
        input_tensor = torch.cat((view1, view2), dim=0) # Shape (2, H, W, D, C)

        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)

        expected_variance = torch.ones(H, W, D, C)
        self.assertTrue(torch.allclose(output_tensor, expected_variance))
        self.assertEqual(output_tensor.shape, (H, W, D, C))

    def test_variance_with_single_view(self):
        """Tests if variance is zero when only one view (N=1) is provided."""
        N, H, W, D, C = 1, 16, 16, 8, 32
        input_tensor = torch.randn(N, H, W, D, C)
        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)

        expected_variance = torch.zeros(H, W, D, C)
        self.assertTrue(torch.allclose(output_tensor, expected_variance, atol=1e-8)) # Use atol for float comparison
        self.assertEqual(output_tensor.shape, (H, W, D, C))

    def test_different_dimensions(self):
        """Tests with diopfferent valid dimensions."""
        N, H, W, D, C = 5, 8, 10, 4, 16
        input_tensor = torch.randn(N, H, W, D, C)
        variance_layer = VarianceLayer()
        output_tensor = variance_layer(input_tensor)
        self.assertEqual(output_tensor.shape, (H, W, D, C))
        # Check if variance calculation runs without error and gives non-negative values
        self.assertTrue(torch.all(output_tensor >= 0))


if __name__ == '__main__':
    unittest.main()