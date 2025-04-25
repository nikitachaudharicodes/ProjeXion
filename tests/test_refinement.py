import unittest
import torch
from models.refinement import Refine

class TestRefine(unittest.TestCase):

    def setUp(self):
        self.model = Refine()

    def test_output_shape(self):
        """Check if the refined depth map has correct shape."""
        B, H, W = 2, 128, 160
        ref_image = torch.randn(B, 3, H, W)
        init_depth = torch.randn(B, 1, H, W)
        output = self.model(ref_image, init_depth)
        self.assertEqual(output.shape, (B, 1, H, W))

    def test_residual_connection(self):
        """Check if residual connection is applied."""
        B, H, W = 1, 64, 64
        ref_image = torch.zeros(B, 3, H, W)
        init_depth = torch.ones(B, 1, H, W) * 2.0
        output = self.model(ref_image, init_depth)
        self.assertFalse(torch.equal(output, init_depth))  # Should be different due to residual

if __name__ == '__main__':
    unittest.main()
