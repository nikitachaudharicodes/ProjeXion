import unittest
import torch
from models.feature_extraction import ImageEncoder
"""
Command to run the test:
python -m unittest -v tests/test_feature_extraction.py
"""
class TestImageEncoder(unittest.TestCase):

    def setUp(self):
        self.model = ImageEncoder()

    def test_output_shape(self):
        """Tests if output shape is correct for 640x512 input."""
        B, C, H, W = 1, 3, 640, 512
        input_tensor = torch.randn(B, C, H, W)
        output_tensor = self.model(input_tensor)
        expected_shape = (B, 32, H // 4, W // 4)
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_channel_count(self):
        """Tests that output has 32 channels."""
        input_tensor = torch.randn(1, 3, 320, 256)
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape[1], 32)

    def test_multiple_input_sizes(self):
        """Tests different valid input sizes divisible by 4."""
        for h, w in [(128, 128), (256, 192), (1024, 768)]:
            with self.subTest(height=h, width=w):
                input_tensor = torch.randn(1, 3, h, w)
                output_tensor = self.model(input_tensor)
                self.assertEqual(output_tensor.shape, (1, 32, h // 4, w // 4))

if __name__ == '__main__':
    unittest.main()
