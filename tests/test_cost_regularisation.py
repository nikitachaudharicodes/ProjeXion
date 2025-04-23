# tests/test_cost_regularization.py

import unittest
import torch
import sys
import os

from models.cost_regularizer import CostRegularizer


class TestCostRegularizer(unittest.TestCase):

    def test_initialization(self):
        """Test if the model initializes without errors."""
        try:
            model = CostRegularizer(in_channels=32)
            self.assertIsInstance(model, CostRegularizer)
        except Exception as e:
            self.fail(f"CostRegularizer initialization failed with exception: {e}")

    def test_forward_pass_shape_cpu(self):
        """Test the forward pass and output shape on CPU."""
        B, C_in, D, H, W = 2, 32, 16, 32, 40 # Example dimensions
        C_out = 1
        expected_shape = (B, C_out, D, H, W)

        model = CostRegularizer(in_channels=C_in)
        dummy_input = torch.randn(B, C_in, D, H, W)

        # Set model to evaluation mode for consistent behavior (e.g., BatchNorm)
        model.eval()
        with torch.no_grad(): # No need to track gradients for shape testing
            output = model(dummy_input)

        self.assertEqual(output.shape, expected_shape,
                         f"Output shape mismatch. Expected {expected_shape}, got {output.shape}")

    def test_forward_pass_various_shapes(self):
        """Test forward pass with different valid input shapes."""
        shapes_to_test = [
            (1, 32, 8, 16, 16),   # Smallest
            (4, 32, 32, 64, 64),  # Larger batch, typical size
            (2, 32, 24, 48, 56)   # Odd dimensions
        ]
        C_out = 1
        model = CostRegularizer(in_channels=32)
        model.eval()

        for shape in shapes_to_test:
            B, C_in, D, H, W = shape
            expected_shape = (B, C_out, D, H, W)
            dummy_input = torch.randn(*shape)
            with torch.no_grad():
                output = model(dummy_input)

            self.assertEqual(output.shape, expected_shape,
                             f"Output shape mismatch for input {shape}. Expected {expected_shape}, got {output.shape}")


    def test_gradient_flow(self):
        """Test if gradients can flow back through the network."""
        B, C_in, D, H, W = 1, 32, 8, 16, 20 # Smaller size for faster test
        C_out = 1

        model = CostRegularizer(in_channels=C_in)
        # Ensure model is in training mode
        model.train()

        # Create input that requires gradients
        dummy_input = torch.randn(B, C_in, D, H, W, requires_grad=True)
        # Create a dummy target with the same shape as the output
        dummy_target = torch.randn(B, C_out, D, H, W)

        # Forward pass
        output = model(dummy_input)

        # Calculate a simple loss
        loss = torch.nn.functional.mse_loss(output, dummy_target)

        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            self.fail(f"loss.backward() failed with exception: {e}")

        # Check if any parameter has received gradients
        grad_found = False
        for param in model.parameters():
            if param.grad is not None:
                # Check if gradient is not all zeros (might happen in edge cases, but unlikely here)
                if torch.abs(param.grad).sum() > 1e-8:
                     grad_found = True
                     break
        self.assertTrue(grad_found, "No gradients found in model parameters after backward pass.")
        self.assertIsNotNone(dummy_input.grad, "Input tensor did not receive gradients.")


    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_forward_pass_shape_gpu(self):
        """Test the forward pass and output shape on GPU if available."""
        B, C_in, D, H, W = 2, 32, 16, 32, 40
        C_out = 1
        expected_shape = (B, C_out, D, H, W)
        device = torch.device("cuda")

        model = CostRegularizer(in_channels=C_in).to(device)
        dummy_input = torch.randn(B, C_in, D, H, W, device=device)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, expected_shape,
                         f"Output shape mismatch on GPU. Expected {expected_shape}, got {output.shape}")
        self.assertEqual(output.device.type, "cuda", "Output tensor is not on CUDA device.")

    def test_training_mode_behavior(self):
        """Check if BatchNorm layers are in training mode by default."""
        model = CostRegularizer(in_channels=32)
        model.train() # Explicitly set to train (should be default)

        # Check a BatchNorm layer (e.g., the first one)
        # Accessing internal layers like this is slightly fragile but common in tests
        self.assertTrue(model.conv0_0.bn.training, "BatchNorm layer is not in training mode after model.train()")

    def test_eval_mode_behavior(self):
        """Check if BatchNorm layers switch to eval mode."""
        model = CostRegularizer(in_channels=32)
        model.eval()

        # Check a BatchNorm layer
        self.assertFalse(model.conv0_0.bn.training, "BatchNorm layer is not in eval mode after model.eval()")
        # Also check forward pass runs in eval mode
        B, C_in, D, H, W = 1, 32, 8, 16, 16
        dummy_input = torch.randn(B, C_in, D, H, W)
        try:
            with torch.no_grad():
                model(dummy_input)
        except Exception as e:
             self.fail(f"Forward pass failed in eval mode: {e}")


# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()