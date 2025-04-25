import torch
from evaluation.evaluate import compute_depth_metrics

def test_compute_metrics_output():
    B, _, H, W = 4, 1, 8, 8
    pred = torch.rand((B, 1, H, W))
    gt = torch.rand((B, 1, H, W)) + 0.1  # avoid division by zero
    mask = gt > 0

    metrics = compute_depth_metrics(pred, gt, mask)
    
    expected_keys = ["Abs Rel", "Sq Rel", "RMSE", "δ < 1.25", "δ < 1.25²", "δ < 1.25³"]
    assert all(key in metrics for key in expected_keys)
    for val in metrics.values():
        assert isinstance(val, torch.Tensor)
        assert val.numel() == 1  # scalar
