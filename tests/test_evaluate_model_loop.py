import torch
from torch.utils.data import DataLoader, Dataset
from evaluation.evaluate import evaluate_model, compute_depth_metrics

class DummyDataset(Dataset):
    def __init__(self, size=4):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        T, C, H, W = 2, 3, 8, 8
        imgs = torch.rand((T, C, H, W))           # Just (C,H,W), batch will be added by DataLoader
        depth = torch.rand((H, W))
        intrinsics = torch.stack([torch.eye(3)] * T)
        extrinsics = torch.rand(4, 4)
        return imgs, intrinsics, extrinsics, depth

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, *args, **kwargs):
        # Expect input x to be (B, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)  # fallback for safety if batching fails
        B, T, C, H, W = x.shape
        return torch.rand(B, 1, H, W)


class DummyLoss(torch.nn.Module):
    def forward(self, pred, target, mask):
        return torch.mean((pred - target)**2 * mask)

def test_evaluate_model_loop():
    model = DummyModel()
    loader = DataLoader(DummyDataset(), batch_size=2)
    loss_fn = DummyLoss()

    val_loss, metrics = evaluate_model(model, loader, loss_fn)

    assert isinstance(val_loss, float)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["Abs Rel", "Sq Rel", "RMSE", "δ < 1.25", "δ < 1.25²", "δ < 1.25³"])
