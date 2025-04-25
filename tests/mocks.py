from torch.utils.data import DataLoader, Dataset
import torch

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
        extrinsics = torch.stack([torch.rand(4, 4)] * T)
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