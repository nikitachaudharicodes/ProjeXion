from models import MVSNet
from .mocks import DummyModel, DummyDataset, DummyLoss
from torch.utils.data import DataLoader

def test_mvsnet():
   dataset = DummyDataset()
   dataloader = DataLoader(dataset, 3)
   images, instrinsics, extrinsics, depth_maps = next(iter(dataloader))
   mvsnet = MVSNet(0, 1, 0.2)
   pred_depth_map = mvsnet(images, instrinsics, extrinsics)