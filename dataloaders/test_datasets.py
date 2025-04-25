from dataloaders.blendedMVS import BlendedMVS
from torch.utils.data import DataLoader
from pathlib import Path

def test_online_sampling():
   data_path = Path('data_sample', 'BlendedMVS', 'dataset_low_res')
   CONTEXT_SIZE = 2
   dataset = BlendedMVS(data_path=data_path, subset=0.7, partition='train', context_size=CONTEXT_SIZE)
   for imgs, depth, K_ref, K_src, Rt_ref, Rt_src in dataset:
      assert imgs.shape == (1 + CONTEXT_SIZE, 3, 512, 640)
      assert depth.shape == (1, 512, 640)

def test_batch_sampling():
   BATCH_SIZE = 2
   SUBSET = 1
   CONTEXT_SIZE = 2
   data_path = Path('data_sample', 'BlendedMVS', 'dataset_low_res')
   dataset = BlendedMVS(data_path=data_path, subset=SUBSET, partition='train', context_size=CONTEXT_SIZE)
   dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
      # Should have batch of images, batch of depth maps, original length of images
      # and original length of depth maps
   for batch in dataloader:
      imgs, depths, K_ref, K_src, Rt_ref, Rt_src = batch
      assert imgs.shape == (BATCH_SIZE, 1 + CONTEXT_SIZE, 3, 512, 640)
      assert depths.shape == (BATCH_SIZE, 1, 512, 640)