from dataloaders.blendedMVS import BlendedMVS
from torch.utils.data import DataLoader
from pathlib import Path

def test_online_sampling():
   data_path = Path('data_sample', 'BlendedMVS', 'dataset_low_res')
   CONTEXT_SIZE = 2
   dataset = BlendedMVS(data_path=data_path, subset=0.7, partition='train', context_size=CONTEXT_SIZE)
   for images, intrinsics, extrinsics, labels in dataset:
      assert images.shape == (1 + CONTEXT_SIZE, 3, 512, 640)
      assert intrinsics.shape == (1 + CONTEXT_SIZE, 3, 3)
      assert extrinsics.shape == (1 + CONTEXT_SIZE, 4, 4)
      assert labels.shape == (1, 512, 640)

def test_batch_sampling():
   BATCH_SIZE = 2
   SUBSET = 1
   CONTEXT_SIZE = 2
   data_path = Path('data_sample', 'BlendedMVS', 'dataset_low_res')
   dataset = BlendedMVS(data_path=data_path, subset=SUBSET, partition='train', context_size=CONTEXT_SIZE)
   dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
   for batch in dataloader:
      # Should have 4 elements: images, intrinsics, extrinsics and depth maps
      assert len(batch) == 4
      batch_images, batch_intrinsics, batch_extrinsics,  batch_depth_maps = batch
      # Check shapes
      assert batch_images.shape == (BATCH_SIZE, 1 + CONTEXT_SIZE, 3, 512, 640)
      assert batch_intrinsics.shape == (BATCH_SIZE, 1 + CONTEXT_SIZE, 3, 3)
      assert batch_extrinsics.shape == (BATCH_SIZE, 1 + CONTEXT_SIZE, 4, 4)
      assert batch_depth_maps.shape == (BATCH_SIZE, 1, 512, 640)
