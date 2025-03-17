from dataloaders.blendedMVS import BlendedMVS
from torch.utils.data import DataLoader
from pathlib import Path

def test_online_sampling():
   data_path = Path('data_sample', 'BlendedMVS', 'dataset_low_res')
   dataset = BlendedMVS(data_path=data_path, subset=0.7)
   for features, labels in dataset:
      assert features.shape[-3:] == (3, 112, 112)
      assert labels.shape[-3:] == (1, 112, 112)

def test_batch_sampling():
   BATCH_SIZE = 2
   SUBSET = 0.7
   data_path = Path('data_sample', 'BlendedMVS', 'dataset_low_res')
   dataset = BlendedMVS(data_path=data_path, subset=SUBSET)
   dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
   for batch in dataloader:
      # Should have batch of images, batch of depth maps, original length of images
      # and original length of depth maps
      assert len(batch) == 4
      batch_images, batch_depth_maps, lengths_images, lengths_depth_maps = batch
      T_images = lengths_images.max()
      assert batch_images.shape == (BATCH_SIZE, T_images, 3, 112, 112)
      T_depth_maps = lengths_depth_maps.max()
      assert batch_depth_maps.shape == (BATCH_SIZE, T_depth_maps, 1, 112, 112)