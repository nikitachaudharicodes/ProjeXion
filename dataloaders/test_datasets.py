from dataloaders.blendedMVS import BlendedMVS
from pathlib import Path

def test_init():
   data_path = Path('data_sample', 'BlendedMVS', 'dataset_low_res')
   dataset = BlendedMVS(data_path=data_path, subset=0.7)
   for features, labels in dataset:
      assert features.shape[-3:] == (3, 112, 112)
      assert labels.shape[-3:] == (1, 112, 112)