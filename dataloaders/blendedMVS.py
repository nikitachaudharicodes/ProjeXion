"""
Creates an iterator over the BlendedMVS dataset that returns a
sequence of images (features) and depth maps (target)
"""

from torch.utils.data import Dataset
import torch
import numpy as np
from utils import load_pfm
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

class BlendedMVS(Dataset):
   def __init__(self, data_path, subset=1, partition='train', seed=42):
      assert partition in ['train', 'val', 'test'], "Parition must be 'train', 'val', or 'test'"
      self.data_path = Path(data_path)
      list_file = self.data_path / f"{partition}_list.txt"
      assert list_file.exists(), f"File {list_file} does not exist"
      
      with open(list_file, 'r') as f:
         valid_objects = set(line.strip() for line in f.readlines())
      
      all_object_paths = sorted([p for p in self.data_path.iterdir() if p.is_dir() and p.name in valid_objects])
      self.subset = subset
      self.length = int(len(all_object_paths) * self.subset)
      self.object_paths = all_object_paths[:self.length]
      
      pil_to_tensor = T.Compose([
         T.Resize((112, 112)),
         T.ToImage(),
         T.ToDtype(torch.float32, scale=True),
         T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
      ])
      depth_map_to_tensor = T.Compose([
         T.Lambda(lambda x: torch.tensor(x)),
         T.Resize((112, 112)),
         T.ToDtype(torch.float32, scale=True),
      ])
      
      object_images = []
      object_depth_maps = []
      
      for object_path in tqdm(self.object_paths, desc=f"Loading {partition} images"):
         # Load images
         images_paths = sorted(list((object_path / 'blended_images').iterdir()))
         images = []
         for image_path in images_paths:
            image = Image.open(image_path).convert('RGB')
            image = pil_to_tensor(image)
            images.append(image)
         images = torch.stack(images, axis=0) # (T, 3, 112, 112)
         object_images.append(images)

         # Load depth maps
         depth_maps_paths = sorted(list((object_path / 'rendered_depth_maps').iterdir()))
         depth_maps = []
         for depth_map_path in depth_maps_paths:
            depth_map = load_pfm(str(depth_map_path))
            depth_map = depth_map_to_tensor(depth_map)
            depth_maps.append(depth_map)
         depth_maps = torch.stack(depth_maps, axis=0) # (T, 1, 112, 112)
         object_depth_maps.append(depth_maps)

      self.object_images = object_images
      self.object_depth_maps = object_depth_maps

   def __len__(self):
      return self.length
   
   def __getitem__(self, index):
      return self.object_images[index], self.object_depth_maps[index]
   
   def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
      # Extract batch of input images and batch of output depth maps separately
      batch_images, batch_depth_maps = list(zip(*batch))
      # Store original lengths
      lengths_images = torch.tensor([image_sequence.shape[0] for image_sequence in batch_images])
      lengths_depth_maps = torch.tensor([depth_map_sequence.shape[0] for depth_map_sequence in batch_depth_maps])
      # Pad the sequences
      batch_images_padded = pad_sequence(batch_images, batch_first=True)
      batch_depth_maps_padded = pad_sequence(batch_depth_maps, batch_first=True)
      return batch_images_padded, batch_depth_maps_padded, lengths_images, lengths_depth_maps
