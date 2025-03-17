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

class BlendedMVS(Dataset):
   def __init__(self, data_path, subset=1):
      self.data_path = Path(data_path)
      self.subset = subset
      all_object_paths = sorted(list(self.data_path.iterdir()))
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
      for object_path in tqdm(self.object_paths, desc="Loading images"):
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
         depth_maps = torch.stack(depth_maps, axis=0) # (T, 3, 112, 112)
         object_depth_maps.append(depth_maps)

      self.object_images = object_images
      self.object_depth_maps = object_depth_maps

   def __len__(self):
      return self.length
   
   def __getitem__(self, index):
      return self.object_images[index], self.object_depth_maps[index]