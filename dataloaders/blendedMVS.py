"""
Creates an iterator over the BlendedMVS dataset that returns a
sequence of images (features) and depth maps (target)
"""

import random
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import load_pfm, parse_pairs, parse_cam
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
from typing import List, Literal, Tuple
from torch.nn.utils.rnn import pad_sequence
from math import ceil

class BlendedMVS(Dataset):
   def __init__(self,
                data_path:str,
                subset:float=1,
                partition:Literal['train', 'val', 'test']='train',
                context_size:int=0):
      """
      :param str data_path: Path to directory with all examples
      :param float subset: Fraction of the data points in the partition to use
      :param str partition: Partition of the data to load
      :param int context_size: Number of images to include in addition to the reference image 
      """
      # Check input
      self.data_path = Path(data_path)
      assert self.data_path.exists()
      assert partition in ['train', 'val', 'test'], "Partition must be 'train', 'val', or 'test'"
      list_file = self.data_path / f"{partition}_list.txt"
      assert list_file.exists(), f"File {list_file} does not exist"
      assert 0 < subset <= 1, f"Subset value should be between (0, 1]"

      # Store configuration
      self.context_size = context_size

      # Parse list file
      with open(list_file, 'r') as f:
         valid_objects = set(line.strip() for line in f.readlines())
      assert valid_objects, f"{list_file} was empty"
      
      # Filter objects in the partition
      all_object_paths = sorted([p for p in self.data_path.iterdir() if p.is_dir() and p.name in valid_objects])
      assert len(all_object_paths) == len(valid_objects), f"Only found {len(all_object_paths)} of the {len(valid_objects)} present in {list_file}"
      
      # Subset the data
      self.subset = subset
      self.length = ceil(len(all_object_paths) * self.subset)
      self.object_paths = all_object_paths[:self.length]
      assert len(self.object_paths) > 0, f"Kept {len(self.object_paths)} after applying the subset factor. At least 1 example should be retained"
      
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
      object_image_pairs = []
      object_intrinsics = []
      object_extrinsics = []
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

         # Load context IDs
         pairs_paths = (object_path / 'cams' / 'pair.txt')
         image_pairs = parse_pairs(pairs_path=pairs_paths)
         object_image_pairs.append(image_pairs)

         # Load camera parameters
         camera_parameters_paths = sorted([
            file for file in (object_path / 'cams').iterdir() if file.name.endswith("_cam.txt")
         ])
         intrinsics = []
         extrinsics = []
         for camera_path in camera_parameters_paths:
            intrinsic, extrinsic = parse_cam(str(camera_path))
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
         intrinsics = torch.stack(intrinsics, axis=0) # (T, 3, 3)
         extrinsics = torch.stack(extrinsics, axis=0) # (T, 4, 4)
         object_intrinsics.append(intrinsics)
         object_extrinsics.append(extrinsics)


      assert (
         len(object_images) == len(object_depth_maps) == len(object_image_pairs)
         == len(object_intrinsics) == len(object_extrinsics) == self.length
      )
      self.object_image_pairs = object_image_pairs
      self.object_images = object_images
      self.object_depth_maps = object_depth_maps
      self.object_intrinsics = object_intrinsics
      self.object_extrinsics = object_extrinsics

   def __len__(self):
      return self.length
   
   def __getitem__(self, object_index):
      # Features
      images = self.object_images[object_index]
      ## Select the reference image randomly
      reference_image_id = random.randint(0, len(images))
      ## Get context for the reference image
      pairs = self.object_image_pairs[object_index]
      context_image_ids = pairs[reference_image_id][:self.context_size]
      image_ids = [reference_image_id] + context_image_ids
      ## Select the final set of images
      images = images[image_ids]

      ## Get camera paramters for images
      extrinsics = self.object_extrinsics[object_index][image_ids]
      intrinsics = self.object_intrinsics[object_index][image_ids]

      # Target
      depth_map = self.object_depth_maps[object_index][reference_image_id]
      return images, intrinsics, extrinsics, depth_map
   
   def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
      # Extract batch of input images and batch of output depth maps separately
      batch_images, batch_intrinsics, batch_extrinsics,  batch_depth_maps = list(zip(*batch))
      batch_images = torch.stack(batch_images)
      batch_intrinsics = torch.stack(batch_intrinsics)
      batch_extrinsics = torch.stack(batch_extrinsics)
      batch_depth_maps = torch.stack(batch_depth_maps)
      return batch_images, batch_intrinsics, batch_extrinsics, batch_depth_maps
