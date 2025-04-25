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
                height: int = 512,       
                width:  int = 640,      
                subset:float=1,
                samples_per_epoch:int = 3000,
                partition:Literal['train', 'val', 'test']='train',
                context_size:int=0,
                ):
      """
      :param str data_path: Path to directory with all examples
      :param Tuple[int,int] img_size: (H, W) to which every RGB & depth map will be resized
      :param float subset: Fraction of the data points in the partition to use
      :param str partition: Partition of the data to load
      :param int context_size: Number of images to include in addition to the reference image 
      :param int height:  target image height after resize
      :param int width:   target image  width after resize
      """
      # Check input
      data_path = Path(data_path)
      assert data_path.exists()
      assert partition in ['train', 'val', 'test'], "Partition must be 'train', 'val', or 'test'"
      list_file = data_path / f"{partition}_list.txt"
      assert list_file.exists(), f"File {list_file} does not exist"
      assert 0 < subset <= 1, f"Subset value should be between (0, 1]"
      assert height % 32 == 0 and width % 32 == 0, "img_size dims must be divisible by 32"
      assert context_size <= 10, "The context size must be <= 10"
      assert samples_per_epoch > 0, f"samples_per_epoch ({samples_per_epoch}) must be greater than 0"
      # Store configuration
      self.data_path = data_path
      self.target_h = height
      self.target_w = width
      self.context_size = context_size
      self.samples_per_epoch = samples_per_epoch

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
         T.Resize((self.target_h, self.target_w), interpolation=Image.BILINEAR),
         T.ToImage(),
         T.ToDtype(torch.float, scale=True),
         T.Normalize([0.5]*3, [0.5]*3),
      ])
      depth_map_to_tensor = T.Compose([
         T.Lambda(lambda x: torch.tensor(x, dtype=torch.float)),
         T.Resize((self.target_h, self.target_w), interpolation=Image.BILINEAR),
      ])


      object_images = []
      object_depth_maps = []
      object_image_pairs = []
      object_intrinsics = []
      object_extrinsics = []
      for object_path in tqdm(self.object_paths, desc=f"Loading {partition} images"):
         # Load images
         images_paths = sorted(list((object_path / 'blended_images').iterdir()))
         images = {}
         for image_path in images_paths:
            image_id = int(image_path.stem)
            image = Image.open(image_path).convert('RGB')
            image = pil_to_tensor(image)
            images[image_id] = image
         object_images.append(images)

         # Load depth maps
         depth_maps_paths = sorted(list((object_path / 'rendered_depth_maps').iterdir()))
         depth_maps = {}
         for depth_map_path in depth_maps_paths:
            depth_map_id = int(depth_map_path.stem)
            depth_map = load_pfm(str(depth_map_path))
            depth_map = depth_map_to_tensor(depth_map)
            depth_maps[depth_map_id] = depth_map
         object_depth_maps.append(depth_maps)

         # Load context IDs
         pairs_paths = (object_path / 'cams' / 'pair.txt')
         image_pairs = parse_pairs(pairs_path=pairs_paths)
         object_image_pairs.append(image_pairs)

         # Load camera parameters
         camera_parameters_paths = sorted([
            file for file in (object_path / 'cams').iterdir() if file.name.endswith("_cam.txt")
         ])
         intrinsics = {}
         extrinsics = {}
         for camera_path in camera_parameters_paths:
            camera_id = int(camera_path.stem[:8])
            intrinsic, extrinsic = parse_cam(str(camera_path))
            # Scale intrinsic parameters
            sx = self.target_w / (2 * intrinsic[0,2].item())
            sy = self.target_h / (2 * intrinsic[1,2].item())
            intrinsic[0,0] *= sx;  intrinsic[0,2] *= sx
            intrinsic[1,1] *= sy;  intrinsic[1,2] *= sy
            intrinsics[camera_id] = intrinsic
            extrinsics[camera_id] = extrinsic
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
      return self.samples_per_epoch
   
   def __getitem__(self, object_index):
      # Ignore the input argument and select and object randomly
      object_index = random.randint(0, self.length - 1)
      # Features
      images = self.object_images[object_index]
      ## Select the reference image randomly
      reference_image_id = random.choice(list(images.keys()))
      ## Get context for the reference image
      pairs = self.object_image_pairs[object_index]
      all_context_images = pairs.get(reference_image_id)
      if all_context_images:
         context_image_ids = pairs[reference_image_id][:self.context_size]
         image_ids = [reference_image_id] + context_image_ids
         ## Select the final set of images
         images = torch.stack([
            images[image_id]
            for image_id in image_ids
         ])
         ## Get camera paramters for those images
         object_extrinsics = self.object_extrinsics[object_index]
         extrinsics = torch.stack([
            object_extrinsics[image_id]
            for image_id in image_ids
         ])
         object_intrinsics = self.object_intrinsics[object_index]
         intrinsics = torch.stack([
            object_intrinsics[image_id]
            for image_id in image_ids
         ])
      else:
         images = None
         extrinsics = None
         intrinsics = None

      # Target
      depth_map = self.object_depth_maps[object_index][reference_image_id]
      return images, intrinsics, extrinsics, depth_map
   
   
   def collate_fn(self, batch: List[tuple]):
      # Filter out reference images that did not have context
      batch = [items for items in batch if items[0] is not None]
      # Extract batch of input images and batch of output depth maps separately
      batch_images, batch_intrinsics, batch_extrinsics,  batch_depth_maps = list(zip(*batch))
      batch_images = torch.nn.utils.rnn.pad_sequence(batch_images, batch_first=True)
      batch_intrinsics = torch.nn.utils.rnn.pad_sequence(batch_intrinsics, batch_first=True)
      batch_extrinsics = torch.nn.utils.rnn.pad_sequence(batch_extrinsics, batch_first=True)
      batch_depth_maps = torch.stack(batch_depth_maps)
      return batch_images, batch_intrinsics, batch_extrinsics, batch_depth_maps
