"""
Creates an iterator over the BlendedMVS dataset that returns a
sequence of images (features) and depth maps (target)
"""

import random
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import load_pfm, parse_pairs
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
                partition:Literal['train', 'val', 'test']='train',
                context_size:int=0,
                img_size: Tuple[int,int] = (512,640)  # (H, W)
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
      self.data_path = Path(data_path)
      assert self.data_path.exists()
      assert partition in ['train', 'val', 'test'], "Partition must be 'train', 'val', or 'test'"
      list_file = self.data_path / f"{partition}_list.txt"
      assert list_file.exists(), f"File {list_file} does not exist"
      assert 0 < subset <= 1, f"Subset value should be between (0, 1]"
      self.img_size = img_size
      H, W = self.img_size
      assert H % 32 == 0 and W % 32 == 0, "img_size dims must be divisible by 32"
      self.target_h = height
      self.target_w = width
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
      
      H, W = self.img_size
      pil_to_tensor = T.Compose([
         T.Resize((H, W), interpolation=Image.BILINEAR),
         T.ToImage(),
         T.ToDtype(torch.float32, scale=True),
         T.Normalize([0.5]*3, [0.5]*3),
      ])
      depth_map_to_tensor = T.Compose([
         T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
         T.Resize(self.img_size, interpolation=Image.BILINEAR),
      ])


      object_images = []
      object_depth_maps = []
      object_image_pairs = []
      self.object_Ks_scaled   = []  # list of (T×3×3)
      self.object_extrinsics  = []  # list of (T×3×4)

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

         # — load & scale cams
         Ks_scene  = []
         Rts_scene = []
         cams_dir = object_path/'cams'
         for cam_file in sorted(cams_dir.iterdir()):
               if cam_file.name == 'pair.txt':
                  continue
               lines = [l.strip() for l in cam_file.open() if l.strip()]
               # extrinsic
               i_ext = lines.index('extrinsic')
               ext_vals = " ".join(lines[i_ext+1:i_ext+4])
               ext_mat = np.fromstring(ext_vals, sep=" ").reshape(3,4)
               Rts_scene.append(torch.from_numpy(ext_mat).float())
               # intrinsic
               i_int = lines.index('intrinsic')
               kin_vals = " ".join(lines[i_int+1:i_int+4])
               K0 = np.fromstring(kin_vals, sep=" ").reshape(3,3)
               # compute scale factors from principal point
               sx = self.target_w / (2 * K0[0,2])
               sy = self.target_h / (2 * K0[1,2])
               K2 = K0.copy()
               K2[0,0] *= sx;  K2[0,2] *= sx
               K2[1,1] *= sy;  K2[1,2] *= sy
               Ks_scene.append(torch.from_numpy(K2).float())
         self.object_extrinsics.append(torch.stack(Rts_scene))  # (T,3,4)
         self.object_Ks_scaled.append(torch.stack(Ks_scene))    # (T,3,3)


      assert len(object_images) == len(object_depth_maps) == len(object_image_pairs) == self.length
      self.object_image_pairs = object_image_pairs
      self.object_images = object_images
      self.object_depth_maps = object_depth_maps

   def __len__(self):
      return self.length
   
   def __getitem__(self, object_index):
      # grab everything for this object
      images   = self.object_images[object_index]      # (T,3,H,W)
      pairs    = self.object_image_pairs[object_index] # list of length T
      Ks_all   = self.object_Ks_scaled[object_index]   # (T,3,3)
      Rts_all  = self.object_extrinsics[object_index]  # (T,3,4)

      T = images.shape[0]
      # pick a valid ref index in [0..T-1]
      ref_id = random.randint(0, T-1)

      # reference + context images
      ref_img = images[ref_id]                         # (3,H,W)
      ctx_ids = pairs[ref_id][:self.context_size]      # up to context_size ints
      # pad or truncate so we always have exactly context_size ids:
      all_ctx = pairs[ref_id]
      if len(all_ctx) < self.context_size:
          all_ctx = all_ctx + [all_ctx[-1]] * (self.context_size - len(all_ctx))
      else:
          all_ctx = all_ctx[:self.context_size]
      ctx_ids = all_ctx
      ctx_imgs = [images[i] for i in ctx_ids]          # list of (3,H,W)
      imgs = torch.stack([ref_img, *ctx_imgs], dim=0)  # (1+ctx,3,H,W)

      # target depth
      depth = self.object_depth_maps[object_index][ref_id]  # (1,H,W)

      # corresponding cams
      K_ref  = Ks_all[ref_id]       # (3,3)
      K_src  = Ks_all[ctx_ids]      # (ctx,3,3)
      Rt_ref = Rts_all[ref_id]      # (3,4)
      Rt_src = Rts_all[ctx_ids]     # (ctx,3,4)

      return imgs, depth, K_ref, K_src, Rt_ref, Rt_src
   
   
   def collate_fn(self, batch: List[tuple]):
      # Extract batch of input images and batch of output depth maps separately
      imgs, depths, K_refs, K_srcs, Rt_refs, Rt_srcs = zip(*batch)
      # (B, T, 3, H, W)
      b_imgs  = torch.stack(imgs,   dim=0)
      # (B, 1, H, W)
      b_depth = torch.stack(depths, dim=0)
      # (B, 3, 3)
      b_K_ref = torch.stack(K_refs, dim=0)
      # pad each (ctx,3,3) up to max(ctx) in the batch → (B, ctx_max, 3, 3)
      b_K_src = pad_sequence(K_srcs, batch_first=True, padding_value=0.0)
      # (B, 3, 4)
      b_Rt_ref = torch.stack(Rt_refs, dim=0)
      # (B, ctx_max, 3, 4)
      b_Rt_src = pad_sequence(Rt_srcs, batch_first=True, padding_value=0.0)
      return b_imgs, b_depth, b_K_ref, b_K_src, b_Rt_ref, b_Rt_src
