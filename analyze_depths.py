from utils import load_pfm
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torchvision.transforms.v2 import Resize
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd

def main(partition:str, by_object:bool):
   depth_maps = get_depth_maps(
      data_path='data/dataset_low_res',
      partition=partition,
   )

   print("Number of objects: ", len(depth_maps))
   print("Number of depth maps by object: ", [len(object) for object in depth_maps])

   # N objects, with V views, of depth maps of (H, W) size
   depth_maps = [
      torch.concat(
         [
            # Removes masks and flattens
            object[view][object[view] > 0].reshape((-1,)) # The view is 2D
            for view in object # Object is a dict
         ]
      )
      for object in depth_maps # Train is a list
   ]
   if not by_object:
      depth_maps=torch.concat(depth_maps)

   # =================================================================
   # Visualize
   # =================================================================

   # Raw
   fig, ax = plt.subplots(1,1, figsize=(4,4))
   if by_object:
      for object in depth_maps:
         ax.hist(object, bins=30, range=(0, 256), histtype='step', density=True)
   else:
      ax.hist(depth_maps, bins=30, density=True)
   ax.set_title(f"{partition.title()}: Raw Distribution")
   ax.set_xlabel('Depth')
   fig.savefig(f'visualizations/{partition}_raw_distribution.jpeg')

   # Log
   fig, ax = plt.subplots(1,1, figsize=(4,4))
   if by_object:
      for object in depth_maps:
         ax.hist(np.log(object), bins=30, range=(-1, 2.4), histtype='step', density=True)
   else:
      ax.hist(np.log(depth_maps), bins=30, density=True)
   ax.set_xlabel('log(Depth)')
   ax.set_title(f"{partition.title()}: Log Distribution")
   fig.savefig(f'visualizations/{partition}_log_distribution.jpeg')


   return


def get_depth_maps(data_path:str, partition:str) -> List[Dict[int, np.ndarray]]:
   resize = Resize((200,200))
   data_path = Path(data_path)
   assert data_path.exists()
   assert partition in ['train', 'val', 'test'], "Partition must be 'train', 'val', or 'test'"
   list_file = data_path / f"{partition}_list.txt"

   # Parse list file
   with open(list_file, 'r') as f:
      valid_objects = set(line.strip() for line in f.readlines())
   assert valid_objects, f"{list_file} was empty"
   
   # Filter objects in the partition
   object_paths = sorted([p for p in data_path.iterdir() if p.is_dir() and p.name in valid_objects])
   assert len(object_paths) == len(valid_objects), f"Only found {len(object_paths)} of the {len(valid_objects)} present in {list_file}"

   object_depth_maps = []
   for object_path in tqdm(object_paths, desc=f"Loading {partition} images"):
      # Load depth maps
      depth_maps_paths = sorted(list((object_path / 'rendered_depth_maps').iterdir()))
      depth_maps = {}
      for depth_map_path in depth_maps_paths:
         depth_map_id = int(depth_map_path.stem)
         depth_map = load_pfm(str(depth_map_path))
         depth_map = resize(torch.tensor(depth_map))
         depth_maps[depth_map_id] = depth_map
      object_depth_maps.append(depth_maps)
   return object_depth_maps



if __name__ == '__main__':
   argparser = ArgumentParser()
   argparser.add_argument('-p', '--partition')
   argparser.add_argument('-o', '--by-object', action='store_true')
   args = argparser.parse_args()
   main(
      partition=args.partition,
      by_object=args.by_object
   )