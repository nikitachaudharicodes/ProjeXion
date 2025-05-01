from dataloaders import BlendedMVS
from torch.utils.data import DataLoader
from models import MVSNet, ProjeXion
import torch
from torchvision.transforms.v2 import Resize
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

@torch.inference_mode()
def main(architecture: str, run_name: str):
   BATCH_SIZE = 32
   DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Model
   if architecture == 'mvsnet':
      model = MVSNet(70).to(DEVICE).eval()
   elif architecture == 'projexion':
      model = ProjeXion(70).to(DEVICE).eval()
   else:
      raise ValueError()
   
   checkpoint_path = Path('checkpoints', run_name, 'best_model.pth')
   checkpoint = torch.load(str(checkpoint_path))
   model.load_state_dict(checkpoint['model_state_dict'])
   del checkpoint
   
   # Data
   data = BlendedMVS(
      'data/dataset_low_res', height=160, width=160, subset=1, partition='test',
      context_size=5
   )
   dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data.collate_fn)

   # Inference
   cumulative_index = 0
   predictions_path = Path('predictions', run_name)
   predictions_path.mkdir(parents=True, exist_ok=True)
   for batch in tqdm(dataloader, desc='Inference'):
      batch = map(lambda x: x.to(DEVICE), batch)
      images, intrinsics, extrinsics, depth_maps = batch
      H, W = depth_maps.shape[-2:]
      pred_to_target_size = Resize((H, W))

      with torch.no_grad():
         initial_depth_map_pred, refined_depth_map_pred = model(images, intrinsics, extrinsics)
         pred_depths = pred_to_target_size(refined_depth_map_pred)

      for i, pred in enumerate(pred_depths):
         image_data = pred.detach().cpu().numpy()
         # Store prediction
         object_id, view_id = data.index[cumulative_index + i]
         object_name = data.object_paths[object_id].name
         object_dir = predictions_path / object_name
         object_dir.mkdir(parents=True, exist_ok=True)
         image_path = predictions_path / object_name / f'{view_id:08d}.npy'
         np.save(str(image_path), image_data)
      cumulative_index += BATCH_SIZE

if __name__ == '__main__':
   argparser = ArgumentParser()
   argparser.add_argument('architecture')
   argparser.add_argument('run_name')
   args = argparser.parse_args()
   main(
      architecture=args.architecture,
      run_name=args.run_name,
   )