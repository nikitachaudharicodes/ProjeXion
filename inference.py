from dataloaders import BlendedMVS
from torch.utils.data import DataLoader
from models import MVSNet
import torch
from torchvision.transforms.v2 import Resize
import matplotlib.pyplot as plt
from pathlib import Path

@torch.inference_mode()
def main():
   predictions_path = Path('predictions')
   predictions_path.mkdir(parents=True, exist_ok=True)
   DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = MVSNet(70).to(DEVICE).eval()
   checkpoint = torch.load('checkpoints/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   data = BlendedMVS(
      'data/dataset_low_res', height=640, width=640, subset=0.2, partition='test',
      context_size=4
   )
   dataloader = DataLoader(data, 6, shuffle=False, collate_fn=data.collate_fn)
   preds = []
   for batch in dataloader:
      batch = map(lambda x: x.to(DEVICE), batch)
      images, intrinsics, extrinsics, depth_maps = batch
      mask = depth_maps > 0
      H, W = depth_maps.shape[-2:]
      pred_to_target_size = Resize((H, W))

      with torch.no_grad():
         initial_depth_map_pred, refined_depth_map_pred = model(images, intrinsics, extrinsics)
         pred_depths = pred_to_target_size(refined_depth_map_pred)
         preds.append(pred_depths)
   preds = torch.concat(preds)
   for i, pred in enumerate(preds):
      # Show the image
      plt.imshow(torch.permute(pred, (1, 2, 0)).detach().cpu().numpy(), cmap="gray")  # Use 'gray' colormap for grayscale images
      plt.axis("off")  # Hide axes
      plt.savefig(predictions_path / f'{i:08d}.jpeg')

if __name__ == '__main__':
   main()