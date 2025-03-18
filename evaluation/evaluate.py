import torch
import numpy as np 
import open3d as o3d 
from tqdm import tqdm


# Depth Estimation Metrics
def abs_rel(pred, gt, mask):
   return torch.mean(torch.abs(pred[mask] - gt[mask]) / gt[mask])

def sq_rel(pred, gt, mask):
   """ Absolute relative error.. """
   return torch.mean((pred[mask] - gt[mask]) ** 2 / gt[mask])

def rmse(pred, gt, mask):
   """ root mean square error"""
   return torch.sqrt(torch.mean((pred[mask] - gt[mask]) ** 2))

def threshold_metric(pred, gt, mask, threshold=1.25):
   """ measures accuracy based on a threshold (δ < threshold)"""
   ratio = torch.max(pred[mask] / gt[mask], gt[mask] / pred[mask])
   return torch.mean((ratio < threshold).float())

def compute_depth_metrics(pred, gt, mask):
   """ computes all depth metrics """
   abs_rel_error = abs_rel(pred, gt, mask)
   sq_rel_error = sq_rel(pred, gt, mask)
   rmse_error = rmse(pred, gt, mask)
   threshold = threshold_metric(pred, gt, mask)
   return {
        "Abs Rel": abs_rel(pred, gt, mask).item(),
        "Sq Rel": sq_rel(pred, gt, mask).item(),
        "RMSE": rmse(pred, gt, mask).item(),
        "δ < 1.25": threshold_metric(pred, gt, mask, 1.25).item(),
        "δ < 1.25²": threshold_metric(pred, gt, mask, 1.25 ** 2).item(),
        "δ < 1.25³": threshold_metric(pred, gt, mask, 1.25 ** 3).item(),
    }


def evaluate_model(model, val_loader, metrics_fn) -> dict:
   model.eval() #set eval mode
   device = next(model.parameters()).device
   depth_metrics_total = []

   with torch.no_grad():
       for batch_images_padded, batch_depth_maps_padded, lengths_images, lengths_depth_maps in tqdm(val_loader, desc="Evaluating"):
            batch_images_padded = batch_images_padded.to(device)
            batch_depth_maps_padded = batch_depth_maps_padded.to(device)

            pred_depths = model(batch_images_padded)  #forward pass

            for i in range(len(batch_images_padded)): 

               pred = pred_depths[i][: lengths_depth_maps[i]]  
               gt = batch_depth_maps_padded[i][: lengths_depth_maps[i]]  
                
               mask = gt > 0  # Define a valid depth mask (assuming no negative depths)
               metrics = metrics_fn(pred, gt, mask)
               depth_metrics_total.append(metrics)
   
   #avg across all batches
   avg_metrics = {key: np.mean([m[key] for m in depth_metrics_total]) for key in depth_metrics_total[0]}

   return avg_metrics



def main(model_checkpoint, data_path, batch_size):
   DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # Use Apple's Metal backend for M1/M2/M3
   print(f"Using device: {DEVICE}")

   # Load trained model
   encoder = ResNetEncoder()
   model = CNNGRUModel(encoder).to(DEVICE)
   model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
   model.eval()
   # Load validation dataset
   val_dataset = BlendedMVS(data_path=data_path, subset=1.0, partition='val')
   val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)

   # Run evaluation
   print("\nEvaluating model...")
   metrics = evaluate_model(model, val_loader, compute_depth_metrics)

   # Print final results
   print("\nEvaluation Results:")
   for metric, value in metrics.items():
      print(f"{metric}: {value:.4f}")

# Command-Line Interface (CLI)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    args = parser.parse_args()

    main(args.model_checkpoint, args.data_path, args.batch_size)