from typing import Dict, List
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


def rmse_log(pred, gt, mask):
    """ Root mean squared error of the log-transformed depths """
    log_pred = torch.log(pred[mask])
    log_gt = torch.log(gt[mask])
    return torch.sqrt(torch.mean((log_pred - log_gt) ** 2))

def log10(pred, gt, mask):
    """ Mean absolute error in log10 space """
    log10_pred = torch.log10(pred[mask])
    log10_gt = torch.log10(gt[mask])
    return torch.mean(torch.abs(log10_pred - log10_gt))

def mae(pred, gt, mask):
    """ Mean absolute error """
    return torch.mean(torch.abs(pred[mask] - gt[mask]))

def compute_depth_metrics(pred, gt, mask):
   """ computes all depth metrics """
   # These functions compute the mean over all valid elements indicated by the mask across the batch.
   return {
        "Abs Rel": abs_rel(pred, gt, mask).item(),
        "Sq Rel": sq_rel(pred, gt, mask).item(),
        "RMSE": rmse(pred, gt, mask).item(),
        "δ < 1.25": threshold_metric(pred, gt, mask, 1.25).item(),
        "δ < 1.25²": threshold_metric(pred, gt, mask, 1.25 ** 2).item(),
        "δ < 1.25³": threshold_metric(pred, gt, mask, 1.25 ** 3).item(),
    }


@torch.inference_mode()
def evaluate_model(model, val_loader, criterion):
   model.eval() #set eval mode
   device = next(model.parameters()).device
   depth_metrics_total = []
   val_loss = 0
   with torch.no_grad():
       for batch_images_padded, batch_depth_maps_padded, lengths_images, lengths_depth_maps in tqdm(val_loader, desc="Evaluating"):
            batch_images_padded = batch_images_padded.to(device)
            batch_depth_maps_padded = batch_depth_maps_padded.to(device)

            pred_depths, pred_lens = model(batch_images_padded, lengths_images)  #forward pass
            
            # Create a mask for valid sequence elements (excluding padding)
            max_len = batch_depth_maps_padded.shape[1] # Assuming shape is B x N_max
            seq_range = torch.arange(max_len, device=device)[None, :] # Shape: 1 x N_max
            lengths_tensor = torch.tensor(lengths_depth_maps, device=device)[:, None] # Shape: B x 1
            seq_mask = seq_range < lengths_tensor # Shape: B x N_max

            # Create a mask for valid ground truth depth values (> 0)
            gt_mask = batch_depth_maps_padded > 0 # Shape: B x N_max
            
            # Combine masks: only consider valid sequence elements with valid ground truth
            valid_mask = gt_mask & seq_mask

            # Compute loss (criterion might need adjustment if it doesn't handle padding)
            # Assuming criterion handles padding or uses the mask internally
            # If not, you might need: loss = criterion(pred_depths[valid_mask], batch_depth_maps_padded[valid_mask])
            # Make sure pred_depths has the same shape B x N_max as batch_depth_maps_padded
            # If pred_depths has a different shape (e.g., requires selection based on pred_lens), adjust accordingly.
            # For now, assuming criterion and shapes match for simplicity.
            # Ensure pred_depths has the shape [B, N_max] for the following metric calculation.
            # If model outputs variable lengths, it needs padding/masking similar to GT.
            # Let's assume pred_depths is already padded/masked appropriately based on pred_lens to match batch_depth_maps_padded shape.

            # If criterion needs unpadded/masked inputs:
            # current_loss = criterion(pred_depths[valid_mask], batch_depth_maps_padded[valid_mask])
            # val_loss += current_loss * valid_mask.sum() # Weight loss by number of valid points
            # total_valid_points += valid_mask.sum() # Accumulate total valid points if needed for averaging later

            # Simplified loss calculation assuming criterion handles batch/mask:
            val_loss += criterion(pred_depths, batch_depth_maps_padded, valid_mask) # Pass mask if needed by criterion

            # Compute metrics for the entire batch using the valid mask
            if valid_mask.any(): # Ensure there are valid elements before computing metrics
                metrics = compute_depth_metrics(pred_depths, batch_depth_maps_padded, valid_mask)
                depth_metrics_total.append(metrics)

   # Average loss and metrics
   # If loss was weighted by valid points: val_loss /= total_valid_points (if accumulated)
   # Else (assuming criterion averages correctly or len(val_loader) is appropriate):
   val_loss /= len(val_loader) # Or adjust depending on criterion behavior

   # Average metrics across batches
   if depth_metrics_total:
        avg_metrics = {key: np.mean([m[key] for m in depth_metrics_total]) for key in depth_metrics_total[0]}
   else:
        avg_metrics = {key: 0 for key in ["Abs Rel", "Sq Rel", "RMSE", "δ < 1.25", "δ < 1.25²", "δ < 1.25³"]} # Default if no valid data

   return val_loss.item(), avg_metrics


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