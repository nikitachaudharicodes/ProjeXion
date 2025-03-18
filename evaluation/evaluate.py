import torch
import numpy as np 
import open3d as o3d 
from tqdm import tqdm


# Depth Estimation Metrics
def abs_rel(pred, gt, mask):
   return torch.mean(torch.abs(pred[mask] - gt[mask]) / gt[mask])

def sq_rel(pred, gt, mask):
   return torch.mean((pred[mask] - gt[mask]) ** 2 / gt[mask])

def rmse(pred, gt, mask):
   return torch.sqrt(torch.mean((pred[mask] - gt[mask]) ** 2))

def threshold_metric(pred, gt, mask, threshold=1.25):
   ratio = torch.max(pred[mask] / gt[mask], gt[mask] / pred[mask])
   return torch.mean((ratio < threshold).float())

def compute_depth_metrics(pred, gt, mask):
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
   model.eval()
   device = next(model.parameters()).device
   depth_metrics_total = []

   with torch.no_grad():
       for batch_images_padded, batch_depth_maps_padded, lengths_images, lengths_depth_maps in tqdm(val_loader, desc="Evaluating"):
            batch_images_padded = batch_images_padded.to(device)
            batch_depth_maps_padded = batch_depth_maps_padded.to(device)

            pred_depths = model(batch_images_padded)  
            for i in range(len(batch_images_padded)):
               pred = pred_depths[i][: lengths_depth_maps[i]]  
               gt = batch_depth_maps_padded[i][: lengths_depth_maps[i]]  
                
               mask = gt > 0  # Define a valid depth mask (assuming no negative depths)
               metrics = metrics_fn(pred, gt, mask)
               depth_metrics_total.append(metrics)
   
   avg_metrics = {key: np.mean([m[key] for m in depth_metrics_total]) for key in depth_metrics_total[0]}

   return avg_metrics
