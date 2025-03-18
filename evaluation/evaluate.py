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

# 3D Point Cloud Metrics 
# def chamfer_distance(pred_points, gt_points):
#    pred_pcd = o3d.geometry.PointCloud()
#    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)

#    gt_pcd = o3d.geometry.PointCloud()
#    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

#    dist1 = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
#    dist2 = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))

#    # might want to experiment after squaring the distances, the formula does it, most implementations dont
#    return np.mean(dist1) + np.mean(dist2)

def evaluate_model(model, val_loader, metrics_fn) -> dict:
   pass