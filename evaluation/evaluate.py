from typing import Dict, List
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from models import ResNet6
from torchvision.transforms.v2 import Resize


# Depth Estimation Metrics
def abs_rel(pred, gt, mask):
    """Absolute relative error"""
    abs_diff = torch.abs(pred - gt)
    rel_error = torch.where(mask, abs_diff / gt, 0)
    return torch.sum(rel_error) / torch.sum(mask)


def sq_rel(pred, gt, mask):
    """Squared relative error"""
    sq_diff = (pred - gt) ** 2
    rel_error = torch.where(mask, sq_diff / gt, 0)
    return torch.sum(rel_error) / torch.sum(mask)


def rmse(pred, gt, mask):
    """root mean square error"""
    sq_diff = (pred - gt) ** 2
    sq_diff = sq_diff * mask
    return torch.sqrt(torch.sum(sq_diff) / torch.sum(mask))


def threshold_metric(pred, gt, mask, threshold=1.25):
    """measures accuracy based on a threshold (δ < threshold)"""
    pred = pred
    gt = gt
    pred_to_gt = torch.where(mask, pred / gt, 0)
    gt_to_pred = torch.where(pred > 0, gt / pred, 0)
    ratio = torch.max(pred_to_gt, gt_to_pred)
    return torch.sum((ratio < threshold) * mask) / torch.sum(mask)


def rmse_log(pred, gt, mask):
    """Root mean squared error of the log-transformed depths"""
    log_pred = torch.log(pred + 1e-7)
    log_gt = torch.log(gt + 1e-7)
    sq_diff = (log_pred - log_gt) ** 2 * mask
    return torch.sqrt(torch.sum(sq_diff) / torch.sum(mask))


def log10(pred, gt, mask):
    """Mean absolute error in log10 space"""
    log10_pred = torch.log10(pred + 1e-7)
    log10_gt = torch.log10(gt + 1e-7)
    abs_diff = torch.abs(log10_pred - log10_gt) * mask
    return torch.sum(abs_diff) / torch.sum(mask)


def mae(pred, gt, mask):
    """Mean absolute error"""
    abs_diff = torch.abs(pred - gt) * mask
    return torch.sum(abs_diff) / torch.sum(mask)


def compute_depth_metrics(pred, gt, mask):
    """computes all depth metrics"""
    abs_rel_error = abs_rel(pred, gt, mask)
    sq_rel_error = sq_rel(pred, gt, mask)
    rmse_error = rmse(pred, gt, mask)
    threshold_metrics = [
        threshold_metric(pred, gt, mask, threshold)
        for threshold in [1.25, 1.25**2, 1.25**3]
    ]
    return {
        "Abs Rel": abs_rel_error,
        "Sq Rel": sq_rel_error,
        "RMSE": rmse_error,
        "δ < 1.25": threshold_metrics[0],
        "δ < 1.25²": threshold_metrics[1],
        "δ < 1.25³": threshold_metrics[2],
    }


@torch.inference_mode()
def evaluate_model(model, val_loader, criterion):
    model.eval()  # set eval mode
    device = next(model.parameters()).device
    depth_metrics_total = []
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = map(lambda x: x.to(device), batch)
            images, intrinsics, extrinsics, depth_maps = batch
            masks = depth_maps > 0

            H, W = depth_maps.shape[-2:]
            pred_to_target_size = Resize((H, W))

            if isinstance(model, ResNet6):
                pred_depths = model(images)
                pred_depths = pred_to_target_size(pred_depths)
                batch_loss = criterion(pred_depths, depth_maps, masks)
                metrics = compute_depth_metrics(pred_depths, depth_maps, masks)
            else:
                initial_depth_map_pred, refined_depth_map_pred = model(
                    images, intrinsics, extrinsics
                )
                initial_depth_map_pred = pred_to_target_size(initial_depth_map_pred)
                refined_depth_map_pred = pred_to_target_size(refined_depth_map_pred)
                batch_loss = criterion(
                    initial_depth_map_pred, depth_maps, masks
                ) + criterion(refined_depth_map_pred, depth_maps, masks)
                metrics = compute_depth_metrics(
                    refined_depth_map_pred, depth_maps, masks
                )
            depth_metrics_total.append(metrics)
            val_loss += batch_loss

    # Average loss and metrics
    # If loss was weighted by valid points: val_loss /= total_valid_points (if accumulated)
    # Else (assuming criterion averages correctly or len(val_loader) is appropriate):
    val_loss /= len(val_loader)  # Or adjust depending on criterion behavior

    # Average metrics across batches
    if depth_metrics_total:
        avg_metrics = {
            key: torch.tensor([m[key] for m in depth_metrics_total]).mean().item()
            for key in depth_metrics_total[0]
        }
    else:
        avg_metrics = {
            key: 0
            for key in [
                "Abs Rel",
                "Sq Rel",
                "RMSE",
                "δ < 1.25",
                "δ < 1.25²",
                "δ < 1.25³",
            ]
        }  # Default if no valid data

    return val_loss.item(), avg_metrics


def main(model_checkpoint, data_path, batch_size):
    DEVICE = (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )  # Use Apple's Metal backend for M1/M2/M3
    print(f"Using device: {DEVICE}")

    # Load trained model
    encoder = ResNetEncoder()
    model = CNNGRUModel(encoder).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()
    # Load validation dataset
    val_dataset = BlendedMVS(data_path=data_path, subset=1.0, partition="val")
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn
    )

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
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    args = parser.parse_args()

    main(args.model_checkpoint, args.data_path, args.batch_size)
