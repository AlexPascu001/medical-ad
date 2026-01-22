"""
Evaluation utilities for anomaly detection
Image-level and pixel-level AUROC, AUPR, operating points
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.utils import resample
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    compute_pixel_auroc: bool = True,
    target_size: Tuple[int, int] = (256, 256)
) -> Dict[str, float]:
    """
    Evaluate anomaly detection model
    
    Args:
        model: AnomalyDetector model
        dataloader: DataLoader with test/val data
        device: Device to run on
        compute_pixel_auroc: Whether to compute pixel-level metrics
        target_size: Image size for upsampling anomaly maps
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_image_scores = []
    all_labels = []
    all_pixel_scores = []
    all_pixel_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Compute anomaly scores
            outputs = model.compute_anomaly_scores(
                images,
                return_maps=compute_pixel_auroc,
                target_size=target_size if compute_pixel_auroc else None
            )
            
            # Collect image-level scores
            image_scores = outputs['image_scores'].cpu().numpy()
            all_image_scores.append(image_scores)
            all_labels.append(labels)
            
            # Collect pixel-level scores if available
            if compute_pixel_auroc and 'pixel_scores' in outputs:
                pixel_scores = outputs['pixel_scores'].cpu().numpy()
                all_pixel_scores.append(pixel_scores)
                
                # Get masks if available
                if 'mask' in batch:
                    masks = batch['mask'].cpu().numpy()
                    all_pixel_masks.append(masks)
    
    # Concatenate all batches
    image_scores = np.concatenate(all_image_scores)
    labels = np.concatenate(all_labels)
    
    # === IMAGE-LEVEL METRICS ===
    image_auroc = roc_auc_score(labels, image_scores)
    image_aupr = average_precision_score(labels, image_scores)
    
    metrics = {
        'image_auroc': image_auroc,
        'image_aupr': image_aupr,
        'num_normal': (labels == 0).sum(),
        'num_anomaly': (labels == 1).sum()
    }
    
    # === PIXEL-LEVEL METRICS ===
    if all_pixel_scores and all_pixel_masks:
        print(f"  Pixel-level data collected: {len(all_pixel_scores)} batches of scores, {len(all_pixel_masks)} batches of masks")
        
        # Concatenate all batches first
        try:
            pixel_scores = np.concatenate(all_pixel_scores)  # (N, H, W)
            pixel_masks = np.concatenate(all_pixel_masks)    # (N, H, W)
            
            print(f"  Concatenated shapes: pixel_scores={pixel_scores.shape}, pixel_masks={pixel_masks.shape}")
            
            # Check if spatial dimensions match
            if pixel_scores.shape[1:] != pixel_masks.shape[1:]:
                print(f"  WARNING: Spatial dimension mismatch! Resizing scores to match masks...")
                print(f"    Scores: {pixel_scores.shape[1:]} -> Masks: {pixel_masks.shape[1:]}")
                
                # Resize scores to match mask dimensions
                from scipy.ndimage import zoom
                scale_h = pixel_masks.shape[1] / pixel_scores.shape[1]
                scale_w = pixel_masks.shape[2] / pixel_scores.shape[2]
                
                # Resize each sample individually
                resized_scores = []
                for i in range(pixel_scores.shape[0]):
                    resized = zoom(pixel_scores[i], (scale_h, scale_w), order=1)  # bilinear
                    resized_scores.append(resized)
                pixel_scores = np.array(resized_scores)
                print(f"  Resized scores to: {pixel_scores.shape}")
            
            # Check if batch dimensions match
            if pixel_scores.shape[0] != pixel_masks.shape[0]:
                print(f"  WARNING: Batch dimension mismatch! Truncating to minimum...")
                min_samples = min(pixel_scores.shape[0], pixel_masks.shape[0])
                pixel_scores = pixel_scores[:min_samples]
                pixel_masks = pixel_masks[:min_samples]
                print(f"  Truncated to {min_samples} samples")
            
            # Flatten for ROC computation
            pixel_scores_flat = pixel_scores.flatten()
            pixel_masks_flat = pixel_masks.flatten()
            
            num_anomaly_pixels = pixel_masks_flat.sum()
            print(f"  Total pixels: {len(pixel_masks_flat)}, Anomalous pixels: {num_anomaly_pixels}")
            
            # Only compute if there are anomalous pixels
            if num_anomaly_pixels > 0:
                pixel_auroc = roc_auc_score(pixel_masks_flat, pixel_scores_flat)
                pixel_aupr = average_precision_score(pixel_masks_flat, pixel_scores_flat)
                
                print(f"  Pixel AUROC: {pixel_auroc:.4f}, Pixel AUPR: {pixel_aupr:.4f}")
                
                metrics.update({
                    'pixel_auroc': pixel_auroc,
                    'pixel_aupr': pixel_aupr
                })
            else:
                print(f"  WARNING: No anomalous pixels found in masks!")
        except Exception as e:
            print(f"  ERROR computing pixel-level metrics: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Pixel-level data not available: scores={len(all_pixel_scores)} batches, masks={len(all_pixel_masks)} batches")
    
    return metrics


def compute_operating_points(
    labels: np.ndarray,
    scores: np.ndarray,
    fpr_targets: list = [0.01, 0.05, 0.1]
) -> Dict[str, Dict]:
    """
    Compute operating points at specific FPR levels
    
    Args:
        labels: Binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores
        fpr_targets: Target FPR levels
    
    Returns:
        Dictionary with threshold and TPR for each FPR target
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    operating_points = {}
    
    for target_fpr in fpr_targets:
        # Find threshold that gives closest FPR to target
        idx = np.argmin(np.abs(fpr - target_fpr))
        
        operating_points[f'fpr_{target_fpr}'] = {
            'threshold': float(thresholds[idx]),
            'fpr': float(fpr[idx]),
            'tpr': float(tpr[idx])
        }
    
    return operating_points


def bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute confidence intervals for AUROC using bootstrap
    
    Args:
        labels: Binary labels
        scores: Anomaly scores
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed
    
    Returns:
        Dictionary with AUROC statistics
    """
    np.random.seed(random_state)
    
    n_samples = len(labels)
    aurocs = []
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        
        labels_boot = labels[indices]
        scores_boot = scores[indices]
        
        # Skip if bootstrap sample has only one class
        if len(np.unique(labels_boot)) < 2:
            continue
        
        auroc = roc_auc_score(labels_boot, scores_boot)
        aurocs.append(auroc)
    
    aurocs = np.array(aurocs)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    return {
        'auroc_mean': float(aurocs.mean()),
        'auroc_std': float(aurocs.std()),
        'auroc_lower': float(np.percentile(aurocs, lower_percentile)),
        'auroc_upper': float(np.percentile(aurocs, upper_percentile)),
        'confidence_level': confidence_level
    }


def evaluate_comprehensive(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    compute_pixel: bool = True,
    target_size: Tuple[int, int] = (256, 256)
) -> Dict:
    """
    Comprehensive evaluation with all metrics and visualizations
    
    Args:
        model: Anomaly detector model
        dataloader: Test dataloader
        device: Device
        save_dir: Directory to save results
        compute_pixel: Whether to compute pixel-level metrics
        target_size: Image size
    
    Returns:
        Complete evaluation results
    """
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic evaluation
    print("Computing metrics...")
    metrics = evaluate_model(
        model, dataloader, device,
        compute_pixel_auroc=compute_pixel,
        target_size=target_size
    )
    
    print(f"Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"Image AUPR: {metrics['image_aupr']:.4f}")
    
    if 'pixel_auroc' in metrics:
        print(f"Pixel AUROC: {metrics['pixel_auroc']:.4f}")
        print(f"Pixel AUPR: {metrics['pixel_aupr']:.4f}")
    
    # Collect scores for additional analysis
    model.eval()
    all_image_scores = []
    all_labels = []
    all_pixel_scores = []
    all_pixel_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Get image scores
            outputs = model.compute_anomaly_scores(images, return_maps=False)
            scores = outputs['image_scores'].cpu().numpy()
            
            all_image_scores.append(scores)
            all_labels.append(labels)
            
            # Get pixel scores if computing pixel metrics
            if compute_pixel:
                pixel_outputs = model.compute_anomaly_scores(
                    images, 
                    return_maps=True, 
                    target_size=target_size
                )
                if 'pixel_scores' in pixel_outputs:
                    pixel_scores = pixel_outputs['pixel_scores'].cpu().numpy()
                    all_pixel_scores.append(pixel_scores)
                    
                    if 'mask' in batch:
                        masks = batch['mask'].cpu().numpy()
                        all_pixel_masks.append(masks)
    
    image_scores = np.concatenate(all_image_scores)
    labels = np.concatenate(all_labels)
    
    # Operating points
    print("\nComputing operating points...")
    op_points = compute_operating_points(labels, image_scores)
    metrics['operating_points'] = op_points
    
    for key, vals in op_points.items():
        print(f"  {key}: TPR={vals['tpr']:.4f} at threshold={vals['threshold']:.4f}")
    
    # Bootstrap confidence intervals
    print("\nComputing confidence intervals...")
    ci_results = bootstrap_auroc(labels, image_scores, n_bootstraps=1000)
    metrics['confidence_intervals'] = ci_results
    
    print(f"  AUROC: {ci_results['auroc_mean']:.4f} ± {ci_results['auroc_std']:.4f}")
    print(f"  95% CI: [{ci_results['auroc_lower']:.4f}, {ci_results['auroc_upper']:.4f}]")
    
    # Plot ROC curve (image-level)
    plot_roc_curve(labels, image_scores, save_dir / 'roc_curve.png', level='Image')
    
    # Plot score distributions (image-level)
    plot_score_distributions(labels, image_scores, save_dir / 'score_distributions.png')
    
    # Plot pixel-level ROC curve if available
    if compute_pixel and all_pixel_scores and all_pixel_masks:
        print("\nPlotting pixel-level ROC curve...")
        try:
            pixel_scores_concat = np.concatenate(all_pixel_scores)
            pixel_masks_concat = np.concatenate(all_pixel_masks)
            
            # Handle size mismatch
            if pixel_scores_concat.shape[1:] != pixel_masks_concat.shape[1:]:
                from scipy.ndimage import zoom
                scale_h = pixel_masks_concat.shape[1] / pixel_scores_concat.shape[1]
                scale_w = pixel_masks_concat.shape[2] / pixel_scores_concat.shape[2]
                
                resized_scores = []
                for i in range(pixel_scores_concat.shape[0]):
                    resized = zoom(pixel_scores_concat[i], (scale_h, scale_w), order=1)
                    resized_scores.append(resized)
                pixel_scores_concat = np.array(resized_scores)
            
            # Flatten for plotting
            pixel_scores_flat = pixel_scores_concat.flatten()
            pixel_masks_flat = pixel_masks_concat.flatten()
            
            # Only plot if we have anomalous pixels
            if pixel_masks_flat.sum() > 0:
                plot_roc_curve(
                    pixel_masks_flat, 
                    pixel_scores_flat, 
                    save_dir / 'pixel_roc_curve.png',
                    level='Pixel'
                )
                print(f"  Pixel-level ROC curve saved")
        except Exception as e:
            print(f"  Warning: Could not plot pixel-level ROC curve: {e}")
    
    # Save metrics to JSON (convert numpy types to native Python types)
    import json
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(save_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)
    
    print(f"\nResults saved to {save_dir}")
    
    return metrics


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, save_path: str, level: str = 'Image'):
    """Plot ROC curve
    
    Args:
        labels: Ground truth labels
        scores: Anomaly scores
        save_path: Path to save plot
        level: 'Image' or 'Pixel' for title
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {level} Level', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distributions(labels: np.ndarray, scores: np.ndarray, save_path: str):
    """Plot anomaly score distributions for normal vs anomalous samples"""
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Anomaly Score Distributions', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 16,
    target_size: Tuple[int, int] = (256, 256)
):
    """
    Visualize sample predictions with anomaly maps
    
    Args:
        model: Anomaly detector
        dataloader: Test dataloader
        device: Device
        save_dir: Save directory
        num_samples: Number of samples to visualize
        target_size: Image size
    """
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    samples_collected = 0
    normal_samples = []
    anomaly_samples = []
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            outputs = model.compute_anomaly_scores(
                images,
                return_maps=True,
                target_size=target_size
            )
            
            image_scores = outputs['image_scores'].cpu().numpy()
            pixel_scores = outputs['pixel_scores'].cpu().numpy() if 'pixel_scores' in outputs else None
            
            # Collect samples
            for i in range(len(images)):
                if samples_collected >= num_samples:
                    break
                
                sample = {
                    'image': images[i].cpu().numpy(),
                    'label': labels[i],
                    'score': image_scores[i],
                    'pixel_map': pixel_scores[i] if pixel_scores is not None else None,
                    'mask': batch['mask'][i].cpu().numpy() if 'mask' in batch else None
                }
                
                if labels[i] == 0 and len(normal_samples) < num_samples // 2:
                    normal_samples.append(sample)
                    samples_collected += 1
                elif labels[i] == 1 and len(anomaly_samples) < num_samples // 2:
                    anomaly_samples.append(sample)
                    samples_collected += 1
    
    # Visualize normal samples
    if normal_samples:
        _plot_samples(normal_samples, save_dir / 'normal_samples.png', 'Normal Samples')
    
    # Visualize anomaly samples
    if anomaly_samples:
        _plot_samples(anomaly_samples, save_dir / 'anomaly_samples.png', 'Anomaly Samples')


def _plot_samples(samples: list, save_path: str, title: str):
    """Helper to plot sample grid"""
    n = len(samples)
    cols = 4
    rows = (n + cols - 1) // cols
    
    # Determine if we have pixel maps
    has_maps = samples[0]['pixel_map'] is not None if len(samples) > 0 else False
    has_masks = samples[0]['mask'] is not None if len(samples) > 0 else False
    
    n_cols = 1 + (1 if has_maps else 0) + (1 if has_masks else 0)
    
    fig, axes = plt.subplots(rows, cols * n_cols, figsize=(cols * n_cols * 3, rows * 3))
    
    if rows == 1 and cols * n_cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols * n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, sample in enumerate(samples):
        row = idx // cols
        col_offset = (idx % cols) * n_cols
        
        # Original image
        img = sample['image'][0]  # Take first channel
        axes[row, col_offset].imshow(img, cmap='gray')
        axes[row, col_offset].set_title(f"Score: {sample['score']:.3f}")
        axes[row, col_offset].axis('off')
        
        col_idx = col_offset + 1
        
        # Anomaly map
        if has_maps and sample['pixel_map'] is not None:
            axes[row, col_idx].imshow(sample['pixel_map'], cmap='jet')
            axes[row, col_idx].set_title('Anomaly Map')
            axes[row, col_idx].axis('off')
            col_idx += 1
        
        # Ground truth mask
        if has_masks and sample['mask'] is not None:
            axes[row, col_idx].imshow(sample['mask'], cmap='gray')
            axes[row, col_idx].set_title('Ground Truth')
            axes[row, col_idx].axis('off')
    
    # Hide unused subplots
    for row in range(rows):
        for col in range(cols * n_cols):
            idx = row * cols + col // n_cols
            if idx >= len(samples):
                axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {save_path}")


def analyze_anchor_assignments(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str
):
    """
    Analyze which anchors are assigned to normal vs anomalous samples
    
    Args:
        model: Anomaly detector
        dataloader: Dataloader
        device: Device
        save_dir: Save directory
    """
    from pathlib import Path
    import pandas as pd
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    anchor_assignments = {
        'normal': np.zeros(model.n_anchors),
        'anomaly': np.zeros(model.n_anchors)
    }
    
    all_distances_normal = []
    all_distances_anomaly = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            outputs = model.compute_anomaly_scores(images, return_maps=False)
            assigned = outputs['assigned_anchors'].cpu().numpy()
            distances = outputs['all_distances'].cpu().numpy()
            
            # Track assignments
            for i, label in enumerate(labels):
                anchor = assigned[i]
                if label == 0:
                    anchor_assignments['normal'][anchor] += 1
                    all_distances_normal.append(distances[i])
                else:
                    anchor_assignments['anomaly'][anchor] += 1
                    all_distances_anomaly.append(distances[i])
    
    # Plot assignment distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    x = np.arange(model.n_anchors)
    width = 0.35
    
    axes[0].bar(x - width/2, anchor_assignments['normal'], width, label='Normal', alpha=0.8)
    axes[0].bar(x + width/2, anchor_assignments['anomaly'], width, label='Anomaly', alpha=0.8)
    axes[0].set_xlabel('Anchor ID', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Anchor Assignments', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Distance distributions
    if all_distances_normal and all_distances_anomaly:
        all_distances_normal = np.array(all_distances_normal)
        all_distances_anomaly = np.array(all_distances_anomaly)
        
        # Plot min distances
        min_dist_normal = all_distances_normal.min(axis=1)
        min_dist_anomaly = all_distances_anomaly.min(axis=1)
        
        axes[1].hist(min_dist_normal, bins=50, alpha=0.6, label='Normal', density=True)
        axes[1].hist(min_dist_anomaly, bins=50, alpha=0.6, label='Anomaly', density=True)
        axes[1].set_xlabel('Min Distance to Anchor', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('Distance Distributions', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'anchor_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats = {
        'normal_assignments': anchor_assignments['normal'].tolist(),
        'anomaly_assignments': anchor_assignments['anomaly'].tolist()
    }
    
    import json
    with open(save_dir / 'anchor_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved anchor analysis to {save_dir}")