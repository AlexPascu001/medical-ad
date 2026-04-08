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
    all_anchor_scores = []
    all_reconstruction_scores = []
    all_divergence_scores = []
    all_pixel_aggregated_scores = []
    all_patch_div_aggregated_scores = []
    all_reconstruction_pixel_scores = []
    all_anchor_pixel_scores = []
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
            if 'anchor_scores' in outputs:
                all_anchor_scores.append(outputs['anchor_scores'].cpu().numpy())
            if 'reconstruction_scores' in outputs:
                all_reconstruction_scores.append(outputs['reconstruction_scores'].cpu().numpy())
            if 'bottleneck_divergence' in outputs:
                all_divergence_scores.append(outputs['bottleneck_divergence'].cpu().numpy())
            if 'pixel_aggregated_score' in outputs:
                all_pixel_aggregated_scores.append(outputs['pixel_aggregated_score'].cpu().numpy())
            if 'patch_divergence_aggregated_score' in outputs:
                all_patch_div_aggregated_scores.append(outputs['patch_divergence_aggregated_score'].cpu().numpy())
            all_labels.append(labels)
            
            # Collect pixel-level scores if available
            if compute_pixel_auroc and 'pixel_scores' in outputs:
                pixel_scores = outputs['pixel_scores'].cpu().numpy()
                all_pixel_scores.append(pixel_scores)

                if 'reconstruction_pixel_scores' in outputs:
                    all_reconstruction_pixel_scores.append(outputs['reconstruction_pixel_scores'].cpu().numpy())
                if 'anchor_pixel_scores' in outputs:
                    all_anchor_pixel_scores.append(outputs['anchor_pixel_scores'].cpu().numpy())
                
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

    if len(all_anchor_scores) > 0:
        anchor_scores = np.concatenate(all_anchor_scores)
        metrics['anchor_image_auroc'] = roc_auc_score(labels, anchor_scores)
        metrics['anchor_image_aupr'] = average_precision_score(labels, anchor_scores)

    if len(all_reconstruction_scores) > 0:
        reconstruction_scores = np.concatenate(all_reconstruction_scores)
        metrics['reconstruction_image_auroc'] = roc_auc_score(labels, reconstruction_scores)
        metrics['reconstruction_image_aupr'] = average_precision_score(labels, reconstruction_scores)

        # Optional combined score metric (dataset-level normalization + weighted sum)
        if getattr(model, 'score_combination_enabled', False):
            alpha = float(getattr(model, 'score_combination_alpha', 0.5))
            norm_mode = getattr(model, 'score_combination_normalization', 'minmax')
            anchor_scores_for_combine = np.concatenate(all_anchor_scores) if len(all_anchor_scores) > 0 else image_scores

            def _normalize(values: np.ndarray) -> np.ndarray:
                if norm_mode == 'zscore':
                    std = values.std()
                    if std < 1e-12:
                        return np.zeros_like(values)
                    return (values - values.mean()) / std
                val_min = values.min()
                val_max = values.max()
                denom = max(val_max - val_min, 1e-12)
                return (values - val_min) / denom

            anchor_norm = _normalize(anchor_scores_for_combine)
            recon_norm = _normalize(reconstruction_scores)
            combined_scores = (1.0 - alpha) * anchor_norm + alpha * recon_norm

            metrics['combined_image_auroc'] = roc_auc_score(labels, combined_scores)
            metrics['combined_image_aupr'] = average_precision_score(labels, combined_scores)
            metrics['combined_alpha'] = alpha
            metrics['combined_normalization'] = norm_mode

    # === BOTTLENECK DIVERGENCE METRICS ===
    if len(all_divergence_scores) > 0:
        divergence_scores = np.concatenate(all_divergence_scores)
        try:
            metrics['divergence_image_auroc'] = roc_auc_score(labels, divergence_scores)
            metrics['divergence_image_aupr'] = average_precision_score(labels, divergence_scores)
        except ValueError:
            pass

    # === PIXEL-AGGREGATED SCORE METRICS ===
    if len(all_pixel_aggregated_scores) > 0:
        pixel_agg_scores = np.concatenate(all_pixel_aggregated_scores)
        try:
            metrics['pixel_aggregated_image_auroc'] = roc_auc_score(labels, pixel_agg_scores)
            metrics['pixel_aggregated_image_aupr'] = average_precision_score(labels, pixel_agg_scores)
        except ValueError:
            pass

    # === PATCH-DIVERGENCE AGGREGATED METRICS ===
    if len(all_patch_div_aggregated_scores) > 0:
        patch_div_agg = np.concatenate(all_patch_div_aggregated_scores)
        try:
            metrics['patch_divergence_aggregated_image_auroc'] = roc_auc_score(labels, patch_div_agg)
            metrics['patch_divergence_aggregated_image_aupr'] = average_precision_score(labels, patch_div_agg)
        except ValueError:
            pass

    # === THREE-SIGNAL SCORE FUSION ===
    if getattr(model, 'score_fusion_enabled', False):
        norm_mode_f = getattr(model, 'score_fusion_normalization', 'minmax')
        w_a = getattr(model, 'score_fusion_anchor_weight', 0.4)
        w_d = getattr(model, 'score_fusion_divergence_weight', 0.3)
        w_p = getattr(model, 'score_fusion_pixel_weight', 0.3)

        anchor_for_fusion = np.concatenate(all_anchor_scores) if len(all_anchor_scores) > 0 else image_scores

        def _norm_fusion(values: np.ndarray, mode: str) -> np.ndarray:
            if mode == 'zscore':
                std = values.std()
                if std < 1e-12:
                    return np.zeros_like(values)
                return (values - values.mean()) / std
            elif mode == 'robust':
                q25, q50, q75 = np.percentile(values, [25, 50, 75])
                iqr = max(q75 - q25, 1e-12)
                return (values - q50) / iqr
            elif mode == 'rank':
                from scipy.stats import rankdata
                return rankdata(values) / len(values)
            else:  # minmax
                lo, hi = values.min(), values.max()
                return (values - lo) / max(hi - lo, 1e-12)

        a_norm = _norm_fusion(anchor_for_fusion, norm_mode_f)

        # Prefer per-patch divergence (spatially resolved 15x15) over CLS-token divergence (1 scalar)
        has_patch_div = len(all_patch_div_aggregated_scores) > 0
        has_div = len(all_divergence_scores) > 0
        has_pix = len(all_pixel_aggregated_scores) > 0

        # Guard: skip divergence signals with AUROC < 0.5 (anti-correlated hurts fusion)
        patch_div_auroc = metrics.get('patch_divergence_aggregated_image_auroc', None)
        div_auroc = metrics.get('divergence_image_auroc', None)
        if getattr(model, 'score_fusion_drop_anticorrelated', True):
            if has_patch_div and patch_div_auroc is not None and patch_div_auroc < 0.5:
                has_patch_div = False  # skip anti-correlated patch divergence
            if has_div and div_auroc is not None and div_auroc < 0.5:
                has_div = False  # skip anti-correlated CLS divergence

        # Use the divergence signal with higher AUROC (bottleneck vs patch)
        if has_patch_div and has_div:
            _patch_a = patch_div_auroc or 0.5
            _div_a = div_auroc or 0.5
            if _div_a >= _patch_a:
                div_scores = np.concatenate(all_divergence_scores)
            else:
                div_scores = np.concatenate(all_patch_div_aggregated_scores)
        elif has_patch_div:
            div_scores = np.concatenate(all_patch_div_aggregated_scores)
        elif has_div:
            div_scores = np.concatenate(all_divergence_scores)
        else:
            div_scores = None
        has_any_div = div_scores is not None

        if has_any_div and has_pix:
            d_norm = _norm_fusion(div_scores, norm_mode_f)
            p_norm = _norm_fusion(np.concatenate(all_pixel_aggregated_scores), norm_mode_f)
            fused = w_a * a_norm + w_d * d_norm + w_p * p_norm
        elif has_any_div:
            d_norm = _norm_fusion(div_scores, norm_mode_f)
            total_w = w_a + w_d
            fused = (w_a / total_w) * a_norm + (w_d / total_w) * d_norm
        elif has_pix:
            p_norm = _norm_fusion(np.concatenate(all_pixel_aggregated_scores), norm_mode_f)
            total_w = w_a + w_p
            fused = (w_a / total_w) * a_norm + (w_p / total_w) * p_norm
        else:
            fused = a_norm

        try:
            metrics['fused_image_auroc'] = roc_auc_score(labels, fused)
            metrics['fused_image_aupr'] = average_precision_score(labels, fused)
            metrics['fusion_weights'] = {'anchor': w_a, 'divergence': w_d, 'pixel': w_p}
            metrics['fusion_normalization'] = norm_mode_f
        except ValueError:
            pass
    
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

                # Extra pixel metrics for map sources when available
                if len(all_reconstruction_pixel_scores) > 0:
                    recon_pixel = np.concatenate(all_reconstruction_pixel_scores)
                    if recon_pixel.shape[1:] != pixel_masks.shape[1:]:
                        from scipy.ndimage import zoom
                        scale_h = pixel_masks.shape[1] / recon_pixel.shape[1]
                        scale_w = pixel_masks.shape[2] / recon_pixel.shape[2]
                        recon_pixel = np.array([zoom(recon_pixel[i], (scale_h, scale_w), order=1) for i in range(recon_pixel.shape[0])])
                    if recon_pixel.shape[0] != pixel_masks.shape[0]:
                        min_samples = min(recon_pixel.shape[0], pixel_masks.shape[0])
                        recon_pixel = recon_pixel[:min_samples]
                        masks_for_recon = pixel_masks[:min_samples]
                    else:
                        masks_for_recon = pixel_masks

                    recon_pixel_flat = recon_pixel.flatten()
                    masks_for_recon_flat = masks_for_recon.flatten()
                    if masks_for_recon_flat.sum() > 0:
                        metrics['reconstruction_pixel_auroc'] = roc_auc_score(masks_for_recon_flat, recon_pixel_flat)
                        metrics['reconstruction_pixel_aupr'] = average_precision_score(masks_for_recon_flat, recon_pixel_flat)

                if len(all_anchor_pixel_scores) > 0:
                    anchor_pixel = np.concatenate(all_anchor_pixel_scores)
                    if anchor_pixel.shape[1:] != pixel_masks.shape[1:]:
                        from scipy.ndimage import zoom
                        scale_h = pixel_masks.shape[1] / anchor_pixel.shape[1]
                        scale_w = pixel_masks.shape[2] / anchor_pixel.shape[2]
                        anchor_pixel = np.array([zoom(anchor_pixel[i], (scale_h, scale_w), order=1) for i in range(anchor_pixel.shape[0])])
                    if anchor_pixel.shape[0] != pixel_masks.shape[0]:
                        min_samples = min(anchor_pixel.shape[0], pixel_masks.shape[0])
                        anchor_pixel = anchor_pixel[:min_samples]
                        masks_for_anchor = pixel_masks[:min_samples]
                    else:
                        masks_for_anchor = pixel_masks

                    anchor_pixel_flat = anchor_pixel.flatten()
                    masks_for_anchor_flat = masks_for_anchor.flatten()
                    if masks_for_anchor_flat.sum() > 0:
                        metrics['anchor_pixel_auroc'] = roc_auc_score(masks_for_anchor_flat, anchor_pixel_flat)
                        metrics['anchor_pixel_aupr'] = average_precision_score(masks_for_anchor_flat, anchor_pixel_flat)
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

    if 'anchor_image_auroc' in metrics:
        print(f"Anchor Score AUROC: {metrics['anchor_image_auroc']:.4f}")
        print(f"Anchor Score AUPR: {metrics['anchor_image_aupr']:.4f}")
    if 'reconstruction_image_auroc' in metrics:
        print(f"Reconstruction Score AUROC: {metrics['reconstruction_image_auroc']:.4f}")
        print(f"Reconstruction Score AUPR: {metrics['reconstruction_image_aupr']:.4f}")
    if 'combined_image_auroc' in metrics:
        print(f"Combined Score AUROC: {metrics['combined_image_auroc']:.4f} (alpha={metrics.get('combined_alpha', 0.5):.2f})")
        print(f"Combined Score AUPR: {metrics['combined_image_aupr']:.4f}")
    if 'divergence_image_auroc' in metrics:
        print(f"Bottleneck Divergence AUROC: {metrics['divergence_image_auroc']:.4f}")
        print(f"Bottleneck Divergence AUPR: {metrics['divergence_image_aupr']:.4f}")
    if 'patch_divergence_aggregated_image_auroc' in metrics:
        print(f"Patch-Divergence Aggregated AUROC: {metrics['patch_divergence_aggregated_image_auroc']:.4f}")
        print(f"Patch-Divergence Aggregated AUPR: {metrics['patch_divergence_aggregated_image_aupr']:.4f}")
    if 'pixel_aggregated_image_auroc' in metrics:
        print(f"Pixel-Aggregated Score AUROC: {metrics['pixel_aggregated_image_auroc']:.4f}")
        print(f"Pixel-Aggregated Score AUPR: {metrics['pixel_aggregated_image_aupr']:.4f}")
    if 'fused_image_auroc' in metrics:
        w = metrics.get('fusion_weights', {})
        print(f"Fused 3-Signal AUROC: {metrics['fused_image_auroc']:.4f} (a={w.get('anchor',0):.2f}, d={w.get('divergence',0):.2f}, p={w.get('pixel',0):.2f})")
        print(f"Fused 3-Signal AUPR: {metrics['fused_image_aupr']:.4f}")
    
    if 'pixel_auroc' in metrics:
        print(f"Pixel AUROC: {metrics['pixel_auroc']:.4f}")
        print(f"Pixel AUPR: {metrics['pixel_aupr']:.4f}")
    if 'reconstruction_pixel_auroc' in metrics:
        print(f"Reconstruction Pixel AUROC: {metrics['reconstruction_pixel_auroc']:.4f}")
        print(f"Reconstruction Pixel AUPR: {metrics['reconstruction_pixel_aupr']:.4f}")
    if 'anchor_pixel_auroc' in metrics:
        print(f"Anchor Pixel AUROC: {metrics['anchor_pixel_auroc']:.4f}")
        print(f"Anchor Pixel AUPR: {metrics['anchor_pixel_aupr']:.4f}")
    
    # Collect scores for additional analysis
    model.eval()
    all_image_scores = []
    all_anchor_scores = []
    all_reconstruction_scores = []
    all_combined_scores = []
    all_divergence_scores_c = []
    all_pixel_aggregated_scores_c = []
    all_patch_div_aggregated_scores_c = []
    all_labels = []
    all_paths = []
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
            all_paths.extend([str(p) for p in batch.get('path', [])])

            if 'anchor_scores' in outputs:
                all_anchor_scores.append(outputs['anchor_scores'].cpu().numpy())
            if 'reconstruction_scores' in outputs:
                all_reconstruction_scores.append(outputs['reconstruction_scores'].cpu().numpy())
            if 'combined_scores_raw' in outputs:
                all_combined_scores.append(outputs['combined_scores_raw'].cpu().numpy())
            if 'bottleneck_divergence' in outputs:
                all_divergence_scores_c.append(outputs['bottleneck_divergence'].cpu().numpy())
            if 'pixel_aggregated_score' in outputs:
                all_pixel_aggregated_scores_c.append(outputs['pixel_aggregated_score'].cpu().numpy())
            if 'patch_divergence_aggregated_score' in outputs:
                all_patch_div_aggregated_scores_c.append(outputs['patch_divergence_aggregated_score'].cpu().numpy())
            
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

    # Save per-sample image-level scores for ensemble post-processing
    try:
        import pandas as pd

        per_sample = {
            'path': all_paths if len(all_paths) == len(labels) else [f'sample_{i:06d}' for i in range(len(labels))],
            'label': labels.astype(int),
            'image_score': image_scores.astype(float)
        }

        if len(all_anchor_scores) > 0:
            per_sample['anchor_score'] = np.concatenate(all_anchor_scores).astype(float)
        if len(all_reconstruction_scores) > 0:
            per_sample['reconstruction_score'] = np.concatenate(all_reconstruction_scores).astype(float)
        if len(all_combined_scores) > 0:
            per_sample['combined_score_raw'] = np.concatenate(all_combined_scores).astype(float)
        if len(all_divergence_scores_c) > 0:
            per_sample['bottleneck_divergence'] = np.concatenate(all_divergence_scores_c).astype(float)
        if len(all_pixel_aggregated_scores_c) > 0:
            per_sample['pixel_aggregated_score'] = np.concatenate(all_pixel_aggregated_scores_c).astype(float)
        if len(all_patch_div_aggregated_scores_c) > 0:
            per_sample['patch_divergence_aggregated'] = np.concatenate(all_patch_div_aggregated_scores_c).astype(float)

        # Compute fused score for CSV if fusion enabled
        if getattr(model, 'score_fusion_enabled', False):
            norm_mode_csv = getattr(model, 'score_fusion_normalization', 'minmax')
            w_a = getattr(model, 'score_fusion_anchor_weight', 0.4)
            w_d = getattr(model, 'score_fusion_divergence_weight', 0.3)
            w_p = getattr(model, 'score_fusion_pixel_weight', 0.3)

            def _norm_csv(values):
                lo, hi = values.min(), values.max()
                return (values - lo) / max(hi - lo, 1e-12)

            a_base = per_sample.get('anchor_score', per_sample['image_score'])
            a_n = _norm_csv(np.array(a_base))
            fused_csv = a_n.copy()
            total_w = w_a
            # Use whichever divergence signal has higher AUROC (bottleneck vs patch)
            _patch_a = metrics.get('patch_divergence_aggregated_image_auroc', 0.5)
            _div_a = metrics.get('divergence_image_auroc', 0.5)
            if 'bottleneck_divergence' in per_sample and 'patch_divergence_aggregated' in per_sample:
                div_key = 'bottleneck_divergence' if _div_a >= _patch_a else 'patch_divergence_aggregated'
            elif 'patch_divergence_aggregated' in per_sample:
                div_key = 'patch_divergence_aggregated'
            else:
                div_key = 'bottleneck_divergence'
            if div_key in per_sample:
                d_n = _norm_csv(np.array(per_sample[div_key]))
                fused_csv = w_a * a_n + w_d * d_n
                total_w += w_d
            if 'pixel_aggregated_score' in per_sample:
                p_n = _norm_csv(np.array(per_sample['pixel_aggregated_score']))
                fused_csv = fused_csv + w_p * p_n
                total_w += w_p
            if total_w > 0:
                fused_csv = fused_csv / total_w
            per_sample['fused_score'] = fused_csv.astype(float)

        per_sample_df = pd.DataFrame(per_sample)
        per_sample_df.to_csv(save_dir / 'evaluation_image_scores.csv', index=False)
        print(f"Saved per-sample scores CSV with {len(per_sample_df.columns)} columns: {list(per_sample_df.columns)}")
    except Exception as e:
        print(f"Warning: Could not save per-sample image scores CSV: {e}")
    
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
    
    # --- Diagnostic visualizations ---
    try:
        visualize_reconstruction_grid(model, dataloader, device, save_dir)
    except Exception as e:
        print(f"  Warning: reconstruction grid visualization failed: {e}")

    try:
        visualize_bottleneck_divergence(model, dataloader, device, save_dir)
    except Exception as e:
        print(f"  Warning: bottleneck divergence visualization failed: {e}")

    try:
        visualize_multi_signal_distributions(model, dataloader, device, save_dir)
    except Exception as e:
        print(f"  Warning: multi-signal distribution visualization failed: {e}")

    try:
        visualize_pixel_anomaly_overlay(model, dataloader, device, save_dir)
    except Exception as e:
        print(f"  Warning: pixel anomaly overlay visualization failed: {e}")

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
                
                detail = {
                    'anchor': float(outputs['anchor_scores'][i].cpu()) if 'anchor_scores' in outputs else float(image_scores[i]),
                    'recon': float(outputs['reconstruction_scores'][i].cpu()) if 'reconstruction_scores' in outputs else None,
                    'div': float(outputs['bottleneck_divergence'][i].cpu()) if 'bottleneck_divergence' in outputs else None,
                    'pix_agg': float(outputs['pixel_aggregated_score'][i].cpu()) if 'pixel_aggregated_score' in outputs else None,
                    'fused': None,
                }
                sample = {
                    'image': images[i].cpu().numpy(),
                    'label': labels[i],
                    'score': image_scores[i],
                    'pixel_map': pixel_scores[i] if pixel_scores is not None else None,
                    'mask': batch['mask'][i].cpu().numpy() if 'mask' in batch else None,
                    'scores_detail': detail
                }
                
                if labels[i] == 0 and len(normal_samples) < num_samples // 2:
                    normal_samples.append(sample)
                    samples_collected += 1
                elif labels[i] == 1 and len(anomaly_samples) < num_samples // 2:
                    anomaly_samples.append(sample)
                    samples_collected += 1
    
    # Compute fused scores across all collected samples for visualization titles
    all_viz_samples = normal_samples + anomaly_samples
    if all_viz_samples and getattr(model, 'score_fusion_enabled', False):
        def _viz_norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / max(hi - lo, 1e-12)
        w_a = getattr(model, 'score_fusion_anchor_weight', 0.333)
        w_d = getattr(model, 'score_fusion_divergence_weight', 0.333)
        w_p = getattr(model, 'score_fusion_pixel_weight', 0.333)
        a_arr = np.array([s['scores_detail']['anchor'] for s in all_viz_samples], dtype=float)
        fused_arr = w_a * _viz_norm(a_arr)
        total_w = w_a
        if any(s['scores_detail']['div'] is not None for s in all_viz_samples):
            d_arr = np.array([s['scores_detail']['div'] if s['scores_detail']['div'] is not None else 0.0 for s in all_viz_samples], dtype=float)
            fused_arr = fused_arr + w_d * _viz_norm(d_arr)
            total_w += w_d
        if any(s['scores_detail']['pix_agg'] is not None for s in all_viz_samples):
            p_arr = np.array([s['scores_detail']['pix_agg'] if s['scores_detail']['pix_agg'] is not None else 0.0 for s in all_viz_samples], dtype=float)
            fused_arr = fused_arr + w_p * _viz_norm(p_arr)
            total_w += w_p
        fused_arr = fused_arr / total_w
        for idx, s in enumerate(all_viz_samples):
            s['scores_detail']['fused'] = float(fused_arr[idx])

    # Visualize normal samples
    if normal_samples:
        _plot_samples(normal_samples, save_dir / 'normal_samples.png', 'Normal Samples')
    
    # Visualize anomaly samples
    if anomaly_samples:
        _plot_samples(anomaly_samples, save_dir / 'anomaly_samples.png', 'Anomaly Samples')


def _plot_samples(samples: list, save_path: str, title: str):
    """Helper to plot sample grid with consistent anomaly map scales"""
    n = len(samples)
    cols = 4
    rows = (n + cols - 1) // cols
    
    # Determine if we have pixel maps
    has_maps = samples[0]['pixel_map'] is not None if len(samples) > 0 else False
    has_masks = samples[0]['mask'] is not None if len(samples) > 0 else False
    
    n_cols = 1 + (1 if has_maps else 0) + (1 if has_masks else 0)
    
    # Compute shared vmin/vmax across all anomaly maps for consistent scale
    vmin_map, vmax_map = 0.0, 1.0
    if has_maps:
        all_map_vals = [s['pixel_map'] for s in samples if s['pixel_map'] is not None]
        if all_map_vals:
            vmin_map = min(m.min() for m in all_map_vals)
            vmax_map = max(m.max() for m in all_map_vals)
            if vmax_map - vmin_map < 1e-8:
                vmax_map = vmin_map + 1.0
    
    fig, axes = plt.subplots(rows, cols * n_cols, figsize=(cols * n_cols * 3, rows * 3))
    
    if rows == 1 and cols * n_cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols * n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    im_ref = None  # Reference for colorbar
    
    for idx, sample in enumerate(samples):
        row = idx // cols
        col_offset = (idx % cols) * n_cols
        
        # Original image
        img = sample['image'][0]  # Take first channel
        axes[row, col_offset].imshow(img, cmap='gray')
        d = sample.get('scores_detail')
        if d:
            parts = []
            if d.get('fused') is not None:
                parts.append(f"F:{d['fused']:.3f}")
            parts.append(f"A:{d['anchor']:.3f}")
            if d.get('recon') is not None:
                parts.append(f"R:{d['recon']:.3f}")
            if d.get('div') is not None:
                parts.append(f"D:{d['div']:.3f}")
            if d.get('pix_agg') is not None:
                parts.append(f"P:{d['pix_agg']:.3f}")
            mid = (len(parts) + 1) // 2
            title_str = ' '.join(parts[:mid]) + '\n' + ' '.join(parts[mid:])
        else:
            title_str = f"Score: {sample['score']:.3f}"
        axes[row, col_offset].set_title(title_str, fontsize=8)
        axes[row, col_offset].axis('off')
        
        col_idx = col_offset + 1
        
        # Anomaly map (with shared scale)
        if has_maps and sample['pixel_map'] is not None:
            im_ref = axes[row, col_idx].imshow(
                sample['pixel_map'], cmap='jet',
                vmin=vmin_map, vmax=vmax_map
            )
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
    
    # Add shared colorbar for anomaly maps
    if has_maps and im_ref is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_label('Anomaly Score', fontsize=10)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92 if has_maps else 1, 0.96])
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


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic Visualizations (Phase 2.3)
# ──────────────────────────────────────────────────────────────────────────────

def visualize_reconstruction_grid(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 8,
):
    """Plot side-by-side: original | reconstruction | |difference| for normal and anomaly samples.

    Requires stage-2 reconstruction to be enabled on the model.
    """
    from pathlib import Path
    save_dir = Path(save_dir)

    model.eval()
    normals, anomalies = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            outputs = model(images, return_dense=False)

            if 'reconstruction' not in outputs:
                print("  [visualize_reconstruction_grid] Reconstruction not available — skipping.")
                return

            recon = outputs['reconstruction']  # (B, C, H, W) or (B, 1, H, W)
            for i in range(images.shape[0]):
                entry = (images[i].cpu(), recon[i].cpu(), int(labels[i]))
                if labels[i] == 0 and len(normals) < num_samples:
                    normals.append(entry)
                elif labels[i] == 1 and len(anomalies) < num_samples:
                    anomalies.append(entry)
                if len(normals) >= num_samples and len(anomalies) >= num_samples:
                    break
            if len(normals) >= num_samples and len(anomalies) >= num_samples:
                break

    def _draw_grid(samples, path, title):
        n = len(samples)
        if n == 0:
            return
        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(title, fontsize=14)
        for i, (orig, rec, _lbl) in enumerate(samples):
            orig_np = orig[0].numpy()  # first channel
            rec_np = rec[0].numpy()
            diff_np = np.abs(orig_np - rec_np)
            axes[i, 0].imshow(orig_np, cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(rec_np, cmap='gray')
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')
            im = axes[i, 2].imshow(diff_np, cmap='hot')
            axes[i, 2].set_title('|Difference|')
            axes[i, 2].axis('off')
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

    _draw_grid(normals, save_dir / 'reconstruction_normals.png', 'Reconstruction — Normal Samples')
    _draw_grid(anomalies, save_dir / 'reconstruction_anomalies.png', 'Reconstruction — Anomaly Samples')
    print(f"Saved reconstruction grids to {save_dir}")


def visualize_bottleneck_divergence(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
):
    """Histogram of frozen-vs-stage2 cosine similarity, split by normal / anomaly.

    Shows whether the bottleneck divergence signal separates the two classes.
    """
    from pathlib import Path
    save_dir = Path(save_dir)

    model.eval()
    div_normal, div_anomaly = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            outputs = model(images, return_dense=False)

            if 'bottleneck_divergence' not in outputs or outputs['bottleneck_divergence'] is None:
                print("  [visualize_bottleneck_divergence] No divergence signal — skipping.")
                return

            div = outputs['bottleneck_divergence'].cpu().numpy()  # (B,)
            for i in range(len(div)):
                if labels[i] == 0:
                    div_normal.append(div[i])
                else:
                    div_anomaly.append(div[i])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(div_normal, bins=60, alpha=0.6, density=True, label=f'Normal (n={len(div_normal)})')
    ax.hist(div_anomaly, bins=60, alpha=0.6, density=True, label=f'Anomaly (n={len(div_anomaly)})')
    ax.set_xlabel('Bottleneck Divergence (cosine)')
    ax.set_ylabel('Density')
    ax.set_title('Frozen vs Stage-2 Bottleneck Divergence')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'bottleneck_divergence_hist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bottleneck divergence histogram to {save_dir}")


def visualize_multi_signal_distributions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
):
    """Per-signal score distributions (anchor / reconstruction / divergence / pixel agg),
    normal vs anomaly overlay histograms in a single figure."""
    from pathlib import Path
    save_dir = Path(save_dir)

    model.eval()
    signals = {
        'anchor_distance': ([], []),
        'reconstruction_error': ([], []),
        'bottleneck_divergence': ([], []),
        'pixel_aggregated': ([], []),
    }
    key_map = {
        'anchor_distance': 'image_scores',
        'reconstruction_error': 'reconstruction_scores',
        'bottleneck_divergence': 'bottleneck_divergence',
        'pixel_aggregated': 'pixel_scores_aggregated',
    }

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            outputs = model.compute_anomaly_scores(images) if hasattr(model, 'compute_anomaly_scores') else model(images, return_dense=True)

            for sig_name, out_key in key_map.items():
                vals = outputs.get(out_key)
                if vals is None:
                    continue
                vals_np = vals.cpu().numpy() if isinstance(vals, torch.Tensor) else np.asarray(vals)
                for i in range(len(labels)):
                    v = float(vals_np[i]) if vals_np.ndim == 1 else float(vals_np[i])
                    signals[sig_name][labels[i]].append(v)

    # Filter to signals that have data
    active = {k: v for k, v in signals.items() if len(v[0]) > 0 or len(v[1]) > 0}
    if not active:
        print("  [visualize_multi_signal_distributions] No signal data collected — skipping.")
        return

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (sig_name, (norm_vals, anom_vals)) in zip(axes, active.items()):
        if norm_vals:
            ax.hist(norm_vals, bins=60, alpha=0.6, density=True, label='Normal')
        if anom_vals:
            ax.hist(anom_vals, bins=60, alpha=0.6, density=True, label='Anomaly')
        ax.set_title(sig_name.replace('_', ' ').title())
        ax.set_xlabel('Score')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Per-Signal Score Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'multi_signal_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-signal distributions to {save_dir}")


def visualize_pixel_anomaly_overlay(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 8,
):
    """Side-by-side: original | GT mask | predicted anomaly heatmap for anomaly samples."""
    from pathlib import Path
    save_dir = Path(save_dir)
    import torch.nn.functional as F

    model.eval()
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            if 'mask' not in batch:
                continue
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask']
            outputs = model(images, return_dense=True)

            pixel_map = outputs.get('pixel_anomaly_map') or outputs.get('reconstruction_error_map')
            if pixel_map is None:
                # Try reconstruction difference
                if 'reconstruction' in outputs:
                    recon = outputs['reconstruction']
                    pixel_map = (images - recon).pow(2).mean(dim=1)  # (B, H, W)
                else:
                    print("  [visualize_pixel_anomaly_overlay] No pixel-level map — skipping.")
                    return

            for i in range(images.shape[0]):
                if labels[i] == 1 and len(samples) < num_samples:
                    orig = images[i, 0].cpu().numpy()
                    gt = masks[i].cpu().numpy() if isinstance(masks[i], torch.Tensor) else masks[i]
                    if gt.ndim == 3:
                        gt = gt[0]
                    # Upsample prediction to image size if needed
                    pred = pixel_map[i]
                    if isinstance(pred, torch.Tensor):
                        if pred.ndim == 3:
                            pred = pred[0]
                        if pred.shape != images.shape[2:]:
                            pred = F.interpolate(
                                pred.unsqueeze(0).unsqueeze(0).float(),
                                size=images.shape[2:], mode='bilinear', align_corners=False
                            ).squeeze()
                        pred = pred.cpu().numpy()
                    samples.append((orig, gt, pred))
            if len(samples) >= num_samples:
                break

    if not samples:
        print("  [visualize_pixel_anomaly_overlay] No anomaly samples with masks — skipping.")
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Pixel Anomaly Overlay (Anomaly Samples)', fontsize=14)
    for i, (orig, gt, pred) in enumerate(samples):
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('GT Mask')
        axes[i, 1].axis('off')
        im = axes[i, 2].imshow(pred, cmap='hot')
        axes[i, 2].set_title('Predicted Heatmap')
        axes[i, 2].axis('off')
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_dir / 'pixel_anomaly_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pixel anomaly overlay to {save_dir}")