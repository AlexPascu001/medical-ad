"""
Pixel-to-Image Anomaly Score Aggregation

Provides multiple strategies to convert a per-pixel anomaly map (H, W) into a
single image-level anomaly score.  This bridges the gap between high pixel-level
AUROC (e.g. 0.92 from reconstruction maps) and comparatively lower image-level
AUROC that results from naïve global-mean aggregation.

Supported methods:
  mean             – global spatial mean (baseline, current default)
  max              – single worst pixel drives the score
  top_k_percentile – q-th percentile of the pixel values (robust max)
  self_normalized  – (top-k mean − median) / IQR, image-content-invariant
  threshold_ratio  – fraction of pixels exceeding a learned threshold

All functions operate on numpy arrays or torch tensors and return a scalar /
batch of scalars.
"""

import torch
import numpy as np
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Torch (batched) implementations – used inside model.compute_anomaly_scores
# ---------------------------------------------------------------------------

def aggregate_pixel_scores_torch(
    pixel_map: torch.Tensor,
    method: str = 'top_k_percentile',
    percentile: float = 95.0,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    """Aggregate a batch of pixel maps to per-image scores.

    Args:
        pixel_map: (B, H, W) per-pixel anomaly scores.
        method: one of 'mean', 'max', 'top_k_percentile', 'self_normalized',
                'threshold_ratio'.
        percentile: percentile value for 'top_k_percentile' and
                    'self_normalized' (0-100).
        threshold: absolute threshold for 'threshold_ratio'. If *None* falls
                   back to ``mean + 3*std`` computed **per sample**.

    Returns:
        (B,) image-level anomaly scores.
    """
    B = pixel_map.shape[0]
    flat = pixel_map.view(B, -1)  # (B, H*W)

    if method == 'mean':
        return flat.mean(dim=1)

    if method == 'max':
        return flat.max(dim=1)[0]

    if method == 'top_k_percentile':
        k = max(1, int((1.0 - percentile / 100.0) * flat.shape[1]))
        topk_vals, _ = flat.topk(k, dim=1)  # (B, k) largest values
        return topk_vals.mean(dim=1)

    if method == 'self_normalized':
        # How many IQRs above median is the hottest region?
        # Removes per-image baseline confound (anatomy, slice position).
        k = max(1, int((1.0 - percentile / 100.0) * flat.shape[1]))
        topk_vals, _ = flat.topk(k, dim=1)
        topk_mean = topk_vals.mean(dim=1)  # (B,)
        median = flat.median(dim=1).values  # (B,)
        q25 = flat.quantile(0.25, dim=1)    # (B,)
        q75 = flat.quantile(0.75, dim=1)    # (B,)
        iqr = (q75 - q25).clamp(min=1e-8)
        return (topk_mean - median) / iqr

    if method == 'threshold_ratio':
        if threshold is not None:
            return (flat > threshold).float().mean(dim=1)
        else:
            # Per-sample adaptive threshold: mean + 3*std
            mu = flat.mean(dim=1, keepdim=True)
            sigma = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
            thr = mu + 3.0 * sigma
            return (flat > thr).float().mean(dim=1)

    raise ValueError(f"Unknown pixel aggregation method: {method}")


# ---------------------------------------------------------------------------
# Numpy (dataset-level) implementations – used in eval / offline analysis
# ---------------------------------------------------------------------------

def aggregate_pixel_scores_numpy(
    pixel_map: np.ndarray,
    method: str = 'top_k_percentile',
    percentile: float = 95.0,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Numpy counterpart of ``aggregate_pixel_scores_torch``.

    Args:
        pixel_map: (N, H, W) or (H, W).
        method: aggregation strategy.
        percentile: for 'top_k_percentile'.
        threshold: for 'threshold_ratio'.

    Returns:
        (N,) or scalar image-level score(s).
    """
    single = pixel_map.ndim == 2
    if single:
        pixel_map = pixel_map[np.newaxis]

    N = pixel_map.shape[0]
    flat = pixel_map.reshape(N, -1)

    if method == 'mean':
        scores = flat.mean(axis=1)
    elif method == 'max':
        scores = flat.max(axis=1)
    elif method == 'top_k_percentile':
        scores = np.percentile(flat, percentile, axis=1)
    elif method == 'self_normalized':
        k = max(1, int((1.0 - percentile / 100.0) * flat.shape[1]))
        # Top-k mean per sample
        idx = np.argpartition(flat, -k, axis=1)[:, -k:]
        topk_vals = np.take_along_axis(flat, idx, axis=1)
        topk_mean = topk_vals.mean(axis=1)
        median = np.median(flat, axis=1)
        q25 = np.percentile(flat, 25, axis=1)
        q75 = np.percentile(flat, 75, axis=1)
        iqr = np.clip(q75 - q25, 1e-8, None)
        scores = (topk_mean - median) / iqr
    elif method == 'threshold_ratio':
        if threshold is not None:
            scores = (flat > threshold).astype(float).mean(axis=1)
        else:
            mu = flat.mean(axis=1, keepdims=True)
            sigma = flat.std(axis=1, keepdims=True).clip(min=1e-8)
            thr = mu + 3.0 * sigma
            scores = (flat > thr).astype(float).mean(axis=1)
    else:
        raise ValueError(f"Unknown pixel aggregation method: {method}")

    return float(scores[0]) if single else scores


# ---------------------------------------------------------------------------
# Training-set statistics helper (for threshold_ratio with global threshold)
# ---------------------------------------------------------------------------

class PixelStatsTracker:
    """Accumulates running mean / std of per-pixel reconstruction errors during
    stage-2 training so that a data-driven threshold can be derived.
    """

    def __init__(self):
        self.n = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, pixel_map: torch.Tensor):
        """Update with a batch of pixel maps (B, H, W)."""
        vals = pixel_map.detach().float()
        count = vals.numel()
        self.n += count
        self._sum += vals.sum().item()
        self._sum_sq += vals.pow(2).sum().item()

    @property
    def mean(self) -> float:
        return self._sum / max(self.n, 1)

    @property
    def std(self) -> float:
        var = self._sum_sq / max(self.n, 1) - self.mean ** 2
        return max(var, 0.0) ** 0.5

    def threshold(self, n_std: float = 3.0) -> float:
        """Return mean + n_std * std."""
        return self.mean + n_std * self.std

    def state_dict(self):
        return {'n': self.n, 'sum': self._sum, 'sum_sq': self._sum_sq}

    def load_state_dict(self, d: dict):
        self.n = d['n']
        self._sum = d['sum']
        self._sum_sq = d['sum_sq']
