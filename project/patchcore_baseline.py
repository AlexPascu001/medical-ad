"""Standalone frozen DINO PatchCore baseline.

This module is intentionally isolated from the anchor/stage-2 training path.
It reuses the existing DINO backbone wrapper and pixel aggregation utilities,
but owns its own memory-bank fitting and kNN inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from pixel_aggregation import aggregate_pixel_scores_torch


@dataclass
class PatchCoreFitSummary:
    """Lightweight metadata about the fitted memory bank."""

    num_images: int
    num_features_before_cap: int
    num_features_after_cap: int
    feature_dim: int
    patches_per_image: int


class PatchCoreBaseline(nn.Module):
    """Patch-token kNN anomaly detector on a frozen feature extractor.

    The class exposes a ``compute_anomaly_scores`` interface compatible with
    the existing evaluation utilities in ``eval.py``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        n_neighbors: int = 1,
        distance_metric: str = 'euclidean',
        patches_per_image: Optional[int] = None,
        max_memory_bank_size: Optional[int] = None,
        query_batch_size: int = 4096,
        aggregation_method: str = 'top_k_percentile',
        aggregation_percentile: float = 95.0,
        aggregation_threshold: Optional[float] = None,
        neighbor_reduction: str = 'mean',
        random_state: int = 42,
    ):
        super().__init__()

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        if query_batch_size < 1:
            raise ValueError(f"query_batch_size must be >= 1, got {query_batch_size}")
        if patches_per_image is not None and patches_per_image < 1:
            raise ValueError(f"patches_per_image must be >= 1, got {patches_per_image}")
        if max_memory_bank_size is not None and max_memory_bank_size < 1:
            raise ValueError(f"max_memory_bank_size must be >= 1, got {max_memory_bank_size}")
        if neighbor_reduction not in {'mean', 'max'}:
            raise ValueError(f"Unsupported neighbor_reduction: {neighbor_reduction}")

        self.backbone = backbone
        self.n_neighbors = int(n_neighbors)
        self.distance_metric = str(distance_metric)
        self.patches_per_image = patches_per_image
        self.max_memory_bank_size = max_memory_bank_size
        self.query_batch_size = int(query_batch_size)
        self.aggregation_method = str(aggregation_method)
        self.aggregation_percentile = float(aggregation_percentile)
        self.aggregation_threshold = aggregation_threshold
        self.neighbor_reduction = neighbor_reduction
        self.random_state = int(random_state)

        self.memory_bank: Optional[torch.Tensor] = None
        self._nn_index: Optional[NearestNeighbors] = None
        self.fit_summary: Optional[PatchCoreFitSummary] = None

        # Keep the baseline inert with respect to existing fusion/combination hooks.
        self.score_fusion_enabled = False
        self.score_combination_enabled = False
        self.supports_pipeline_visualization = False

    def forward(
        self,
        x: torch.Tensor,
        return_dense: bool = False,
        return_maps: Optional[bool] = None,
        target_size: Optional[tuple[int, int]] = None,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        """Accept the legacy detector forward contract used by visualizers.

        Existing evaluation helpers call ``model(images, return_dense=...)`` on
        anchor/reconstruction detectors. PatchCore has no separate dense branch,
        so ``return_dense`` simply maps to whether pixel maps are returned.
        """
        if return_maps is None:
            return_maps = return_dense
        return self.compute_anomaly_scores(x, return_maps=return_maps, target_size=target_size)

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract normalized dense patch features as ``(B, H, W, D)``."""
        outputs = self.backbone(images)
        if 'dense' not in outputs:
            raise KeyError("Backbone output must contain a 'dense' feature map.")

        dense = outputs['dense']
        if dense.ndim != 4:
            raise ValueError(f"Expected dense features with 4 dims, got shape {tuple(dense.shape)}")

        return F.normalize(dense.float(), dim=-1)

    def _sample_dense_batch(
        self,
        dense_features: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Subsample a dense feature batch to a flat memory-bank chunk."""
        batch_size, height, width, feature_dim = dense_features.shape
        flat = dense_features.reshape(batch_size, height * width, feature_dim)

        if self.patches_per_image is None or self.patches_per_image >= height * width:
            return flat.reshape(-1, feature_dim).cpu()

        sampled = []
        for image_features in flat:
            indices = rng.choice(
                image_features.shape[0],
                size=self.patches_per_image,
                replace=False,
            )
            sampled.append(image_features[torch.from_numpy(indices).long()])

        return torch.cat(sampled, dim=0).cpu()

    def _build_index(self) -> None:
        """Create the sklearn neighbor index on the CPU memory bank."""
        if self.memory_bank is None or self.memory_bank.numel() == 0:
            raise RuntimeError('Cannot build index without a fitted memory bank.')

        self._nn_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='auto',
            metric=self.distance_metric,
            n_jobs=-1,
        )
        self._nn_index.fit(self.memory_bank.numpy())

    def set_memory_bank(self, memory_bank: torch.Tensor) -> None:
        """Set a normalized CPU memory bank and build the neighbor index."""
        if memory_bank.ndim != 2:
            raise ValueError(f"Expected memory bank shape (N, D), got {tuple(memory_bank.shape)}")
        if memory_bank.shape[0] == 0:
            raise ValueError('Memory bank must contain at least one feature vector.')

        normalized = F.normalize(memory_bank.detach().float().cpu(), dim=1)
        self.memory_bank = normalized.contiguous()
        self._build_index()

    def fit(
        self,
        dataloader: DataLoader,
        *,
        device: torch.device,
        max_images: Optional[int] = None,
    ) -> PatchCoreFitSummary:
        """Build a patch memory bank from normal training images."""
        self.eval()

        rng = np.random.default_rng(self.random_state)
        collected_batches = []
        num_images = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Fitting PatchCore memory bank'):
                images = batch['image']
                if max_images is not None:
                    remaining = max_images - num_images
                    if remaining <= 0:
                        break
                    images = images[:remaining]
                    if images.numel() == 0:
                        break

                images = images.to(device)
                dense_features = self._extract_patch_features(images)
                sampled = self._sample_dense_batch(dense_features, rng)
                collected_batches.append(sampled)
                num_images += int(images.shape[0])

        if not collected_batches:
            raise RuntimeError('No PatchCore features were collected from the dataloader.')

        memory_bank = torch.cat(collected_batches, dim=0)
        num_features_before_cap = int(memory_bank.shape[0])

        if self.max_memory_bank_size is not None and num_features_before_cap > self.max_memory_bank_size:
            selected = rng.choice(num_features_before_cap, size=self.max_memory_bank_size, replace=False)
            selected = np.sort(selected)
            memory_bank = memory_bank[torch.from_numpy(selected).long()]

        self.set_memory_bank(memory_bank)

        patches_per_image = self.patches_per_image
        if patches_per_image is None:
            patches_per_image = num_features_before_cap // max(num_images, 1)

        self.fit_summary = PatchCoreFitSummary(
            num_images=num_images,
            num_features_before_cap=num_features_before_cap,
            num_features_after_cap=int(self.memory_bank.shape[0]),
            feature_dim=int(self.memory_bank.shape[1]),
            patches_per_image=int(patches_per_image or 0),
        )
        return self.fit_summary

    def _reduce_neighbor_distances(self, distances: np.ndarray) -> np.ndarray:
        """Reduce kNN distances to a single scalar per query patch."""
        if self.neighbor_reduction == 'max':
            return distances.max(axis=1)
        return distances.mean(axis=1)

    def _query_memory_bank(self, flat_features: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Query the CPU memory bank in chunks and return distances on ``device``."""
        if self._nn_index is None:
            raise RuntimeError('PatchCoreBaseline must be fitted before inference.')

        flat_np = flat_features.detach().float().cpu().numpy()
        chunk_scores = []
        for start in range(0, flat_np.shape[0], self.query_batch_size):
            stop = start + self.query_batch_size
            chunk = flat_np[start:stop]
            distances, _ = self._nn_index.kneighbors(chunk, return_distance=True)
            reduced = self._reduce_neighbor_distances(distances)
            chunk_scores.append(torch.from_numpy(reduced).float())

        return torch.cat(chunk_scores, dim=0).to(device)

    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        return_maps: bool = True,
        target_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return PatchCore-style image and pixel anomaly scores."""
        dense_features = self._extract_patch_features(x)
        batch_size, height, width, feature_dim = dense_features.shape
        flat_features = dense_features.reshape(batch_size * height * width, feature_dim)
        flat_scores = self._query_memory_bank(flat_features, device=x.device)

        patch_map = flat_scores.reshape(batch_size, height, width)
        pixel_scores = patch_map
        if target_size is not None and patch_map.shape[1:] != target_size:
            pixel_scores = F.interpolate(
                patch_map.unsqueeze(1),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

        image_scores = aggregate_pixel_scores_torch(
            pixel_scores,
            method=self.aggregation_method,
            percentile=self.aggregation_percentile,
            threshold=self.aggregation_threshold,
        )

        result: Dict[str, torch.Tensor] = {
            'image_scores': image_scores,
            'anchor_scores': image_scores,
            'pixel_aggregated_score': image_scores,
        }

        if return_maps:
            result['pixel_scores'] = pixel_scores
            result['anchor_pixel_scores'] = pixel_scores
            result['pixel_anomaly_map'] = pixel_scores
            result['pixel_scores_source'] = 'patchcore_patch_knn'
            result['anchor_pixel_scores_source'] = 'patchcore_patch_knn'

        return result