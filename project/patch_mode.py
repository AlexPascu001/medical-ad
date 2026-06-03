"""Patch-anchor mode built on dense anchor maps.

This module keeps the existing global pipeline intact and provides separate
stage-1 detector/loss paths selected via anchor.mode='patch'.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AnomalyDetector, DINOv3Backbone
from pixel_aggregation import aggregate_pixel_scores_torch


def get_patch_variant(config: dict) -> str:
    """Return the configured patch-mode variant with backward-compatible defaulting."""
    return str(config.get('anchor', {}).get('patch', {}).get('variant', 'legacy')).lower()


def _validate_patch_reduction(name: str, reduction: str) -> None:
    if reduction not in {'mean', 'max', 'percentile'}:
        raise ValueError(f"Unsupported {name}: {reduction}. Expected 'mean', 'max', or 'percentile'.")


def _validate_location_kmeans_config(config: dict) -> Tuple[str, str, str, float]:
    """Reject settings that still depend on the legacy image-level patch path."""
    train_augment_mode = config.get('data', {}).get('train_augment_mode', 'full')
    if train_augment_mode != 'none':
        raise ValueError(
            "anchor.patch.variant='location_kmeans' requires data.train_augment_mode='none' "
            "to preserve strict patch-location alignment."
        )

    stage2_cfg = config.get('stage2', {})
    if stage2_cfg.get('enabled', False) and stage2_cfg.get('alignment_target', 'sample') == 'anchor':
        raise ValueError(
            "anchor.patch.variant='location_kmeans' does not support stage2.alignment_target='anchor'; "
            "use 'sample' or 'local_anchor_pool'."
        )

    training_cfg = config.get('training', {})
    if training_cfg.get('fixed_pseudo_labels', False):
        raise ValueError(
            "anchor.patch.variant='location_kmeans' does not use image-level pseudo-labels; "
            "set training.fixed_pseudo_labels=false."
        )

    if config.get('pretraining', {}).get('enabled', False):
        raise ValueError("anchor.patch.variant='location_kmeans' does not support projection pretraining in v1.")

    anchor_cfg = config.get('anchor', {})
    if anchor_cfg.get('strategy', 'kmeans') != 'kmeans':
        raise ValueError("anchor.patch.variant='location_kmeans' currently requires anchor.strategy='kmeans'.")

    if anchor_cfg.get('representation', 'closest_samples') != 'centroids':
        raise ValueError(
            "anchor.patch.variant='location_kmeans' currently requires anchor.representation='centroids'."
        )

    if not anchor_cfg.get('use_embedding_space', False):
        raise ValueError(
            "anchor.patch.variant='location_kmeans' requires anchor.use_embedding_space=true "
            "to build the local bank from frozen DINO patch tokens."
        )

    projection_hidden_dims = config.get('model', {}).get('projection_hidden_dims', None)
    projection_dim = projection_hidden_dims[-1] if projection_hidden_dims is not None else config.get('model', {}).get('projection_dim', None)
    if projection_dim is not None and not anchor_cfg.get('reproject_anchors', False):
        raise ValueError(
            "anchor.patch.variant='location_kmeans' currently requires anchor.reproject_anchors=true "
            "when a projection head is enabled."
        )

    if anchor_cfg.get('learnable', False):
        raise ValueError("anchor.patch.variant='location_kmeans' uses a fixed local centroid bank in v1.")

    patch_cfg = anchor_cfg.get('patch', {})
    local_distance_metric = patch_cfg.get('local_distance_metric', 'euclidean').lower()
    if local_distance_metric not in {'euclidean', 'cosine'}:
        raise ValueError(
            f"Unsupported anchor.patch.local_distance_metric: {local_distance_metric}. "
            "Expected 'euclidean' or 'cosine'."
        )

    _validate_patch_reduction('anchor.patch.score_reduction', patch_cfg.get('score_reduction', 'mean'))
    local_score_reduction = patch_cfg.get('local_score_reduction', patch_cfg.get('score_reduction', 'percentile'))
    _validate_patch_reduction('anchor.patch.local_score_reduction', local_score_reduction)
    local_score_percentile = float(patch_cfg.get('local_score_percentile', 95.0))
    if not 0.0 <= local_score_percentile <= 100.0:
        raise ValueError(
            f"anchor.patch.local_score_percentile must be in [0, 100], got {local_score_percentile}."
        )

    return train_augment_mode, local_distance_metric, local_score_reduction, local_score_percentile


def prepare_location_kmeans_anchors(
    train_images: list,
    preprocessor,
    config: dict,
    save_dir: Path,
    backbone: DINOv3Backbone,
    device: torch.device,
) -> tuple:
    """Build a same-location local centroid bank from frozen DINO patch tokens."""
    _, local_distance_metric, _, local_score_percentile = _validate_location_kmeans_config(config)

    max_images = config['anchor'].get('max_images_for_pca', 5000)
    if max_images is None:
        max_images = len(train_images)
    selected_paths = train_images[:max_images]

    print("\n" + "=" * 80)
    print("PATCH LOCATION_KMEANS BANK GENERATION")
    print("=" * 80)
    print("Strategy: per-location local k-means in frozen DINO patch space")
    print(f"  Training images used : {len(selected_paths)}")
    print(f"  Local centroids / pos: {config['anchor']['n_anchors']}")
    print(f"  Local metric         : {local_distance_metric}")

    import cv2
    from anchors import _grayscale_batch_to_tensor
    from sklearn.cluster import KMeans

    images_list = []
    for img_path in selected_paths:
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images_list.append(preprocessor.preprocess(img))

    if not images_list:
        raise ValueError("No training images could be loaded for location_kmeans bank generation.")

    images_np = np.array(images_list)
    n_images = int(images_np.shape[0])
    n_anchors = int(config['anchor']['n_anchors'])
    if n_images < n_anchors:
        raise ValueError(
            f"anchor.patch.variant='location_kmeans' requires at least as many training images as centroids: "
            f"got {n_images} images for {n_anchors} centroids."
        )

    print(f"Loaded {n_images} images, shape: {images_np.shape}")

    patch_batches = []
    norm_mode = config['data'].get('normalization', 'zscore_only')
    batch_size = 64
    backbone.eval()

    with torch.no_grad():
        for start_idx in range(0, n_images, batch_size):
            batch_imgs = images_np[start_idx:start_idx + batch_size]
            batch_tensor = _grayscale_batch_to_tensor(
                batch_imgs,
                device,
                apply_imagenet_norm=(norm_mode == 'minmax_imagenet'),
            )
            features = backbone.backbone.forward_features(batch_tensor)
            patch_tokens = features[:, 1 + backbone.num_register_tokens:]
            h_patches = batch_tensor.shape[2] // backbone.patch_size
            w_patches = batch_tensor.shape[3] // backbone.patch_size
            patch_batches.append(patch_tokens.view(batch_tensor.shape[0], h_patches, w_patches, -1).cpu())

    patch_tensor = torch.cat(patch_batches, dim=0).float()
    if local_distance_metric == 'cosine':
        patch_tensor = F.normalize(patch_tensor, dim=-1)

    _, h_patches, w_patches, embed_dim = patch_tensor.shape
    print(f"Extracted patch bank tensor: {tuple(patch_tensor.shape)}")

    centroids = torch.empty((n_anchors, h_patches, w_patches, embed_dim), dtype=patch_tensor.dtype)
    location_inertias = []

    for row in range(h_patches):
        for col in range(w_patches):
            location_vectors = patch_tensor[:, row, col, :].numpy()
            kmeans = KMeans(n_clusters=n_anchors, random_state=config['seed'], n_init=10)
            kmeans.fit(location_vectors)
            centers = torch.from_numpy(kmeans.cluster_centers_).to(dtype=patch_tensor.dtype)
            if local_distance_metric == 'cosine':
                centers = F.normalize(centers, dim=1)
            centroids[:, row, col, :] = centers
            location_inertias.append(float(kmeans.inertia_))

    anchor_dense = centroids.contiguous()
    anchor_global = anchor_dense.mean(dim=(1, 2))
    anchor_global = F.normalize(anchor_global, dim=1)

    anchor_metadata = {
        'variant': 'location_kmeans',
        'representation': 'centroids',
        'initial_k': n_anchors,
        'effective_k': n_anchors,
        'bank_grid': [int(h_patches), int(w_patches)],
        'embedding_dim': int(embed_dim),
        'num_train_images': n_images,
        'local_distance_metric': local_distance_metric,
        'local_score_percentile': local_score_percentile,
        'mean_location_inertia': float(np.mean(location_inertias)),
        'std_location_inertia': float(np.std(location_inertias)),
    }

    torch.save({
        'anchor_images': None,
        'anchor_global': anchor_global,
        'anchor_dense': anchor_dense,
        'embedding_dim': int(embed_dim),
        'projection_dim': config['model'].get('projection_hidden_dims', [config['model'].get('projection_dim', None)])[-1],
        'is_projected': False,
        'generation_method': 'patch_location_kmeans',
        'anchor_metadata': anchor_metadata,
    }, save_dir / 'anchor_embeddings.pt')

    print("\nLocation-kmeans bank preparation complete!")
    print(f"  Local centroid bank: {tuple(anchor_dense.shape)}")
    print(f"  Summary anchors    : {tuple(anchor_global.shape)}")

    return None, anchor_global, anchor_dense


class PatchAnomalyDetector(AnomalyDetector):
    """Dense-anchor detector that scores images from corresponding patch distances."""

    def __init__(
        self,
        backbone: DINOv3Backbone,
        anchor_global_embeddings: torch.Tensor,
        anchor_dense_embeddings: torch.Tensor,
        distance_metric: str = 'cosine',
        learnable_anchors: bool = False,
        target_size: tuple[int, int] = (240, 240),
        anchors_already_projected: bool = False,
        score_reduction: str = 'mean',
    ):
        if anchor_dense_embeddings is None:
            raise ValueError("anchor.mode='patch' requires dense anchor embeddings.")

        super().__init__(
            backbone=backbone,
            anchor_global_embeddings=anchor_global_embeddings,
            anchor_dense_embeddings=anchor_dense_embeddings,
            distance_metric=distance_metric,
            learnable_anchors=learnable_anchors,
            use_pixel_decoder=False,
            target_size=target_size,
            anchors_already_projected=anchors_already_projected,
        )
        self.anchor_mode = 'patch'
        self.score_reduction = score_reduction

    def _get_projected_anchors(self):
        projected_anchors = super()._get_projected_anchors()
        anchor_dense = projected_anchors[1]
        if anchor_dense is None:
            raise RuntimeError("Patch mode requires dense anchor embeddings after projection.")

        anchor_summary = anchor_dense.mean(dim=(1, 2))
        anchor_summary = F.normalize(anchor_summary, dim=1)
        return anchor_summary, anchor_dense

    def _compute_dense_distances(
        self,
        dense_feat: torch.Tensor,
        anchor_dense: torch.Tensor,
    ) -> torch.Tensor:
        if dense_feat.shape[1:3] != anchor_dense.shape[1:3]:
            raise ValueError(
                f"Patch mode requires matching spatial grids, got sample {dense_feat.shape[1:3]} "
                f"and anchor {anchor_dense.shape[1:3]}."
            )

        if self.distance_metric == 'cosine':
            dense_feat = F.normalize(dense_feat, dim=-1)
            anchor_dense = F.normalize(anchor_dense, dim=-1)
            return 1.0 - (dense_feat.unsqueeze(1) * anchor_dense.unsqueeze(0)).sum(dim=-1)

        return torch.norm(dense_feat.unsqueeze(1) - anchor_dense.unsqueeze(0), p=2, dim=-1)

    def _extract_raw_dense_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen pre-projection patch features for semantic patch assignments."""
        features = self.backbone.backbone.forward_features(x)
        num_register_tokens = self.backbone.num_register_tokens
        patch_tokens = features[:, 1 + num_register_tokens:]
        height, width = x.shape[2:]
        h_patches = height // self.backbone.patch_size
        w_patches = width // self.backbone.patch_size
        return patch_tokens.view(x.shape[0], h_patches, w_patches, -1)

    def _reduce_dense_scores(self, dense_distances: torch.Tensor) -> torch.Tensor:
        flat = dense_distances.view(dense_distances.shape[0], dense_distances.shape[1], -1)
        if self.score_reduction == 'max':
            return flat.max(dim=-1)[0]
        return flat.mean(dim=-1)

    def compute_label_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-image anchor assignment distances for fixed patch pseudo-labels."""
        if getattr(self, 'anchor_dense_raw', None) is not None:
            dense_feat = self._extract_raw_dense_features(x)
            anchor_dense = self.anchor_dense_raw
            dense_distances = self._compute_dense_distances(dense_feat, anchor_dense)
            return self._reduce_dense_scores(dense_distances)

        outputs = self.forward(x, return_dense=False)
        return outputs['global_distances']

    def forward(self, x: torch.Tensor, return_dense: bool = False) -> Dict[str, torch.Tensor]:
        features = self.backbone(x, return_multi_scale=False)
        dense_feat = features['dense']
        _, anchor_dense = self._get_projected_anchors()

        dense_distances = self._compute_dense_distances(dense_feat, anchor_dense)
        global_distances = self._reduce_dense_scores(dense_distances)

        pooled_feat = dense_feat.mean(dim=(1, 2))
        pooled_feat = F.normalize(pooled_feat, dim=1)

        outputs = {
            'global_feat': pooled_feat,
            'global_distances': global_distances,
            'dense_feat': dense_feat,
            'dense_distances': dense_distances,
        }

        if self.reconstruction_enabled and self.stage2_projection is not None:
            _, _, height, width = x.shape
            global_raw = features['global_raw']
            stage2_feat = self.stage2_projection(global_raw)
            stage2_feat = F.normalize(stage2_feat, dim=1)

            decoupled = getattr(self, '_recon_dim', None) is not None and self._recon_dim != self._anchor_dim

            frozen_feat = None
            bottleneck_divergence = None
            if self.use_frozen_bottleneck and self.frozen_projection is not None:
                with torch.no_grad():
                    frozen_feat = self.frozen_projection(global_raw)
                    frozen_feat = F.normalize(frozen_feat, dim=1)
                if decoupled:
                    bottleneck_divergence = 1.0 - (frozen_feat * pooled_feat).sum(dim=1)
                else:
                    bottleneck_divergence = 1.0 - (frozen_feat * stage2_feat).sum(dim=1)

            patch_divergence_map = None
            if self.use_frozen_bottleneck and self.frozen_projection is not None and 'patch_raw' in features:
                patch_raw = features['patch_raw']
                batch_p, num_patches, dim_p = patch_raw.shape
                with torch.no_grad():
                    frozen_patches = self.frozen_projection(patch_raw.reshape(batch_p * num_patches, dim_p))
                    frozen_patches = F.normalize(frozen_patches, dim=1).view(batch_p, num_patches, -1)
                if decoupled:
                    if self.backbone.projection is not None:
                        with torch.no_grad():
                            stage1_patches = self.backbone.projection(patch_raw.reshape(batch_p * num_patches, dim_p))
                            stage1_patches = F.normalize(stage1_patches, dim=1).view(batch_p, num_patches, -1)
                    else:
                        stage1_patches = F.normalize(patch_raw, dim=2)
                else:
                    stage1_patches = self.stage2_projection(patch_raw.reshape(batch_p * num_patches, dim_p))
                    stage1_patches = F.normalize(stage1_patches, dim=1).view(batch_p, num_patches, -1)

                per_patch_div = 1.0 - (frozen_patches * stage1_patches).sum(dim=-1)
                h_patches = height // self.backbone.patch_size
                w_patches = width // self.backbone.patch_size
                patch_divergence_map = F.interpolate(
                    per_patch_div.view(batch_p, 1, h_patches, w_patches),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            assigned_anchor_idx = global_distances.argmin(dim=1)
            anchor_global, _ = self._get_projected_anchors()
            assigned_anchor_embeddings = anchor_global[assigned_anchor_idx]
            if self.stage2_freeze_anchor_target:
                assigned_anchor_embeddings = assigned_anchor_embeddings.detach()

            if self.no_fuser:
                recon_latent = stage2_feat
            else:
                if self.anchor_reproject is not None:
                    assigned_anchor_embeddings = self.anchor_reproject(assigned_anchor_embeddings)
                recon_latent = self.stage2_fuser(torch.cat([assigned_anchor_embeddings, stage2_feat], dim=1))

            reconstruction = self.reconstruction_decoder(recon_latent)
            reconstruction_error = (reconstruction - x).pow(2).mean(dim=(1, 2, 3))

            reconstruction_pixel_map = None
            if self.stage2_pixel_map_enabled:
                if self.stage2_pixel_map_type == 'reconstruction_l1':
                    reconstruction_pixel_map = (reconstruction - x).abs().mean(dim=1)
                else:
                    reconstruction_pixel_map = (reconstruction - x).pow(2).mean(dim=1)

            outputs.update({
                'stage2_feat': stage2_feat,
                'frozen_feat': frozen_feat,
                'bottleneck_divergence': bottleneck_divergence,
                'patch_divergence_map': patch_divergence_map,
                'recon_latent': recon_latent,
                'assigned_anchor_idx': assigned_anchor_idx,
                'assigned_anchor_embeddings': assigned_anchor_embeddings,
                'reconstruction': reconstruction,
                'reconstruction_error': reconstruction_error,
                'reconstruction_pixel_map': reconstruction_pixel_map
            })

        return outputs


class PatchAnchorLoss(nn.Module):
    """Stage-1 loss for patch mode using per-anchor dense maps as the attractor."""

    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        delta: float = 0.0,
        diversity_temperature: float = 0.1,
        distance_metric: str = 'euclidean',
        spatial_reduction: str = 'mean',
    ):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.diversity_temperature = diversity_temperature
        self.distance_metric = distance_metric
        self.spatial_reduction = spatial_reduction
        self.global_loss = self

    def _reduce_spatial(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.view(values.shape[0], -1)
        if self.spatial_reduction == 'max':
            return flat.max(dim=1)[0]
        return flat.mean(dim=1)

    def forward(self, outputs: Dict[str, torch.Tensor], anchor_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        distances = outputs['global_distances']
        fixed_assignments = outputs.get('fixed_assignments')

        if fixed_assignments is not None:
            assigned_anchors = fixed_assignments
        else:
            assigned_anchors = distances.argmin(dim=1)

        batch_indices = torch.arange(distances.shape[0], device=distances.device)
        assigned_dense = outputs['dense_distances'][batch_indices, assigned_anchors]
        assigned_scores = self._reduce_spatial(assigned_dense)
        loss_attract = 0.5 * (assigned_scores ** 2).mean()

        loss_repel = torch.tensor(0.0, device=distances.device, dtype=distances.dtype)
        if anchor_embeddings.shape[0] > 1 and self.beta > 0:
            if self.distance_metric == 'cosine':
                anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
                anchor_distances = 1.0 - (anchor_embeddings @ anchor_embeddings.t())
            else:
                anchor_distances = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)

            mask = ~torch.eye(anchor_distances.shape[0], dtype=torch.bool, device=anchor_distances.device)
            violations = torch.relu(2 * self.margin - anchor_distances)
            loss_repel = 0.5 * (violations[mask] ** 2).mean()

        loss_diversity = torch.tensor(0.0, device=distances.device, dtype=distances.dtype)
        if self.delta > 0 and distances.shape[1] > 1:
            soft_assignments = torch.softmax(-distances / self.diversity_temperature, dim=1)
            avg_assignments = soft_assignments.mean(dim=0)
            entropy = -(avg_assignments * torch.log(avg_assignments + 1e-8)).sum()
            max_entropy = math.log(distances.shape[1])
            loss_diversity = 1.0 - (entropy / max(max_entropy, 1e-12))

        total_loss = self.alpha * loss_attract + self.beta * loss_repel + self.delta * loss_diversity
        return {
            'loss': total_loss,
            'loss_global': total_loss,
            'loss_global_attract': loss_attract.item(),
            'loss_global_repel': loss_repel.item(),
            'loss_global_norm': 0.0,
            'loss_global_diversity': loss_diversity.item() if self.delta > 0 else 0.0,
            'assigned_anchors': assigned_anchors,
        }


class LocationKMeansPatchAnomalyDetector(PatchAnomalyDetector):
    """Same-location local centroid bank for patch-wise matching."""

    def __init__(
        self,
        backbone: DINOv3Backbone,
        anchor_global_embeddings: torch.Tensor,
        anchor_dense_embeddings: torch.Tensor,
        distance_metric: str = 'euclidean',
        learnable_anchors: bool = False,
        target_size: tuple[int, int] = (240, 240),
        anchors_already_projected: bool = False,
        score_reduction: str = 'mean',
        local_score_reduction: str = 'mean',
        local_score_percentile: float = 95.0,
    ):
        super().__init__(
            backbone=backbone,
            anchor_global_embeddings=anchor_global_embeddings,
            anchor_dense_embeddings=anchor_dense_embeddings,
            distance_metric=distance_metric,
            learnable_anchors=learnable_anchors,
            target_size=target_size,
            anchors_already_projected=anchors_already_projected,
            score_reduction=score_reduction,
        )
        self.patch_variant = 'location_kmeans'
        self.local_score_reduction = local_score_reduction
        self.local_score_percentile = float(local_score_percentile)

    def _pool_local_anchor_guidance(
        self,
        anchor_dense: torch.Tensor,
        local_assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather assigned local anchors and pool them into one guidance vector per image."""
        batch_size, height_patches, width_patches = local_assignments.shape
        anchor_dense_by_location = anchor_dense.permute(1, 2, 0, 3).contiguous()
        anchor_dense_flat = anchor_dense_by_location.view(height_patches * width_patches, anchor_dense.shape[0], -1)

        assignment_flat = local_assignments.view(batch_size, height_patches * width_patches)
        location_indices = torch.arange(height_patches * width_patches, device=anchor_dense.device)
        location_indices = location_indices.view(1, height_patches * width_patches).expand(batch_size, -1)

        local_anchor_map = anchor_dense_flat[location_indices, assignment_flat]
        local_anchor_map = local_anchor_map.view(batch_size, height_patches, width_patches, -1)
        pooled_guidance = F.normalize(local_anchor_map.mean(dim=(1, 2)), dim=1)
        return local_anchor_map, pooled_guidance

    def _reduce_local_scores(self, local_scores: torch.Tensor) -> torch.Tensor:
        flat = local_scores.view(local_scores.shape[0], -1)
        if self.local_score_reduction == 'max':
            return flat.max(dim=1)[0]
        if self.local_score_reduction == 'percentile':
            return torch.quantile(flat, self.local_score_percentile / 100.0, dim=1)
        return flat.mean(dim=1)

    def forward(self, x: torch.Tensor, return_dense: bool = False) -> Dict[str, torch.Tensor]:
        features = self.backbone(x, return_multi_scale=False)
        dense_feat = features['dense']
        _, anchor_dense = self._get_projected_anchors()

        dense_distances = self._compute_dense_distances(dense_feat, anchor_dense)
        global_distances = self._reduce_dense_scores(dense_distances)
        local_min_distances, local_assignments = dense_distances.min(dim=1)

        pooled_feat = dense_feat.mean(dim=(1, 2))
        pooled_feat = F.normalize(pooled_feat, dim=1)

        outputs = {
            'global_feat': pooled_feat,
            'global_distances': global_distances,
            'dense_feat': dense_feat,
            'dense_distances': dense_distances,
            'local_min_distances': local_min_distances,
            'local_assignments': local_assignments,
            'assigned_anchors': global_distances.argmin(dim=1),
            'image_scores': self._reduce_local_scores(local_min_distances),
        }

        if self.reconstruction_enabled and self.stage2_projection is not None:
            _, _, height, width = x.shape
            global_raw = features['global_raw']
            stage2_feat = self.stage2_projection(global_raw)
            stage2_feat = F.normalize(stage2_feat, dim=1)

            decoupled = getattr(self, '_recon_dim', None) is not None and self._recon_dim != self._anchor_dim

            frozen_feat = None
            bottleneck_divergence = None
            if self.use_frozen_bottleneck and self.frozen_projection is not None:
                with torch.no_grad():
                    frozen_feat = self.frozen_projection(global_raw)
                    frozen_feat = F.normalize(frozen_feat, dim=1)
                if decoupled:
                    bottleneck_divergence = 1.0 - (frozen_feat * pooled_feat).sum(dim=1)
                else:
                    bottleneck_divergence = 1.0 - (frozen_feat * stage2_feat).sum(dim=1)

            patch_divergence_map = None
            if self.use_frozen_bottleneck and self.frozen_projection is not None and 'patch_raw' in features:
                patch_raw = features['patch_raw']
                batch_p, num_patches, dim_p = patch_raw.shape
                with torch.no_grad():
                    frozen_patches = self.frozen_projection(patch_raw.reshape(batch_p * num_patches, dim_p))
                    frozen_patches = F.normalize(frozen_patches, dim=1).view(batch_p, num_patches, -1)
                if decoupled:
                    if self.backbone.projection is not None:
                        with torch.no_grad():
                            stage1_patches = self.backbone.projection(patch_raw.reshape(batch_p * num_patches, dim_p))
                            stage1_patches = F.normalize(stage1_patches, dim=1).view(batch_p, num_patches, -1)
                    else:
                        stage1_patches = F.normalize(patch_raw, dim=2)
                else:
                    stage1_patches = self.stage2_projection(patch_raw.reshape(batch_p * num_patches, dim_p))
                    stage1_patches = F.normalize(stage1_patches, dim=1).view(batch_p, num_patches, -1)

                per_patch_div = 1.0 - (frozen_patches * stage1_patches).sum(dim=-1)
                h_patches = height // self.backbone.patch_size
                w_patches = width // self.backbone.patch_size
                patch_divergence_map = F.interpolate(
                    per_patch_div.view(batch_p, 1, h_patches, w_patches),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            local_anchor_map, pooled_local_guidance = self._pool_local_anchor_guidance(anchor_dense, local_assignments)
            assigned_anchor_embeddings = pooled_local_guidance
            if self.stage2_freeze_anchor_target:
                assigned_anchor_embeddings = assigned_anchor_embeddings.detach()

            if self.no_fuser:
                recon_latent = stage2_feat
                stage2_guidance = assigned_anchor_embeddings
            else:
                stage2_guidance = assigned_anchor_embeddings
                if self.anchor_reproject is not None:
                    stage2_guidance = self.anchor_reproject(stage2_guidance)
                recon_latent = self.stage2_fuser(torch.cat([stage2_guidance, stage2_feat], dim=1))

            reconstruction = self.reconstruction_decoder(recon_latent)
            reconstruction_error = (reconstruction - x).pow(2).mean(dim=(1, 2, 3))

            reconstruction_pixel_map = None
            if self.stage2_pixel_map_enabled:
                if self.stage2_pixel_map_type == 'reconstruction_l1':
                    reconstruction_pixel_map = (reconstruction - x).abs().mean(dim=1)
                else:
                    reconstruction_pixel_map = (reconstruction - x).pow(2).mean(dim=1)

            outputs.update({
                'stage2_feat': stage2_feat,
                'frozen_feat': frozen_feat,
                'bottleneck_divergence': bottleneck_divergence,
                'patch_divergence_map': patch_divergence_map,
                'recon_latent': recon_latent,
                'assigned_anchor_embeddings': stage2_guidance,
                'stage2_guidance': stage2_guidance,
                'local_anchor_guidance_map': local_anchor_map,
                'reconstruction': reconstruction,
                'reconstruction_error': reconstruction_error,
                'reconstruction_pixel_map': reconstruction_pixel_map,
            })

        return outputs
    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        return_maps: bool = True,
        target_size: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.forward(x, return_dense=return_maps)
            image_scores = outputs['image_scores']
            result = {
                'image_scores': image_scores,
                'anchor_scores': image_scores,
                'assigned_anchors': outputs['assigned_anchors'],
                'all_distances': outputs['global_distances'],
            }

            if 'reconstruction_error' in outputs:
                result['reconstruction_scores'] = outputs['reconstruction_error']
                result['reconstruction'] = outputs.get('reconstruction')

            if outputs.get('bottleneck_divergence') is not None:
                result['bottleneck_divergence'] = outputs['bottleneck_divergence']

            if outputs.get('patch_divergence_map') is not None:
                patch_div_map = outputs['patch_divergence_map']
                if target_size is not None and patch_div_map.shape[1:] != target_size:
                    patch_div_map = F.interpolate(
                        patch_div_map.unsqueeze(1),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(1)
                if return_maps:
                    result['patch_divergence_map'] = patch_div_map
                patch_div_aggregated = aggregate_pixel_scores_torch(
                    patch_div_map,
                    method=self.pixel_aggregation_method,
                    percentile=self.pixel_aggregation_percentile,
                    threshold=self.pixel_aggregation_threshold,
                )
                result['patch_divergence_aggregated_score'] = patch_div_aggregated

            if outputs.get('reconstruction_pixel_map') is not None:
                reconstruction_pixel_scores = outputs['reconstruction_pixel_map']
                if target_size is not None and reconstruction_pixel_scores.shape[1:] != target_size:
                    reconstruction_pixel_scores = F.interpolate(
                        reconstruction_pixel_scores.unsqueeze(1),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(1)
                if return_maps:
                    result['reconstruction_pixel_scores'] = reconstruction_pixel_scores
                    result['pixel_scores'] = reconstruction_pixel_scores
                    result['pixel_scores_source'] = 'reconstruction_pixel_map'

                pixel_aggregated = aggregate_pixel_scores_torch(
                    reconstruction_pixel_scores,
                    method=self.pixel_aggregation_method,
                    percentile=self.pixel_aggregation_percentile,
                    threshold=self.pixel_aggregation_threshold,
                )
                result['pixel_aggregated_score'] = pixel_aggregated

            if return_maps:
                pixel_scores = outputs['local_min_distances']
                if target_size is not None and pixel_scores.shape[1:] != target_size:
                    pixel_scores = F.interpolate(
                        pixel_scores.unsqueeze(1),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(1)
                result['anchor_pixel_scores'] = pixel_scores
                result['anchor_pixel_scores_source'] = 'dense_patch_upsampled'
                if 'pixel_scores' not in result:
                    result['pixel_scores'] = pixel_scores
                    result['pixel_scores_source'] = 'dense_patch_upsampled'

            if self.score_combination_enabled and ('anchor_scores' in result) and ('reconstruction_scores' in result):
                alpha = self.score_combination_alpha
                result['combined_scores_raw'] = (1.0 - alpha) * result['anchor_scores'] + alpha * result['reconstruction_scores']

            return result


class LocationKMeansPatchLoss(PatchAnchorLoss):
    """Stage-1 loss that attracts each patch to its nearest same-location centroid."""

    def forward(self, outputs: Dict[str, torch.Tensor], anchor_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        if outputs.get('fixed_assignments') is not None:
            raise ValueError(
                "Location-kmeans patch mode does not support image-level fixed pseudo-label assignments."
            )

        distances = outputs['global_distances']
        local_min_distances = outputs['local_min_distances']
        assigned_anchors = outputs.get('assigned_anchors', distances.argmin(dim=1))

        assigned_scores = self._reduce_spatial(local_min_distances)
        loss_attract = 0.5 * (assigned_scores ** 2).mean()

        loss_repel = torch.tensor(0.0, device=distances.device, dtype=distances.dtype)
        if anchor_embeddings.shape[0] > 1 and self.beta > 0:
            if self.distance_metric == 'cosine':
                anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
                anchor_distances = 1.0 - (anchor_embeddings @ anchor_embeddings.t())
            else:
                anchor_distances = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)

            mask = ~torch.eye(anchor_distances.shape[0], dtype=torch.bool, device=anchor_distances.device)
            violations = torch.relu(2 * self.margin - anchor_distances)
            loss_repel = 0.5 * (violations[mask] ** 2).mean()

        loss_diversity = torch.tensor(0.0, device=distances.device, dtype=distances.dtype)
        if self.delta > 0 and distances.shape[1] > 1:
            soft_assignments = torch.softmax(-distances / self.diversity_temperature, dim=1)
            avg_assignments = soft_assignments.mean(dim=0)
            entropy = -(avg_assignments * torch.log(avg_assignments + 1e-8)).sum()
            max_entropy = math.log(distances.shape[1])
            loss_diversity = 1.0 - (entropy / max(max_entropy, 1e-12))

        total_loss = self.alpha * loss_attract + self.beta * loss_repel + self.delta * loss_diversity
        return {
            'loss': total_loss,
            'loss_global': total_loss,
            'loss_global_attract': loss_attract.item(),
            'loss_global_repel': loss_repel.item(),
            'loss_global_norm': 0.0,
            'loss_global_diversity': loss_diversity.item() if self.delta > 0 else 0.0,
            'assigned_anchors': assigned_anchors,
        }


def _build_backbone(config: dict) -> DINOv3Backbone:
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    multi_scale_indices = config['model'].get('multi_scale_indices', [2, 5, 8, 11])
    projection_hidden_dims = config['model'].get('projection_hidden_dims', None)
    projection_dim = projection_hidden_dims[-1] if projection_hidden_dims is not None else config['model'].get('projection_dim', None)

    return DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=projection_dim,
        pretrained=True,
        multi_scale_indices=multi_scale_indices if use_pixel_decoder else None,
        projection_hidden_dims=projection_hidden_dims,
    )


def create_patch_detector(
    config: dict,
    anchor_global: torch.Tensor,
    anchor_dense: torch.Tensor,
    backbone: Optional[DINOv3Backbone] = None,
) -> PatchAnomalyDetector:
    patch_variant = get_patch_variant(config)
    if config['model'].get('use_pixel_decoder', False):
        raise ValueError("anchor.mode='patch' currently uses dense anchor maps directly; disable model.use_pixel_decoder.")
    train_augment_mode = config.get('data', {}).get('train_augment_mode', 'full')
    if train_augment_mode == 'full':
        raise ValueError(
            "anchor.mode='patch' is incompatible with data.train_augment_mode='full'; "
            "use 'none' or 'flip_only'."
        )
    stage2_cfg = config.get('stage2', {})
    if stage2_cfg.get('enabled', False) and stage2_cfg.get('alignment_target', 'sample') == 'anchor':
        if not config.get('training', {}).get('fixed_pseudo_labels', False):
            raise ValueError(
                "anchor.mode='patch' with stage2.alignment_target='anchor' requires training.fixed_pseudo_labels=true."
            )
    if anchor_dense is None:
        raise ValueError(
            "anchor.mode='patch' requires dense anchor embeddings. Regenerate anchors without the pretraining shortcut."
        )

    if backbone is None:
        backbone = _build_backbone(config)

    use_embedding_space = config['anchor'].get('use_embedding_space', False)
    projection_hidden_dims = config['model'].get('projection_hidden_dims', None)
    projection_dim = projection_hidden_dims[-1] if projection_hidden_dims is not None else config['model'].get('projection_dim', None)
    anchors_already_projected = False if use_embedding_space else (projection_dim is not None)
    patch_cfg = config['anchor'].get('patch', {})

    if patch_variant == 'location_kmeans':
        _, local_distance_metric, local_score_reduction, local_score_percentile = _validate_location_kmeans_config(config)
        print("  Patch variant: location_kmeans")
        print("  Same-location local centroid bank with hard nearest-centroid patch assignments.")
        print(f"  Dense anchors: {tuple(anchor_dense.shape)}")
        return LocationKMeansPatchAnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_global,
            anchor_dense_embeddings=anchor_dense,
            distance_metric=local_distance_metric,
            learnable_anchors=False,
            target_size=tuple(config['data']['target_size']),
            anchors_already_projected=anchors_already_projected,
            score_reduction=patch_cfg.get('score_reduction', 'mean'),
            local_score_reduction=local_score_reduction,
            local_score_percentile=local_score_percentile,
        )

    print("  Patch mode uses dense anchor maps and aggregates per-location distances per anchor.")
    print(f"  Dense anchors: {tuple(anchor_dense.shape)}")
    if train_augment_mode != 'none':
        print(f"  Training augmentation preset: {train_augment_mode}")
    if stage2_cfg.get('enabled', False):
        print(f"  Stage-2 enabled for patch mode (alignment_target={stage2_cfg.get('alignment_target', 'sample')})")

    return PatchAnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_global,
        anchor_dense_embeddings=anchor_dense,
        distance_metric=config['loss']['distance_metric'],
        learnable_anchors=config['anchor'].get('learnable', False),
        target_size=tuple(config['data']['target_size']),
        anchors_already_projected=anchors_already_projected,
        score_reduction=patch_cfg.get('score_reduction', 'mean'),
    )


def create_patch_criterion(config: dict) -> PatchAnchorLoss:
    patch_variant = get_patch_variant(config)
    if patch_variant == 'location_kmeans':
        local_distance_metric = config['anchor'].get('patch', {}).get('local_distance_metric', 'euclidean')
        print("\nCreating loss function: patch/location_kmeans")
        return LocationKMeansPatchLoss(
            margin=config['loss']['margin'],
            alpha=config['loss']['alpha'],
            beta=config['loss']['beta'],
            delta=config['loss'].get('delta', 0.0),
            diversity_temperature=config['loss'].get('diversity_temperature', 0.1),
            distance_metric=local_distance_metric,
            spatial_reduction=config['loss'].get('spatial_reduction', 'mean'),
        )

    print("\nCreating loss function: patch")
    return PatchAnchorLoss(
        margin=config['loss']['margin'],
        alpha=config['loss']['alpha'],
        beta=config['loss']['beta'],
        delta=config['loss'].get('delta', 0.0),
        diversity_temperature=config['loss'].get('diversity_temperature', 0.1),
        distance_metric=config['loss']['distance_metric'],
        spatial_reduction=config['loss'].get('spatial_reduction', 'mean'),
    )