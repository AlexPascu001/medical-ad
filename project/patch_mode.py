"""Patch-anchor mode built on dense anchor maps.

This module keeps the existing global pipeline intact and provides a separate
stage-1 detector/loss path selected via anchor.mode='patch'.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AnomalyDetector, DINOv3Backbone


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
        score_reduction=config['anchor'].get('patch', {}).get('score_reduction', 'mean'),
    )


def create_patch_criterion(config: dict) -> PatchAnchorLoss:
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