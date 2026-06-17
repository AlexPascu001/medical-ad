"""
Training Loop for Anomaly Detector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
import json
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

from data import BMADDataset
from model import AnomalyDetector
from loss import CombinedAnchorLoss
from eval import evaluate_model


def _get_model_anchor_mode(model: AnomalyDetector) -> str:
    """Return the model anchor mode for checkpoint compatibility checks."""
    return str(getattr(model, 'anchor_mode', 'global')).lower()


def _validate_checkpoint_anchor_mode(checkpoint: Dict, expected_anchor_mode: str, checkpoint_path: str) -> None:
    """Reject checkpoints saved from a different anchor mode."""
    checkpoint_anchor_mode = str(checkpoint.get('anchor_mode', 'global')).lower()
    if checkpoint_anchor_mode != expected_anchor_mode:
        raise ValueError(
            f"Checkpoint anchor_mode mismatch for {checkpoint_path}: "
            f"expected '{expected_anchor_mode}', found '{checkpoint_anchor_mode}'."
        )


def _build_non_augmented_dataset(dataset) -> DataLoader:
    """Create a deterministic, augmentation-free dataset view for pseudo-label precomputation."""
    if not isinstance(dataset, BMADDataset):
        return dataset

    normalize_mode = getattr(dataset.preprocessor, 'normalize_mode', 'zscore_only')
    return BMADDataset(
        image_paths=list(dataset.image_paths),
        labels=list(dataset.labels) if dataset.labels is not None else None,
        mask_paths=list(dataset.mask_paths) if dataset.mask_paths is not None else None,
        preprocessor=dataset.preprocessor,
        augment=False,
        is_training=False,
        normalize_mode=normalize_mode,
        augment_mode='none',
        use_timm_transforms=getattr(dataset, 'use_timm_transforms', False),
        timm_data_config=getattr(dataset, 'timm_data_config', None)
    )


def _compute_capacitated_assignments(distances: torch.Tensor, max_per_anchor: int) -> torch.Tensor:
    """Assign each sample to an anchor while respecting a hard maximum occupancy."""
    if max_per_anchor <= 0:
        raise ValueError("max_per_anchor must be positive for capacitated assignment.")

    num_samples, num_anchors = distances.shape
    total_capacity = max_per_anchor * num_anchors
    if total_capacity < num_samples:
        raise ValueError(
            f"Capacity infeasible: {num_anchors} anchors * {max_per_anchor} < {num_samples} samples."
        )

    ranked_anchors = torch.argsort(distances, dim=1).cpu().numpy()
    if num_anchors > 1:
        top2 = torch.topk(-distances, k=2, dim=1).values.neg().cpu().numpy()
        priority = top2[:, 1] - top2[:, 0]
    else:
        priority = np.full(num_samples, np.inf, dtype=np.float32)

    sample_order = np.argsort(-priority, kind='mergesort')
    remaining_capacity = np.full(num_anchors, max_per_anchor, dtype=np.int64)
    assignments = np.full(num_samples, -1, dtype=np.int64)

    for sample_idx in sample_order.tolist():
        for anchor_idx in ranked_anchors[sample_idx].tolist():
            if remaining_capacity[anchor_idx] > 0:
                assignments[sample_idx] = anchor_idx
                remaining_capacity[anchor_idx] -= 1
                break

        if assignments[sample_idx] < 0:
            fallback_anchor = int(np.argmax(remaining_capacity))
            if remaining_capacity[fallback_anchor] <= 0:
                raise RuntimeError("No remaining anchor capacity while computing fixed pseudo-labels.")
            assignments[sample_idx] = fallback_anchor
            remaining_capacity[fallback_anchor] -= 1

    return torch.from_numpy(assignments).long()


class Trainer:
    """Training manager for anomaly detector"""
    
    def __init__(
        self,
        model: AnomalyDetector,
        criterion: Optional[CombinedAnchorLoss],
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: Path,
        use_amp: bool = True,
        log_interval: int = 50,
        val_interval: int = 1,
        fixed_pseudo_labels: bool = False,
        pseudo_label_assignment: str = 'nearest',
        capacity_multiplier: float = 2.0,
        dynamic_reassignment: bool = False,
        reassignment_interval: int = 5,
        save_checkpoints: bool = True,
        stage2_mode: bool = False,
        stage2_config: Optional[Dict] = None
    ):
        """
        Args:
            model: AnomalyDetector model
            criterion: Loss function
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            save_dir: Directory to save checkpoints
            use_amp: Use mixed precision training
            log_interval: Logging frequency (batches)
            val_interval: Validation frequency (epochs)
            fixed_pseudo_labels: If True, compute pseudo-labels once before training
            pseudo_label_assignment: Strategy for initial fixed pseudo-labeling ('nearest' or 'capacitated')
            capacity_multiplier: Hard-cap multiplier for capacitated pseudo-labeling
            dynamic_reassignment: If True, recompute pseudo-labels every reassignment_interval epochs
            reassignment_interval: Epochs between pseudo-label recomputation (only if dynamic_reassignment=True)
            save_checkpoints: If True, save periodic checkpoints; if False, only save best and final
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda') if use_amp else None
        
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_checkpoints = save_checkpoints
        self.stage2_mode = stage2_mode
        self.stage2_config = stage2_config or {}
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_auroc = 0.0

        # Fixed set for TSNE tracking
        self.tsne_samples = self._prepare_tsne_samples(normal_count=500, anomaly_count=100)
        self.fixed_assignments = None
        self.fixed_pseudo = fixed_pseudo_labels
        self.pseudo_label_assignment = pseudo_label_assignment
        self.capacity_multiplier = float(capacity_multiplier)
        
        # Dynamic label reassignment for learnable anchors
        self.dynamic_reassignment = dynamic_reassignment
        self.reassignment_interval = reassignment_interval

        if self.fixed_pseudo and self.dynamic_reassignment and self.pseudo_label_assignment != 'nearest':
            raise ValueError(
                "dynamic_reassignment is only supported with training.pseudo_label_assignment='nearest'."
            )
        
        # Enhanced history tracking
        self.history = {
            # Training metrics
            'train_loss': [],
            'train_loss_attract': [],
            'train_loss_repel': [],
            'train_loss_dense': [],
            'train_loss_dense_attract': [],
            
            # Validation metrics
            'val_loss': [],
            'val_loss_attract': [],
            'val_loss_repel': [],
            'val_loss_norm': [],
            'val_loss_dense': [],
            'val_image_auroc': [],
            'val_pixel_auroc': [],
            'val_cluster_all_effective_anchors_used': [],
            'val_cluster_all_assignment_entropy': [],
            'val_cluster_all_assignment_entropy_normalized': [],
            'val_cluster_all_largest_anchor_share': [],
            'val_cluster_all_max_min_nonzero_ratio': [],
            'val_cluster_all_assignment_counts': [],
            'val_cluster_normal_effective_anchors_used': [],
            'val_cluster_normal_assignment_entropy': [],
            'val_cluster_normal_assignment_entropy_normalized': [],
            'val_cluster_normal_largest_anchor_share': [],
            'val_cluster_normal_max_min_nonzero_ratio': [],
            'val_cluster_normal_assignment_counts': [],
            'val_cluster_normal_mean_nearest_distance': [],
            'val_cluster_normal_mean_second_nearest_distance': [],
            'val_cluster_normal_mean_margin_d2_minus_d1': [],
            'val_cluster_normal_mean_ratio_d1_over_d2': [],
            'val_cluster_anomaly_assignment_counts': [],
            
            # Per-epoch statistics
            'epochs': [],
            'learning_rates': []
        }

        if self.fixed_pseudo:
            self._precompute_pseudo_labels()

        # Precompute pseudo-labels if enabled in config
        self.fixed_pseudo = getattr(self, 'fixed_pseudo', False)

        # Pixel-stats tracker for stage-2 reconstruction threshold
        self.pixel_stats_tracker = None

    def _compute_stage2_losses(self, outputs: Dict[str, torch.Tensor], images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reconstruction + alignment losses for stage-2 training.

        Loss components:
          1. **Reconstruction loss** (MSE or L1) – faithfully reconstruct normal data.
          2. **Alignment loss** – keep the trainable stage-2 bottleneck close to the
             frozen stage-1 bottleneck on *normal* training data.  This replaces the
             old consistency-to-anchor loss with a more informative regulariser.
             If no frozen bottleneck is available, falls back to the legacy
             consistency loss (cosine / L2 to assigned anchor embedding).
        """
        if 'reconstruction' not in outputs:
            raise RuntimeError("Stage-2 training requires model outputs to include 'reconstruction'.")

        recon_weight = float(self.stage2_config.get('recon_weight', 1.0))
        alignment_weight = float(self.stage2_config.get('alignment_weight',
                                    self.stage2_config.get('consistency_weight', 0.1)))
        recon_type = self.stage2_config.get('recon_loss', 'mse')

        reconstruction = outputs['reconstruction']
        recon_target = images

        if recon_type == 'l1':
            recon_loss = F.l1_loss(reconstruction, recon_target)
        else:
            recon_loss = F.mse_loss(reconstruction, recon_target)

        # ----- alignment / consistency loss -----
        stage2_feat = outputs['stage2_feat']
        frozen_feat = outputs.get('frozen_feat')
        alignment_target = self.stage2_config.get('alignment_target', 'sample')

        if alignment_target == 'local_anchor_pool':
            anchor_target = outputs.get('stage2_guidance')
            if anchor_target is None:
                raise RuntimeError("Stage-2 local_anchor_pool alignment requires 'stage2_guidance' in model outputs.")
            if self.stage2_config.get('consistency_loss', 'cosine') == 'l2':
                alignment_loss = F.mse_loss(stage2_feat, anchor_target.detach())
            else:
                alignment_loss = 1.0 - torch.cosine_similarity(stage2_feat, anchor_target.detach(), dim=1).mean()
        elif alignment_target == 'anchor':
            fixed_assignments = outputs.get('fixed_assignments')
            if fixed_assignments is None:
                raise RuntimeError("Stage-2 anchor alignment requires fixed pseudo-label assignments.")
            if getattr(self.model, '_recon_dim', None) != getattr(self.model, '_anchor_dim', None):
                raise NotImplementedError(
                    "stage2.alignment_target='anchor' currently supports only recon_dim == anchor_dim."
                )

            projected_anchor_targets, _ = self.model._get_projected_anchors()
            anchor_target = projected_anchor_targets[fixed_assignments].detach()
            if self.stage2_config.get('consistency_loss', 'cosine') == 'l2':
                alignment_loss = F.mse_loss(stage2_feat, anchor_target)
            else:
                alignment_loss = 1.0 - torch.cosine_similarity(stage2_feat, anchor_target, dim=1).mean()
        elif frozen_feat is not None:
            # Dual-bottleneck alignment: penalise divergence from frozen projection
            alignment_loss = 1.0 - torch.cosine_similarity(stage2_feat, frozen_feat.detach(), dim=1).mean()
        else:
            # Legacy fallback: consistency to assigned anchor embedding
            consistency_type = self.stage2_config.get('consistency_loss', 'cosine')
            anchor_target = outputs['assigned_anchor_embeddings']
            if consistency_type == 'l2':
                alignment_loss = F.mse_loss(stage2_feat, anchor_target)
            else:
                alignment_loss = 1.0 - torch.cosine_similarity(stage2_feat, anchor_target, dim=1).mean()

        total_loss = recon_weight * recon_loss + alignment_weight * alignment_loss

        return {
            'loss': total_loss,
            'loss_recon': recon_loss,
            'loss_consistency': alignment_loss,   # keep key name for history compat
            'loss_alignment': alignment_loss,
        }

    def _prepare_tsne_samples(self, normal_count: int = 500, anomaly_count: int = 100):
        """Collect a fixed pool of samples for visualization (CPU tensors)."""
        # No longer used - we visualize ALL training samples instead
        return None

    def _visualize_training_samples(
        self,
        epoch: int,
        max_samples: int = 2000,
        max_lines_per_anchor: int = 150,
        save_name: Optional[str] = None
    ):
        """
        Visualize ALL training samples with lines to their assigned anchors.
        Uses both t-SNE and PCA projections.
        
        Args:
            epoch: Current epoch number
            max_samples: Maximum samples to visualize (for memory/speed)
            max_lines_per_anchor: Maximum lines to draw per anchor (for visibility)
        """
        print(f"\n  Generating training visualization (epoch {epoch})...")
        
        self.model.eval()
        save_dir = self.save_dir / 'visualizations'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Collect all training embeddings and assignments
        all_embeddings = []
        all_assignments = []
        
        # Use non-shuffled loader
        eval_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        
        with torch.no_grad():
            anchor_global, _ = self.model._get_projected_anchors()
            
            for batch in tqdm(eval_loader, desc='    Collecting embeddings', leave=False):
                images = batch['image'].to(self.device)
                outputs = self.model(images, return_dense=False)
                embeddings = outputs['global_feat']
                distances = outputs['global_distances']
                assignments = distances.argmin(dim=1)
                
                all_embeddings.append(embeddings.cpu())
                all_assignments.append(assignments.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_assignments = torch.cat(all_assignments, dim=0)
        anchors = anchor_global.cpu()
        
        n_samples = all_embeddings.shape[0]
        n_anchors = anchors.shape[0]
        
        # Subsample if too many
        if n_samples > max_samples:
            indices = torch.randperm(n_samples)[:max_samples]
            all_embeddings = all_embeddings[indices]
            all_assignments = all_assignments[indices]
            n_samples = max_samples
        
        emb_np = all_embeddings.numpy()
        anc_np = anchors.numpy()
        assign_np = all_assignments.numpy()
        
        # Combine for dimensionality reduction
        all_points = np.vstack([emb_np, anc_np])
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_anchors, 10)))

        # Adaptive visualization for large number of anchors
        large_anchor_mode = n_anchors > 64
        top_anchor_k = min(24, n_anchors)
        anchor_counts = np.bincount(assign_np, minlength=n_anchors)
        top_anchor_ids = np.argsort(anchor_counts)[::-1][:top_anchor_k]
        top_anchor_set = set(top_anchor_ids.tolist())
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # === Plot 1: t-SNE ===
        ax = axes[0]
        perplexity = min(30, len(all_points) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        coords_2d = tsne.fit_transform(all_points)
        
        sample_coords = coords_2d[:n_samples]
        anchor_coords = coords_2d[n_samples:]
        
        if large_anchor_mode:
            # Samples in neutral color to reduce clutter
            ax.scatter(sample_coords[:, 0], sample_coords[:, 1], c='steelblue', s=10, alpha=0.35, label=f'Samples (N={n_samples})', zorder=2)

            # Plot all anchors as small gray points
            ax.scatter(anchor_coords[:, 0], anchor_coords[:, 1], c='black', s=10, alpha=0.35, marker='o', label=f'All anchors (K={n_anchors})', zorder=3)

            # Highlight top-used anchors only
            for rank, k in enumerate(top_anchor_ids):
                color = colors[rank % len(colors)]
                ax.scatter(anchor_coords[k, 0], anchor_coords[k, 1], c=[color], s=220, marker='*', edgecolors='black', linewidths=1.0, zorder=8)
                if rank < 12:
                    ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]), fontsize=9, fontweight='bold', ha='center', va='center', zorder=9)

            # Draw assignment lines only for top-used anchors (sparsified)
            for rank, k in enumerate(top_anchor_ids):
                color = colors[rank % len(colors)]
                mask = assign_np == k
                indices = np.where(mask)[0]
                if len(indices) > max_lines_per_anchor:
                    indices = np.random.choice(indices, max_lines_per_anchor, replace=False)
                for idx in indices:
                    ax.plot(
                        [sample_coords[idx, 0], anchor_coords[k, 0]],
                        [sample_coords[idx, 1], anchor_coords[k, 1]],
                        c=color, alpha=0.10, linewidth=0.4, zorder=1
                    )
        else:
            # Draw lines from samples to their assigned anchors
            for k in range(n_anchors):
                mask = assign_np == k
                indices = np.where(mask)[0]
                if len(indices) > max_lines_per_anchor:
                    indices = np.random.choice(indices, max_lines_per_anchor, replace=False)
                for idx in indices:
                    ax.plot(
                        [sample_coords[idx, 0], anchor_coords[k, 0]],
                        [sample_coords[idx, 1], anchor_coords[k, 1]],
                        c=colors[k % len(colors)], alpha=0.15, linewidth=0.5, zorder=1
                    )

            # Plot samples colored by assignment
            for k in range(n_anchors):
                mask = assign_np == k
                count = mask.sum()
                if count > 0:
                    ax.scatter(sample_coords[mask, 0], sample_coords[mask, 1], c=[colors[k % len(colors)]], s=15, alpha=0.6, label=f'Anchor {k} (n={count})', zorder=2)

            # Plot anchors (big stars)
            for k in range(n_anchors):
                ax.scatter(anchor_coords[k, 0], anchor_coords[k, 1], c=[colors[k % len(colors)]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
                ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]), fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
        ax.set_title(f't-SNE (N={n_samples})', fontsize=13, fontweight='bold')
        if large_anchor_mode:
            ax.legend(fontsize=9, loc='upper right')
        else:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # === Plot 2: PCA ===
        ax = axes[1]
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(all_points)
        
        sample_pca = coords_pca[:n_samples]
        anchor_pca = coords_pca[n_samples:]
        
        if large_anchor_mode:
            ax.scatter(sample_pca[:, 0], sample_pca[:, 1], c='steelblue', s=10, alpha=0.35, label=f'Samples (N={n_samples})', zorder=2)
            ax.scatter(anchor_pca[:, 0], anchor_pca[:, 1], c='black', s=10, alpha=0.35, marker='o', label=f'All anchors (K={n_anchors})', zorder=3)

            for rank, k in enumerate(top_anchor_ids):
                color = colors[rank % len(colors)]
                ax.scatter(anchor_pca[k, 0], anchor_pca[k, 1], c=[color], s=220, marker='*', edgecolors='black', linewidths=1.0, zorder=8)
                if rank < 12:
                    ax.annotate(f'A{k}', (anchor_pca[k, 0], anchor_pca[k, 1]), fontsize=9, fontweight='bold', ha='center', va='center', zorder=9)

            for rank, k in enumerate(top_anchor_ids):
                color = colors[rank % len(colors)]
                mask = assign_np == k
                indices = np.where(mask)[0]
                if len(indices) > max_lines_per_anchor:
                    indices = np.random.choice(indices, max_lines_per_anchor, replace=False)
                for idx in indices:
                    ax.plot(
                        [sample_pca[idx, 0], anchor_pca[k, 0]],
                        [sample_pca[idx, 1], anchor_pca[k, 1]],
                        c=color, alpha=0.10, linewidth=0.4, zorder=1
                    )
        else:
            # Draw lines
            for k in range(n_anchors):
                mask = assign_np == k
                indices = np.where(mask)[0]
                if len(indices) > max_lines_per_anchor:
                    indices = np.random.choice(indices, max_lines_per_anchor, replace=False)
                for idx in indices:
                    ax.plot(
                        [sample_pca[idx, 0], anchor_pca[k, 0]],
                        [sample_pca[idx, 1], anchor_pca[k, 1]],
                        c=colors[k % len(colors)], alpha=0.15, linewidth=0.5, zorder=1
                    )

            # Plot samples
            for k in range(n_anchors):
                mask = assign_np == k
                count = mask.sum()
                if count > 0:
                    ax.scatter(sample_pca[mask, 0], sample_pca[mask, 1], c=[colors[k % len(colors)]], s=15, alpha=0.6, label=f'Anchor {k} (n={count})', zorder=2)

            # Plot anchors
            for k in range(n_anchors):
                ax.scatter(anchor_pca[k, 0], anchor_pca[k, 1], c=[colors[k % len(colors)]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
                ax.annotate(f'A{k}', (anchor_pca[k, 0], anchor_pca[k, 1]), fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
        # Mark origin
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        ax.scatter([0], [0], c='red', s=100, marker='x', zorder=5, label='Origin')
        
        ax.set_title(f'PCA (N={n_samples})', fontsize=13, fontweight='bold')
        if large_anchor_mode:
            ax.legend(fontsize=9, loc='upper right')
        else:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        title_suffix = save_name if save_name is not None else f'Epoch {epoch}'
        plt.suptitle(f'Training Samples - {title_suffix}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_name = save_name if save_name is not None else f'train_epoch_{epoch:03d}'
        plt.savefig(save_dir / f'{output_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {save_dir / f'{output_name}.png'}")

    def _visualize_test_samples(self, test_loader: DataLoader, save_name: str = 'test_final'):
        """
        Visualize test samples (normal and anomaly) with anchors.
        Shows how normal vs anomaly samples relate to the learned anchors.
        
        Args:
            test_loader: DataLoader with test samples (has both normal and anomaly)
            save_name: Name for the saved file
        """
        print(f"\n  Generating test visualization...")
        
        self.model.eval()
        save_dir = self.save_dir / 'visualizations'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Collect all test embeddings
        all_embeddings = []
        all_labels = []  # 0=normal, 1=anomaly
        all_distances = []  # Distance to nearest anchor
        all_assignments = []
        
        with torch.no_grad():
            anchor_global, _ = self.model._get_projected_anchors()
            
            for batch in tqdm(test_loader, desc='    Collecting test embeddings', leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                outputs = self.model(images, return_dense=False)
                embeddings = outputs['global_feat']
                distances = outputs['global_distances']
                min_dist = distances.min(dim=1)[0]  # Distance to nearest anchor
                assigned = distances.argmin(dim=1)
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                all_distances.append(min_dist.cpu())
                all_assignments.append(assigned.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_distances = torch.cat(all_distances, dim=0)
        all_assignments = torch.cat(all_assignments, dim=0)
        anchors = anchor_global.cpu()
        
        n_samples = all_embeddings.shape[0]
        n_anchors = anchors.shape[0]
        n_normal = (all_labels == 0).sum().item()
        n_anomaly = (all_labels == 1).sum().item()
        
        emb_np = all_embeddings.numpy()
        anc_np = anchors.numpy()
        labels_np = all_labels.numpy()
        distances_np = all_distances.numpy()
        assignments_np = all_assignments.numpy()

        large_anchor_mode = n_anchors > 64
        top_anchor_k = min(24, n_anchors)
        anchor_counts = np.bincount(assignments_np, minlength=n_anchors)
        top_anchor_ids = np.argsort(anchor_counts)[::-1][:top_anchor_k]
        
        # Combine for dimensionality reduction
        all_points = np.vstack([emb_np, anc_np])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Colors for normal/anomaly
        normal_color = 'steelblue'
        anomaly_color = 'crimson'
        anchor_colors = plt.cm.tab10(np.linspace(0, 1, min(n_anchors, 10)))
        
        # === Plot 1: t-SNE ===
        ax = axes[0]
        perplexity = min(30, len(all_points) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        coords_2d = tsne.fit_transform(all_points)
        
        sample_coords = coords_2d[:n_samples]
        anchor_coords = coords_2d[n_samples:]
        
        # Plot normal samples
        normal_mask = labels_np == 0
        ax.scatter(sample_coords[normal_mask, 0], sample_coords[normal_mask, 1], 
                  c=normal_color, s=15, alpha=0.5, label=f'Normal (n={n_normal})', zorder=2)
        
        # Plot anomaly samples
        anomaly_mask = labels_np == 1
        ax.scatter(sample_coords[anomaly_mask, 0], sample_coords[anomaly_mask, 1], 
                  c=anomaly_color, s=15, alpha=0.5, label=f'Anomaly (n={n_anomaly})', zorder=3)
        
        if large_anchor_mode:
            ax.scatter(anchor_coords[:, 0], anchor_coords[:, 1], c='black', s=10, alpha=0.35, marker='o', label=f'All anchors (K={n_anchors})', zorder=4)
            for rank, k in enumerate(top_anchor_ids):
                color = anchor_colors[rank % len(anchor_colors)]
                ax.scatter(anchor_coords[k, 0], anchor_coords[k, 1], c=[color], s=220, marker='*', edgecolors='black', linewidths=1.0, zorder=10)
                if rank < 12:
                    ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]), fontsize=9, fontweight='bold', ha='center', va='center', zorder=11)
        else:
            # Plot anchors (big stars)
            for k in range(n_anchors):
                ax.scatter(anchor_coords[k, 0], anchor_coords[k, 1], c=[anchor_colors[k % len(anchor_colors)]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
                ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]), fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
        ax.set_title(f't-SNE (N={n_samples})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # === Plot 2: PCA ===
        ax = axes[1]
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(all_points)
        
        sample_pca = coords_pca[:n_samples]
        anchor_pca = coords_pca[n_samples:]
        
        # Plot normal samples
        ax.scatter(sample_pca[normal_mask, 0], sample_pca[normal_mask, 1], 
                  c=normal_color, s=15, alpha=0.5, label=f'Normal (n={n_normal})', zorder=2)
        
        # Plot anomaly samples
        ax.scatter(sample_pca[anomaly_mask, 0], sample_pca[anomaly_mask, 1], 
                  c=anomaly_color, s=15, alpha=0.5, label=f'Anomaly (n={n_anomaly})', zorder=3)
        
        if large_anchor_mode:
            ax.scatter(anchor_pca[:, 0], anchor_pca[:, 1], c='black', s=10, alpha=0.35, marker='o', label=f'All anchors (K={n_anchors})', zorder=4)
            for rank, k in enumerate(top_anchor_ids):
                color = anchor_colors[rank % len(anchor_colors)]
                ax.scatter(anchor_pca[k, 0], anchor_pca[k, 1], c=[color], s=220, marker='*', edgecolors='black', linewidths=1.0, zorder=10)
                if rank < 12:
                    ax.annotate(f'A{k}', (anchor_pca[k, 0], anchor_pca[k, 1]), fontsize=9, fontweight='bold', ha='center', va='center', zorder=11)
        else:
            # Plot anchors
            for k in range(n_anchors):
                ax.scatter(anchor_pca[k, 0], anchor_pca[k, 1], c=[anchor_colors[k % len(anchor_colors)]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
                ax.annotate(f'A{k}', (anchor_pca[k, 0], anchor_pca[k, 1]), fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
        # Mark origin
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_title(f'PCA (N={n_samples})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Compute and display statistics
        normal_dist_mean = distances_np[normal_mask].mean()
        anomaly_dist_mean = distances_np[anomaly_mask].mean()
        
        fig.text(0.5, 0.02, 
                f'Mean distance to nearest anchor - Normal: {normal_dist_mean:.4f}, Anomaly: {anomaly_dist_mean:.4f}',
                ha='center', fontsize=11, style='italic')
        
        plt.suptitle(f'Test Samples: Normal vs Anomaly', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(save_dir / f'{save_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {save_dir / f'{save_name}.png'}")
        print(f"    Normal samples mean dist: {normal_dist_mean:.4f}")
        print(f"    Anomaly samples mean dist: {anomaly_dist_mean:.4f}")
        
        # Also create a histogram of distances
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(distances_np[normal_mask], bins=50, alpha=0.7, color=normal_color, 
                label=f'Normal (n={n_normal}, μ={normal_dist_mean:.3f})')
        ax.hist(distances_np[anomaly_mask], bins=50, alpha=0.7, color=anomaly_color,
                label=f'Anomaly (n={n_anomaly}, μ={anomaly_dist_mean:.3f})')
        ax.set_xlabel('Distance to Nearest Anchor', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Distances to Nearest Anchor', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'{save_name}_histogram.png', dpi=150)
        plt.close()
        
        print(f"    Saved: {save_dir / f'{save_name}_histogram.png'}")

    def _save_tsne(self, epoch: int, final: bool = False):
        """Legacy method - redirects to new visualization."""
        # Call the new comprehensive visualization
        self._visualize_training_samples(epoch=epoch)

    def _precompute_pseudo_labels(self):
        """Compute fixed nearest-anchor assignments once before training."""
        is_patch_mode = _get_model_anchor_mode(self.model) == 'patch'
        print("\n" + "="*80)
        if is_patch_mode:
            print("COMPUTING FIXED PATCH PSEUDO-LABELS")
            print("="*80)
            print("Using patch-to-anchor assignment distances on unaugmented images")
            print("Ensures fixed labels follow patch-anchor scoring semantics")
        else:
            print("COMPUTING PSEUDO-LABELS IN 384D DINOV3 SPACE (SOLUTION A)")
            print("="*80)
            print("Using RAW 384D DINOv3 embeddings (frozen semantic features)")
            print("Ensures pseudo-labels based on SEMANTIC similarity, not random projection")
        print("="*80)
        
        self.model.eval()
        mapping = {}
        
        # Use a non-dropping, non-shuffling loader to cover all samples
        preload_dataset = _build_non_augmented_dataset(self.train_loader.dataset)
        preload = DataLoader(
            preload_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory,
            drop_last=False
        )

        all_paths = []
        all_distances = []
        with torch.no_grad():
            progress_desc = 'Computing patch pseudo-labels' if is_patch_mode else 'Computing pseudo-labels in 384D space'
            for batch in tqdm(preload, desc=progress_desc):
                images = batch['image'].to(self.device)
                paths = batch['path']

                if is_patch_mode and hasattr(self.model, 'compute_label_distances'):
                    batch_distances = self.model.compute_label_distances(images).cpu()
                else:
                    anchor_embeddings_384d = self.model.get_semantic_anchors().to(self.device)  # (K, 384)
                    features_384d = self.model.backbone.backbone.forward_features(images)
                    embeddings_384d = features_384d[:, 0]
                    batch_distances = torch.cdist(embeddings_384d, anchor_embeddings_384d).cpu()

                all_distances.append(batch_distances)
                all_paths.extend([str(path) for path in paths])

        distances = torch.cat(all_distances, dim=0)
        print(f"\nAssignment distance matrix: {tuple(distances.shape)}")

        if self.pseudo_label_assignment == 'capacitated':
            n_samples = int(distances.shape[0])
            n_anchors = int(distances.shape[1])
            max_per_anchor = int(np.ceil(self.capacity_multiplier * n_samples / max(n_anchors, 1)))
            print(
                f"Applying capacity-constrained pseudo-labeling: "
                f"max_per_anchor={max_per_anchor} (multiplier={self.capacity_multiplier:.3f})"
            )
            assigned_tensor = _compute_capacitated_assignments(distances, max_per_anchor=max_per_anchor)
            min_distances = distances[torch.arange(n_samples), assigned_tensor]
        else:
            min_distances, assigned_tensor = distances.min(dim=1)

        for path, assignment in zip(all_paths, assigned_tensor.tolist()):
            mapping[path] = int(assignment)

        self.fixed_assignments = mapping
        all_min_distances = min_distances
        
        # Statistics
        label_list = list(mapping.values())
        unique_labels = set(label_list)
        counts = {label: label_list.count(label) for label in unique_labels}
        
        print(f"\n{'='*80}")
        print("PSEUDO-LABEL STATISTICS (384D SPACE)")
        print(f"{'='*80}")
        print(f"Total samples: {len(mapping)}")
        print(f"Anchors used: {len(unique_labels)} / {distances.shape[1]}")
        print(f"Assignment mode: {self.pseudo_label_assignment}")
        print(f"\nDistribution:")
        for label in sorted(unique_labels):
            count = counts[label]
            percentage = 100.0 * count / len(mapping)
            print(f"  Anchor {label}: {count:5d} samples ({percentage:5.2f}%)")
        
        print(f"\nDistance statistics:")
        print(f"  Mean: {all_min_distances.mean():.4f}")
        print(f"  Std:  {all_min_distances.std():.4f}")
        print(f"  Min:  {all_min_distances.min():.4f}")
        print(f"  Max:  {all_min_distances.max():.4f}")
        
        # Warning if imbalanced
        max_count = max(counts.values())
        min_count = min(counts.values())
        if self.pseudo_label_assignment == 'capacitated':
            n_samples = len(mapping)
            max_allowed = int(np.ceil(self.capacity_multiplier * n_samples / max(distances.shape[1], 1)))
            print(f"  Capacity cap: {max_allowed}")
            if max_count > max_allowed:
                raise RuntimeError(f"Capacitated pseudo-labeling violated max_per_anchor={max_allowed}.")

        if max_count > 3 * min_count:
            print(f"\n⚠️  Warning: Imbalanced distribution (max={max_count}, min={min_count})")
            print("   Consider using diversity loss if this persists during training.")
        else:
            print(f"\n✓ Distribution looks balanced (max={max_count}, min={min_count})")
        
        print(f"{'='*80}\n")

    def _resolve_batch_assignments(self, batch_paths, batch_images: torch.Tensor) -> torch.Tensor:
        """Resolve anchor assignments for a batch.

        Training samples reuse the fixed precomputed mapping. Any unseen paths
        (for example validation batches during stage-2 anchor alignment) fall
        back to nearest semantic-anchor assignment in frozen 384D space.
        """
        if not self.fixed_pseudo or self.fixed_assignments is None:
            raise RuntimeError("Fixed pseudo-label assignments were requested but are not available.")

        resolved = [self.fixed_assignments.get(str(path)) for path in batch_paths]
        missing_indices = [index for index, assignment in enumerate(resolved) if assignment is None]

        if missing_indices:
            with torch.no_grad():
                if _get_model_anchor_mode(self.model) == 'patch' and hasattr(self.model, 'compute_label_distances'):
                    batch_distances = self.model.compute_label_distances(batch_images)
                    missing_distances = batch_distances[missing_indices]
                else:
                    anchor_embeddings_384d = self.model.get_semantic_anchors().to(self.device)
                    batch_embeddings_384d = self.model.backbone.backbone.forward_features(batch_images)[:, 0]
                    missing_embeddings = batch_embeddings_384d[missing_indices]
                    missing_distances = torch.cdist(missing_embeddings, anchor_embeddings_384d)
                missing_assignments = missing_distances.argmin(dim=1).cpu().tolist()

            for batch_index, assignment in zip(missing_indices, missing_assignments):
                resolved[batch_index] = int(assignment)

        return torch.tensor(resolved, device=self.device, dtype=torch.long)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'loss_attract': 0.0,
            'loss_repel': 0.0,
            'loss_norm': 0.0,
            'loss_diversity': 0.0,
            'loss_dense': 0.0,
            'loss_dense_attract': 0.0
        }
        
        anchor_assignments = np.zeros(self.model.n_anchors)
        
        # Check if model has pixel decoder for dense/pixel loss
        use_pixel_decoder = getattr(self.model, 'use_pixel_decoder', False)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            fixed_assign = None
            if self.fixed_pseudo:
                fixed_assign = self._resolve_batch_assignments(batch['path'], images)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get fresh anchor embeddings for this batch
            # This must be inside the loop to avoid reusing computation graph
            anchor_global, _ = self.model._get_projected_anchors()
            
            if self.use_amp:
                with autocast('cuda'):
                    # Enable dense features if pixel decoder is available
                    outputs = self.model(images, return_dense=use_pixel_decoder)
                    if fixed_assign is not None:
                        outputs['fixed_assignments'] = fixed_assign
                    loss_dict = self.criterion(outputs, anchor_global)
                    loss = loss_dict['loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, return_dense=use_pixel_decoder)
                if fixed_assign is not None:
                    outputs['fixed_assignments'] = fixed_assign
                loss_dict = self.criterion(outputs, anchor_global)
                loss = loss_dict['loss']
                
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            epoch_metrics['loss'] += loss.item()
            
            # Handle different loss types (CAM vs Contrastive)
            # CAM loss: loss_global_attract, loss_global_repel, loss_global_norm, loss_global_diversity
            # Contrastive: loss_global_loss_center, loss_global_loss_infonce, loss_global_loss_repel
            if 'loss_global_attract' in loss_dict:
                # CAM loss
                epoch_metrics['loss_attract'] += loss_dict['loss_global_attract']
                epoch_metrics['loss_repel'] += loss_dict['loss_global_repel']
                epoch_metrics['loss_norm'] += loss_dict.get('loss_global_norm', 0.0)
                epoch_metrics['loss_diversity'] += loss_dict.get('loss_global_diversity', 0.0)
            else:
                # Contrastive loss - aggregate all components
                epoch_metrics['loss_attract'] += loss_dict.get('loss_global_loss_center', 0.0)
                epoch_metrics['loss_attract'] += loss_dict.get('loss_global_loss_infonce', 0.0)
                epoch_metrics['loss_repel'] += loss_dict.get('loss_global_loss_repel', 0.0)
                epoch_metrics['loss_norm'] += 0.0  # No norm loss in contrastive
                epoch_metrics['loss_diversity'] += 0.0  # No diversity loss in contrastive
            
            # Dense/pixel loss tracking
            if 'loss_dense' in loss_dict:
                epoch_metrics['loss_dense'] += loss_dict['loss_dense'].item() if isinstance(loss_dict['loss_dense'], torch.Tensor) else loss_dict['loss_dense']
            if 'loss_dense_attract' in loss_dict:
                epoch_metrics['loss_dense_attract'] += loss_dict['loss_dense_attract']
            
            # Track anchor assignments
            assigned = loss_dict['assigned_anchors'].cpu().numpy()
            for a in assigned:
                anchor_assignments[a] += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                }
                
                # Add loss components based on loss type
                if 'loss_global_attract' in loss_dict:
                    # CAM loss
                    postfix_dict['attr'] = f"{loss_dict['loss_global_attract']:.4f}"
                    postfix_dict['rep'] = f"{loss_dict['loss_global_repel']:.4f}"
                    if loss_dict.get('loss_global_norm', 0.0) > 0:
                        postfix_dict['norm'] = f"{loss_dict['loss_global_norm']:.4f}"
                else:
                    # Contrastive loss
                    if 'loss_global_loss_center' in loss_dict:
                        postfix_dict['ctr'] = f"{loss_dict['loss_global_loss_center']:.4f}"
                    if 'loss_global_loss_infonce' in loss_dict:
                        postfix_dict['inf'] = f"{loss_dict['loss_global_loss_infonce']:.4f}"
                    if 'loss_global_loss_repel' in loss_dict:
                        postfix_dict['rep'] = f"{loss_dict['loss_global_loss_repel']:.4f}"
                
                if 'loss_dense' in loss_dict and loss_dict['loss_dense'] > 0:
                    loss_dense_val = loss_dict['loss_dense'].item() if isinstance(loss_dict['loss_dense'], torch.Tensor) else loss_dict['loss_dense']
                    postfix_dict['pix'] = f"{loss_dense_val:.4f}"
                pbar.set_postfix(postfix_dict)
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Normalize anchor assignments
        anchor_assignments = anchor_assignments / anchor_assignments.sum()
        epoch_metrics['anchor_balance'] = anchor_assignments.tolist()
        
        return epoch_metrics
    
    def _summarize_assignment_subset(
        self,
        distances: np.ndarray,
        assigned: np.ndarray,
        subset_mask: np.ndarray,
    ) -> Dict[str, object]:
        subset_distances = distances[subset_mask]
        subset_assigned = assigned[subset_mask]
        n_anchors = int(self.model.n_anchors)

        counts = np.bincount(subset_assigned, minlength=n_anchors).astype(np.int64) if len(subset_assigned) > 0 else np.zeros(n_anchors, dtype=np.int64)
        nonzero = counts[counts > 0]
        total = int(counts.sum())

        if total > 0:
            probs = counts[counts > 0].astype(np.float64) / total
            entropy = float(-(probs * np.log(probs)).sum())
            max_entropy = np.log(n_anchors) if n_anchors > 1 else 1.0
            normalized_entropy = float(entropy / max(max_entropy, 1e-12))
            largest_anchor_share = float(counts.max() / total)
        else:
            entropy = 0.0
            normalized_entropy = 0.0
            largest_anchor_share = 0.0

        summary: Dict[str, object] = {
            'effective_anchors_used': int((counts > 0).sum()),
            'assignment_entropy': entropy,
            'assignment_entropy_normalized': normalized_entropy,
            'largest_anchor_share': largest_anchor_share,
            'max_min_nonzero_ratio': float(nonzero.max() / max(nonzero.min(), 1)) if len(nonzero) > 0 else 0.0,
            'assignment_counts': counts.tolist(),
        }

        if len(subset_distances) == 0:
            summary.update({
                'mean_nearest_distance': None,
                'mean_second_nearest_distance': None,
                'mean_margin_d2_minus_d1': None,
                'mean_ratio_d1_over_d2': None,
            })
            return summary

        sorted_distances = np.sort(subset_distances, axis=1)
        nearest = sorted_distances[:, 0]
        summary['mean_nearest_distance'] = float(nearest.mean())

        if subset_distances.shape[1] > 1:
            second = sorted_distances[:, 1]
            margin = second - nearest
            ratio = nearest / np.maximum(second, 1e-12)
            summary['mean_second_nearest_distance'] = float(second.mean())
            summary['mean_margin_d2_minus_d1'] = float(margin.mean())
            summary['mean_ratio_d1_over_d2'] = float(ratio.mean())
        else:
            summary['mean_second_nearest_distance'] = None
            summary['mean_margin_d2_minus_d1'] = None
            summary['mean_ratio_d1_over_d2'] = None

        return summary

    def _compute_stage1_cluster_diagnostics(self) -> Dict[str, object]:
        self.model.eval()
        all_labels: List[np.ndarray] = []
        all_assigned: List[np.ndarray] = []
        all_distances: List[np.ndarray] = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                outputs = self.model.compute_anomaly_scores(images, return_maps=False)
                all_labels.append(batch['label'].cpu().numpy())
                all_assigned.append(outputs['assigned_anchors'].cpu().numpy())
                all_distances.append(outputs['all_distances'].cpu().numpy())

        labels = np.concatenate(all_labels)
        assigned = np.concatenate(all_assigned)
        distances = np.concatenate(all_distances)

        diagnostics = {
            'all': self._summarize_assignment_subset(distances, assigned, np.ones_like(labels, dtype=bool)),
            'normal': self._summarize_assignment_subset(distances, assigned, labels == 0),
            'anomaly': self._summarize_assignment_subset(distances, assigned, labels == 1),
        }
        return diagnostics

    def validate(self) -> Dict[str, float]:
        """Run validation - compute both metrics and loss"""
        print("\nRunning validation...")
        
        # Compute validation loss
        self.model.eval()
        val_loss_metrics = {
            'loss': 0.0,
            'loss_attract': 0.0,
            'loss_repel': 0.0,
            'loss_dense': 0.0
        }
        
        # Get anchors for validation loss computation
        # No need to detach since we're inside torch.no_grad() anyway
        anchor_global, _ = self.model._get_projected_anchors()
        
        # Check if model has pixel decoder
        use_pixel_decoder = getattr(self.model, 'use_pixel_decoder', False)
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                
                # Compute loss (enable dense if pixel decoder available)
                outputs = self.model(images, return_dense=use_pixel_decoder)
                loss_dict = self.criterion(outputs, anchor_global)
                
                val_loss_metrics['loss'] += loss_dict['loss'].item()
                
                # Handle different loss types
                if 'loss_global_attract' in loss_dict:
                    val_loss_metrics['loss_attract'] += loss_dict['loss_global_attract']
                    val_loss_metrics['loss_repel'] += loss_dict['loss_global_repel']
                else:
                    val_loss_metrics['loss_attract'] += loss_dict.get('loss_global_loss_center', 0.0)
                    val_loss_metrics['loss_attract'] += loss_dict.get('loss_global_loss_infonce', 0.0)
                    val_loss_metrics['loss_repel'] += loss_dict.get('loss_global_loss_repel', 0.0)
                
                # Dense loss disabled
        
        # Average loss metrics
        num_batches = len(self.val_loader)
        for key in val_loss_metrics:
            val_loss_metrics[key] /= num_batches
        
        # Compute AUROC metrics
        target_size = tuple(getattr(self.model, 'target_size', (256, 256)))
        auroc_metrics = evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            compute_pixel_auroc=True,
            target_size=target_size,
        )

        cluster_diagnostics = self._compute_stage1_cluster_diagnostics()
        
        # Combine metrics
        val_metrics = {**val_loss_metrics, **auroc_metrics}
        for subset_name, subset_metrics in cluster_diagnostics.items():
            for metric_name, metric_value in subset_metrics.items():
                val_metrics[f'cluster_{subset_name}_{metric_name}'] = metric_value
        
        return val_metrics

    def train_epoch_stage2(self) -> Dict[str, float]:
        """Train one epoch for stage-2 reconstruction branch."""
        self.model.train()
        # Ensure frozen projection stays in eval mode
        if hasattr(self.model, 'frozen_projection') and self.model.frozen_projection is not None:
            self.model.frozen_projection.eval()

        epoch_metrics = {
            'loss': 0.0,
            'loss_recon': 0.0,
            'loss_consistency': 0.0,
            'pixel_residual_mean': 0.0,
            'pixel_residual_max': 0.0,
            'bottleneck_divergence_mean': 0.0,
        }

        # Lazy-init pixel stats tracker
        if self.pixel_stats_tracker is None:
            from pixel_aggregation import PixelStatsTracker
            self.pixel_stats_tracker = PixelStatsTracker()

        pbar = tqdm(self.train_loader, desc=f'Stage2 Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            self.optimizer.zero_grad()
            fixed_assign = None
            if self.fixed_pseudo:
                fixed_assign = self._resolve_batch_assignments(batch['path'], images)

            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, return_dense=False)
                    if fixed_assign is not None:
                        outputs['fixed_assignments'] = fixed_assign
                    loss_dict = self._compute_stage2_losses(outputs, images)
                    loss = loss_dict['loss']

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, return_dense=False)
                if fixed_assign is not None:
                    outputs['fixed_assignments'] = fixed_assign
                loss_dict = self._compute_stage2_losses(outputs, images)
                loss = loss_dict['loss']
                loss.backward()
                self.optimizer.step()

            epoch_metrics['loss'] += float(loss_dict['loss'].item())
            epoch_metrics['loss_recon'] += float(loss_dict['loss_recon'].item())
            epoch_metrics['loss_consistency'] += float(loss_dict['loss_consistency'].item())

            if outputs.get('reconstruction_pixel_map') is not None:
                pmap = outputs['reconstruction_pixel_map']
                epoch_metrics['pixel_residual_mean'] += float(pmap.mean().item())
                epoch_metrics['pixel_residual_max'] += float(pmap.max().item())
                self.pixel_stats_tracker.update(pmap)

            if outputs.get('bottleneck_divergence') is not None:
                epoch_metrics['bottleneck_divergence_mean'] += float(outputs['bottleneck_divergence'].mean().item())

            if batch_idx % self.log_interval == 0:
                postfix = {
                    'loss': f"{loss_dict['loss'].item():.4f}",
                    'recon': f"{loss_dict['loss_recon'].item():.4f}",
                    'align': f"{loss_dict['loss_consistency'].item():.4f}",
                }
                if outputs.get('reconstruction_pixel_map') is not None:
                    postfix['pix_mean'] = f"{outputs['reconstruction_pixel_map'].mean().item():.4f}"
                if outputs.get('bottleneck_divergence') is not None:
                    postfix['div'] = f"{outputs['bottleneck_divergence'].mean().item():.4f}"
                pbar.set_postfix(postfix)

            self.global_step += 1

        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        return epoch_metrics

    def validate_stage2(self) -> Dict[str, float]:
        """Validate stage-2 with reconstruction losses, divergence, and separate AUROCs."""
        self.model.eval()
        val_loss = {'loss': 0.0, 'loss_recon': 0.0, 'loss_consistency': 0.0}

        all_labels = []
        all_anchor_scores = []
        all_recon_scores = []
        all_divergence_scores = []
        all_pixel_aggregated = []
        all_recon_pixel_scores = []
        all_pixel_masks = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()

                outputs = self.model(images, return_dense=False)
                if self.fixed_pseudo:
                    outputs['fixed_assignments'] = self._resolve_batch_assignments(batch['path'], images)
                loss_dict = self._compute_stage2_losses(outputs, images)

                val_loss['loss'] += float(loss_dict['loss'].item())
                val_loss['loss_recon'] += float(loss_dict['loss_recon'].item())
                val_loss['loss_consistency'] += float(loss_dict['loss_consistency'].item())

                scores = self.model.compute_anomaly_scores(images, return_maps=True, target_size=images.shape[-2:])
                all_labels.append(labels)
                all_anchor_scores.append(scores['anchor_scores'].cpu().numpy())
                if 'reconstruction_scores' in scores:
                    all_recon_scores.append(scores['reconstruction_scores'].cpu().numpy())
                if 'bottleneck_divergence' in scores:
                    all_divergence_scores.append(scores['bottleneck_divergence'].cpu().numpy())
                if 'pixel_aggregated_score' in scores:
                    all_pixel_aggregated.append(scores['pixel_aggregated_score'].cpu().numpy())
                if 'reconstruction_pixel_scores' in scores:
                    all_recon_pixel_scores.append(scores['reconstruction_pixel_scores'].cpu().numpy())
                if 'mask' in batch:
                    all_pixel_masks.append(batch['mask'].cpu().numpy())

        num_batches = len(self.val_loader)
        for key in val_loss:
            val_loss[key] /= num_batches

        labels = np.concatenate(all_labels)
        anchor_scores = np.concatenate(all_anchor_scores)

        metrics = {
            **val_loss,
            'anchor_image_auroc': roc_auc_score(labels, anchor_scores)
        }

        if len(all_recon_scores) > 0:
            recon_scores = np.concatenate(all_recon_scores)
            metrics['reconstruction_image_auroc'] = roc_auc_score(labels, recon_scores)
            metrics['reconstruction_image_aupr'] = average_precision_score(labels, recon_scores)

        if len(all_divergence_scores) > 0:
            div_scores = np.concatenate(all_divergence_scores)
            try:
                metrics['divergence_image_auroc'] = roc_auc_score(labels, div_scores)
            except ValueError:
                pass  # single class

        if len(all_pixel_aggregated) > 0:
            pix_agg = np.concatenate(all_pixel_aggregated)
            try:
                metrics['pixel_aggregated_image_auroc'] = roc_auc_score(labels, pix_agg)
            except ValueError:
                pass

        # Three-signal combined AUROC for early stopping
        if len(all_divergence_scores) > 0 and len(all_pixel_aggregated) > 0:
            from pixel_aggregation import aggregate_pixel_scores_numpy
            div_scores = np.concatenate(all_divergence_scores)
            pix_agg = np.concatenate(all_pixel_aggregated)
            # minmax normalise each
            def _mm(arr):
                lo, hi = arr.min(), arr.max()
                return (arr - lo) / max(hi - lo, 1e-12)
            w = self.model.score_fusion_anchor_weight if hasattr(self.model, 'score_fusion_anchor_weight') else 0.4
            wd = self.model.score_fusion_divergence_weight if hasattr(self.model, 'score_fusion_divergence_weight') else 0.3
            wp = self.model.score_fusion_pixel_weight if hasattr(self.model, 'score_fusion_pixel_weight') else 0.3
            combined = w * _mm(anchor_scores) + wd * _mm(div_scores) + wp * _mm(pix_agg)
            try:
                metrics['combined_image_auroc'] = roc_auc_score(labels, combined)
            except ValueError:
                pass

        pixel_metrics_enabled = self.stage2_config.get('pixel_metrics', {}).get('enabled', True)
        if pixel_metrics_enabled and len(all_recon_pixel_scores) > 0 and len(all_pixel_masks) > 0:
            try:
                recon_pixel_scores = np.concatenate(all_recon_pixel_scores)
                pixel_masks = np.concatenate(all_pixel_masks)

                if recon_pixel_scores.shape[1:] != pixel_masks.shape[1:]:
                    from scipy.ndimage import zoom
                    scale_h = pixel_masks.shape[1] / recon_pixel_scores.shape[1]
                    scale_w = pixel_masks.shape[2] / recon_pixel_scores.shape[2]
                    recon_pixel_scores = np.array([
                        zoom(recon_pixel_scores[i], (scale_h, scale_w), order=1)
                        for i in range(recon_pixel_scores.shape[0])
                    ])

                if recon_pixel_scores.shape[0] != pixel_masks.shape[0]:
                    min_samples = min(recon_pixel_scores.shape[0], pixel_masks.shape[0])
                    recon_pixel_scores = recon_pixel_scores[:min_samples]
                    pixel_masks = pixel_masks[:min_samples]

                scores_flat = recon_pixel_scores.flatten()
                masks_flat = pixel_masks.flatten()
                if masks_flat.sum() > 0:
                    metrics['reconstruction_pixel_auroc'] = roc_auc_score(masks_flat, scores_flat)
                    metrics['reconstruction_pixel_aupr'] = average_precision_score(masks_flat, scores_flat)
            except Exception as e:
                print(f"Warning: stage-2 reconstruction pixel metric computation failed: {e}")

        return metrics
    
    def train(
        self,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10,
        min_epochs_before_early_stopping: int = 0,
    ):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
            early_stopping_patience: Epochs without improvement before stopping
            min_epochs_before_early_stopping: Earliest epoch count at which early stopping may trigger
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Stage-1 early stopping: patience={early_stopping_patience}, min_epochs={min_epochs_before_early_stopping}")
        
        # Print pseudo-label configuration
        if self.fixed_pseudo:
            if self.dynamic_reassignment:
                print(f"Pseudo-labels: DYNAMIC (reassign every {self.reassignment_interval} epochs)")
            else:
                print(f"Pseudo-labels: FIXED (computed once before training)")
        else:
            print(f"Pseudo-labels: NONE (use nearest anchor per batch)")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Dynamic pseudo-label reassignment (for learnable anchors)
            if self.dynamic_reassignment and self.fixed_pseudo:
                if epoch > 0 and epoch % self.reassignment_interval == 0:
                    print(f"\n  >> Recomputing pseudo-labels (epoch {epoch})...")
                    self._precompute_pseudo_labels()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log training metrics to history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_loss_attract'].append(train_metrics['loss_attract'])
            self.history['train_loss_repel'].append(train_metrics['loss_repel'])
            # Dense loss disabled; keep zeros for compatibility
            self.history['train_loss_dense'].append(0.0)
            self.history['train_loss_dense_attract'].append(0.0)
            self.history['epochs'].append(epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    Attractor: {train_metrics['loss_attract']:.4f}")
            
            # Check if repeller is enabled (different attribute names for different losses)
            repeller_enabled = False
            if hasattr(self.criterion.global_loss, 'beta'):
                repeller_enabled = self.criterion.global_loss.beta > 0
            elif hasattr(self.criterion.global_loss, 'lambda_repel'):
                repeller_enabled = self.criterion.global_loss.lambda_repel > 0
            
            if repeller_enabled:
                print(f"    Repeller: {train_metrics['loss_repel']:.4f}")
            else:
                print(f"    Repeller: {train_metrics['loss_repel']:.4f} (disabled)")
                
            if train_metrics.get('loss_dense', 0.0) > 0:
                print(f"    Dense: {train_metrics['loss_dense']:.4f} (Attr: {train_metrics.get('loss_dense_attract', 0.0):.4f})")
            print(f"  Anchor Balance: min={np.min(train_metrics['anchor_balance']):.3f}, max={np.max(train_metrics['anchor_balance']):.3f}, std={np.std(train_metrics['anchor_balance']):.3f}")
            print(f"  Time: {time.time() - start_time:.1f}s")
            
            # Validation
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate()
                
                # Log validation metrics to history
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_loss_attract'].append(val_metrics['loss_attract'])
                self.history['val_loss_repel'].append(val_metrics['loss_repel'])
                self.history['val_loss_dense'].append(0.0)
                self.history['val_image_auroc'].append(val_metrics['image_auroc'])
                if 'pixel_auroc' in val_metrics:
                    self.history['val_pixel_auroc'].append(val_metrics['pixel_auroc'])
                self.history['val_cluster_all_effective_anchors_used'].append(val_metrics['cluster_all_effective_anchors_used'])
                self.history['val_cluster_all_assignment_entropy'].append(val_metrics['cluster_all_assignment_entropy'])
                self.history['val_cluster_all_assignment_entropy_normalized'].append(val_metrics['cluster_all_assignment_entropy_normalized'])
                self.history['val_cluster_all_largest_anchor_share'].append(val_metrics['cluster_all_largest_anchor_share'])
                self.history['val_cluster_all_max_min_nonzero_ratio'].append(val_metrics['cluster_all_max_min_nonzero_ratio'])
                self.history['val_cluster_all_assignment_counts'].append(val_metrics['cluster_all_assignment_counts'])
                self.history['val_cluster_normal_effective_anchors_used'].append(val_metrics['cluster_normal_effective_anchors_used'])
                self.history['val_cluster_normal_assignment_entropy'].append(val_metrics['cluster_normal_assignment_entropy'])
                self.history['val_cluster_normal_assignment_entropy_normalized'].append(val_metrics['cluster_normal_assignment_entropy_normalized'])
                self.history['val_cluster_normal_largest_anchor_share'].append(val_metrics['cluster_normal_largest_anchor_share'])
                self.history['val_cluster_normal_max_min_nonzero_ratio'].append(val_metrics['cluster_normal_max_min_nonzero_ratio'])
                self.history['val_cluster_normal_assignment_counts'].append(val_metrics['cluster_normal_assignment_counts'])
                self.history['val_cluster_normal_mean_nearest_distance'].append(val_metrics['cluster_normal_mean_nearest_distance'])
                self.history['val_cluster_normal_mean_second_nearest_distance'].append(val_metrics['cluster_normal_mean_second_nearest_distance'])
                self.history['val_cluster_normal_mean_margin_d2_minus_d1'].append(val_metrics['cluster_normal_mean_margin_d2_minus_d1'])
                self.history['val_cluster_normal_mean_ratio_d1_over_d2'].append(val_metrics['cluster_normal_mean_ratio_d1_over_d2'])
                self.history['val_cluster_anomaly_assignment_counts'].append(val_metrics['cluster_anomaly_assignment_counts'])
                
                print(f"\n  Validation Results:")
                print(f"    Val Loss: {val_metrics['loss']:.4f}")
                print(f"      Attractor: {val_metrics['loss_attract']:.4f}")
                
                # Check if repeller is enabled (different attribute names for different losses)
                repeller_enabled = False
                if hasattr(self.criterion.global_loss, 'beta'):
                    repeller_enabled = self.criterion.global_loss.beta > 0
                elif hasattr(self.criterion.global_loss, 'lambda_repel'):
                    repeller_enabled = self.criterion.global_loss.lambda_repel > 0
                
                if repeller_enabled:
                    print(f"      Repeller: {val_metrics['loss_repel']:.4f}")
                else:
                    print(f"      Repeller: {val_metrics['loss_repel']:.4f} (disabled)")
                    
                if val_metrics.get('loss_dense', 0.0) > 0:
                    print(f"      Dense: {val_metrics['loss_dense']:.4f}")
                print(f"    Image AUROC: {val_metrics['image_auroc']:.4f}")
                if 'pixel_auroc' in val_metrics:
                    print(f"    Pixel AUROC: {val_metrics['pixel_auroc']:.4f}")
                print(
                    "    Cluster Normals: "
                    f"used={val_metrics['cluster_normal_effective_anchors_used']}/{self.model.n_anchors}, "
                    f"entropy={val_metrics['cluster_normal_assignment_entropy_normalized']:.4f}, "
                    f"largest_share={val_metrics['cluster_normal_largest_anchor_share']:.4f}, "
                    f"max/min={val_metrics['cluster_normal_max_min_nonzero_ratio']:.2f}"
                )
                second_distance_value = val_metrics['cluster_normal_mean_second_nearest_distance']
                margin_value = val_metrics['cluster_normal_mean_margin_d2_minus_d1']
                ratio_value = val_metrics['cluster_normal_mean_ratio_d1_over_d2']
                print(
                    "    Cluster Distances (normals): "
                    f"d1={val_metrics['cluster_normal_mean_nearest_distance']:.4f}, "
                    f"d2={(second_distance_value if second_distance_value is not None else float('nan')):.4f}, "
                    f"margin={(margin_value if margin_value is not None else float('nan')):.4f}, "
                    f"ratio={(ratio_value if ratio_value is not None else float('nan')):.4f}"
                )
                
                # Save best model
                if val_metrics['image_auroc'] > self.best_val_auroc:
                    self.best_val_auroc = val_metrics['image_auroc']
                    self.save_checkpoint('best_model.pth', val_metrics)
                    print(f"  ✓ New best model! AUROC: {self.best_val_auroc:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if (epoch + 1) >= min_epochs_before_early_stopping and patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    # Save TSNE snapshot before breaking
                    self._save_tsne(epoch=epoch+1, final=True)
                    break
            
            # Save regular checkpoint (only if enabled)
            if self.save_checkpoints and (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                self._save_tsne(epoch=epoch+1)
            
            # Update scheduler
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                self.history['learning_rates'].append(current_lr)
                print(f"  LR: {current_lr:.6f}")
            
            print("-" * 80)
        
        # Save final checkpoint and plot training curves
        self.save_checkpoint('final_model.pth')
        self.save_history()
        self.plot_training_curves()
        self._save_tsne(epoch=self.epoch + 1, final=True)
        
        # Create training summary
        self._create_training_summary(num_epochs, early_stopping_patience, min_epochs_before_early_stopping)
        
        print(f"\nTraining complete!")
        print(f"Best validation AUROC: {self.best_val_auroc:.4f}")

    def train_stage2(
        self,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10
    ):
        """Stage-2 training loop (reconstruction branch)."""
        print(f"Starting stage-2 reconstruction training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Stage-2 trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        if self.model.use_frozen_bottleneck:
            print(f"  Dual-bottleneck enabled (frozen → alignment loss)")

        best_stage2_metric = -1e9
        patience_counter = 0

        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            train_metrics = self.train_epoch_stage2()
            val_metrics = self.validate_stage2() if (epoch + 1) % self.val_interval == 0 else {}

            self.history.setdefault('stage2_train_loss', []).append(train_metrics['loss'])
            self.history.setdefault('stage2_train_recon', []).append(train_metrics['loss_recon'])
            self.history.setdefault('stage2_train_consistency', []).append(train_metrics['loss_consistency'])
            if train_metrics.get('bottleneck_divergence_mean', 0) > 0:
                self.history.setdefault('stage2_train_divergence', []).append(train_metrics['bottleneck_divergence_mean'])

            print(f"\nStage-2 Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    Reconstruction: {train_metrics['loss_recon']:.4f}")
            print(f"    Alignment: {train_metrics['loss_consistency']:.4f}")
            print(f"    Pixel Residual Mean/Max: {train_metrics['pixel_residual_mean']:.4f} / {train_metrics['pixel_residual_max']:.4f}")
            if train_metrics.get('bottleneck_divergence_mean', 0) > 0:
                print(f"    Bottleneck Divergence: {train_metrics['bottleneck_divergence_mean']:.4f}")

            if val_metrics:
                self.history.setdefault('stage2_val_loss', []).append(val_metrics['loss'])
                self.history.setdefault('stage2_val_recon', []).append(val_metrics['loss_recon'])
                self.history.setdefault('stage2_val_consistency', []).append(val_metrics['loss_consistency'])
                self.history.setdefault('stage2_val_anchor_auroc', []).append(val_metrics['anchor_image_auroc'])

                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Anchor AUROC: {val_metrics['anchor_image_auroc']:.4f}")
                if 'reconstruction_image_auroc' in val_metrics:
                    self.history.setdefault('stage2_val_recon_auroc', []).append(val_metrics['reconstruction_image_auroc'])
                    print(f"  Val Reconstruction AUROC: {val_metrics['reconstruction_image_auroc']:.4f}")
                if 'divergence_image_auroc' in val_metrics:
                    self.history.setdefault('stage2_val_divergence_auroc', []).append(val_metrics['divergence_image_auroc'])
                    print(f"  Val Divergence AUROC: {val_metrics['divergence_image_auroc']:.4f}")
                if 'pixel_aggregated_image_auroc' in val_metrics:
                    self.history.setdefault('stage2_val_pixel_agg_auroc', []).append(val_metrics['pixel_aggregated_image_auroc'])
                    print(f"  Val Pixel-Aggregated AUROC: {val_metrics['pixel_aggregated_image_auroc']:.4f}")
                if 'fused_image_auroc' in val_metrics:
                    self.history.setdefault('stage2_val_fused_auroc', []).append(val_metrics['fused_image_auroc'])
                    print(f"  Val Fused AUROC: {val_metrics['fused_image_auroc']:.4f}")
                if 'combined_image_auroc' in val_metrics:
                    self.history.setdefault('stage2_val_combined_auroc', []).append(val_metrics['combined_image_auroc'])
                    print(f"  Val Combined (3-signal) AUROC: {val_metrics['combined_image_auroc']:.4f}")
                if 'reconstruction_pixel_auroc' in val_metrics:
                    self.history.setdefault('stage2_val_recon_pixel_auroc', []).append(val_metrics['reconstruction_pixel_auroc'])
                    print(f"  Val Reconstruction Pixel AUROC: {val_metrics['reconstruction_pixel_auroc']:.4f}")

                monitor_name = self.stage2_config.get('early_stopping_metric', 'pixel_aggregated_image_auroc')
                if monitor_name not in val_metrics:
                    # Fallback chain: pixel_agg → combined → recon → anchor
                    for fallback in ['pixel_aggregated_image_auroc', 'combined_image_auroc', 'reconstruction_image_auroc', 'anchor_image_auroc']:
                        if fallback in val_metrics:
                            monitor_name = fallback
                            break
                current_metric = float(val_metrics.get(monitor_name, 0.0))

                if current_metric > best_stage2_metric:
                    best_stage2_metric = current_metric
                    patience_counter = 0
                    self.save_checkpoint('best_stage2_model.pth', val_metrics)
                    print(f"  ✓ New best stage-2 metric ({monitor_name}): {best_stage2_metric:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping stage-2 after {epoch + 1} epochs")
                    break

            if scheduler is not None:
                scheduler.step()

            print(f"  Time: {time.time() - start_time:.1f}s")
            print("-" * 80)

        # Save pixel stats for threshold_ratio aggregation
        if self.pixel_stats_tracker is not None:
            pixel_stats_path = self.save_dir / 'pixel_stats.json'
            import json as _json
            with open(pixel_stats_path, 'w') as f:
                stats = self.pixel_stats_tracker.state_dict()
                stats['threshold_3std'] = self.pixel_stats_tracker.threshold(3.0)
                stats['mean'] = self.pixel_stats_tracker.mean
                stats['std'] = self.pixel_stats_tracker.std
                _json.dump(stats, f, indent=2)
            print(f"Saved pixel stats: mean={self.pixel_stats_tracker.mean:.6f}, std={self.pixel_stats_tracker.std:.6f}, threshold(3σ)={self.pixel_stats_tracker.threshold(3.0):.6f}")

        self.save_checkpoint('final_stage2_model.pth')
        self.save_history()
        print("\nStage-2 training complete!")
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auroc': self.best_val_auroc,
            'history': self.history,
            'anchor_mode': _get_model_anchor_mode(self.model)
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        _validate_checkpoint_anchor_mode(
            checkpoint,
            expected_anchor_mode=_get_model_anchor_mode(self.model),
            checkpoint_path=str(checkpoint_path)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_auroc = checkpoint['best_val_auroc']
        self.history = checkpoint['history']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def save_history(self):
        """Save training history"""
        # Convert any tensors or numpy types to native Python types
        def convert_to_serializable(obj):
            """Recursively convert tensors and numpy types to Python natives"""
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
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().tolist() if obj.numel() > 1 else obj.item()
            else:
                return obj
        
        history_serializable = convert_to_serializable(self.history)
        
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"Saved training history: {history_path}")
    
    def _create_training_summary(self, total_epochs: int, early_stopping_patience: int, min_epochs_before_early_stopping: int):
        """Create a summary JSON file with training information and best model details"""
        import json
        
        # Find best epoch based on validation AUROC
        best_epoch = -1
        best_auroc = 0.0
        if len(self.history['val_image_auroc']) > 0:
            val_aurocs = np.array(self.history['val_image_auroc'])
            best_epoch = int(np.argmax(val_aurocs))
            best_auroc = float(val_aurocs[best_epoch])
        
        # Determine if early stopping occurred
        actual_epochs = len(self.history['epochs'])
        early_stopped = actual_epochs < total_epochs
        
        summary = {
            'training_completed': True,
            'total_epochs_configured': total_epochs,
            'actual_epochs_trained': actual_epochs,
            'early_stopping_enabled': early_stopping_patience < 1000,
            'early_stopping_patience': early_stopping_patience,
            'min_epochs_before_early_stopping': min_epochs_before_early_stopping,
            'early_stopped': early_stopped,
            'best_model': {
                'epoch': best_epoch + 1,  # 1-indexed for readability
                'image_auroc': best_auroc,
                'saved_as': 'best_model.pth',
                'description': f'Best model achieved at epoch {best_epoch + 1} with image AUROC {best_auroc:.4f}'
            },
            'final_model': {
                'epoch': actual_epochs,
                'saved_as': 'final_model.pth',
                'description': f'Final model after {actual_epochs} epochs'
            },
            'checkpoints_saved': self.save_checkpoints,
            'model_files': ['best_model.pth', 'final_model.pth']
        }
        
        # Add checkpoint files if they were saved
        if self.save_checkpoints:
            checkpoint_epochs = [e + 1 for e in range(actual_epochs) if (e + 1) % 5 == 0]
            summary['model_files'].extend([f'checkpoint_epoch_{e}.pth' for e in checkpoint_epochs])
        
        summary_path = self.save_dir / 'training_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\nTraining Summary:")
        print(f"  Best model: Epoch {best_epoch + 1}, AUROC {best_auroc:.4f}")
        print(f"  Final model: Epoch {actual_epochs}")
        if early_stopped:
            print(f"  Early stopping triggered after {actual_epochs} epochs")
        print(f"  Saved summary: {summary_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves - simplified version"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style('whitegrid')
        
        # Convert any tensors to numpy arrays
        def to_numpy(data):
            """Convert tensor or list to numpy array"""
            if isinstance(data, list):
                return np.array([to_numpy(x) for x in data])
            elif isinstance(data, torch.Tensor):
                return data.cpu().detach().numpy()
            elif isinstance(data, (np.ndarray, float, int)):
                return data
            else:
                return np.array(data)
        
        # Determine number of validation points
        n_val_points = len(self.history['val_image_auroc'])
        if n_val_points == 0:
            print("No validation data to plot")
            return
        
        # Create validation epoch indices
        val_epochs = list(range(0, len(self.history['epochs']), self.val_interval))[:n_val_points]
        
        # Create figure with 2x2 subplots (simplified)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Total Loss (train + val)
        ax = axes[0, 0]
        ax.plot(to_numpy(self.history['epochs']), to_numpy(self.history['train_loss']), 'b-', label='Train', linewidth=2, alpha=0.7)
        if self.history['val_loss']:
            ax.plot(to_numpy(val_epochs), to_numpy(self.history['val_loss']), 'r-', label='Val', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Total Loss', fontsize=11)
        ax.set_title('Total Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Attractor Loss
        ax = axes[0, 1]
        ax.plot(to_numpy(self.history['epochs']), to_numpy(self.history['train_loss_attract']), 'b-', label='Train', linewidth=2, alpha=0.7)
        if self.history['val_loss_attract']:
            ax.plot(to_numpy(val_epochs), to_numpy(self.history['val_loss_attract']), 'r-', label='Val', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Attractor Loss', fontsize=11)
        ax.set_title('Attractor Loss (Pull to Anchors)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Image AUROC
        ax = axes[1, 0]
        val_auroc_np = to_numpy(self.history['val_image_auroc'])
        best_val_auroc = float(np.max(val_auroc_np))
        ax.plot(to_numpy(val_epochs), val_auroc_np, 'g-o', linewidth=2, markersize=5, alpha=0.7)
        ax.axhline(y=best_val_auroc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_val_auroc:.4f}')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Image AUROC', fontsize=11)
        ax.set_title('Validation Image AUROC', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 4. Pixel AUROC
        ax = axes[1, 1]
        if self.history['val_pixel_auroc'] and len(self.history['val_pixel_auroc']) > 0:
            val_pixel_np = to_numpy(self.history['val_pixel_auroc'])
            # Filter out zeros/None
            val_pixel_filtered = [x for x in val_pixel_np if x > 0]
            if val_pixel_filtered:
                best_pixel = float(np.max(val_pixel_filtered))
                ax.plot(to_numpy(val_epochs[:len(val_pixel_np)]), val_pixel_np, 'purple', linewidth=2, marker='s', markersize=5, alpha=0.7)
                ax.axhline(y=best_pixel, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_pixel:.4f}')
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Pixel AUROC', fontsize=11)
                ax.set_title('Validation Pixel AUROC', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.set_ylim([0, 1])
            else:
                ax.text(0.5, 0.5, 'Pixel AUROC Not Computed\n(No anomalous pixels in validation)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Pixel AUROC Not Computed\n(No anomalous pixels in validation)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Progress - Medical Anomaly Detection', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves: {save_path}")
