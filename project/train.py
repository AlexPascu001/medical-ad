"""
Training Loop for Anomaly Detector
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
import json
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from model import AnomalyDetector
from loss import CombinedAnchorLoss
from eval import evaluate_model


class Trainer:
    """Training manager for anomaly detector"""
    
    def __init__(
        self,
        model: AnomalyDetector,
        criterion: CombinedAnchorLoss,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: Path,
        use_amp: bool = True,
        log_interval: int = 50,
        val_interval: int = 1,
        fixed_pseudo_labels: bool = False,
        dynamic_reassignment: bool = False,
        reassignment_interval: int = 5
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
            dynamic_reassignment: If True, recompute pseudo-labels every reassignment_interval epochs
            reassignment_interval: Epochs between pseudo-label recomputation (only if dynamic_reassignment=True)
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
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_auroc = 0.0

        # Fixed set for TSNE tracking
        self.tsne_samples = self._prepare_tsne_samples(normal_count=500, anomaly_count=100)
        self.fixed_assignments = None
        self.fixed_pseudo = fixed_pseudo_labels
        
        # Dynamic label reassignment for learnable anchors
        self.dynamic_reassignment = dynamic_reassignment
        self.reassignment_interval = reassignment_interval
        
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
            
            # Per-epoch statistics
            'epochs': [],
            'learning_rates': []
        }

        if self.fixed_pseudo:
            self._precompute_pseudo_labels()

        # Precompute pseudo-labels if enabled in config
        self.fixed_pseudo = getattr(self, 'fixed_pseudo', False)

    def _prepare_tsne_samples(self, normal_count: int = 500, anomaly_count: int = 100):
        """Collect a fixed pool of samples for visualization (CPU tensors)."""
        # No longer used - we visualize ALL training samples instead
        return None

    def _visualize_training_samples(self, epoch: int, max_samples: int = 2000, max_lines_per_anchor: int = 150):
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
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # === Plot 1: t-SNE ===
        ax = axes[0]
        perplexity = min(30, len(all_points) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        coords_2d = tsne.fit_transform(all_points)
        
        sample_coords = coords_2d[:n_samples]
        anchor_coords = coords_2d[n_samples:]
        
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
                    c=colors[k], alpha=0.15, linewidth=0.5, zorder=1
                )
        
        # Plot samples colored by assignment
        for k in range(n_anchors):
            mask = assign_np == k
            count = mask.sum()
            if count > 0:
                ax.scatter(sample_coords[mask, 0], sample_coords[mask, 1], 
                          c=[colors[k]], s=15, alpha=0.6, label=f'Anchor {k} (n={count})', zorder=2)
        
        # Plot anchors (big stars)
        for k in range(n_anchors):
            ax.scatter(anchor_coords[k, 0], anchor_coords[k, 1],
                      c=[colors[k]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
            ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]),
                       fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
        ax.set_title(f't-SNE (N={n_samples})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # === Plot 2: PCA ===
        ax = axes[1]
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(all_points)
        
        sample_pca = coords_pca[:n_samples]
        anchor_pca = coords_pca[n_samples:]
        
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
                    c=colors[k], alpha=0.15, linewidth=0.5, zorder=1
                )
        
        # Plot samples
        for k in range(n_anchors):
            mask = assign_np == k
            count = mask.sum()
            if count > 0:
                ax.scatter(sample_pca[mask, 0], sample_pca[mask, 1],
                          c=[colors[k]], s=15, alpha=0.6, label=f'Anchor {k} (n={count})', zorder=2)
        
        # Plot anchors
        for k in range(n_anchors):
            ax.scatter(anchor_pca[k, 0], anchor_pca[k, 1],
                      c=[colors[k]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
            ax.annotate(f'A{k}', (anchor_pca[k, 0], anchor_pca[k, 1]),
                       fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
        # Mark origin
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        ax.scatter([0], [0], c='red', s=100, marker='x', zorder=5, label='Origin')
        
        ax.set_title(f'PCA (N={n_samples})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Samples - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'train_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {save_dir / f'train_epoch_{epoch:03d}.png'}")

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
        
        with torch.no_grad():
            anchor_global, _ = self.model._get_projected_anchors()
            
            for batch in tqdm(test_loader, desc='    Collecting test embeddings', leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                outputs = self.model(images, return_dense=False)
                embeddings = outputs['global_feat']
                distances = outputs['global_distances']
                min_dist = distances.min(dim=1)[0]  # Distance to nearest anchor
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                all_distances.append(min_dist.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_distances = torch.cat(all_distances, dim=0)
        anchors = anchor_global.cpu()
        
        n_samples = all_embeddings.shape[0]
        n_anchors = anchors.shape[0]
        n_normal = (all_labels == 0).sum().item()
        n_anomaly = (all_labels == 1).sum().item()
        
        emb_np = all_embeddings.numpy()
        anc_np = anchors.numpy()
        labels_np = all_labels.numpy()
        distances_np = all_distances.numpy()
        
        # Combine for dimensionality reduction
        all_points = np.vstack([emb_np, anc_np])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Colors for normal/anomaly
        normal_color = 'steelblue'
        anomaly_color = 'crimson'
        anchor_colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
        
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
        
        # Plot anchors (big stars)
        for k in range(n_anchors):
            ax.scatter(anchor_coords[k, 0], anchor_coords[k, 1],
                      c=[anchor_colors[k]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
            ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]),
                       fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
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
        
        # Plot anchors
        for k in range(n_anchors):
            ax.scatter(anchor_pca[k, 0], anchor_pca[k, 1],
                      c=[anchor_colors[k]], s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
            ax.annotate(f'A{k}', (anchor_pca[k, 0], anchor_pca[k, 1]),
                       fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
        
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
        print("\n" + "="*80)
        print("COMPUTING PSEUDO-LABELS IN 384D DINOV3 SPACE (SOLUTION A)")
        print("="*80)
        print("Using RAW 384D DINOv3 embeddings (frozen semantic features)")
        print("Ensures pseudo-labels based on SEMANTIC similarity, not random projection")
        print("="*80)
        
        self.model.eval()
        mapping = {}

        # Get SEMANTIC anchors (384D) for labeling
        # EXPERT'S APPROACH: Use frozen DINOv3 embeddings for pseudo-label computation
        anchor_embeddings_384d = self.model.get_semantic_anchors().to(self.device)  # (K, 384)
        print(f"\nSemantic anchor embeddings (384D): {anchor_embeddings_384d.shape}")
        
        # Use a non-dropping, non-shuffling loader to cover all samples
        preload = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory,
            drop_last=False
        )

        all_min_distances = []
        with torch.no_grad():
            for batch in tqdm(preload, desc='Computing pseudo-labels in 384D space'):
                images = batch['image'].to(self.device)
                paths = batch['path']

                # Extract RAW 384D DINOv3 features (NO projection)
                # Access backbone directly: backbone.backbone is the DINOv2Model
                features_384d = self.model.backbone.backbone.forward_features(images)  # (B, N_patches, 384)
                embeddings_384d = features_384d[:, 0]  # CLS token: (B, 384)
                
                # Compute distances in 384D space
                distances = torch.cdist(embeddings_384d, anchor_embeddings_384d)  # (B, K)
                assigned = distances.argmin(dim=1).cpu().tolist()
                min_distances = distances.min(dim=1)[0]  # (B,)

                for p, a in zip(paths, assigned):
                    mapping[str(p)] = int(a)
                
                all_min_distances.append(min_distances.cpu())

        self.fixed_assignments = mapping
        all_min_distances = torch.cat(all_min_distances, dim=0)
        
        # Statistics
        label_list = list(mapping.values())
        unique_labels = set(label_list)
        counts = {label: label_list.count(label) for label in unique_labels}
        
        print(f"\n{'='*80}")
        print("PSEUDO-LABEL STATISTICS (384D SPACE)")
        print(f"{'='*80}")
        print(f"Total samples: {len(mapping)}")
        print(f"Anchors used: {len(unique_labels)} / {anchor_embeddings_384d.shape[0]}")
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
        if max_count > 3 * min_count:
            print(f"\n⚠️  Warning: Imbalanced distribution (max={max_count}, min={min_count})")
            print("   Consider using diversity loss if this persists during training.")
        else:
            print(f"\n✓ Distribution looks balanced (max={max_count}, min={min_count})")
        
        print(f"{'='*80}\n")
    
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
                fixed_assign = torch.tensor(
                    [self.fixed_assignments[str(p)] for p in batch['path']],
                    device=self.device,
                    dtype=torch.long
                )
            
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
        auroc_metrics = evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            compute_pixel_auroc=True
        )
        
        # Combine metrics
        val_metrics = {**val_loss_metrics, **auroc_metrics}
        
        return val_metrics
    
    def train(
        self,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10
    ):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
            early_stopping_patience: Epochs without improvement before stopping
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
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
                
                # Save best model
                if val_metrics['image_auroc'] > self.best_val_auroc:
                    self.best_val_auroc = val_metrics['image_auroc']
                    self.save_checkpoint('best_model.pth', val_metrics)
                    print(f"  ✓ New best model! AUROC: {self.best_val_auroc:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    # Save TSNE snapshot before breaking
                    self._save_tsne(epoch=epoch+1, final=True)
                    break
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
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
        
        print(f"\nTraining complete!")
        print(f"Best validation AUROC: {self.best_val_auroc:.4f}")
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auroc': self.best_val_auroc,
            'history': self.history
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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