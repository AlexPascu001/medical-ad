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
        fixed_pseudo_labels: bool = False
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
        """Collect a fixed pool of samples for TSNE visualization (CPU tensors)."""
        normals = []
        anomalies = []

        # Use val_loader (has both classes) to grab a stable subset
        for batch in self.val_loader:
            images = batch['image']  # already preprocessed tensors
            labels = batch['label']
            for img, lbl in zip(images, labels):
                if lbl.item() == 0 and len(normals) < normal_count:
                    normals.append(img.cpu())
                elif lbl.item() == 1 and len(anomalies) < anomaly_count:
                    anomalies.append(img.cpu())
            if len(normals) >= normal_count and len(anomalies) >= anomaly_count:
                break

        if len(normals) == 0:
            print("TSNE prep warning: no normal samples collected")
        if len(anomalies) == 0:
            print("TSNE prep warning: no anomalous samples collected")

        images = normals + anomalies
        labels = [0] * len(normals) + [1] * len(anomalies)

        if len(images) == 0:
            return None

        return {
            'images': torch.stack(images),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def _precompute_pseudo_labels(self):
        """Compute fixed nearest-anchor assignments once before training."""
        print("\nPrecomputing fixed pseudo-labels (nearest anchors)...")
        self.model.eval()
        mapping = {}

        # Use a non-dropping, non-shuffling loader to cover all samples
        preload = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory,
            drop_last=False
        )

        with torch.no_grad():
            for batch in tqdm(preload, desc='Pseudo-labeling'):
                images = batch['image'].to(self.device)
                paths = batch['path']

                outputs = self.model(images, return_dense=False)
                distances = outputs['global_distances']  # (B, K)
                assigned = distances.argmin(dim=1).cpu().tolist()

                for p, a in zip(paths, assigned):
                    mapping[str(p)] = int(a)

        self.fixed_assignments = mapping
        print(f"✓ Pseudo-labels computed for {len(mapping)} samples")

    def _save_tsne(self, epoch: int, final: bool = False):
        """Compute embeddings for fixed samples + anchors and save TSNE plot."""
        if self.tsne_samples is None:
            return

        self.model.eval()
        save_dir = self.save_dir / 'tsne'
        save_dir.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            imgs = self.tsne_samples['images'].to(self.device)
            labels = self.tsne_samples['labels']

            outputs = self.model(imgs, return_dense=False)
            sample_embeds = outputs['global_feat'].detach().cpu().numpy()

            anchor_global, _ = self.model._get_projected_anchors()
            anchor_np = anchor_global.detach().cpu().numpy()

        data = np.vstack([sample_embeds, anchor_np])
        label_vec = np.concatenate([
            labels.numpy(),
            np.full(anchor_np.shape[0], 2, dtype=np.int64)  # 2 = anchor
        ])

        # TSNE parameters
        total_points = data.shape[0]
        perplexity = max(5, min(30, (total_points - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        coords = tsne.fit_transform(data)

        # Plot
        plt.figure(figsize=(8, 6))
        colors = {0: 'steelblue', 1: 'crimson', 2: 'orange'}
        labels_map = {0: 'normal', 1: 'anomaly', 2: 'anchor'}
        for cls in [0, 1, 2]:
            mask = label_vec == cls
            if mask.any():
                plt.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7, c=colors[cls], label=labels_map[cls])
        plt.legend()
        plt.title(f'TSNE epoch {epoch}' if not final else 'TSNE final')
        plt.tight_layout()

        fname = 'tsne_final.png' if final else f'tsne_epoch_{epoch:03d}.png'
        plt.savefig(save_dir / fname, dpi=150)
        plt.close()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'loss_attract': 0.0,
            'loss_repel': 0.0,
            'loss_norm': 0.0,
            'loss_dense': 0.0,
            'loss_dense_attract': 0.0
        }
        
        anchor_assignments = np.zeros(self.model.n_anchors)
        
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
                    outputs = self.model(images, return_dense=False)  # Dense loss disabled
                    if fixed_assign is not None:
                        outputs['fixed_assignments'] = fixed_assign
                    loss_dict = self.criterion(outputs, anchor_global)
                    loss = loss_dict['loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, return_dense=False)  # Dense loss disabled
                if fixed_assign is not None:
                    outputs['fixed_assignments'] = fixed_assign
                loss_dict = self.criterion(outputs, anchor_global)
                loss = loss_dict['loss']
                
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            epoch_metrics['loss'] += loss.item()
            
            # Handle different loss types (CAM vs Contrastive)
            # CAM loss: loss_global_attract, loss_global_repel, loss_global_norm
            # Contrastive: loss_global_loss_center, loss_global_loss_infonce, loss_global_loss_repel
            if 'loss_global_attract' in loss_dict:
                # CAM loss
                epoch_metrics['loss_attract'] += loss_dict['loss_global_attract']
                epoch_metrics['loss_repel'] += loss_dict['loss_global_repel']
                epoch_metrics['loss_norm'] += loss_dict.get('loss_global_norm', 0.0)
            else:
                # Contrastive loss - aggregate all components
                epoch_metrics['loss_attract'] += loss_dict.get('loss_global_loss_center', 0.0)
                epoch_metrics['loss_attract'] += loss_dict.get('loss_global_loss_infonce', 0.0)
                epoch_metrics['loss_repel'] += loss_dict.get('loss_global_loss_repel', 0.0)
                epoch_metrics['loss_norm'] += 0.0  # No norm loss in contrastive
            
            # Dense path disabled; keep zero defaults
            
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
                
                if 'loss_dense' in loss_dict:
                    postfix_dict['dense'] = f"{loss_dict['loss_dense']:.4f}"
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
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                
                # Compute loss
                outputs = self.model(images, return_dense=False)
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
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
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