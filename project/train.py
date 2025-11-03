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
        val_interval: int = 1
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
            'val_loss_dense': [],
            'val_image_auroc': [],
            'val_pixel_auroc': [],
            
            # Per-epoch statistics
            'epochs': [],
            'learning_rates': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'loss_attract': 0.0,
            'loss_repel': 0.0,
            'loss_dense': 0.0,
            'loss_dense_attract': 0.0
        }
        
        anchor_assignments = np.zeros(self.model.n_anchors)
        
        # Get projected anchor embeddings for loss computation
        # Detach to prevent backprop through anchor computation
        anchor_global, _ = self.model._get_projected_anchors()
        anchor_global = anchor_global.detach()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, return_dense=True)  # Enable dense for loss
                    loss_dict = self.criterion(outputs, anchor_global)
                    loss = loss_dict['loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, return_dense=True)  # Enable dense for loss
                loss_dict = self.criterion(outputs, anchor_global)
                loss = loss_dict['loss']
                
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['loss_attract'] += loss_dict['loss_global_attract']
            epoch_metrics['loss_repel'] += loss_dict['loss_global_repel']
            
            # Track dense metrics if available
            if 'loss_dense' in loss_dict:
                epoch_metrics['loss_dense'] += loss_dict['loss_dense']
                epoch_metrics['loss_dense_attract'] += loss_dict['loss_dense_attract']
            
            # Track anchor assignments
            assigned = loss_dict['assigned_anchors'].cpu().numpy()
            for a in assigned:
                anchor_assignments[a] += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                    'attr': f"{loss_dict['loss_global_attract']:.4f}",
                    'rep': f"{loss_dict['loss_global_repel']:.4f}"
                }
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
        
        anchor_global, _ = self.model._get_projected_anchors()
        anchor_global = anchor_global.detach()
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                
                # Compute loss
                outputs = self.model(images, return_dense=True)
                loss_dict = self.criterion(outputs, anchor_global)
                
                val_loss_metrics['loss'] += loss_dict['loss'].item()
                val_loss_metrics['loss_attract'] += loss_dict['loss_global_attract']
                val_loss_metrics['loss_repel'] += loss_dict['loss_global_repel']
                
                if 'loss_dense' in loss_dict:
                    val_loss_metrics['loss_dense'] += loss_dict['loss_dense']
        
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
            self.history['train_loss_dense'].append(train_metrics.get('loss_dense', 0.0))
            self.history['train_loss_dense_attract'].append(train_metrics.get('loss_dense_attract', 0.0))
            self.history['epochs'].append(epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    Attractor: {train_metrics['loss_attract']:.4f}")
            if self.criterion.global_loss.beta > 0:
                print(f"    Repeller: {train_metrics['loss_repel']:.4f}")
            else:
                print(f"    Repeller: {train_metrics['loss_repel']:.4f} (disabled, beta=0)")
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
                self.history['val_loss_dense'].append(val_metrics.get('loss_dense', 0.0))
                self.history['val_image_auroc'].append(val_metrics['image_auroc'])
                if 'pixel_auroc' in val_metrics:
                    self.history['val_pixel_auroc'].append(val_metrics['pixel_auroc'])
                
                print(f"\n  Validation Results:")
                print(f"    Val Loss: {val_metrics['loss']:.4f}")
                print(f"      Attractor: {val_metrics['loss_attract']:.4f}")
                if self.criterion.global_loss.beta > 0:
                    print(f"      Repeller: {val_metrics['loss_repel']:.4f}")
                else:
                    print(f"      Repeller: {val_metrics['loss_repel']:.4f} (disabled, beta=0)")
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
                    break
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
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