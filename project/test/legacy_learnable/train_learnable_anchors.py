"""
Train model with learnable anchors using CAM loss

This script:
1. Loads pre-computed anchors from a previous experiment (eigenface/kmeans/random)
2. Initializes learnable anchors with those fixed anchors
3. Trains the model with CAM loss (attractor + repeller + min-norm)
4. Fine-tunes both the projection head AND the anchors
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
import wandb

from model import DINOv3Backbone, AnomalyDetector
from data import BMADDataset, BMADPreprocessor
from learnable_anchors import LearnableAnchors, CAMLoss, assign_to_nearest_anchor
from eval import evaluate_model
from main import load_dataset_paths, set_seed


class LearnableAnchorTrainer:
    """Trainer for models with learnable anchors using CAM loss"""
    
    def __init__(
        self,
        model: AnomalyDetector,
        learnable_anchors: LearnableAnchors,
        criterion: CAMLoss,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        device: torch.device,
        save_dir: Path,
        distance_metric: str = 'euclidean',
        use_amp: bool = True,
        log_interval: int = 50,
        val_interval: int = 1
    ):
        self.model = model
        self.learnable_anchors = learnable_anchors
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.distance_metric = distance_metric
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.history = {
            'train_loss': [],
            'train_attractor': [],
            'train_repeller': [],
            'train_norm': [],
            'val_image_auroc': [],
            'val_pixel_auroc': [],
            'anchor_norms': [],
            'anchor_distances': []
        }
        self.best_auroc = 0.0
        self.epochs_no_improve = 0
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': [],
            'attractor': [],
            'repeller': [],
            'norm': []
        }
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    embeddings = outputs['global_feat']  # (B, D)
                    
                    # Get current learnable anchors
                    anchors = self.learnable_anchors()  # (K, D)
                    
                    # Assign each embedding to nearest anchor
                    assignments = assign_to_nearest_anchor(
                        embeddings, anchors, self.distance_metric
                    )
                    
                    # Compute CAM loss
                    loss, loss_dict = self.criterion(embeddings, anchors, assignments)
            else:
                outputs = self.model(images)
                embeddings = outputs['global_feat']
                anchors = self.learnable_anchors()
                assignments = assign_to_nearest_anchor(
                    embeddings, anchors, self.distance_metric
                )
                loss, loss_dict = self.criterion(embeddings, anchors, assignments)
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Log losses
            epoch_losses['total'].append(loss_dict['total'])
            epoch_losses['attractor'].append(loss_dict['attractor'])
            epoch_losses['repeller'].append(loss_dict['repeller'])
            epoch_losses['norm'].append(loss_dict['norm'])
            
            # Print progress
            if batch_idx % self.log_interval == 0:
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"(Att: {loss_dict['attractor']:.4f}, "
                      f"Rep: {loss_dict['repeller']:.4f}, "
                      f"Norm: {loss_dict['norm']:.4f})")
        
        # Compute epoch averages
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, epoch: int):
        """Validate model"""
        print(f"\n  Validating...")
        
        # Get current anchors
        with torch.no_grad():
            current_anchors = self.learnable_anchors()
            
            # Update model's anchor embeddings
            self.model.anchor_global_embeddings = current_anchors.cpu()
        
        # Evaluate
        results = evaluate_model(
            model=self.model,
            test_loader=self.val_loader,
            device=self.device,
            compute_pixel=False,
            distance_metric=self.distance_metric
        )
        
        image_auroc = results['image_auroc']
        
        print(f"  Val Image AUROC: {image_auroc:.4f}")
        
        # Log anchor statistics
        with torch.no_grad():
            anchor_norms = self.learnable_anchors.get_anchor_norms()
            anchor_dists = self.learnable_anchors.get_pairwise_distances()
            
            # Get off-diagonal distances (actual anchor separations)
            K = anchor_dists.shape[0]
            mask = ~torch.eye(K, dtype=torch.bool, device=anchor_dists.device)
            off_diag = anchor_dists[mask]
            
            min_dist = off_diag.min().item()
            mean_dist = off_diag.mean().item()
            
            print(f"  Anchor norms: min={anchor_norms.min().item():.3f}, "
                  f"max={anchor_norms.max().item():.3f}, "
                  f"mean={anchor_norms.mean().item():.3f}")
            print(f"  Anchor distances: min={min_dist:.3f}, mean={mean_dist:.3f}")
        
        return image_auroc, anchor_norms.cpu().numpy(), off_diag.cpu().numpy()
    
    def train(
        self,
        num_epochs: int,
        scheduler=None,
        early_stopping_patience: int = 10
    ):
        """Train for multiple epochs"""
        print(f"\nTraining for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            avg_losses = self.train_epoch(epoch)
            
            print(f"\n  Train Loss: {avg_losses['total']:.4f} "
                  f"(Att: {avg_losses['attractor']:.4f}, "
                  f"Rep: {avg_losses['repeller']:.4f}, "
                  f"Norm: {avg_losses['norm']:.4f})")
            
            # Update history
            self.history['train_loss'].append(avg_losses['total'])
            self.history['train_attractor'].append(avg_losses['attractor'])
            self.history['train_repeller'].append(avg_losses['repeller'])
            self.history['train_norm'].append(avg_losses['norm'])
            
            # Validate
            if (epoch + 1) % self.val_interval == 0:
                image_auroc, anchor_norms, anchor_dists = self.validate(epoch)
                
                self.history['val_image_auroc'].append(image_auroc)
                self.history['anchor_norms'].append(anchor_norms.tolist())
                self.history['anchor_distances'].append(anchor_dists.tolist())
                
                # Check for improvement
                if image_auroc > self.best_auroc:
                    self.best_auroc = image_auroc
                    self.epochs_no_improve = 0
                    
                    # Save best model
                    self.save_checkpoint('best_model.pth', epoch, image_auroc)
                    print(f"  ✓ New best AUROC: {self.best_auroc:.4f}")
                else:
                    self.epochs_no_improve += 1
                    print(f"  No improvement for {self.epochs_no_improve} epochs")
                
                # Early stopping
                if self.epochs_no_improve >= early_stopping_patience:
                    print(f"\n  Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, image_auroc)
        
        # Save final model
        self.save_checkpoint('final_model.pth', num_epochs-1, self.best_auroc)
        
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training complete! Best AUROC: {self.best_auroc:.4f}")
    
    def save_checkpoint(self, filename: str, epoch: int, auroc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'anchors': self.learnable_anchors.anchors.data.cpu(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auroc': auroc,
            'history': self.history
        }
        
        torch.save(checkpoint, self.save_dir / filename)


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override from args
    if args.init_from:
        config['learnable_anchors']['init_from'] = args.init_from
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("LEARNABLE ANCHORS TRAINING WITH CAM LOSS")
    print("="*80)
    print(f"Initializing from: {config['learnable_anchors']['init_from']}")
    print(f"Device: {device}")
    
    # Create save directory
    save_dir = Path(config['output_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load data
    data_root = config['data']['data_root']
    train_paths, val_paths, val_labels, val_masks, test_paths, test_labels, test_masks = load_dataset_paths(data_root)
    
    preprocessor = BMADPreprocessor(target_size=tuple(config['data']['target_size']))
    
    train_dataset = BMADDataset(train_paths, [0]*len(train_paths), None, preprocessor, augment=True, is_training=True)
    val_dataset = BMADDataset(val_paths, val_labels, val_masks, preprocessor, augment=False, is_training=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    print(f"\nDataset: {len(train_paths)} train, {len(val_paths)} val")
    
    # Load initial anchors from previous experiment
    print("\n" + "="*80)
    print("LOADING INITIAL ANCHORS")
    print("="*80)
    
    init_exp_dir = Path(config['learnable_anchors']['init_from'])
    anchor_path = init_exp_dir / 'anchor_embeddings.pt'
    
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")
    
    anchor_data = torch.load(anchor_path, map_location='cpu', weights_only=False)
    if isinstance(anchor_data, dict):
        initial_anchors = anchor_data.get('global', anchor_data.get('anchor_global'))
    else:
        initial_anchors = anchor_data
    
    print(f"✓ Loaded anchors: {initial_anchors.shape}")
    
    # Create backbone and model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=config['model']['projection_dim'],
        pretrained=True
    ).to(device)
    
    # Create learnable anchors
    learnable_anchors = LearnableAnchors(
        initial_anchors=initial_anchors,
        freeze=config['learnable_anchors'].get('freeze_anchors', False)
    ).to(device)
    
    # Create model (will use fixed anchors initially, we'll update them during training)
    model = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=initial_anchors,
        anchor_dense_embeddings=None,
        distance_metric=config['loss']['distance_metric']
    ).to(device)
    
    # Create CAM loss
    print("\n" + "="*80)
    print("CREATING CAM LOSS")
    print("="*80)
    
    criterion = CAMLoss(
        lambda_attractor=config['learnable_anchors']['lambda_attractor'],
        lambda_repeller=config['learnable_anchors']['lambda_repeller'],
        lambda_norm=config['learnable_anchors']['lambda_norm'],
        margin=config['learnable_anchors']['margin'],
        min_norm=config['learnable_anchors']['min_norm'],
        distance_metric=config['loss']['distance_metric']
    )
    
    # Create optimizer (optimize both projection head AND anchors)
    print("\n" + "="*80)
    print("CREATING OPTIMIZER")
    print("="*80)
    
    trainable_params = list(model.parameters()) + list(learnable_anchors.parameters())
    trainable_params = [p for p in trainable_params if p.requires_grad]
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['lr'] * 0.01
    )
    
    # Train
    trainer = LearnableAnchorTrainer(
        model=model,
        learnable_anchors=learnable_anchors,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        distance_metric=config['loss']['distance_metric'],
        use_amp=config['training']['use_amp'],
        log_interval=config['training']['log_interval'],
        val_interval=config['training']['val_interval']
    )
    
    trainer.train(
        num_epochs=config['training']['epochs'],
        scheduler=scheduler,
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    print("\n✓ Learnable anchor training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--init-from', type=str, help='Experiment dir to init anchors from')
    
    args = parser.parse_args()
    main(args)
