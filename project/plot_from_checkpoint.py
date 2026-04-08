"""
Load a trained experiment checkpoint and generate visualizations without retraining.

Capabilities:
1) Plot training curves from training_history.json
2) Rebuild model from config + anchors and load best_model.pth (or custom checkpoint)
3) Generate scalable train/test sample visualizations from checkpoint weights
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

from data import create_dataloaders
from main import load_dataset_paths
from model import DINOv3Backbone, AnomalyDetector
from train import Trainer


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


def plot_training_curves(history, save_dir, val_interval=1):
    """Plot training curves from history - simplified version"""
    sns.set_style('whitegrid')
    
    # Determine number of validation points
    n_val_points = len(history.get('val_image_auroc', []))
    if n_val_points == 0:
        print("No validation data to plot")
        return
    
    # Create validation epoch indices
    val_epochs = list(range(0, len(history['epochs']), val_interval))[:n_val_points]
    
    # Create figure with 2x2 subplots (simplified)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Total Loss (train + val)
    ax = axes[0, 0]
    ax.plot(to_numpy(history['epochs']), to_numpy(history['train_loss']), 'b-', label='Train', linewidth=2, alpha=0.7)
    if history['val_loss']:
        ax.plot(to_numpy(val_epochs), to_numpy(history['val_loss']), 'r-', label='Val', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Attractor Loss
    ax = axes[0, 1]
    ax.plot(to_numpy(history['epochs']), to_numpy(history['train_loss_attract']), 'b-', label='Train', linewidth=2, alpha=0.7)
    if history['val_loss_attract']:
        ax.plot(to_numpy(val_epochs), to_numpy(history['val_loss_attract']), 'r-', label='Val', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Attractor Loss', fontsize=11)
    ax.set_title('Attractor Loss (Pull to Anchors)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Image AUROC
    ax = axes[1, 0]
    val_auroc_np = to_numpy(history['val_image_auroc'])
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
    if history['val_pixel_auroc'] and len(history['val_pixel_auroc']) > 0:
        val_pixel_np = to_numpy(history['val_pixel_auroc'])
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
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training curves: {save_path}")


def _history_is_empty(history: dict) -> bool:
    """Return True if stage-1 training history is effectively empty."""
    return len(history.get('epochs', [])) == 0 and len(history.get('train_loss', [])) == 0


def load_history_with_fallback(checkpoint_dir: Path, checkpoint_name: str) -> Optional[dict]:
    """Load history from JSON; if empty/missing, fallback to history embedded in checkpoint."""
    history_path = checkpoint_dir / 'training_history.json'
    history = None

    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

    if history is not None and not _history_is_empty(history):
        return history

    # Fallback to checkpoint history
    ckpt_candidates = [checkpoint_dir / checkpoint_name, checkpoint_dir / 'best_model.pth', checkpoint_dir / 'best_stage2_model.pth']
    for ckpt in ckpt_candidates:
        if ckpt.exists():
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            ckpt_history = checkpoint.get('history', None)
            if isinstance(ckpt_history, dict) and not _history_is_empty(ckpt_history):
                print(f"Using history fallback from checkpoint: {ckpt.name}")
                return ckpt_history

    return history


def build_model_from_experiment(config: dict, anchor_data: dict, device: torch.device) -> AnomalyDetector:
    """Rebuild model architecture exactly from experiment config + stored anchors."""
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    multi_scale_indices = config['model'].get('multi_scale_indices', [2, 5, 8, 11])
    projection_dim = config['model'].get('projection_dim', None)

    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=projection_dim,
        pretrained=True,
        multi_scale_indices=multi_scale_indices if use_pixel_decoder else None
    ).to(device)

    target_size = tuple(config['data']['target_size'])
    learnable_anchors = config['anchor'].get('learnable', False)
    use_embedding_space = config['anchor'].get('use_embedding_space', False)
    reproject_anchors = config['anchor'].get('reproject_anchors', False)
    use_decoupled = use_embedding_space and (not reproject_anchors) and ('anchor_geometric' in anchor_data)

    anchor_global = anchor_data.get('anchor_global', None)
    anchor_dense = anchor_data.get('anchor_dense', None)

    if use_decoupled:
        model = AnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_global,
            anchor_dense_embeddings=anchor_dense,
            distance_metric=config['loss']['distance_metric'],
            learnable_anchors=learnable_anchors,
            use_pixel_decoder=use_pixel_decoder,
            decoder_hidden_dim=config['model'].get('decoder_hidden_dim', 256),
            target_size=target_size,
            anchor_semantic_embeddings=anchor_data['anchor_semantic'],
            anchor_geometric_targets=anchor_data['anchor_geometric'],
            use_decoupled_anchors=True
        ).to(device)
    else:
        if use_embedding_space:
            anchors_already_projected = False
        elif projection_dim:
            anchors_already_projected = True
        else:
            anchors_already_projected = False

        model = AnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_global,
            anchor_dense_embeddings=anchor_dense,
            distance_metric=config['loss']['distance_metric'],
            learnable_anchors=learnable_anchors,
            use_pixel_decoder=use_pixel_decoder,
            decoder_hidden_dim=config['model'].get('decoder_hidden_dim', 256),
            target_size=target_size,
            anchors_already_projected=anchors_already_projected
        ).to(device)

    return model


def generate_sample_visualizations(
    checkpoint_dir: Path,
    checkpoint_name: str,
    device: torch.device,
    train_max_samples: int,
    train_max_lines_per_anchor: int,
    train_save_name: str,
    test_save_name: str
):
    """Generate train/test visualization PNGs from an existing checkpoint."""
    config_path = checkpoint_dir / 'config.yaml'
    anchor_path = checkpoint_dir / 'anchor_embeddings.pt'
    ckpt_path = checkpoint_dir / checkpoint_name

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not anchor_path.exists():
        raise FileNotFoundError(f"Missing anchors: {anchor_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_anchor_data = torch.load(anchor_path, map_location='cpu', weights_only=False)
    if isinstance(raw_anchor_data, dict):
        anchor_data = {
            'anchor_global': raw_anchor_data.get('anchor_global', raw_anchor_data.get('global')),
            'anchor_dense': raw_anchor_data.get('anchor_dense', raw_anchor_data.get('dense')),
            'anchor_semantic': raw_anchor_data.get('anchor_semantic', None),
            'anchor_geometric': raw_anchor_data.get('anchor_geometric', None)
        }
    else:
        anchor_data = {
            'anchor_global': raw_anchor_data,
            'anchor_dense': None,
            'anchor_semantic': None,
            'anchor_geometric': None
        }

    model = build_model_from_experiment(config, anchor_data, device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    data_root = config['data'].get('data_root', './data/BraTS2021_slice')
    train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(data_root)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        val_mask_paths=val_mask_paths,
        test_mask_paths=test_mask_paths,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        target_size=tuple(config['data']['target_size'])
    )

    # Create a lightweight trainer shell to reuse visualization methods
    dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
    dummy_optimizer = torch.optim.SGD([dummy_param], lr=0.1)

    viz_trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=dummy_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=checkpoint_dir,
        use_amp=False,
        log_interval=999999,
        val_interval=1,
        fixed_pseudo_labels=False,
        dynamic_reassignment=False,
        reassignment_interval=999999,
        save_checkpoints=False
    )

    print(f"Loaded checkpoint for visualization: {ckpt_path}")
    viz_trainer._visualize_training_samples(
        epoch=0,
        max_samples=train_max_samples,
        max_lines_per_anchor=train_max_lines_per_anchor,
        save_name=train_save_name
    )
    viz_trainer._visualize_test_samples(test_loader=test_loader, save_name=test_save_name)
    print(f"✓ Saved sample visualizations under: {checkpoint_dir / 'visualizations'}")


def main(args):
    """Load checkpoint and generate plots"""
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.plot_curves:
        history = load_history_with_fallback(checkpoint_dir, args.checkpoint_name)
        if history is None:
            history_path = checkpoint_dir / 'training_history.json'
            print(f"Warning: Training history not found at {history_path} (skipping curves)")
        else:
            print(f"Loaded training history for plotting")

            print("Generating training curves...")
            plot_training_curves(history, checkpoint_dir, val_interval=args.val_interval)

            print("\n" + "="*80)
            print("TRAINING SUMMARY")
            print("="*80)
            total_epochs = len(history.get('epochs', []))
            print(f"Total epochs: {total_epochs}")
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            val_image_auroc = history.get('val_image_auroc', [])
            val_pixel_auroc = history.get('val_pixel_auroc', [])

            if len(train_loss) > 0:
                print(f"Final train loss: {train_loss[-1]:.6f}")
            else:
                print("Final train loss: n/a")

            if len(val_loss) > 0:
                print(f"Final val loss: {val_loss[-1]:.6f}")

            if len(val_image_auroc) > 0:
                best_auroc = max(val_image_auroc)
                print(f"Best validation Image AUROC: {best_auroc:.4f}")

            if len(val_pixel_auroc) > 0:
                best_pixel = max(val_pixel_auroc)
                print(f"Best validation Pixel AUROC: {best_pixel:.4f}")

            if len(history.get('stage2_train_loss', [])) > 0:
                print(f"Stage-2 epochs: {len(history.get('stage2_train_loss', []))}")
                stage2_best = history.get('stage2_val_recon_auroc', [])
                if len(stage2_best) > 0:
                    print(f"Best stage-2 reconstruction AUROC: {max(stage2_best):.4f}")

            print("="*80)

    if args.plot_samples:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generate_sample_visualizations(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=args.checkpoint_name,
            device=device,
            train_max_samples=args.train_max_samples,
            train_max_lines_per_anchor=args.train_max_lines_per_anchor,
            train_save_name=args.train_save_name,
            test_save_name=args.test_save_name
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate plots from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='experiments/bmad_fixed',
                        help='Path to checkpoint directory')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validation interval used during training')
    parser.add_argument('--checkpoint-name', type=str, default='best_model.pth',
                        help='Checkpoint file name inside checkpoint-dir (e.g., best_model.pth, best_stage2_model.pth)')
    parser.add_argument('--plot-curves', action='store_true',
                        help='Generate training curves from training_history.json')
    parser.add_argument('--plot-samples', action='store_true',
                        help='Generate train/test sample visualizations from checkpoint weights')
    parser.add_argument('--train-max-samples', type=int, default=2000,
                        help='Max training samples used for train embedding visualization')
    parser.add_argument('--train-max-lines-per-anchor', type=int, default=150,
                        help='Max assignment lines per anchor in train visualization')
    parser.add_argument('--train-save-name', type=str, default='train_best_model',
                        help='Output PNG name for train visualization (without extension)')
    parser.add_argument('--test-save-name', type=str, default='test_best_model',
                        help='Output PNG name for test visualization (without extension)')
    
    args = parser.parse_args()
    if not args.plot_curves and not args.plot_samples:
        args.plot_curves = True
        args.plot_samples = True
    main(args)
