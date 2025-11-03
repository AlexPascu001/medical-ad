"""
Load a trained checkpoint and generate training curves without retraining
"""

import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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
    n_val_points = len(history['val_image_auroc'])
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


def main(args):
    """Load checkpoint and generate plots"""
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Load training history
    history_path = checkpoint_dir / 'training_history.json'
    if not history_path.exists():
        print(f"Error: Training history not found at {history_path}")
        return
    
    print(f"Loading training history from {history_path}")
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # Generate plots
    print("Generating training curves...")
    plot_training_curves(history, checkpoint_dir, val_interval=args.val_interval)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total epochs: {len(history['epochs'])}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    if history['val_image_auroc']:
        best_auroc = max(history['val_image_auroc'])
        print(f"Best validation Image AUROC: {best_auroc:.4f}")
    if history['val_pixel_auroc']:
        best_pixel = max(history['val_pixel_auroc'])
        print(f"Best validation Pixel AUROC: {best_pixel:.4f}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate plots from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='experiments/bmad_fixed',
                        help='Path to checkpoint directory')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validation interval used during training')
    
    args = parser.parse_args()
    main(args)
