"""
Comprehensive Visualization from Existing Checkpoints

DEPRECATED: Use visualize_pipeline.py instead, which provides a unified
pipeline visualization covering all 10 diagnostic steps.
    python visualize_pipeline.py --experiment <experiment_dir>

This script loads a trained model from a checkpoint directory and generates:
1. t-SNE visualization of embeddings (anchors, normal samples, anomaly samples)
2. Training curves (if history is available)
3. Anchor statistics and analysis
4. Model summary and configuration

Usage:
    python visualize_from_checkpoint.py --experiment experiments/bmad_random_k8_l2
    python visualize_from_checkpoint.py --experiment experiments/bmad_eigenface_k8_l2 --n-samples 500
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
import yaml
import json
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

from model import DINOv3Backbone, AnomalyDetector
from data import BMADDataset, BMADPreprocessor, create_dataloaders
from main import load_dataset_paths


def load_experiment(exp_dir: Path, device: torch.device) -> Tuple[AnomalyDetector, dict, dict]:
    """
    Load model, config, and anchor data from an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory (e.g., experiments/bmad_random_k8_l2)
        device: Torch device
    
    Returns:
        model: Loaded AnomalyDetector model
        config: Configuration dictionary
        anchor_data: Dictionary with anchor images, embeddings
    """
    exp_dir = Path(exp_dir)
    
    # Check required files exist
    required_files = ['config.yaml', 'best_model.pth', 'anchor_embeddings.pt']
    for f in required_files:
        if not (exp_dir / f).exists():
            raise FileNotFoundError(f"Required file not found: {exp_dir / f}")
    
    # Load config
    with open(exp_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"Loading Experiment: {exp_dir.name}")
    print(f"{'='*60}")
    print(f"  Anchor Strategy: {config['anchor']['strategy']}")
    print(f"  Number of Anchors: {config['anchor']['n_anchors']}")
    print(f"  Distance Metric: {config['loss']['distance_metric']}")
    print(f"  Learnable Anchors: {config['anchor'].get('learnable', False)}")
    
    # Load anchor data
    anchor_data = torch.load(exp_dir / 'anchor_embeddings.pt', map_location=device, weights_only=False)
    
    if isinstance(anchor_data, dict):
        anchor_global = anchor_data.get('anchor_global', anchor_data.get('global'))
        anchor_dense = anchor_data.get('anchor_dense', anchor_data.get('dense'))
        anchor_images = anchor_data.get('anchor_images', None)
    else:
        anchor_global = anchor_data
        anchor_dense = None
        anchor_images = None
    
    print(f"  Anchor Embeddings Shape: {anchor_global.shape}")
    
    # Create backbone
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=config['model'].get('projection_dim', None),
        pretrained=True
    ).to(device)
    
    # Create model
    model = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_global,
        anchor_dense_embeddings=anchor_dense,
        distance_metric=config['loss'].get('distance_metric', 'euclidean'),
        learnable_anchors=config['anchor'].get('learnable', False)
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(exp_dir / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"  Loaded from Epoch: {checkpoint['epoch']}")
    
    return model, config, {
        'anchor_global': anchor_global,
        'anchor_dense': anchor_dense,
        'anchor_images': anchor_images
    }


def extract_embeddings_with_labels(
    model: AnomalyDetector,
    dataloader,
    device: torch.device,
    n_normal: int = 500,
    n_anomaly: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings from dataloader, separating normal and anomaly samples.
    
    Returns:
        normal_embeddings: (N_normal, D) numpy array
        anomaly_embeddings: (N_anomaly, D) numpy array
        anchor_embeddings: (K, D) numpy array
    """
    model.eval()
    
    normal_embeds = []
    anomaly_embeds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['image'].to(device)
            labels = batch['label']
            
            outputs = model(images, return_dense=False)
            embeddings = outputs['global_feat'].cpu().numpy()
            
            for emb, label in zip(embeddings, labels):
                if label == 0 and len(normal_embeds) < n_normal:
                    normal_embeds.append(emb)
                elif label == 1 and len(anomaly_embeds) < n_anomaly:
                    anomaly_embeds.append(emb)
            
            # Early stop if we have enough
            if len(normal_embeds) >= n_normal and len(anomaly_embeds) >= n_anomaly:
                break
    
    # Get anchor embeddings
    anchor_global, _ = model._get_projected_anchors()
    anchor_embeds = anchor_global.detach().cpu().numpy()
    
    normal_embeds = np.array(normal_embeds) if normal_embeds else np.zeros((0, anchor_embeds.shape[1]))
    anomaly_embeds = np.array(anomaly_embeds) if anomaly_embeds else np.zeros((0, anchor_embeds.shape[1]))
    
    print(f"\nExtracted embeddings:")
    print(f"  Normal: {len(normal_embeds)}")
    print(f"  Anomaly: {len(anomaly_embeds)}")
    print(f"  Anchors: {len(anchor_embeds)}")
    
    return normal_embeds, anomaly_embeds, anchor_embeds


def visualize_tsne(
    normal_embeddings: np.ndarray,
    anomaly_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    save_path: Path,
    title: str = "t-SNE Embedding Visualization",
    perplexity: int = 30,
    random_state: int = 42
):
    """
    Create t-SNE visualization of embeddings with anchors, normal, and anomaly samples.
    """
    print(f"\nGenerating t-SNE visualization...")
    
    # Combine all embeddings
    all_embeddings = np.vstack([
        anchor_embeddings,
        normal_embeddings,
        anomaly_embeddings
    ])
    
    n_anchors = len(anchor_embeddings)
    n_normal = len(normal_embeddings)
    n_anomaly = len(anomaly_embeddings)
    total = len(all_embeddings)
    
    # Adjust perplexity if needed
    perplexity = min(perplexity, total // 3, 50)
    perplexity = max(perplexity, 5)
    
    print(f"  Running t-SNE with perplexity={perplexity}...")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        init='pca'
    )
    
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split back
    anchor_2d = embeddings_2d[:n_anchors]
    normal_2d = embeddings_2d[n_anchors:n_anchors+n_normal]
    anomaly_2d = embeddings_2d[n_anchors+n_normal:]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot normal samples (light blue, small)
    if n_normal > 0:
        ax.scatter(
            normal_2d[:, 0], normal_2d[:, 1],
            c='steelblue', s=25, alpha=0.5,
            label=f'Normal (n={n_normal})',
            edgecolors='none'
        )
    
    # Plot anomaly samples (red, small)
    if n_anomaly > 0:
        ax.scatter(
            anomaly_2d[:, 0], anomaly_2d[:, 1],
            c='crimson', s=30, alpha=0.6,
            label=f'Anomaly (n={n_anomaly})',
            edgecolors='darkred', linewidths=0.5
        )
    
    # Plot anchors (large stars, distinct colors)
    colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
    for i in range(n_anchors):
        ax.scatter(
            anchor_2d[i, 0], anchor_2d[i, 1],
            c=[colors[i]], s=400, marker='*',
            edgecolors='black', linewidths=1.5,
            label=f'Anchor {i}', zorder=10
        )
        # Add anchor number label
        ax.annotate(
            str(i), (anchor_2d[i, 0], anchor_2d[i, 1]),
            fontsize=8, fontweight='bold',
            ha='center', va='center'
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Create legend with custom handling for many anchors
    if n_anchors <= 8:
        ax.legend(loc='best', fontsize=9, framealpha=0.8)
    else:
        # Just show normal/anomaly in legend for many anchors
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], loc='best', fontsize=10, framealpha=0.8)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def visualize_anchor_distances(
    model: AnomalyDetector,
    dataloader,
    device: torch.device,
    save_path: Path,
    n_samples: int = 1000
):
    """
    Visualize distance distributions to anchors for normal vs anomaly samples.
    """
    print(f"\nGenerating anchor distance analysis...")
    
    model.eval()
    
    normal_distances = []
    anomaly_distances = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing distances"):
            images = batch['image'].to(device)
            labels = batch['label']
            
            outputs = model(images, return_dense=False)
            distances = outputs['global_distances']  # (B, K)
            min_distances = distances.min(dim=1)[0].cpu().numpy()  # (B,)
            
            for dist, label in zip(min_distances, labels):
                if label == 0:
                    normal_distances.append(dist)
                else:
                    anomaly_distances.append(dist)
            
            if len(normal_distances) + len(anomaly_distances) >= n_samples:
                break
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution plot
    ax = axes[0]
    if normal_distances:
        ax.hist(normal_distances, bins=50, alpha=0.6, color='steelblue', 
                label=f'Normal (n={len(normal_distances)})', density=True)
    if anomaly_distances:
        ax.hist(anomaly_distances, bins=50, alpha=0.6, color='crimson',
                label=f'Anomaly (n={len(anomaly_distances)})', density=True)
    ax.set_xlabel('Min Distance to Anchor', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distance Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    data_to_plot = []
    labels_to_plot = []
    if normal_distances:
        data_to_plot.append(normal_distances)
        labels_to_plot.append('Normal')
    if anomaly_distances:
        data_to_plot.append(anomaly_distances)
        labels_to_plot.append('Anomaly')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        colors = ['steelblue', 'crimson'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel('Min Distance to Anchor', fontsize=12)
    ax.set_title('Distance Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")
    
    # Print statistics
    if normal_distances:
        print(f"  Normal distances: mean={np.mean(normal_distances):.4f}, std={np.std(normal_distances):.4f}")
    if anomaly_distances:
        print(f"  Anomaly distances: mean={np.mean(anomaly_distances):.4f}, std={np.std(anomaly_distances):.4f}")


def visualize_anchor_images(
    anchor_images: Optional[np.ndarray],
    save_path: Path
):
    """
    Visualize the actual anchor images (if available).
    """
    if anchor_images is None:
        print("  No anchor images available to visualize")
        return
    
    print(f"\nVisualizing anchor images...")
    
    K = len(anchor_images)
    cols = min(4, K)
    rows = (K + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).flatten() if K > 1 else [axes]
    
    for i in range(K):
        axes[i].imshow(anchor_images[i], cmap='gray')
        axes[i].set_title(f'Anchor {i}', fontsize=11, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(K, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Anchor Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def visualize_anchor_separation(
    anchor_embeddings: np.ndarray,
    save_path: Path,
    distance_metric: str = 'euclidean'
):
    """
    Visualize pairwise distances between anchors as a heatmap.
    """
    print(f"\nVisualizing anchor separation...")
    
    K = len(anchor_embeddings)
    
    # Compute pairwise distances
    if distance_metric == 'euclidean':
        from scipy.spatial.distance import cdist
        distances = cdist(anchor_embeddings, anchor_embeddings, metric='euclidean')
    else:  # cosine
        # Normalize and compute cosine distance
        norms = np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
        normalized = anchor_embeddings / (norms + 1e-8)
        similarities = normalized @ normalized.T
        distances = 1 - similarities
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(
        distances,
        annot=True if K <= 10 else False,
        fmt='.3f',
        cmap='viridis',
        square=True,
        ax=ax,
        xticklabels=[f'A{i}' for i in range(K)],
        yticklabels=[f'A{i}' for i in range(K)]
    )
    
    ax.set_title(f'Anchor Pairwise Distances ({distance_metric})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Anchor', fontsize=11)
    ax.set_ylabel('Anchor', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    mask = ~np.eye(K, dtype=bool)
    off_diag = distances[mask]
    print(f"  Min separation: {off_diag.min():.4f}")
    print(f"  Mean separation: {off_diag.mean():.4f}")
    print(f"  Max separation: {off_diag.max():.4f}")
    print(f"  Saved: {save_path}")


def load_training_history(exp_dir: Path) -> Optional[dict]:
    """
    Attempt to load training history from checkpoint.
    """
    checkpoint_path = exp_dir / 'best_model.pth'
    if not checkpoint_path.exists():
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint.get('history', None)


def visualize_training_curves(
    history: dict,
    save_path: Path
):
    """
    Plot training curves from history dictionary.
    """
    print(f"\nVisualizing training curves...")
    
    if not history or 'train_loss' not in history:
        print("  No training history available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history.get('epochs', list(range(len(history['train_loss']))))
    
    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, alpha=0.8)
    if history.get('val_loss'):
        val_epochs = np.linspace(0, max(epochs), len(history['val_loss']))
        ax.plot(val_epochs, history['val_loss'], 'r-', label='Val', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Attractor Loss
    ax = axes[0, 1]
    if history.get('train_loss_attract'):
        ax.plot(epochs, history['train_loss_attract'], 'b-', label='Train', linewidth=2, alpha=0.8)
    if history.get('val_loss_attract'):
        val_epochs = np.linspace(0, max(epochs), len(history['val_loss_attract']))
        ax.plot(val_epochs, history['val_loss_attract'], 'r-', label='Val', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Attractor Loss', fontsize=11)
    ax.set_title('Attractor Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Image AUROC
    ax = axes[1, 0]
    if history.get('val_image_auroc'):
        val_auroc = np.array(history['val_image_auroc'])
        val_epochs = np.linspace(0, max(epochs), len(val_auroc))
        ax.plot(val_epochs, val_auroc, 'g-o', linewidth=2, markersize=4, alpha=0.8)
        best_auroc = val_auroc.max()
        ax.axhline(y=best_auroc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_auroc:.4f}')
        ax.legend(fontsize=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Image AUROC', fontsize=11)
    ax.set_title('Validation Image AUROC', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 4. Pixel AUROC (if available)
    ax = axes[1, 1]
    if history.get('val_pixel_auroc'):
        val_pixel = np.array([x for x in history['val_pixel_auroc'] if x > 0])
        if len(val_pixel) > 0:
            val_epochs = np.linspace(0, max(epochs), len(val_pixel))
            ax.plot(val_epochs, val_pixel, 'purple', linewidth=2, marker='s', markersize=4, alpha=0.8)
            best_pixel = val_pixel.max()
            ax.axhline(y=best_pixel, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_pixel:.4f}')
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Pixel AUROC Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No Pixel AUROC Data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Pixel AUROC', fontsize=11)
    ax.set_title('Validation Pixel AUROC', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.suptitle('Training Progress', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def generate_summary_report(
    exp_dir: Path,
    config: dict,
    anchor_data: dict,
    model: AnomalyDetector,
    save_path: Path
):
    """
    Generate a text summary report of the experiment.
    """
    print(f"\nGenerating summary report...")
    
    lines = []
    lines.append("="*60)
    lines.append(f"EXPERIMENT SUMMARY: {exp_dir.name}")
    lines.append("="*60)
    lines.append("")
    
    # Configuration
    lines.append("CONFIGURATION:")
    lines.append(f"  Anchor Strategy: {config['anchor']['strategy']}")
    lines.append(f"  Number of Anchors: {config['anchor']['n_anchors']}")
    lines.append(f"  Distance Metric: {config['loss']['distance_metric']}")
    lines.append(f"  Learnable Anchors: {config['anchor'].get('learnable', False)}")
    lines.append(f"  Projection Dim: {config['model'].get('projection_dim', 'None')}")
    lines.append(f"  Backbone: {config['model']['backbone']}")
    lines.append("")
    
    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append("MODEL:")
    lines.append(f"  Total Parameters: {n_params:,}")
    lines.append(f"  Trainable Parameters: {n_trainable:,}")
    lines.append("")
    
    # Anchor info
    anchor_global = anchor_data['anchor_global']
    if isinstance(anchor_global, torch.Tensor):
        anchor_global = anchor_global.cpu().numpy()
    
    lines.append("ANCHORS:")
    lines.append(f"  Shape: {anchor_global.shape}")
    lines.append(f"  Norm (mean): {np.linalg.norm(anchor_global, axis=1).mean():.4f}")
    lines.append(f"  Norm (std): {np.linalg.norm(anchor_global, axis=1).std():.4f}")
    lines.append("")
    
    # Training info (if available)
    history = load_training_history(exp_dir)
    if history and 'val_image_auroc' in history:
        val_auroc = np.array(history['val_image_auroc'])
        lines.append("RESULTS:")
        lines.append(f"  Best Image AUROC: {val_auroc.max():.4f}")
        lines.append(f"  Final Image AUROC: {val_auroc[-1]:.4f}")
        if history.get('val_pixel_auroc'):
            val_pixel = [x for x in history['val_pixel_auroc'] if x > 0]
            if val_pixel:
                lines.append(f"  Best Pixel AUROC: {max(val_pixel):.4f}")
    lines.append("")
    lines.append("="*60)
    
    # Write report
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved: {save_path}")
    
    # Also print to console
    print('\n'.join(lines))


def main(args):
    """Main visualization pipeline."""
    exp_dir = Path(args.experiment)
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = exp_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # Load experiment
    model, config, anchor_data = load_experiment(exp_dir, device)
    
    # Load dataset for embedding extraction
    data_root = config['data'].get('data_root', './data/BraTS2021_slice')
    _, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(data_root)
    
    # Create dataloader
    from data import BMADDataset
    target_size = tuple(config['data']['target_size'])
    
    # Use test set for visualization
    test_dataset = BMADDataset(
        image_paths=test_paths,
        labels=test_labels,
        mask_paths=test_mask_paths,
        target_size=target_size
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Extract embeddings
    normal_embeds, anomaly_embeds, anchor_embeds = extract_embeddings_with_labels(
        model=model,
        dataloader=test_loader,
        device=device,
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly
    )
    
    # Generate visualizations
    exp_name = exp_dir.name
    
    # 1. t-SNE
    visualize_tsne(
        normal_embeddings=normal_embeds,
        anomaly_embeddings=anomaly_embeds,
        anchor_embeddings=anchor_embeds,
        save_path=output_dir / f'{exp_name}_tsne.png',
        title=f't-SNE: {exp_name}',
        perplexity=args.perplexity
    )
    
    # 2. Anchor distance analysis
    visualize_anchor_distances(
        model=model,
        dataloader=test_loader,
        device=device,
        save_path=output_dir / f'{exp_name}_distances.png',
        n_samples=args.n_normal + args.n_anomaly
    )
    
    # 3. Anchor images (if available)
    visualize_anchor_images(
        anchor_images=anchor_data.get('anchor_images'),
        save_path=output_dir / f'{exp_name}_anchor_images.png'
    )
    
    # 4. Anchor separation heatmap
    visualize_anchor_separation(
        anchor_embeddings=anchor_embeds,
        save_path=output_dir / f'{exp_name}_anchor_separation.png',
        distance_metric=config['loss']['distance_metric']
    )
    
    # 5. Training curves (if history available)
    history = load_training_history(exp_dir)
    if history:
        visualize_training_curves(
            history=history,
            save_path=output_dir / f'{exp_name}_training_curves.png'
        )
    
    # 6. Summary report
    generate_summary_report(
        exp_dir=exp_dir,
        config=config,
        anchor_data=anchor_data,
        model=model,
        save_path=output_dir / f'{exp_name}_summary.txt'
    )
    
    print(f"\n{'='*60}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate visualizations from trained checkpoint')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to experiment directory (e.g., experiments/bmad_random_k8_l2)')
    parser.add_argument('--n-normal', type=int, default=500,
                        help='Number of normal samples for t-SNE')
    parser.add_argument('--n-anomaly', type=int, default=100,
                        help='Number of anomaly samples for t-SNE')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    
    args = parser.parse_args()
    main(args)
