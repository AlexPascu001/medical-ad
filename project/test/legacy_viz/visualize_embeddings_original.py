"""
Visualize anchors and images in the ORIGINAL 384D embedding space (before projection).
This shows where anchors truly are, not where the projection head maps them.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse
import yaml
from typing import Dict, Tuple
import pickle

from model import DINOv3Backbone, AnomalyDetector
from data import BMADDataset, BMADPreprocessor
from main import load_dataset_paths


def load_model_and_anchors(checkpoint_path: str, device: torch.device) -> Tuple[AnomalyDetector, Dict]:
    """Load trained model and anchor information"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config from experiment directory
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load anchors
    anchor_path = checkpoint_dir / 'anchor_embeddings.pt'
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor embeddings not found: {anchor_path}")
    
    anchor_data = torch.load(anchor_path, map_location=device, weights_only=False)
    
    # Handle different anchor data formats
    if isinstance(anchor_data, dict):
        anchor_global = anchor_data.get('global', anchor_data.get('anchor_global', None))
        anchor_dense = anchor_data.get('dense', anchor_data.get('anchor_dense', None))
    else:
        # If it's just a tensor, assume it's the global embeddings
        anchor_global = anchor_data
        anchor_dense = None
    
    if anchor_global is None:
        raise ValueError(f"Could not find global anchor embeddings in {anchor_path}")
    
    # Create backbone
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=config['model']['projection_dim'],
        pretrained=True
    ).to(device)
    
    # Create model
    model = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_global,
        anchor_dense_embeddings=anchor_dense,
        distance_metric=config['loss'].get('distance_metric', 'euclidean')
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, config


def extract_original_embeddings(
    model: AnomalyDetector,
    dataset: BMADDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Extract embeddings from the ORIGINAL 384D space (before projection head).
    
    Returns:
        embeddings: (N, 384) numpy array - original DINOv3 embeddings
    """
    embeddings_list = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Get batch
            batch_images = []
            for idx in batch_indices:
                sample = dataset[idx]
                batch_images.append(sample['image'])
            
            batch_images = torch.stack(batch_images).to(device)
            
            # Extract features from backbone BEFORE projection
            features = model.backbone.backbone.forward_features(batch_images)
            
            # DINOv3: features shape (B, N_tokens, D)
            # Token order: [CLS, REG1, REG2, REG3, REG4, PATCH1, ...]
            cls_token = features[:, 0]  # (B, 384) - original embedding
            
            # Normalize (same as backbone does)
            cls_token = torch.nn.functional.normalize(cls_token, dim=1)
            
            embeddings_list.append(cls_token.cpu().numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    return embeddings


def visualize_tsne(
    anchor_embeddings: np.ndarray,
    normal_embeddings: np.ndarray,
    anomaly_embeddings: np.ndarray,
    save_path: str,
    space_name: str,
    perplexity: int = 30,
    random_state: int = 42
):
    """
    Create t-SNE visualization of embeddings
    
    Args:
        anchor_embeddings: (K, D) anchor embeddings
        normal_embeddings: (N_normal, D) normal image embeddings
        anomaly_embeddings: (N_anomaly, D) anomaly image embeddings
        save_path: Path to save figure
        space_name: Name of the space (e.g., "384D Original Space")
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
    """
    print(f"\n{'='*80}")
    print(f"t-SNE VISUALIZATION - {space_name}")
    print(f"{'='*80}")
    
    # Combine all embeddings
    all_embeddings = np.vstack([
        anchor_embeddings,
        normal_embeddings,
        anomaly_embeddings
    ])
    
    n_anchors = len(anchor_embeddings)
    n_normal = len(normal_embeddings)
    n_anomaly = len(anomaly_embeddings)
    
    print(f"Embeddings:")
    print(f"  Anchors: {n_anchors}")
    print(f"  Normal: {n_normal}")
    print(f"  Anomaly: {n_anomaly}")
    print(f"  Total: {len(all_embeddings)}")
    print(f"  Dimension: {all_embeddings.shape[1]}")
    
    # Run t-SNE
    print(f"\nRunning t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(all_embeddings) - 1),
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split back into groups
    anchor_2d = embeddings_2d[:n_anchors]
    normal_2d = embeddings_2d[n_anchors:n_anchors+n_normal]
    anomaly_2d = embeddings_2d[n_anchors+n_normal:]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot normal images (small, light blue)
    ax.scatter(
        normal_2d[:, 0], normal_2d[:, 1],
        c='lightblue', s=30, alpha=0.5,
        label=f'Normal (n={n_normal})',
        edgecolors='none'
    )
    
    # Plot anomaly images (small, red)
    if n_anomaly > 0:
        ax.scatter(
            anomaly_2d[:, 0], anomaly_2d[:, 1],
            c='red', s=30, alpha=0.6,
            label=f'Anomaly (n={n_anomaly})',
            edgecolors='darkred', linewidths=0.5
        )
    
    # Plot anchors (large, gold stars)
    ax.scatter(
        anchor_2d[:, 0], anchor_2d[:, 1],
        c='gold', s=400, alpha=1.0,
        marker='*',
        label=f'Anchors (K={n_anchors})',
        edgecolors='black', linewidths=2
    )
    
    # Annotate anchors
    for i, (x, y) in enumerate(anchor_2d):
        ax.annotate(
            f'{i}',
            (x, y),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center'
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f'Embedding Space Visualization - {space_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.close()
    
    # Also create a zoomed version
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot with smaller range around anchors
    anchor_center = anchor_2d.mean(axis=0)
    anchor_range = np.abs(anchor_2d - anchor_center).max() * 2.5
    
    # Plot normal images
    ax.scatter(
        normal_2d[:, 0], normal_2d[:, 1],
        c='lightblue', s=40, alpha=0.6,
        label=f'Normal (n={n_normal})',
        edgecolors='blue', linewidths=0.3
    )
    
    # Plot anomaly images
    if n_anomaly > 0:
        ax.scatter(
            anomaly_2d[:, 0], anomaly_2d[:, 1],
            c='red', s=40, alpha=0.7,
            label=f'Anomaly (n={n_anomaly})',
            edgecolors='darkred', linewidths=0.5
        )
    
    # Plot anchors
    ax.scatter(
        anchor_2d[:, 0], anchor_2d[:, 1],
        c='gold', s=500, alpha=1.0,
        marker='*',
        label=f'Anchors (K={n_anchors})',
        edgecolors='black', linewidths=2
    )
    
    # Annotate anchors
    for i, (x, y) in enumerate(anchor_2d):
        ax.annotate(
            f'{i}',
            (x, y),
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='center'
        )
    
    ax.set_xlim(anchor_center[0] - anchor_range, anchor_center[0] + anchor_range)
    ax.set_ylim(anchor_center[1] - anchor_range, anchor_center[1] + anchor_range)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f'Embedding Space - Zoomed on Anchors - {space_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    zoomed_path = save_path.replace('.png', '_zoomed.png')
    plt.savefig(zoomed_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved zoomed visualization to: {zoomed_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings in ORIGINAL 384D space (before projection)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--n-normal', type=int, default=500,
                       help='Number of normal images to visualize')
    parser.add_argument('--n-anomaly', type=int, default=100,
                       help='Number of anomaly images to visualize')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: same as checkpoint dir)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--indices-file', type=str, default='cache/tsne_indices.pkl',
                       help='Path to save/load consistent sample indices')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and config
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_model_and_anchors(args.checkpoint, device)
    
    # Load data
    print("\nLoading datasets...")
    data_root = Path(config['data']['data_root'])
    if not data_root.is_absolute():
        # Config paths are relative to workspace root
        if str(data_root).startswith('..'):
            data_root = Path(str(data_root).replace('..', '.'))
        data_root = Path.cwd() / data_root
        data_root = data_root.resolve()
    
    preprocessor = BMADPreprocessor(target_size=tuple(config['data']['target_size']))
    
    # Load validation and test data
    (train_paths, val_images, val_labels, val_masks,
     test_images, test_labels, test_masks) = load_dataset_paths(str(data_root))
    
    # Combine validation and test for sampling
    all_images = val_images + test_images
    all_labels = val_labels + test_labels
    all_masks = (val_masks + test_masks) if val_masks and test_masks else None
    
    combined_dataset = BMADDataset(
        image_paths=all_images,
        labels=all_labels,
        mask_paths=all_masks,
        preprocessor=preprocessor,
        augment=False,
        is_training=False
    )
    
    # Load or generate consistent sample indices
    indices_path = Path(args.indices_file)
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    
    normal_indices = [i for i, label in enumerate(all_labels) if label == 0]
    anomaly_indices = [i for i, label in enumerate(all_labels) if label == 1]
    
    if indices_path.exists():
        print(f"\n✓ Loading consistent sample indices from: {indices_path}")
        with open(indices_path, 'rb') as f:
            saved_indices = pickle.load(f)
        normal_subset_indices = saved_indices['normal']
        anomaly_subset_indices = saved_indices['anomaly']
        print(f"  Using {len(normal_subset_indices)} normal + {len(anomaly_subset_indices)} anomaly samples")
        print(f"  This ensures fair comparison across all experiments!")
    else:
        print(f"\n→ Sample indices file not found. Using default sampling.")
        np.random.seed(args.seed)
        normal_subset_indices = np.random.choice(
            normal_indices, 
            size=min(args.n_normal, len(normal_indices)), 
            replace=False
        )
        if len(anomaly_indices) > 0:
            anomaly_subset_indices = np.random.choice(
                anomaly_indices, 
                size=min(args.n_anomaly, len(anomaly_indices)), 
                replace=False
            )
        else:
            anomaly_subset_indices = np.array([], dtype=int)
    
    # Extract embeddings in ORIGINAL 384D space
    print(f"\nExtracting embeddings in ORIGINAL 384D space (before projection)...")
    print(f"  Normal images: {len(normal_subset_indices)}")
    normal_embeddings = extract_original_embeddings(
        model, combined_dataset, normal_subset_indices, device
    )
    
    print(f"  Anomaly images: {len(anomaly_subset_indices)}")
    if len(anomaly_subset_indices) > 0:
        anomaly_embeddings = extract_original_embeddings(
            model, combined_dataset, anomaly_subset_indices, device
        )
    else:
        anomaly_embeddings = np.array([]).reshape(0, 384)
    
    # Get anchor embeddings in ORIGINAL 384D space (before projection)
    print(f"  Anchors: {model.n_anchors}")
    with torch.no_grad():
        # Get anchors before projection
        original_anchors = model.anchor_global.cpu().numpy()
        # Normalize (same as images)
        original_anchors = original_anchors / np.linalg.norm(original_anchors, axis=1, keepdims=True)
    
    print(f"\n✓ Extracted embeddings:")
    print(f"  Normal: {normal_embeddings.shape}")
    print(f"  Anomaly: {anomaly_embeddings.shape}")
    print(f"  Anchors: {original_anchors.shape}")
    
    # Create visualization
    if args.output is None:
        output_path = Path(args.checkpoint).parent / 'tsne_original_384d.png'
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_tsne(
        anchor_embeddings=original_anchors,
        normal_embeddings=normal_embeddings,
        anomaly_embeddings=anomaly_embeddings,
        save_path=str(output_path),
        space_name="384D Original Space (Before Projection)",
        perplexity=args.perplexity,
        random_state=args.seed
    )
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNote: This shows the ORIGINAL 384D embedding space.")
    print(f"Anchors should be well-separated here (if using fixed anchors).")
    print(f"Compare with the 128D projected space to see the collapse effect.")


if __name__ == '__main__':
    main()
