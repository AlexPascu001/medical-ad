"""
Compare multiple anchor strategies with consistent t-SNE visualization

This script ensures fair comparison by:
1. Using the same sample indices across all experiments
2. Fitting t-SNE ONCE on combined data from all strategies
3. Creating side-by-side visualizations with identical coordinate systems
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse
import yaml
from typing import Dict, Tuple, List
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def extract_embeddings_for_samples(
    model: AnomalyDetector,
    dataset: BMADDataset,
    sample_indices: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """Extract embeddings for specific sample indices"""
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            outputs = model(image)
            embeddings.append(outputs['global_feat'].cpu().numpy())
    
    return np.vstack(embeddings)


def plot_single_strategy(
    ax,
    anchor_2d: np.ndarray,
    normal_2d: np.ndarray,
    anomaly_2d: np.ndarray,
    strategy_name: str,
    show_legend: bool = False
):
    """Plot embeddings for a single strategy"""
    n_anchors = len(anchor_2d)
    n_normal = len(normal_2d)
    n_anomaly = len(anomaly_2d)
    
    # Plot normal images (small, light blue)
    ax.scatter(
        normal_2d[:, 0], normal_2d[:, 1],
        c='lightblue', s=20, alpha=0.5,
        label=f'Normal (n={n_normal})',
        edgecolors='none'
    )
    
    # Plot anomaly images (small, red)
    if n_anomaly > 0:
        ax.scatter(
            anomaly_2d[:, 0], anomaly_2d[:, 1],
            c='red', s=20, alpha=0.6,
            label=f'Anomaly (n={n_anomaly})',
            edgecolors='darkred', linewidths=0.5
        )
    
    # Plot anchors (large, gold stars)
    ax.scatter(
        anchor_2d[:, 0], anchor_2d[:, 1],
        c='gold', s=300, alpha=1.0,
        marker='*',
        label=f'Anchors (K={n_anchors})',
        edgecolors='black', linewidths=1.5
    )
    
    # Annotate anchors
    for i, (x, y) in enumerate(anchor_2d):
        ax.annotate(
            f'{i}',
            (x, y),
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center'
        )
    
    ax.set_title(strategy_name, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc='best', fontsize=9, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser(description='Compare anchor strategies with consistent t-SNE')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints to compare')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                       help='Names for each strategy (same order as checkpoints)')
    parser.add_argument('--n-normal', type=int, default=500,
                       help='Number of normal images to visualize')
    parser.add_argument('--n-anomaly', type=int, default=100,
                       help='Number of anomaly images to visualize')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity')
    parser.add_argument('--output', type=str, default='project/experiments/anchor_comparison.png',
                       help='Output path for comparison figure')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--indices-file', type=str, default='cache/tsne_indices.pkl',
                       help='Path to save/load consistent sample indices')
    parser.add_argument('--force-new-indices', action='store_true',
                       help='Force generation of new sample indices')
    
    args = parser.parse_args()
    
    if len(args.checkpoints) != len(args.names):
        raise ValueError("Number of checkpoints must match number of names")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n{'='*80}")
    print(f"COMPARING {len(args.checkpoints)} ANCHOR STRATEGIES")
    print(f"{'='*80}")
    
    # Load first model to get data config
    print(f"\nLoading first checkpoint to get data config...")
    _, config = load_model_and_anchors(args.checkpoints[0], device)
    
    # Load data
    print("Loading datasets...")
    data_root = Path(config['data']['data_root'])
    if not data_root.is_absolute():
        checkpoint_dir = Path(args.checkpoints[0]).parent
        data_root = (checkpoint_dir / '..' / '..' / data_root).resolve()
    
    preprocessor = BMADPreprocessor(target_size=tuple(config['data']['target_size']))
    
    # Load validation and test data
    (train_paths, val_images, val_labels, val_masks,
     test_images, test_labels, test_masks) = load_dataset_paths(str(data_root))
    
    # Combine validation and test
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
    
    if indices_path.exists() and not args.force_new_indices:
        print(f"\n✓ Loading consistent sample indices from: {indices_path}")
        with open(indices_path, 'rb') as f:
            saved_indices = pickle.load(f)
        normal_subset_indices = saved_indices['normal']
        anomaly_subset_indices = saved_indices['anomaly']
        print(f"  Using {len(normal_subset_indices)} normal + {len(anomaly_subset_indices)} anomaly samples")
    else:
        print(f"\n→ Generating new sample indices (seed={args.seed})...")
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
        
        with open(indices_path, 'wb') as f:
            pickle.dump({
                'normal': normal_subset_indices,
                'anomaly': anomaly_subset_indices,
                'seed': args.seed,
                'n_normal': len(normal_subset_indices),
                'n_anomaly': len(anomaly_subset_indices)
            }, f)
        print(f"✓ Saved indices to: {indices_path}")
    
    # Extract embeddings for all strategies
    all_strategy_data = []
    
    for checkpoint_path, strategy_name in zip(args.checkpoints, args.names):
        print(f"\n{'='*80}")
        print(f"Processing: {strategy_name}")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Load model
        model, _ = load_model_and_anchors(checkpoint_path, device)
        
        # Extract anchor embeddings
        with torch.no_grad():
            anchor_embeddings = model._get_projected_anchors()[0].cpu().numpy()
        
        # Extract sample embeddings
        print(f"Extracting embeddings for {len(normal_subset_indices)} normal samples...")
        normal_embeddings = extract_embeddings_for_samples(
            model, combined_dataset, normal_subset_indices, device
        )
        
        print(f"Extracting embeddings for {len(anomaly_subset_indices)} anomaly samples...")
        if len(anomaly_subset_indices) > 0:
            anomaly_embeddings = extract_embeddings_for_samples(
                model, combined_dataset, anomaly_subset_indices, device
            )
        else:
            anomaly_embeddings = np.array([]).reshape(0, normal_embeddings.shape[1])
        
        all_strategy_data.append({
            'name': strategy_name,
            'anchor_embeddings': anchor_embeddings,
            'normal_embeddings': normal_embeddings,
            'anomaly_embeddings': anomaly_embeddings
        })
        
        print(f"✓ Extracted: {len(anchor_embeddings)} anchors, {len(normal_embeddings)} normal, {len(anomaly_embeddings)} anomaly")
    
    # Combine ALL embeddings from ALL strategies for a single t-SNE fit
    print(f"\n{'='*80}")
    print("FITTING SINGLE t-SNE ON COMBINED DATA")
    print(f"{'='*80}")
    
    all_embeddings_list = []
    strategy_offsets = []
    
    for data in all_strategy_data:
        offset_start = len(all_embeddings_list)
        
        all_embeddings_list.extend(data['anchor_embeddings'])
        all_embeddings_list.extend(data['normal_embeddings'])
        all_embeddings_list.extend(data['anomaly_embeddings'])
        
        n_anchors = len(data['anchor_embeddings'])
        n_normal = len(data['normal_embeddings'])
        n_anomaly = len(data['anomaly_embeddings'])
        
        strategy_offsets.append({
            'name': data['name'],
            'start': offset_start,
            'anchor_range': (offset_start, offset_start + n_anchors),
            'normal_range': (offset_start + n_anchors, offset_start + n_anchors + n_normal),
            'anomaly_range': (offset_start + n_anchors + n_normal, offset_start + n_anchors + n_normal + n_anomaly)
        })
    
    all_embeddings = np.vstack(all_embeddings_list)
    
    print(f"Total embeddings: {len(all_embeddings)} ({all_embeddings.shape[1]}-dimensional)")
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, len(all_embeddings) - 1),
        random_state=args.seed,
        max_iter=1000,
        verbose=1
    )
    
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    print(f"\n✓ t-SNE complete! KL divergence: {tsne.kl_divergence_:.3f}")
    
    # Create comparison figure
    print(f"\n{'='*80}")
    print("CREATING COMPARISON VISUALIZATION")
    print(f"{'='*80}")
    
    n_strategies = len(all_strategy_data)
    fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))
    
    if n_strategies == 1:
        axes = [axes]
    
    for i, (ax, offset_info) in enumerate(zip(axes, strategy_offsets)):
        anchor_2d = embeddings_2d[offset_info['anchor_range'][0]:offset_info['anchor_range'][1]]
        normal_2d = embeddings_2d[offset_info['normal_range'][0]:offset_info['normal_range'][1]]
        anomaly_2d = embeddings_2d[offset_info['anomaly_range'][0]:offset_info['anomaly_range'][1]]
        
        plot_single_strategy(
            ax,
            anchor_2d,
            normal_2d,
            anomaly_2d,
            offset_info['name'],
            show_legend=(i == n_strategies - 1)  # Only show legend on last plot
        )
    
    fig.suptitle(
        f'Anchor Strategy Comparison (t-SNE, KL={tsne.kl_divergence_:.3f})',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Saved comparison to: {output_path}")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Strategies compared: {', '.join(args.names)}")
    print(f"Same samples: {len(normal_subset_indices)} normal + {len(anomaly_subset_indices)} anomaly")
    print(f"Same t-SNE fit: All strategies use identical 2D coordinate system")
    print(f"Result: Fair, apples-to-apples comparison! ✓")


if __name__ == '__main__':
    main()
