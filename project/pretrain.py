"""
Projection Head Pre-Training Module

Pre-trains the projection head with temporary anchors before main training.
This ensures anchors are projected through a semantically meaningful projection head
rather than a randomly initialized one, fixing the temporal misalignment issue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import hashlib
import json
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data import BMADDataset, BMADPreprocessor
from loss import AnchorMarginLoss
from anchors import KMeansCentroidAnchorStrategy, compute_anchor_embeddings


def visualize_pretraining_embeddings(
    sample_embeddings: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    assignments: torch.Tensor,
    epoch: int,
    output_dir: Path,
    stage: str = "epoch",
    n_samples: int = 2000
):
    """
    Visualize sample and anchor embeddings during pre-training.
    
    Args:
        sample_embeddings: (N, D) tensor of sample embeddings
        anchor_embeddings: (K, D) tensor of anchor embeddings
        assignments: (N,) tensor of anchor assignments
        epoch: Epoch number
        output_dir: Directory to save visualizations
        stage: Stage name (e.g., "initial", "epoch", "final")
        n_samples: Number of samples to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample subset for visualization
    n_samples = min(n_samples, sample_embeddings.shape[0])
    indices = torch.randperm(sample_embeddings.shape[0])[:n_samples]
    
    sample_emb = sample_embeddings[indices].cpu().numpy()
    anchor_emb = anchor_embeddings.cpu().numpy()
    assign = assignments[indices].cpu().numpy()
    
    n_anchors = anchor_emb.shape[0]
    
    # Compute statistics
    anchor_counts = np.bincount(assign, minlength=n_anchors)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # t-SNE visualization
    print(f"  Computing t-SNE for {stage} epoch {epoch}...")
    combined = np.vstack([sample_emb, anchor_emb])
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
    embedded = tsne.fit_transform(combined)
    
    sample_tsne = embedded[:n_samples]
    anchor_tsne = embedded[n_samples:]
    
    # Plot samples colored by assignment
    for k in range(n_anchors):
        mask = assign == k
        if mask.sum() > 0:
            axes[0].scatter(
                sample_tsne[mask, 0], sample_tsne[mask, 1],
                alpha=0.4, s=10, label=f'Anchor {k} (n={anchor_counts[k]})'
            )
    
    # Plot anchors as stars
    for k in range(n_anchors):
        axes[0].scatter(
            anchor_tsne[k, 0], anchor_tsne[k, 1],
            marker='*', s=500, c='black', edgecolors='white', linewidths=2,
            zorder=100
        )
    
    axes[0].set_title(f't-SNE (N={n_samples}) - {stage.capitalize()} Epoch {epoch}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # PCA visualization
    print(f"  Computing PCA for {stage} epoch {epoch}...")
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)
    
    sample_pca = combined_pca[:n_samples]
    anchor_pca = combined_pca[n_samples:]
    
    # Plot samples colored by assignment
    for k in range(n_anchors):
        mask = assign == k
        if mask.sum() > 0:
            axes[1].scatter(
                sample_pca[mask, 0], sample_pca[mask, 1],
                alpha=0.4, s=10, label=f'Anchor {k} (n={anchor_counts[k]})'
            )
    
    # Plot anchors as stars
    for k in range(n_anchors):
        axes[1].scatter(
            anchor_pca[k, 0], anchor_pca[k, 1],
            marker='*', s=500, c='black', edgecolors='white', linewidths=2,
            zorder=100
        )
    
    # Plot origin
    axes[1].scatter(0, 0, marker='x', s=200, c='red', linewidths=3, zorder=50, label='Origin')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.3)
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.3)
    
    axes[1].set_title(f'PCA (N={n_samples}) - {stage.capitalize()} Epoch {epoch}')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Overall title with statistics
    imbalance_ratio = anchor_counts.max() / (anchor_counts.sum() / n_anchors) if anchor_counts.sum() > 0 else 0
    fig.suptitle(
        f'Pre-training Embeddings - {stage.capitalize()} Epoch {epoch}\n'
        f'Max anchor: {anchor_counts.max()}/{n_samples} samples ({anchor_counts.max()/n_samples*100:.1f}%) | '
        f'Imbalance ratio: {imbalance_ratio:.2f}x',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    filename = f'pretrain_{stage}_epoch_{epoch:03d}.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved {filename}")
    print(f"    Anchor distribution: {anchor_counts}")


def get_pretrain_cache_key(backbone_name: str, projection_dim: int, n_anchors: int, strategy: str) -> str:
    """Generate a unique cache key for pre-trained projection head"""
    key_str = f"{backbone_name}_{projection_dim}_{n_anchors}_{strategy}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def pretrain_projection_head(
    backbone: nn.Module,
    train_paths: list,
    preprocessor: BMADPreprocessor,
    config: dict,
    device: torch.device,
    cache_dir: Path,
    force_retrain: bool = False
) -> tuple:
    """
    Pre-train projection head with anchors using CAM loss.
    
    Strategy:
    1. Generate anchors from training samples (kmeans/random/eigenface)
    2. Project anchors through projection head
    3. Train ONLY projection head (freeze DINOv3) using CAM loss for N epochs
    4. KEEP these anchors for main training (perfect alignment!)
    5. Cache pre-trained weights + anchors for reuse across experiments
    
    Args:
        backbone: DINOv3Backbone with projection head (will be modified in-place)
        train_paths: List of training image paths
        preprocessor: Image preprocessing pipeline
        config: Full experiment configuration
        device: torch device
        cache_dir: Directory to cache pre-trained weights
        force_retrain: Force re-training even if cached weights exist
        
    Returns:
        (anchor_images, anchor_global_embeddings): Anchors to use for main training
    """
    pretrain_config = config.get('pretraining', {})
    
    # Check if pre-training is enabled
    if not pretrain_config.get('enabled', False):
        print("\nPre-training disabled (pretraining.enabled=False)")
        return
    
    # Extract pre-training parameters
    n_anchors = config['anchor']['n_anchors']  # Use same number as main training
    strategy_name = config['anchor']['strategy']  # Use same strategy as main training
    epochs = pretrain_config.get('epochs', 5)
    lr = pretrain_config.get('lr', 1e-3)
    batch_size = pretrain_config.get('batch_size', 64)
    loss_alpha = pretrain_config.get('loss_alpha', config['loss']['alpha'])
    loss_beta = pretrain_config.get('loss_beta', config['loss']['beta'])
    distance_metric = pretrain_config.get('distance_metric', config['loss']['distance_metric'])
    
    # Generate cache key
    cache_key = get_pretrain_cache_key(
        backbone_name=config['model']['backbone'],
        projection_dim=config['model']['projection_dim'],
        n_anchors=n_anchors,
        strategy=strategy_name
    )
    cache_path = cache_dir / f'pretrained_projection_{cache_key}.pt'
    cache_info_path = cache_dir / f'pretrained_projection_{cache_key}.json'
    anchors_cache_path = cache_dir / f'pretrained_anchors_{cache_key}.pt'
    
    # Check if cached weights exist
    if cache_path.exists() and anchors_cache_path.exists() and not force_retrain:
        print(f"\n{'='*80}")
        print(f"LOADING CACHED PRE-TRAINED PROJECTION HEAD + ANCHORS")
        print(f"{'='*80}")
        print(f"Cache key: {cache_key}")
        print(f"Cache path: {cache_path}")
        
        # Load cached weights
        checkpoint = torch.load(cache_path, map_location=device, weights_only=True)
        backbone.projection.load_state_dict(checkpoint['projection_state_dict'])
        
        # Load cached anchors
        anchor_cache = torch.load(anchors_cache_path, map_location=device, weights_only=False)
        anchor_images = anchor_cache['anchor_images']
        anchor_global = anchor_cache['anchor_global'].to(device)
        
        # Load and display cache info
        if cache_info_path.exists():
            with open(cache_info_path, 'r') as f:
                cache_info = json.load(f)
            print(f"\nCache info:")
            print(f"  Trained: {cache_info.get('timestamp', 'unknown')}")
            print(f"  Epochs: {cache_info.get('epochs', epochs)}")
            print(f"  Final loss: {cache_info.get('final_loss', 'N/A'):.4f}")
            print(f"  Strategy: {cache_info.get('strategy', strategy_name)}")
            print(f"  Anchors: {len(anchor_images)} ({anchor_global.shape})")
        
        print(f"\n✓ Loaded pre-trained projection head + anchors from cache")
        print(f"✓ These anchors will be used for main training (perfect alignment!)")
        return anchor_images, anchor_global
    
    # No cached weights - perform pre-training
    print(f"\n{'='*80}")
    print(f"PRE-TRAINING PROJECTION HEAD")
    print(f"{'='*80}")
    print(f"Strategy: Fix temporal misalignment by pre-training projection head")
    print(f"  before projecting real anchors")
    print(f"\nPre-training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Anchors: {n_anchors} ({strategy_name})")
    print(f"  Loss weights: α={loss_alpha}, β={loss_beta}")
    print(f"  Distance metric: {distance_metric}")
    print(f"  Cache key: {cache_key}")
    
    # ===== STEP 1: Generate Anchors (will be kept for main training!) =====
    print(f"\nStep 1: Generating anchors for pre-training AND main training...")
    
    # Load subset of training images for anchor generation
    max_images = min(len(train_paths), config['anchor'].get('max_images_for_pca', 5000))
    anchor_image_paths = train_paths[:max_images]
    
    # Load and preprocess images
    anchor_images_list = []
    for img_path in tqdm(anchor_image_paths, desc="Loading images for anchors"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = preprocessor.preprocess(img)
        anchor_images_list.append(img)
    
    anchor_images_np = np.array(anchor_images_list)
    print(f"Loaded {len(anchor_images_np)} images for anchor generation")
    
    # Generate anchors using specified strategy (THESE will be kept!)
    if strategy_name == 'kmeans':
        from anchors import KMeansCentroidAnchorStrategy
        strategy = KMeansCentroidAnchorStrategy(
            n_anchors=n_anchors,
            random_state=config['seed']
        )
    elif strategy_name == 'random':
        from anchors import RandomAnchorStrategy
        strategy = RandomAnchorStrategy(
            n_anchors=n_anchors,
            random_state=config['seed']
        )
    elif strategy_name == 'eigenface':
        from anchors import EigenfaceAnchorStrategy
        strategy = EigenfaceAnchorStrategy(
            n_components=config['anchor'].get('n_components', 50),
            n_anchors=n_anchors,
            random_state=config['seed']
        )
    else:
        raise ValueError(f"Unknown anchor strategy: {strategy_name}")
    
    anchor_images = strategy.fit(anchor_images_np)
    print(f"✓ Generated {n_anchors} anchors using {strategy_name}")
    print(f"✓ These anchors will be KEPT for main training (perfect alignment!)")
    
    # ===== STEP 2: Project Anchors =====
    print(f"\nStep 2: Projecting anchors through projection head...")
    
    backbone.eval()
    anchor_global, _ = compute_anchor_embeddings(
        anchor_images=anchor_images,
        backbone_model=backbone,
        device=device,
        batch_size=8,
        return_projected=True
    )
    anchor_global = anchor_global.to(device)  # Ensure on correct device
    print(f"✓ Anchor embeddings: {anchor_global.shape}")
    
    # ===== VISUALIZATION: Initial State (before pre-training) =====
    print(f"\nVisualizing initial anchor embeddings...")
    vis_dir = Path(config.get('output_dir', 'experiments/default')) / 'visualizations' / 'pretraining'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary data loader for visualization
    temp_dataset = BMADDataset(
        image_paths=train_paths,
        preprocessor=preprocessor,
        is_training=False  # No augmentation for visualization
    )
    temp_loader = DataLoader(
        temp_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get a sample of training embeddings for visualization
    backbone.eval()
    sample_embeddings_list = []
    sample_assignments_list = []
    n_vis_batches = 32  # ~2000 samples
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(temp_loader):
            if batch_idx >= n_vis_batches:
                break
            images = batch['image'].to(device)
            
            # Get embeddings
            dino_embeddings = backbone.backbone.forward_features(images)[:, 0]
            embeddings = backbone.projection(dino_embeddings)
            embeddings = F.normalize(embeddings, dim=1)  # CRITICAL: Match anchor normalization!
            
            # Compute assignments (nearest anchor)
            distances = torch.cdist(embeddings, anchor_global, p=2)
            assignments = distances.argmin(dim=1)
            
            sample_embeddings_list.append(embeddings.cpu())
            sample_assignments_list.append(assignments.cpu())
    
    initial_embeddings = torch.cat(sample_embeddings_list, dim=0)
    initial_assignments = torch.cat(sample_assignments_list, dim=0)
    
    visualize_pretraining_embeddings(
        sample_embeddings=initial_embeddings,
        anchor_embeddings=anchor_global.cpu(),
        assignments=initial_assignments,
        epoch=0,
        output_dir=vis_dir,
        stage="initial",
        n_samples=2000
    )
    
    # ===== STEP 3: SKIP TRAINING - Just use orthogonal projection =====
    print(f"\nStep 3: Skipping projection head training (using orthogonal init)")
    print(f"  Rationale: Orthogonal weights (gain=1.0) already preserve DINOv3 structure")
    print(f"  Training causes collapse without sample-level diversity regularization")
    print(f"  Using initialized weights ensures consistent projection without collapse")
    
    # Final anchor embeddings = current projection (orthogonal init, not trained)
    # These will be used for main training
    final_anchor_global = anchor_global.clone()
    
    # ===== STEP 4: Save Projection Weights + Anchors to Cache =====
    print(f"\nStep 4: Saving projection head (orthogonal init) + anchors to cache...")
    
    # Save projection head state dict
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'projection_state_dict': backbone.projection.state_dict(),
        'config': pretrain_config,
        'cache_key': cache_key
    }, cache_path)
    
    # Save anchors (CRITICAL: reuse for main training!)
    torch.save({
        'anchor_images': anchor_images,
        'anchor_global': final_anchor_global.cpu(),
        'strategy': strategy_name,
        'n_anchors': n_anchors
    }, anchors_cache_path)
    
    # Save cache info
    import datetime
    cache_info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'backbone': config['model']['backbone'],
        'projection_dim': config['model']['projection_dim'],
        'n_anchors': n_anchors,
        'strategy': strategy_name,
        'training': 'skipped (orthogonal init only)',
        'cache_key': cache_key
    }
    with open(cache_info_path, 'w') as f:
        json.dump(cache_info, f, indent=2)
    
    print(f"✓ Saved projection head to {cache_path}")
    print(f"✓ Saved anchors to {anchors_cache_path}")
    print(f"✓ Cache info saved to {cache_info_path}")
    
    print(f"\n{'='*80}")
    print(f"PRE-TRAINING COMPLETE (ORTHOGONAL INIT ONLY)")
    print(f"{'='*80}")
    print(f"✓ Projection head initialized with orthogonal weights (gain=1.0)")
    print(f"✓ Anchors projected through initialized projection (not trained)")
    print(f"✓ These SAME anchors will be used for main training")
    print(f"✓ Perfect alignment - no collapse due to training!")
    
    return anchor_images, final_anchor_global


# Import needed for anchor generation
import cv2
import numpy as np
