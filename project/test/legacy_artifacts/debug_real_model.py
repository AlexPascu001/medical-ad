"""
Debug Real Model: Test anchor behavior with actual DINO backbone on real data

Proper training setup:
1. Generate 4 fixed random anchors from training samples
2. Precompute fixed assignments for ALL samples (closest anchor)
3. Train with normal batching through epochs (attractor-only loss since anchors fixed)
4. Visualize ALL samples with lines to their assigned anchors

Key: We want to verify samples assigned to an anchor at init gravitate toward it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import DINOv3Backbone, AnomalyDetector
from loss import AnchorMarginLoss
from data import BMADPreprocessor, BMADDataset


def load_all_training_data(data_root):
    """Load ALL training data paths"""
    data_root = Path(data_root)
    train_dir = data_root / 'train' / 'good'
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training data not found at {train_dir}")
    
    all_paths = sorted([str(p) for p in train_dir.glob('*.png')])
    
    if len(all_paths) == 0:
        raise ValueError("No training images found")
    
    print(f"Found {len(all_paths)} training images")
    return all_paths


def visualize_all_samples(
    embeddings: torch.Tensor,
    anchors: torch.Tensor,
    assignments: torch.Tensor,
    step: int,
    save_dir: Path,
    show_lines: bool = True,
    max_lines_per_anchor: int = 100  # Limit lines for visibility
):
    """
    Visualize ALL samples and anchors with lines from samples to assigned anchors.
    
    Uses both t-SNE and PCA to show the embedding space.
    """
    emb_np = embeddings.detach().cpu().numpy()
    anc_np = anchors.detach().cpu().numpy()
    assign_np = assignments.cpu().numpy()
    
    n_samples = emb_np.shape[0]
    n_anchors = anc_np.shape[0]
    
    print(f"  Visualizing {n_samples} samples with {n_anchors} anchors...")
    
    # Combine for dimensionality reduction
    all_points = np.vstack([emb_np, anc_np])
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Plot 1: t-SNE ===
    ax = axes[0]
    perplexity = min(30, len(all_points) - 1)
    print(f"  Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', max_iter=1000)
    coords_2d = tsne.fit_transform(all_points)
    
    sample_coords = coords_2d[:n_samples]
    anchor_coords = coords_2d[n_samples:]
    
    # Draw lines from samples to their assigned anchors
    if show_lines:
        for k in range(n_anchors):
            mask = assign_np == k
            indices = np.where(mask)[0]
            # Subsample if too many
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
                   fontsize=14, fontweight='bold', ha='center', va='center', zorder=11)
    
    ax.set_title(f'Step {step}: t-SNE (N={n_samples})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # === Plot 2: PCA ===
    ax = axes[1]
    print(f"  Running PCA...")
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(all_points)
    
    sample_pca = coords_pca[:n_samples]
    anchor_pca = coords_pca[n_samples:]
    
    # Draw lines
    if show_lines:
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
                   fontsize=14, fontweight='bold', ha='center', va='center', zorder=11)
    
    # Mark origin
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    ax.scatter([0], [0], c='red', s=100, marker='x', zorder=5, label='Origin')
    
    ax.set_title(f'Step {step}: PCA (N={n_samples})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'All Samples Visualization - Step {step}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'all_samples_step_{step:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_dir / f'all_samples_step_{step:04d}.png'}")


def compute_metrics(embeddings, anchors, assignments):
    """Compute training metrics"""
    n_anchors = anchors.shape[0]
    
    # Distance from each sample to its assigned anchor
    dists_to_assigned = []
    for i in range(len(embeddings)):
        k = assignments[i].item()
        dist = torch.norm(embeddings[i] - anchors[k]).item()
        dists_to_assigned.append(dist)
    
    mean_dist = np.mean(dists_to_assigned)
    
    # Cosine similarity to assigned anchor
    emb_norm = F.normalize(embeddings, dim=1)
    anc_norm = F.normalize(anchors, dim=1)
    cos_sims = []
    for i in range(len(embeddings)):
        k = assignments[i].item()
        cos_sim = (emb_norm[i] @ anc_norm[k]).item()
        cos_sims.append(cos_sim)
    mean_cos_sim = np.mean(cos_sims)
    
    # Anchor statistics
    anc_norms = anchors.norm(dim=1).cpu().tolist()
    
    # Pairwise anchor distances
    anc_dists = torch.cdist(anchors.unsqueeze(0), anchors.unsqueeze(0))[0]
    anc_min_dist = anc_dists[anc_dists > 0].min().item() if (anc_dists > 0).any() else 0
    
    return {
        'mean_dist_to_assigned': mean_dist,
        'mean_cos_sim': mean_cos_sim,
        'anc_norms': anc_norms,
        'anc_min_dist': anc_min_dist
    }


def run_real_model_debug(
    data_root: str = '../data/BraTS2021_slice',
    n_anchors: int = 4,
    n_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    proj_dim: int = 128,
    alpha: float = 1.0,  # Attractor weight
    beta: float = 0.0,   # Repeller weight (0 since anchors are fixed)
    margin: float = 1.0,
    distance_metric: str = 'euclidean',
    output_dir: str = './debug_output/real_model_proper',
    seed: int = 42,
    visualize_every: int = 1,  # Visualize every N epochs
    max_lines: int = 200,      # Max lines per anchor in visualization
    visualize_batch_only: bool = False  # If True, only visualize one batch (not all samples)
):
    """
    Proper training setup:
    1. Load ALL training samples
    2. Generate 4 fixed random anchors from samples
    3. Precompute fixed assignments (each sample -> nearest anchor)
    4. Train with normal epoch-based training
    5. Only attractor loss (anchors are fixed, not learnable)
    6. Visualize ALL samples at intervals
    """
    print("="*80)
    print("REAL MODEL DEBUG: Proper Training Setup")
    print("="*80)
    
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # === 1. Load ALL training data ===
    print(f"\n1. Loading ALL training data...")
    all_paths = load_all_training_data(data_root)
    n_samples = len(all_paths)
    
    preprocessor = BMADPreprocessor(target_size=(240, 240))
    dataset = BMADDataset(
        image_paths=all_paths,
        labels=[0] * n_samples,
        mask_paths=None,
        preprocessor=preprocessor,
        augment=False
    )
    
    # === 2. Create backbone and extract RAW embeddings for all samples ===
    print(f"\n2. Creating model and extracting embeddings for all {n_samples} samples...")
    
    backbone = DINOv3Backbone(
        model_name='vit_small_patch16_dinov3.lvd1689m',
        freeze_backbone=True,
        projection_dim=proj_dim,
        pretrained=True
    ).to(device)
    
    # Extract raw CLS tokens (384D) for anchor initialization and assignment computation
    raw_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"   Extracting raw backbone features...")
    all_raw_embeddings = []
    with torch.no_grad():
        for batch in tqdm(raw_dataloader, desc="   Extracting"):
            images = batch['image'].to(device)
            features = backbone.backbone.forward_features(images)
            cls_token = features[:, 0]  # (B, 384)
            all_raw_embeddings.append(cls_token.cpu())
    all_raw_embeddings = torch.cat(all_raw_embeddings, dim=0)  # (N, 384)
    print(f"   Raw embeddings shape: {all_raw_embeddings.shape}")
    
    # === 3. Initialize 4 random anchors DIRECTLY in projected 128D space ===
    print(f"\n3. Initializing {n_anchors} random anchors in PROJECTED space...")
    anchor_indices = torch.randperm(n_samples)[:n_anchors]
    print(f"   Anchor indices: {anchor_indices.tolist()}")
    
    # Project the anchor samples through the projection head ONCE
    # These will be our FIXED anchors in 128D - they won't be re-projected
    anchors_raw = all_raw_embeddings[anchor_indices].to(device)  # (K, 384)
    with torch.no_grad():
        anchors_projected_fixed = backbone.projection(anchors_raw)  # (K, 128)
    print(f"   Anchors shape (projected): {anchors_projected_fixed.shape}")
    print(f"   Anchor norms: {anchors_projected_fixed.norm(dim=1).cpu().tolist()}")
    
    # These anchors are FIXED in 128D space - they don't go through projection again
    # We'll use them directly in the loss, not through the model
    
    # === 4. Project ALL samples to get initial projected embeddings ===
    print(f"\n4. Projecting all samples and computing assignments...")
    
    all_projected = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_raw = all_raw_embeddings[i:i+batch_size].to(device)
            projected = backbone.projection(batch_raw)
            all_projected.append(projected.cpu())
    all_projected = torch.cat(all_projected, dim=0)  # (N, 128)
    
    # Compute distances and assignments using the FIXED projected anchors
    print(f"   Computing distances to anchors...")
    distances = torch.cdist(all_projected, anchors_projected_fixed.cpu())  # (N, K)
    fixed_assignments = distances.argmin(dim=1)  # (N,)
    
    assignment_counts = torch.bincount(fixed_assignments, minlength=n_anchors)
    print(f"   Assignment distribution: {assignment_counts.tolist()}")
    for k in range(n_anchors):
        print(f"      Anchor {k}: {assignment_counts[k].item()} samples ({100*assignment_counts[k].item()/n_samples:.1f}%)")
    
    # === 6. Create loss (attractor only since anchors are fixed) ===
    print(f"\n5. Loss configuration:")
    print(f"   Alpha (attractor): {alpha}")
    print(f"   Beta (repeller): {beta} (should be 0 since anchors fixed)")
    print(f"   Distance metric: {distance_metric}")
    
    loss_fn = AnchorMarginLoss(
        margin=margin,
        alpha=alpha,
        beta=beta,
        distance_metric=distance_metric
    )
    
    # === 5. Create optimizer (only projection head is trainable) ===
    trainable_params = list(backbone.projection.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n6. Optimizer: Adam, lr={lr}")
    print(f"   Trainable params: {n_trainable:,} (projection head only)")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    # === 8. Create dataloader for training (with shuffling) ===
    # We need sample indices to look up their fixed assignments
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
        def __len__(self):
            return len(self.base_dataset)
        def __getitem__(self, idx):
            item = self.base_dataset[idx]
            item['idx'] = idx
            return item
    
    indexed_dataset = IndexedDataset(dataset)
    train_dataloader = torch.utils.data.DataLoader(
        indexed_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    # === 9. Training loop ===
    print(f"\n7. Training for {n_epochs} epochs ({n_samples} samples, batch_size={batch_size})...")
    print(f"   Visualization mode: {'BATCH ONLY' if visualize_batch_only else 'ALL SAMPLES'}")
    
    history = {
        'epoch': [],
        'loss': [],
        'mean_dist': [],
        'mean_cos_sim': [],
        'anc_norms': [],
        'anc_min_dist': []
    }
    
    # For batch-only visualization, we'll use the first batch indices
    if visualize_batch_only:
        viz_indices = torch.arange(min(batch_size, n_samples))
        viz_assignments = fixed_assignments[viz_indices]
        print(f"   Will visualize first {len(viz_indices)} samples only")
    
    # Visualize initial state (step 0)
    print(f"\n   === Epoch 0 (Before Training) ===")
    with torch.no_grad():
        # Get all projected embeddings (through the trainable projection head)
        all_proj_embeddings = []
        for i in range(0, n_samples, batch_size):
            batch_raw = all_raw_embeddings[i:i+batch_size].to(device)
            projected = backbone.projection(batch_raw)
            all_proj_embeddings.append(projected.cpu())
        all_proj_embeddings = torch.cat(all_proj_embeddings, dim=0)
        
        # Use the FIXED projected anchors (not re-projected!)
        # Compute metrics
        metrics = compute_metrics(all_proj_embeddings, anchors_projected_fixed.cpu(), fixed_assignments)
        print(f"   Mean dist to assigned: {metrics['mean_dist_to_assigned']:.4f}")
        print(f"   Mean cosine sim: {metrics['mean_cos_sim']:.4f}")
        print(f"   Anchor norms (FIXED): {[f'{n:.3f}' for n in metrics['anc_norms']]}")
        
        history['epoch'].append(0)
        history['loss'].append(0)
        history['mean_dist'].append(metrics['mean_dist_to_assigned'])
        history['mean_cos_sim'].append(metrics['mean_cos_sim'])
        history['anc_norms'].append(metrics['anc_norms'])
        history['anc_min_dist'].append(metrics['anc_min_dist'])
        
        # Visualize with FIXED anchors
        if visualize_batch_only:
            visualize_all_samples(
                all_proj_embeddings[viz_indices], anchors_projected_fixed.cpu(), viz_assignments,
                step=0, save_dir=save_dir, show_lines=True, max_lines_per_anchor=max_lines
            )
        else:
            visualize_all_samples(
                all_proj_embeddings, anchors_projected_fixed.cpu(), fixed_assignments,
                step=0, save_dir=save_dir, show_lines=True, max_lines_per_anchor=max_lines
            )
    
    # Training epochs
    for epoch in range(1, n_epochs + 1):
        print(f"\n   === Epoch {epoch}/{n_epochs} ===")
        
        epoch_loss = 0.0
        n_batches = 0
        
        backbone.projection.train()  # Only projection head trains
        pbar = tqdm(train_dataloader, desc=f"   Epoch {epoch}")
        for batch in pbar:
            images = batch['image'].to(device)
            indices = batch['idx']  # Original sample indices
            
            # Get fixed assignments for this batch
            batch_assignments = fixed_assignments[indices].to(device)
            
            # Forward pass - get raw features and project them
            with torch.no_grad():
                features = backbone.backbone.forward_features(images)
                raw_cls = features[:, 0]  # (B, 384)
            
            # Project through TRAINABLE projection head
            embeddings = backbone.projection(raw_cls)  # (B, 128)
            
            # Use FIXED projected anchors (not re-projected!)
            # Compute loss with FIXED assignments
            loss_dict = loss_fn(
                embeddings, anchors_projected_fixed,  # FIXED anchors!
                return_components=True,
                fixed_assignments=batch_assignments
            )
            
            loss = loss_dict['loss']
            epoch_loss += loss.item()
            n_batches += 1
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / n_batches
        print(f"   Avg loss: {avg_loss:.4f}")
        
        # Evaluate on ALL samples
        backbone.projection.eval()
        with torch.no_grad():
            all_proj_embeddings = []
            for i in range(0, n_samples, batch_size):
                batch_raw = all_raw_embeddings[i:i+batch_size].to(device)
                projected = backbone.projection(batch_raw)
                all_proj_embeddings.append(projected.cpu())
            all_proj_embeddings = torch.cat(all_proj_embeddings, dim=0)
            
            # Use FIXED projected anchors
            metrics = compute_metrics(all_proj_embeddings, anchors_projected_fixed.cpu(), fixed_assignments)
            print(f"   Mean dist to assigned: {metrics['mean_dist_to_assigned']:.4f}")
            print(f"   Mean cosine sim: {metrics['mean_cos_sim']:.4f}")
            print(f"   Anchor norms (FIXED): {[f'{n:.3f}' for n in metrics['anc_norms']]}")
            
            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            history['mean_dist'].append(metrics['mean_dist_to_assigned'])
            history['mean_cos_sim'].append(metrics['mean_cos_sim'])
            history['anc_norms'].append(metrics['anc_norms'])
            history['anc_min_dist'].append(metrics['anc_min_dist'])
            
            # Visualize with FIXED anchors
            if epoch % visualize_every == 0 or epoch == n_epochs:
                if visualize_batch_only:
                    visualize_all_samples(
                        all_proj_embeddings[viz_indices], anchors_projected_fixed.cpu(), viz_assignments,
                        step=epoch, save_dir=save_dir, show_lines=True, max_lines_per_anchor=max_lines
                    )
                else:
                    visualize_all_samples(
                        all_proj_embeddings, anchors_projected_fixed.cpu(), fixed_assignments,
                        step=epoch, save_dir=save_dir, show_lines=True, max_lines_per_anchor=max_lines
                    )
    
    # === 10. Plot training curves ===
    print(f"\n8. Plotting training curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = history['epoch']
    
    ax = axes[0, 0]
    ax.plot(epochs[1:], history['loss'][1:], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss (Attractor Only)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(epochs, history['mean_dist'], 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean Distance', fontsize=11)
    ax.set_title('Mean Distance to Assigned Anchor', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(epochs, history['mean_cos_sim'], 'm-o', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title('Mean Cosine Similarity to Assigned Anchor', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for k in range(n_anchors):
        norms = [h[k] for h in history['anc_norms']]
        ax.plot(epochs, norms, '-o', linewidth=2, markersize=6, label=f'Anchor {k}')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Norm', fontsize=11)
    ax.set_title('Anchor Norms (Projected Space)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Metrics (N={n_samples}, K={n_anchors})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # === Summary ===
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Initial mean distance: {history['mean_dist'][0]:.4f}")
    print(f"Final mean distance:   {history['mean_dist'][-1]:.4f}")
    print(f"Change: {100*(history['mean_dist'][-1] - history['mean_dist'][0])/history['mean_dist'][0]:.1f}%")
    print(f"\nInitial cosine sim: {history['mean_cos_sim'][0]:.4f}")
    print(f"Final cosine sim:   {history['mean_cos_sim'][-1]:.4f}")
    print(f"\nOutputs saved to: {save_dir}")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug real model with proper training')
    parser.add_argument('--data-root', type=str, default='../data/BraTS2021_slice')
    parser.add_argument('--n-anchors', type=int, default=4)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1.0, help='Attractor weight')
    parser.add_argument('--beta', type=float, default=0.0, help='Repeller weight (0 for fixed anchors)')
    parser.add_argument('--output', type=str, default='./debug_output/real_model_proper')
    parser.add_argument('--visualize-every', type=int, default=1, help='Visualize every N epochs')
    parser.add_argument('--max-lines', type=int, default=200, help='Max lines per anchor in viz')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-only', action='store_true', help='Only visualize one batch (not all samples)')
    
    args = parser.parse_args()
    
    run_real_model_debug(
        data_root=args.data_root,
        n_anchors=args.n_anchors,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        output_dir=args.output,
        visualize_every=args.visualize_every,
        max_lines=args.max_lines,
        seed=args.seed,
        visualize_batch_only=args.batch_only
    )
