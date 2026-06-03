"""
Debug Script: Investigate Anchor Migration to Center

This script creates a controlled toy example to debug why anchors appear to
migrate towards the center of the distribution in t-SNE visualizations.

Hypothesis to test:
- Each sample should gravitate toward its assigned anchor
- We should see K mini-clusters forming around each anchor
- If anchors migrate to center instead, there's a bug in the loss/training

Test setup:
- 4 learnable anchors in 2D embedding space (for easy visualization)
- 100 synthetic samples distributed in 4 clusters
- Train for ~50 steps with attractor + repeller loss
- Visualize embeddings every 5 steps

What to look for:
1. Do samples move toward their assigned anchor? (Expected: YES)
2. Do anchors stay in place or move? (Expected: Stay if fixed, move appropriately if learnable)
3. Do anchors collapse toward center? (Bug indicator: YES)
4. Does repeller loss keep anchors separated? (Expected: YES)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ============================================================================
# Simple Loss Functions (matching project implementation)
# ============================================================================

class SimpleAnchorLoss(nn.Module):
    """
    Simplified anchor loss for debugging.
    Matches the project's AnchorMarginLoss implementation.
    
    Attractor: L_A = 0.5 * ||e_i - c_{nearest}||^2  (pull samples to nearest anchor)
    Repeller:  L_R = 0.5 * max(0, 2m - ||c_j - c_k||)^2  (push anchors apart)
    """
    
    def __init__(self, margin=1.0, alpha=1.0, beta=1.0, distance_metric='euclidean'):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # attractor weight
        self.beta = beta    # repeller weight
        self.distance_metric = distance_metric
    
    def forward(self, embeddings, anchors, fixed_assignments=None, return_details=False):
        """
        Args:
            embeddings: (N, D) sample embeddings (from projection head)
            anchors: (K, D) anchor embeddings (potentially learnable)
            fixed_assignments: (N,) pre-computed anchor assignments (optional)
            return_details: return per-sample distances for analysis
        """
        B, D = embeddings.shape
        K, _ = anchors.shape
        
        # === COMPUTE DISTANCES ===
        if self.distance_metric == 'euclidean':
            # L2 distances: (B, K)
            distances = torch.cdist(embeddings, anchors, p=2)
            anchors_for_repeller = anchors
        else:  # cosine
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            anchors_norm = F.normalize(anchors, p=2, dim=1)
            similarities = embeddings_norm @ anchors_norm.T
            distances = 1.0 - similarities
            anchors_for_repeller = anchors_norm
        
        # === FIND NEAREST ANCHOR ===
        if fixed_assignments is not None:
            assigned_anchors = fixed_assignments
            min_distances = distances[torch.arange(B, device=embeddings.device), assigned_anchors]
        else:
            min_distances, assigned_anchors = distances.min(dim=1)
        
        # === ATTRACTOR LOSS (from project's loss.py) ===
        # Paper formulation: 0.5 * ||e_i - c_{y_i}||^2
        loss_attract = 0.5 * (min_distances ** 2).mean()
        
        # === REPELLER LOSS (from project's loss.py) ===
        # Push anchors apart: 0.5 * max(0, 2m - ||c_j - c_k||)^2
        if self.distance_metric == 'euclidean':
            anchor_distances = torch.cdist(anchors_for_repeller, anchors_for_repeller, p=2)
        else:
            anchor_sims = anchors_for_repeller @ anchors_for_repeller.T
            anchor_distances = 1.0 - anchor_sims
        
        mask = ~torch.eye(K, dtype=torch.bool, device=anchor_distances.device)
        violations = torch.relu(2 * self.margin - anchor_distances)
        loss_repel = 0.5 * (violations[mask] ** 2).mean() if violations[mask].numel() > 0 else torch.tensor(0.0)
        
        # === TOTAL LOSS ===
        total_loss = self.alpha * loss_attract + self.beta * loss_repel
        
        result = {
            'loss': total_loss,
            'loss_attract': loss_attract.item(),
            'loss_repel': loss_repel.item(),
            'assigned_anchors': assigned_anchors,
            'min_distances': min_distances.detach()
        }
        
        if return_details:
            result['all_distances'] = distances.detach()
        
        return result


# ============================================================================
# Toy Model (mimics project's architecture)
# ============================================================================

class ToyProjectionHead(nn.Module):
    """Simple projection head like the project uses"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ToyModel(nn.Module):
    """
    Simplified model matching project architecture:
    - "Backbone" that produces embeddings (we use synthetic data)
    - Learnable projection head
    - Learnable or fixed anchors
    """
    def __init__(self, embed_dim, proj_dim, n_anchors, learnable_anchors=True, initial_anchors=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.n_anchors = n_anchors
        
        # Projection head (trainable)
        self.projection = ToyProjectionHead(embed_dim, embed_dim, proj_dim)
        
        # Initialize anchors
        if initial_anchors is not None:
            anchor_init = initial_anchors.clone()
        else:
            # Random anchors in projected space
            anchor_init = torch.randn(n_anchors, proj_dim)
            anchor_init = F.normalize(anchor_init, dim=1) * 2.0  # Scale to spread them out
        
        if learnable_anchors:
            self.anchors = nn.Parameter(anchor_init)
            print(f"  ✓ Anchors are LEARNABLE ({n_anchors} × {proj_dim}D)")
        else:
            self.register_buffer('anchors', anchor_init)
            print(f"  ✓ Anchors are FIXED ({n_anchors} × {proj_dim}D)")
        
        self.learnable_anchors = learnable_anchors
    
    def forward(self, raw_embeddings):
        """
        Args:
            raw_embeddings: (B, embed_dim) - "backbone" outputs
        Returns:
            projected: (B, proj_dim) - projected embeddings
            anchors: (K, proj_dim) - anchor embeddings (potentially projected)
        """
        # Project embeddings through trainable head
        projected = self.projection(raw_embeddings)
        
        # Project anchors if they're in raw space (matching the real model's _get_projected_anchors)
        # In real model, anchors are stored in raw backbone dimension and projected
        # For simplicity here, we store anchors directly in projected dimension
        
        return projected, self.anchors


# ============================================================================
# Synthetic Data Generator
# ============================================================================

def generate_synthetic_data(n_samples_per_cluster=25, n_clusters=4, dim=32, cluster_std=0.3, seed=42):
    """
    Generate synthetic embeddings organized in clusters.
    This simulates what the frozen DINO backbone would output.
    
    Args:
        n_samples_per_cluster: samples per cluster
        n_clusters: number of clusters (should match n_anchors)
        dim: embedding dimension
        cluster_std: standard deviation within clusters
        seed: random seed
    
    Returns:
        embeddings: (N, dim) tensor
        cluster_ids: (N,) tensor with ground truth cluster assignments
        cluster_centers: (K, dim) tensor with cluster centers
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create cluster centers spread out in the space
    cluster_centers = torch.randn(n_clusters, dim)
    cluster_centers = F.normalize(cluster_centers, dim=1) * 3.0  # Spread them out
    
    embeddings = []
    cluster_ids = []
    
    for i in range(n_clusters):
        # Generate samples around each cluster center
        center = cluster_centers[i]
        samples = center.unsqueeze(0) + torch.randn(n_samples_per_cluster, dim) * cluster_std
        embeddings.append(samples)
        cluster_ids.extend([i] * n_samples_per_cluster)
    
    embeddings = torch.cat(embeddings, dim=0)
    cluster_ids = torch.tensor(cluster_ids, dtype=torch.long)
    
    return embeddings, cluster_ids, cluster_centers


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_2d_space(
    projected_embeddings,
    projected_anchors,
    assigned_anchors,
    step,
    save_dir,
    title_suffix="",
    gt_cluster_ids=None,
    show_arrows=False
):
    """
    Visualize embeddings and anchors in 2D projected space.
    
    Color scheme:
    - Samples: colored by their assigned anchor (from loss computation)
    - Anchors: large stars with same colors
    - Optional: show arrows from samples to their assigned anchors
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    n_anchors = projected_anchors.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_anchors))
    
    # === Left plot: Colored by assigned anchor ===
    ax = axes[0]
    
    # Plot samples
    for k in range(n_anchors):
        mask = assigned_anchors == k
        if mask.sum() > 0:
            ax.scatter(
                projected_embeddings[mask, 0],
                projected_embeddings[mask, 1],
                c=[colors[k]],
                s=50,
                alpha=0.6,
                label=f'Assigned to Anchor {k} (n={mask.sum()})'
            )
            
            # Draw arrows from samples to their assigned anchor
            if show_arrows and mask.sum() > 0:
                anchor_pos = projected_anchors[k]
                for emb in projected_embeddings[mask]:
                    ax.annotate(
                        '', 
                        xy=(anchor_pos[0], anchor_pos[1]),
                        xytext=(emb[0], emb[1]),
                        arrowprops=dict(arrowstyle='->', color=colors[k], alpha=0.2, lw=0.5)
                    )
    
    # Plot anchors
    for k in range(n_anchors):
        ax.scatter(
            projected_anchors[k, 0],
            projected_anchors[k, 1],
            c=[colors[k]],
            s=400,
            marker='*',
            edgecolors='black',
            linewidths=2,
            zorder=10
        )
        ax.annotate(
            f'A{k}',
            (projected_anchors[k, 0], projected_anchors[k, 1]),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center'
        )
    
    ax.set_title(f'Step {step}: Assigned Anchors{title_suffix}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Projected Dim 1')
    ax.set_ylabel('Projected Dim 2')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # === Right plot: Colored by ground truth cluster (if available) ===
    ax = axes[1]
    
    if gt_cluster_ids is not None:
        for k in range(n_anchors):
            mask = gt_cluster_ids == k
            if mask.sum() > 0:
                ax.scatter(
                    projected_embeddings[mask, 0],
                    projected_embeddings[mask, 1],
                    c=[colors[k]],
                    s=50,
                    alpha=0.6,
                    label=f'GT Cluster {k}'
                )
    else:
        ax.scatter(
            projected_embeddings[:, 0],
            projected_embeddings[:, 1],
            c='blue',
            s=50,
            alpha=0.6
        )
    
    # Plot anchors
    for k in range(n_anchors):
        ax.scatter(
            projected_anchors[k, 0],
            projected_anchors[k, 1],
            c=[colors[k]],
            s=400,
            marker='*',
            edgecolors='black',
            linewidths=2,
            zorder=10
        )
    
    # Mark origin (center)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.scatter([0], [0], c='red', s=100, marker='x', zorder=10, label='Origin (Center)')
    
    ax.set_title(f'Step {step}: Ground Truth Clusters', fontsize=12, fontweight='bold')
    ax.set_xlabel('Projected Dim 1')
    ax.set_ylabel('Projected Dim 2')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_path = save_dir / f'step_{step:04d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_metrics_history(history, save_dir):
    """Plot training metrics over time"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    steps = history['steps']
    
    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(steps, history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    
    # 2. Attractor Loss
    ax = axes[0, 1]
    ax.plot(steps, history['loss_attract'], 'g-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Attractor Loss')
    ax.set_title('Attractor Loss (pull to anchors)')
    ax.grid(True, alpha=0.3)
    
    # 3. Repeller Loss
    ax = axes[0, 2]
    ax.plot(steps, history['loss_repel'], 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Repeller Loss')
    ax.set_title('Repeller Loss (push anchors apart)')
    ax.grid(True, alpha=0.3)
    
    # 4. Mean distance to assigned anchor
    ax = axes[1, 0]
    ax.plot(steps, history['mean_min_dist'], 'purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Distance')
    ax.set_title('Mean Distance to Assigned Anchor')
    ax.grid(True, alpha=0.3)
    
    # 5. Anchor spread (mean pairwise distance between anchors)
    ax = axes[1, 1]
    ax.plot(steps, history['anchor_spread'], 'orange', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Anchor Spread')
    ax.set_title('Mean Pairwise Anchor Distance')
    ax.grid(True, alpha=0.3)
    
    # 6. Anchor distance to origin (center)
    ax = axes[1, 2]
    for k in range(len(history['anchor_norms'][0])):
        norms = [h[k] for h in history['anchor_norms']]
        ax.plot(steps, norms, linewidth=2, label=f'Anchor {k}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Origin')
    ax.set_title('Anchor Distance to Origin (Center)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_gradients(model, loss_dict, step):
    """Analyze gradients on anchors and projection head"""
    print(f"\n  [Step {step}] Gradient Analysis:")
    
    # Check anchor gradients
    if model.learnable_anchors:
        if model.anchors.grad is not None:
            grad_norm = model.anchors.grad.norm().item()
            grad_mean = model.anchors.grad.abs().mean().item()
            grad_per_anchor = model.anchors.grad.norm(dim=1).cpu().numpy()
            print(f"    Anchor gradients: norm={grad_norm:.6f}, mean_abs={grad_mean:.6f}")
            print(f"    Per-anchor grad norms: {grad_per_anchor}")
        else:
            print(f"    ❌ Anchor gradients are None!")
    else:
        print(f"    Anchors are fixed (no gradients)")
    
    # Check projection head gradients
    total_proj_grad = 0.0
    for name, param in model.projection.named_parameters():
        if param.grad is not None:
            total_proj_grad += param.grad.norm().item()
    print(f"    Projection head total grad norm: {total_proj_grad:.6f}")


# ============================================================================
# Main Debug Function
# ============================================================================

def run_debug_experiment(
    n_anchors=4,
    n_samples_per_cluster=25,
    raw_dim=32,
    proj_dim=2,  # 2D for easy visualization
    learnable_anchors=True,
    use_fixed_assignments=False,  # Whether to fix sample->anchor assignments
    n_steps=100,
    lr=0.01,
    margin=1.0,
    alpha=1.0,  # attractor weight
    beta=1.0,   # repeller weight
    distance_metric='euclidean',
    visualize_every=5,
    seed=42,
    output_dir='./debug_output'
):
    """
    Run the debugging experiment.
    
    Key parameters to vary:
    - learnable_anchors: True/False - does the issue happen with fixed anchors?
    - use_fixed_assignments: True/False - does reassigning to nearest each step cause issues?
    - alpha, beta: loss weights - is the issue with attractor or repeller?
    - proj_dim: 2 for visualization, higher to test if issue is dimension-specific
    """
    
    print("="*80)
    print("ANCHOR MIGRATION DEBUG EXPERIMENT")
    print("="*80)
    
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # === Generate synthetic data ===
    print(f"\n1. Generating synthetic data:")
    print(f"   {n_anchors} clusters × {n_samples_per_cluster} samples = {n_anchors * n_samples_per_cluster} total")
    print(f"   Raw dim: {raw_dim}, Projected dim: {proj_dim}")
    
    raw_embeddings, gt_cluster_ids, cluster_centers = generate_synthetic_data(
        n_samples_per_cluster=n_samples_per_cluster,
        n_clusters=n_anchors,
        dim=raw_dim,
        cluster_std=0.5,
        seed=seed
    )
    raw_embeddings = raw_embeddings.to(device)
    gt_cluster_ids = gt_cluster_ids.to(device)
    
    # === Initialize anchors ===
    print(f"\n2. Initializing model:")
    
    # Initialize anchors in projected space (spread out)
    torch.manual_seed(seed + 100)
    initial_anchors = torch.randn(n_anchors, proj_dim)
    initial_anchors = F.normalize(initial_anchors, dim=1) * 2.0  # Spread out from origin
    initial_anchors = initial_anchors.to(device)
    
    print(f"   Initial anchor positions:")
    for k in range(n_anchors):
        print(f"     Anchor {k}: {initial_anchors[k].cpu().numpy()}")
    
    # === Create model ===
    model = ToyModel(
        embed_dim=raw_dim,
        proj_dim=proj_dim,
        n_anchors=n_anchors,
        learnable_anchors=learnable_anchors,
        initial_anchors=initial_anchors
    ).to(device)
    
    # === Create loss ===
    print(f"\n3. Loss configuration:")
    print(f"   Alpha (attractor): {alpha}")
    print(f"   Beta (repeller): {beta}")
    print(f"   Margin: {margin}")
    print(f"   Distance metric: {distance_metric}")
    
    loss_fn = SimpleAnchorLoss(
        margin=margin,
        alpha=alpha,
        beta=beta,
        distance_metric=distance_metric
    )
    
    # === Create optimizer ===
    print(f"\n4. Optimizer: Adam, lr={lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # === Pre-compute fixed assignments if requested ===
    fixed_assignments = None
    if use_fixed_assignments:
        print(f"\n5. Computing fixed assignments (samples -> nearest initial anchor)...")
        with torch.no_grad():
            proj_emb, proj_anc = model(raw_embeddings)
            if distance_metric == 'euclidean':
                distances = torch.cdist(proj_emb, proj_anc, p=2)
            else:
                proj_emb_norm = F.normalize(proj_emb, dim=1)
                proj_anc_norm = F.normalize(proj_anc, dim=1)
                distances = 1.0 - proj_emb_norm @ proj_anc_norm.T
            fixed_assignments = distances.argmin(dim=1)
        
        print(f"   Assignment distribution: {torch.bincount(fixed_assignments, minlength=n_anchors).cpu().numpy()}")
    else:
        print(f"\n5. Using dynamic assignments (nearest anchor per step)")
    
    # === Training loop ===
    print(f"\n6. Training for {n_steps} steps...")
    
    history = {
        'steps': [],
        'loss': [],
        'loss_attract': [],
        'loss_repel': [],
        'mean_min_dist': [],
        'anchor_spread': [],
        'anchor_norms': [],
        'anchor_positions': []
    }
    
    for step in range(n_steps + 1):
        model.train()
        
        # Forward pass
        proj_embeddings, proj_anchors = model(raw_embeddings)
        
        # Compute loss
        loss_dict = loss_fn(
            proj_embeddings, 
            proj_anchors, 
            fixed_assignments=fixed_assignments,
            return_details=True
        )
        
        # Compute metrics
        with torch.no_grad():
            # Anchor spread: mean pairwise distance
            anchor_dists = torch.cdist(proj_anchors, proj_anchors, p=2)
            mask = ~torch.eye(n_anchors, dtype=torch.bool, device=device)
            anchor_spread = anchor_dists[mask].mean().item()
            
            # Anchor norms (distance to origin)
            anchor_norms = proj_anchors.norm(dim=1).cpu().numpy().tolist()
        
        # Record history
        history['steps'].append(step)
        history['loss'].append(loss_dict['loss'].item())
        history['loss_attract'].append(loss_dict['loss_attract'])
        history['loss_repel'].append(loss_dict['loss_repel'])
        history['mean_min_dist'].append(loss_dict['min_distances'].mean().item())
        history['anchor_spread'].append(anchor_spread)
        history['anchor_norms'].append(anchor_norms)
        history['anchor_positions'].append(proj_anchors.detach().cpu().numpy().copy())
        
        # Visualize
        if step % visualize_every == 0:
            print(f"\n  Step {step}:")
            print(f"    Loss: {loss_dict['loss'].item():.4f} (attract: {loss_dict['loss_attract']:.4f}, repel: {loss_dict['loss_repel']:.4f})")
            print(f"    Mean dist to anchor: {loss_dict['min_distances'].mean().item():.4f}")
            print(f"    Anchor spread: {anchor_spread:.4f}")
            print(f"    Anchor norms: {[f'{n:.3f}' for n in anchor_norms]}")
            
            # Check if anchors are collapsing to center
            mean_anchor_norm = np.mean(anchor_norms)
            if step > 0 and mean_anchor_norm < history['anchor_norms'][0][0] * 0.5:
                print(f"    ⚠️  WARNING: Anchors may be collapsing toward center!")
            
            # Create visualization
            with torch.no_grad():
                assigned = loss_dict['assigned_anchors'].cpu().numpy()
                visualize_2d_space(
                    projected_embeddings=proj_embeddings.detach().cpu().numpy(),
                    projected_anchors=proj_anchors.detach().cpu().numpy(),
                    assigned_anchors=assigned,
                    step=step,
                    save_dir=save_dir,
                    gt_cluster_ids=gt_cluster_ids.cpu().numpy(),
                    show_arrows=(step % 20 == 0)
                )
        
        # Backward pass (skip step 0 which is just initialization visualization)
        if step < n_steps:
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            
            # Analyze gradients at first few steps
            if step < 5 or step % 20 == 0:
                analyze_gradients(model, loss_dict, step)
            
            optimizer.step()
    
    # === Final analysis ===
    print(f"\n7. Final Analysis:")
    
    initial_anchor_norms = history['anchor_norms'][0]
    final_anchor_norms = history['anchor_norms'][-1]
    
    print(f"   Initial anchor norms: {[f'{n:.3f}' for n in initial_anchor_norms]}")
    print(f"   Final anchor norms:   {[f'{n:.3f}' for n in final_anchor_norms]}")
    
    norm_ratio = np.mean(final_anchor_norms) / np.mean(initial_anchor_norms)
    print(f"   Norm ratio (final/initial): {norm_ratio:.3f}")
    
    if norm_ratio < 0.5:
        print(f"\n   ❌ ISSUE DETECTED: Anchors collapsed toward center!")
        print(f"      This explains why t-SNE shows anchors in the middle.")
    elif norm_ratio > 1.5:
        print(f"\n   ⚠️  Anchors moved away from center (norm increased)")
    else:
        print(f"\n   ✓ Anchor norms stayed relatively stable")
    
    # Check anchor spread
    initial_spread = history['anchor_spread'][0]
    final_spread = history['anchor_spread'][-1]
    print(f"\n   Initial anchor spread: {initial_spread:.3f}")
    print(f"   Final anchor spread: {final_spread:.3f}")
    
    if final_spread < initial_spread * 0.5:
        print(f"   ❌ ISSUE: Anchors collapsed together!")
    elif final_spread < margin:
        print(f"   ⚠️  Anchors closer than margin ({margin})")
    else:
        print(f"   ✓ Repeller loss keeping anchors apart")
    
    # Check if samples are clustering around anchors
    final_mean_dist = history['mean_min_dist'][-1]
    initial_mean_dist = history['mean_min_dist'][0]
    print(f"\n   Initial mean dist to anchor: {initial_mean_dist:.3f}")
    print(f"   Final mean dist to anchor: {final_mean_dist:.3f}")
    
    if final_mean_dist < initial_mean_dist:
        print(f"   ✓ Attractor loss is working (samples closer to anchors)")
    else:
        print(f"   ❌ Samples not getting closer to anchors!")
    
    # === Save plots ===
    plot_metrics_history(history, save_dir)
    print(f"\n8. Saved outputs to: {save_dir}")
    print(f"   - Step visualizations: step_XXXX.png")
    print(f"   - Metrics history: metrics_history.png")
    
    return history


# ============================================================================
# Run multiple experiments to isolate the issue
# ============================================================================

def run_diagnostic_suite():
    """Run multiple experiments to diagnose the issue"""
    
    base_dir = Path('./debug_output')
    base_dir.mkdir(exist_ok=True)
    
    experiments = [
        # Experiment 1: Baseline - learnable anchors, dynamic assignments
        {
            'name': 'exp1_learnable_dynamic',
            'learnable_anchors': True,
            'use_fixed_assignments': False,
            'alpha': 1.0,
            'beta': 1.0,
        },
        # Experiment 2: Learnable anchors with FIXED assignments
        {
            'name': 'exp2_learnable_fixed_assign',
            'learnable_anchors': True,
            'use_fixed_assignments': True,
            'alpha': 1.0,
            'beta': 1.0,
        },
        # Experiment 3: FIXED anchors (only train projection)
        {
            'name': 'exp3_fixed_anchors',
            'learnable_anchors': False,
            'use_fixed_assignments': False,
            'alpha': 1.0,
            'beta': 1.0,
        },
        # Experiment 4: No repeller loss (beta=0)
        {
            'name': 'exp4_no_repeller',
            'learnable_anchors': True,
            'use_fixed_assignments': False,
            'alpha': 1.0,
            'beta': 0.0,  # No repeller
        },
        # Experiment 5: Strong repeller (beta=2.0)
        {
            'name': 'exp5_strong_repeller',
            'learnable_anchors': True,
            'use_fixed_assignments': False,
            'alpha': 1.0,
            'beta': 2.0,  # Strong repeller
        },
        # Experiment 6: Only repeller, no attractor
        {
            'name': 'exp6_only_repeller',
            'learnable_anchors': True,
            'use_fixed_assignments': False,
            'alpha': 0.0,  # No attractor
            'beta': 1.0,
        },
    ]
    
    print("="*80)
    print("DIAGNOSTIC SUITE: Running 6 experiments to isolate the issue")
    print("="*80)
    
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"Running: {exp['name']}")
        print(f"{'='*80}")
        
        run_debug_experiment(
            n_anchors=4,
            n_samples_per_cluster=25,
            raw_dim=32,
            proj_dim=2,
            learnable_anchors=exp['learnable_anchors'],
            use_fixed_assignments=exp['use_fixed_assignments'],
            n_steps=100,
            lr=0.01,
            margin=1.0,
            alpha=exp['alpha'],
            beta=exp['beta'],
            distance_metric='euclidean',
            visualize_every=10,
            seed=42,
            output_dir=str(base_dir / exp['name'])
        )
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUITE COMPLETE")
    print("="*80)
    print("\nCheck the following directories for results:")
    for exp in experiments:
        print(f"  - {base_dir / exp['name']}")
    print("\nCompare step_0000.png (initial) vs step_0100.png (final) for each experiment")
    print("to see how anchors and samples evolve under different configurations.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug anchor migration issue')
    parser.add_argument('--suite', action='store_true', help='Run full diagnostic suite')
    parser.add_argument('--learnable', type=bool, default=True, help='Use learnable anchors')
    parser.add_argument('--fixed-assign', type=bool, default=False, help='Use fixed assignments')
    parser.add_argument('--alpha', type=float, default=1.0, help='Attractor weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Repeller weight')
    parser.add_argument('--steps', type=int, default=100, help='Training steps')
    parser.add_argument('--output', type=str, default='./debug_output/single_run', help='Output directory')
    
    args = parser.parse_args()
    
    if args.suite:
        run_diagnostic_suite()
    else:
        run_debug_experiment(
            learnable_anchors=args.learnable,
            use_fixed_assignments=args.fixed_assign,
            alpha=args.alpha,
            beta=args.beta,
            n_steps=args.steps,
            output_dir=args.output
        )
