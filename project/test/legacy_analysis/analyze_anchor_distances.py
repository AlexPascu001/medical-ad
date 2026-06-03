"""
Analyze anchor separation in original embedding space (not t-SNE)
"""

import torch
import numpy as np
from pathlib import Path
import argparse

def analyze_anchors(checkpoint_path: str):
    """Analyze anchor distances in original embedding space"""
    
    # Load anchors
    checkpoint_dir = Path(checkpoint_path).parent
    anchor_path = checkpoint_dir / 'anchor_embeddings.pt'
    
    anchor_data = torch.load(anchor_path, weights_only=False)
    if isinstance(anchor_data, dict):
        anchors = anchor_data.get('global', anchor_data.get('anchor_global', None))
    else:
        anchors = anchor_data
    
    anchors = anchors.numpy()  # (K, D)
    K, D = anchors.shape
    
    print(f"\n{'='*80}")
    print(f"ANCHOR ANALYSIS: {checkpoint_dir.name}")
    print(f"{'='*80}")
    print(f"Number of anchors: {K}")
    print(f"Embedding dimension: {D}")
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    
    # Euclidean distances
    distances = squareform(pdist(anchors, metric='euclidean'))
    
    print(f"\n--- Pairwise Euclidean Distances ---")
    print(f"Min distance: {distances[distances > 0].min():.4f}")
    print(f"Max distance: {distances.max():.4f}")
    print(f"Mean distance: {distances[distances > 0].mean():.4f}")
    print(f"Std distance: {distances[distances > 0].std():.4f}")
    
    # Show distance matrix
    print(f"\nDistance matrix:")
    print("     ", end="")
    for i in range(K):
        print(f"  [{i}]  ", end="")
    print()
    
    for i in range(K):
        print(f"[{i}]", end="  ")
        for j in range(K):
            if i == j:
                print("  -   ", end=" ")
            else:
                print(f"{distances[i,j]:5.2f}", end=" ")
        print()
    
    # Compute anchor norms
    norms = np.linalg.norm(anchors, axis=1)
    print(f"\n--- Anchor Norms (L2) ---")
    print(f"Min norm: {norms.min():.4f}")
    print(f"Max norm: {norms.max():.4f}")
    print(f"Mean norm: {norms.mean():.4f}")
    print(f"Std norm: {norms.std():.4f}")
    
    print(f"\nIndividual norms:")
    for i, norm in enumerate(norms):
        print(f"  Anchor [{i}]: {norm:.4f}")
    
    # Check if anchors are well-separated
    min_dist = distances[distances > 0].min()
    mean_dist = distances[distances > 0].mean()
    
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}")
    
    if min_dist < 0.1:
        print(f"⚠️  WARNING: Some anchors are very close (min={min_dist:.4f})")
        print("   This suggests potential anchor collapse.")
    elif min_dist < 0.5:
        print(f"⚠️  MODERATE: Anchors are somewhat close (min={min_dist:.4f})")
        print("   Consider using repeller loss (beta > 0) to push them apart.")
    else:
        print(f"✓ GOOD: Anchors are well-separated (min={min_dist:.4f})")
    
    if mean_dist > 2.0:
        print(f"✓ Anchors cover a wide region (mean={mean_dist:.4f})")
    else:
        print(f"→ Anchors are relatively clustered (mean={mean_dist:.4f})")
        print("   This is OK if they still cover the normal data manifold.")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints')
    args = parser.parse_args()
    
    for checkpoint_path in args.checkpoints:
        analyze_anchors(checkpoint_path)
