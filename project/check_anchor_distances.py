"""
Check actual distances between anchors in the original embedding space
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def compute_anchor_statistics(anchor_path: str):
    """Compute distances and statistics for anchors"""
    print(f"\nLoading anchors from: {anchor_path}")
    
    # Load anchor embeddings
    anchor_data = torch.load(anchor_path, map_location='cpu', weights_only=False)
    
    # Handle different formats
    if isinstance(anchor_data, dict):
        anchors = anchor_data.get('global', anchor_data.get('anchor_global', None))
    else:
        anchors = anchor_data
    
    if anchors is None:
        raise ValueError("Could not find anchor embeddings")
    
    anchors = anchors.cpu().numpy()
    K, D = anchors.shape
    
    print(f"\nAnchor Shape: {K} anchors × {D} dimensions")
    
    # Compute pairwise distances
    print(f"\nComputing pairwise distances...")
    distances = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            distances[i, j] = np.linalg.norm(anchors[i] - anchors[j])
    
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones((K, K)), k=1).astype(bool)
    pairwise_dists = distances[mask]
    
    print(f"\n{'='*80}")
    print("ANCHOR DISTANCE STATISTICS (L2 norm)")
    print(f"{'='*80}")
    print(f"Mean distance:     {pairwise_dists.mean():.4f}")
    print(f"Std distance:      {pairwise_dists.std():.4f}")
    print(f"Min distance:      {pairwise_dists.min():.4f}")
    print(f"Max distance:      {pairwise_dists.max():.4f}")
    print(f"Median distance:   {np.median(pairwise_dists):.4f}")
    
    # Compute anchor norms
    norms = np.linalg.norm(anchors, axis=1)
    print(f"\n{'='*80}")
    print("ANCHOR NORMS")
    print(f"{'='*80}")
    print(f"Mean norm:         {norms.mean():.4f}")
    print(f"Std norm:          {norms.std():.4f}")
    print(f"Min norm:          {norms.min():.4f}")
    print(f"Max norm:          {norms.max():.4f}")
    
    # Show distance matrix
    print(f"\n{'='*80}")
    print("PAIRWISE DISTANCE MATRIX")
    print(f"{'='*80}")
    print("     ", end="")
    for i in range(K):
        print(f"  A{i}  ", end="")
    print()
    
    for i in range(K):
        print(f"A{i}  ", end="")
        for j in range(K):
            if i == j:
                print("  -   ", end="")
            else:
                print(f"{distances[i, j]:5.2f} ", end="")
        print()
    
    print(f"\n{'='*80}")
    
    return {
        'mean_dist': pairwise_dists.mean(),
        'std_dist': pairwise_dists.std(),
        'min_dist': pairwise_dists.min(),
        'max_dist': pairwise_dists.max(),
        'mean_norm': norms.mean(),
        'std_norm': norms.std()
    }


def main():
    parser = argparse.ArgumentParser(description='Check anchor distances')
    parser.add_argument('--experiments', type=str, nargs='+',
                       default=[
                           'experiments/multi_trial_results/best_models/eigenface',
                           'experiments/multi_trial_results/best_models/kmeans',
                           'experiments/multi_trial_results/best_models/random'
                       ],
                       help='Experiment directories to check')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("ANCHOR DISTANCE ANALYSIS")
    print(f"{'='*80}")
    
    results = {}
    for exp_dir in args.experiments:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        anchor_path = exp_path / 'anchor_embeddings.pt'
        
        if not anchor_path.exists():
            print(f"\n✗ Anchor file not found: {anchor_path}")
            continue
        
        print(f"\n\n{'#'*80}")
        print(f"# {exp_name.upper()}")
        print(f"{'#'*80}")
        
        stats = compute_anchor_statistics(str(anchor_path))
        results[exp_name] = stats
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Strategy':<15} {'Mean Dist':<12} {'Std Dist':<12} {'Min Dist':<12} {'Max Dist':<12}")
        print("-" * 80)
        for strategy, stats in results.items():
            print(f"{strategy:<15} {stats['mean_dist']:<12.4f} {stats['std_dist']:<12.4f} "
                  f"{stats['min_dist']:<12.4f} {stats['max_dist']:<12.4f}")
        
        print(f"\n{'Strategy':<15} {'Mean Norm':<12} {'Std Norm':<12}")
        print("-" * 80)
        for strategy, stats in results.items():
            print(f"{strategy:<15} {stats['mean_norm']:<12.4f} {stats['std_norm']:<12.4f}")


if __name__ == '__main__':
    main()
