"""
Analyze anchor coverage and utilization against actual data distribution.
This checks if anchors are too clumped, too spread, or just right.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import yaml
from typing import Dict, Tuple
from tqdm import tqdm

from model import DINOv3Backbone, AnomalyDetector
from data import BMADDataset, BMADPreprocessor
from main import load_dataset_paths


def analyze_anchor_perspective(anchors, data_embeddings, strategy_name="Unknown"):
    """
    Comprehensive anchor quality analysis.
    
    Args:
        anchors: Tensor [K, D] - anchor embeddings
        data_embeddings: Tensor [N, D] - normal training data embeddings
        strategy_name: Name of the anchor initialization strategy
    """
    
    # Ensure normalization
    anchors = torch.nn.functional.normalize(anchors, p=2, dim=1)
    data_embeddings = torch.nn.functional.normalize(data_embeddings, p=2, dim=1)
    
    K = anchors.shape[0]
    N = data_embeddings.shape[0]
    
    print(f"\n{'='*80}")
    print(f"ANCHOR PERSPECTIVE ANALYSIS - {strategy_name.upper()}")
    print(f"{'='*80}")
    print(f"Anchors: {K}")
    print(f"Data samples: {N}")
    print(f"Embedding dimension: {anchors.shape[1]}")
    
    # 1. Compute Data-to-Nearest-Anchor Distances
    print("\nComputing data-to-anchor distances...")
    dists = torch.cdist(data_embeddings, anchors, p=2)
    
    # For every data point, how far is the CLOSEST anchor?
    min_dists, nearest_anchor_indices = torch.min(dists, dim=1)
    
    mean_data_to_anchor = min_dists.mean().item()
    std_data_to_anchor = min_dists.std().item()
    median_data_to_anchor = min_dists.median().item()
    
    # 2. Compute Inter-Anchor Distances
    anchor_dists = torch.cdist(anchors, anchors, p=2)
    mask = ~torch.eye(K, dtype=torch.bool)
    inter_anchor_dists = anchor_dists[mask]
    mean_inter_anchor = inter_anchor_dists.mean().item()
    std_inter_anchor = inter_anchor_dists.std().item()
    
    # 3. Coverage Ratio
    coverage_ratio = mean_data_to_anchor / mean_inter_anchor
    
    # 4. Angular interpretation
    # For normalized vectors: d^2 = 2(1 - cos(theta))
    # cos(theta) = 1 - d^2/2
    mean_anchor_cosine = 1 - (mean_inter_anchor ** 2) / 2
    mean_data_cosine = 1 - (mean_data_to_anchor ** 2) / 2
    
    # Convert to degrees
    mean_anchor_angle = np.arccos(np.clip(mean_anchor_cosine, -1, 1)) * 180 / np.pi
    mean_data_angle = np.arccos(np.clip(mean_data_cosine, -1, 1)) * 180 / np.pi
    
    print(f"\n{'='*80}")
    print("DISTANCE STATISTICS")
    print(f"{'='*80}")
    print(f"\nInter-Anchor Distances:")
    print(f"  Mean:   {mean_inter_anchor:.4f}  (≈ {mean_anchor_angle:.1f}° angular separation)")
    print(f"  Std:    {std_inter_anchor:.4f}")
    print(f"  Range:  [{inter_anchor_dists.min():.4f}, {inter_anchor_dists.max():.4f}]")
    
    print(f"\nData-to-Nearest-Anchor Distances:")
    print(f"  Mean:   {mean_data_to_anchor:.4f}  (≈ {mean_data_angle:.1f}° from nearest anchor)")
    print(f"  Std:    {std_data_to_anchor:.4f}")
    print(f"  Median: {median_data_to_anchor:.4f}")
    print(f"  Range:  [{min_dists.min():.4f}, {min_dists.max():.4f}]")
    
    print(f"\n{'='*80}")
    print("COVERAGE RATIO")
    print(f"{'='*80}")
    print(f"Ratio (Data-Dist / Anchor-Dist): {coverage_ratio:.4f}")
    print()
    
    # Diagnosis
    if coverage_ratio > 1.2:
        diagnosis = "⚠️  ANCHORS TOO CLUMPED"
        explanation = "Data is spread OUTSIDE the anchor region.\n" \
                     "The anchors huddle in the center while data surrounds them like a shell.\n" \
                     "This means anchors fail to capture the edges of the normal distribution."
        recommendation = "Consider using K-Means initialization to better span the data."
    elif coverage_ratio < 0.3:
        diagnosis = "⚠️  ANCHORS TOO SPREAD"
        explanation = "Anchors are spread WIDER than the actual data distribution.\n" \
                     "Some anchors may be outliers that don't represent real data patterns."
        recommendation = "Anchors might be initialized too far from data centroid."
    else:
        diagnosis = "✓ GOOD BALANCE"
        explanation = "Anchors span the width of the normal data manifold appropriately.\n" \
                     "Data points are roughly as far from anchors as anchors are from each other."
        recommendation = "Anchor initialization quality looks good!"
    
    print(f"DIAGNOSIS: {diagnosis}")
    print(f"\n{explanation}")
    print(f"\n💡 {recommendation}")
    
    # 5. Anchor Utilization
    print(f"\n{'='*80}")
    print("ANCHOR UTILIZATION")
    print(f"{'='*80}")
    
    unique_indices, counts = torch.unique(nearest_anchor_indices, return_counts=True)
    utilization = {idx.item(): count.item() for idx, count in zip(unique_indices, counts)}
    
    # Build full usage array
    usage_counts = [0] * K
    usage_percents = [0.0] * K
    for idx, count in utilization.items():
        usage_counts[idx] = count
        usage_percents[idx] = (count / N) * 100
    
    empty_anchors = K - len(utilization)
    
    print(f"Active Anchors: {len(utilization)} / {K}")
    if empty_anchors > 0:
        print(f"⚠️  Empty Anchors: {empty_anchors} (NO data points assigned!)")
    
    print(f"\nUsage Distribution:")
    for i in range(K):
        bar_length = int(usage_percents[i] / 2)
        bar = '█' * bar_length
        print(f"  Anchor {i}: {usage_percents[i]:5.1f}% ({usage_counts[i]:5d} samples) {bar}")
    
    # Check for mode collapse
    max_usage = max(usage_percents)
    if max_usage > 80:
        print(f"\n⚠️  WARNING: Anchor {usage_percents.index(max_usage)} dominates with {max_usage:.1f}%!")
        print("    This indicates MODE COLLAPSE - most data assigned to single anchor.")
        print("    Other anchors are being ignored by the model.")
    elif max_usage > 50:
        print(f"\n⚠️  One anchor captures {max_usage:.1f}% of data (slight imbalance).")
    else:
        print(f"\n✓ Balanced utilization (max usage: {max_usage:.1f}%)")
    
    # 6. Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Theoretical maximum distance on unit sphere
    max_possible_dist = 2.0  # Opposite vectors
    orthogonal_dist = np.sqrt(2)  # 90 degree vectors
    
    coverage_quality = "Good" if 0.3 <= coverage_ratio <= 1.2 else "Poor"
    utilization_quality = "Good" if empty_anchors == 0 and max_usage < 50 else "Poor"
    
    print(f"Coverage Quality:    {coverage_quality}")
    print(f"Utilization Quality: {utilization_quality}")
    print(f"\nAnchor Spread: {mean_inter_anchor:.4f} / {orthogonal_dist:.4f} (0.71 = orthogonal)")
    print(f"Coverage Ratio: {coverage_ratio:.4f} (target: 0.3-1.2)")
    
    return {
        'mean_inter_anchor': mean_inter_anchor,
        'mean_data_to_anchor': mean_data_to_anchor,
        'coverage_ratio': coverage_ratio,
        'utilization': usage_percents,
        'empty_anchors': empty_anchors,
        'max_usage': max_usage,
        'diagnosis': diagnosis,
        'mean_anchor_angle': mean_anchor_angle,
        'mean_data_angle': mean_data_angle
    }


def extract_training_embeddings(model, dataset, n_samples=2000, batch_size=32, device='cuda'):
    """Extract embeddings from training data"""
    print(f"\nExtracting embeddings from {n_samples} training samples...")
    
    # Sample indices
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    embeddings_list = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(indices), batch_size), desc="Extracting"):
            batch_indices = indices[i:i+batch_size]
            
            batch_images = []
            for idx in batch_indices:
                sample = dataset[idx]
                batch_images.append(sample['image'])
            
            batch_images = torch.stack(batch_images).to(device)
            # Model forward returns dict with 'global_feat', 'global_distances', etc.
            outputs = model(batch_images)
            embeddings_list.append(outputs['global_feat'].cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0)
    print(f"✓ Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Analyze anchor coverage against data distribution')
    parser.add_argument('--experiments', type=str, nargs='+',
                       default=[
                           'project/experiments/bmad_kmeans_k8_l2',
                           'project/experiments/bmad_random_k8_l2',
                           'project/experiments/bmad_eigenface_k8_l2'
                        #    'experiments/multi_trial_results/best_models/eigenface',
                        #    'experiments/multi_trial_results/best_models/kmeans',
                        #    'experiments/multi_trial_results/best_models/random'
                       ],
                       help='Experiment directories to analyze')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Number of training samples to analyze')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding extraction')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    for exp_dir in args.experiments:
        exp_path = Path(exp_dir)
        strategy_name = exp_path.name
        
        checkpoint_path = exp_path / 'best_model.pth'
        config_path = exp_path / 'config.yaml'
        anchor_path = exp_path / 'anchor_embeddings.pt'
        
        if not checkpoint_path.exists():
            print(f"\n✗ Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n\n{'#'*80}")
        print(f"# {strategy_name.upper()}")
        print(f"{'#'*80}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model
        print(f"\nLoading model...")
        backbone = DINOv3Backbone(
            model_name=config['model']['backbone'],
            freeze_backbone=config['model']['freeze_backbone'],
            projection_dim=config['model']['projection_dim'],
            pretrained=True
        ).to(device)
        
        # Load anchors
        anchor_data = torch.load(anchor_path, map_location=device, weights_only=False)
        if isinstance(anchor_data, dict):
            anchor_embeddings = anchor_data.get('global', anchor_data.get('anchor_global'))
        else:
            anchor_embeddings = anchor_data
        
        model = AnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_embeddings,
            anchor_dense_embeddings=None,
            distance_metric=config['loss'].get('distance_metric', 'euclidean')
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        # Load training data
        print(f"Loading training data...")
        data_root = Path(config['data']['data_root'])
        if not data_root.is_absolute():
            # Config has relative path like ../data or ./data
            # Resolve from workspace root (cwd)
            if str(data_root).startswith('..'):
                # ../data -> ./data (we're running from workspace root)
                data_root = Path(str(data_root).replace('..', '.'))
            data_root = Path.cwd() / data_root
            data_root = data_root.resolve()
        
        preprocessor = BMADPreprocessor(target_size=tuple(config['data']['target_size']))
        
        (train_paths, val_images, val_labels, val_masks,
         test_images, test_labels, test_masks) = load_dataset_paths(str(data_root))
        
        train_dataset = BMADDataset(
            image_paths=train_paths,
            labels=[0] * len(train_paths),
            mask_paths=None,
            preprocessor=preprocessor,
            augment=False,
            is_training=False
        )
        
        print(f"✓ Loaded {len(train_dataset)} training images")
        
        # Extract embeddings
        with torch.no_grad():
            data_embeddings = extract_training_embeddings(
                model,
                train_dataset,
                n_samples=min(args.n_samples, len(train_dataset)),
                batch_size=args.batch_size,
                device=device
            )
        
        # Get projected anchors from model
        with torch.no_grad():
            projected_anchors, _ = model._get_projected_anchors()
        
        # Analyze
        stats = analyze_anchor_perspective(
            projected_anchors.cpu(),
            data_embeddings,
            strategy_name
        )
        
        results[strategy_name] = stats
    
    # Comparative summary
    if len(results) > 1:
        print(f"\n\n{'='*80}")
        print("COMPARATIVE SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Strategy':<15} {'Coverage':<12} {'Max Usage':<12} {'Empty':<8} {'Diagnosis'}")
        print("-" * 80)
        for strategy, stats in results.items():
            print(f"{strategy:<15} {stats['coverage_ratio']:<12.4f} "
                  f"{stats['max_usage']:<12.1f}% {stats['empty_anchors']:<8} "
                  f"{stats['diagnosis']}")
        
        print(f"\n{'Strategy':<15} {'Inter-Anchor':<15} {'Data-to-Anchor':<15} {'Ratio'}")
        print("-" * 80)
        for strategy, stats in results.items():
            print(f"{strategy:<15} "
                  f"{stats['mean_inter_anchor']:.4f} ({stats['mean_anchor_angle']:.0f}°)    "
                  f"{stats['mean_data_to_anchor']:.4f} ({stats['mean_data_angle']:.0f}°)    "
                  f"{stats['coverage_ratio']:.4f}")


if __name__ == '__main__':
    main()
