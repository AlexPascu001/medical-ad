"""
Comprehensive Analysis Script to Identify Issues in the Model
Checks for data leakage, distribution shift, overfitting, and implementation bugs
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import argparse

from data import create_dataloaders
from model import DINOv3Backbone, AnomalyDetector
from main import load_dataset_paths, load_config


def analyze_data_distribution(train_loader, val_loader, test_loader, save_dir):
    """Analyze if there's distribution shift between train/val/test"""
    print("\n" + "="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics for each split
    def compute_stats(loader, name):
        means = []
        stds = []
        mins = []
        maxs = []
        
        for batch in tqdm(loader, desc=f'Analyzing {name}'):
            images = batch['image'].numpy()
            
            means.extend(images.mean(axis=(1,2,3)).tolist())
            stds.extend(images.std(axis=(1,2,3)).tolist())
            mins.extend(images.min(axis=(1,2,3)).tolist())
            maxs.extend(images.max(axis=(1,2,3)).tolist())
        
        return {
            'means': np.array(means),
            'stds': np.array(stds),
            'mins': np.array(mins),
            'maxs': np.array(maxs)
        }
    
    train_stats = compute_stats(train_loader, 'Train')
    val_stats = compute_stats(val_loader, 'Val')
    test_stats = compute_stats(test_loader, 'Test')
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean intensity
    axes[0, 0].hist(train_stats['means'], bins=50, alpha=0.5, label='Train', density=True)
    axes[0, 0].hist(val_stats['means'], bins=50, alpha=0.5, label='Val', density=True)
    axes[0, 0].hist(test_stats['means'], bins=50, alpha=0.5, label='Test', density=True)
    axes[0, 0].set_xlabel('Mean Intensity')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Mean Intensity Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std dev
    axes[0, 1].hist(train_stats['stds'], bins=50, alpha=0.5, label='Train', density=True)
    axes[0, 1].hist(val_stats['stds'], bins=50, alpha=0.5, label='Val', density=True)
    axes[0, 1].hist(test_stats['stds'], bins=50, alpha=0.5, label='Test', density=True)
    axes[0, 1].set_xlabel('Std Dev')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Std Dev Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min values
    axes[1, 0].hist(train_stats['mins'], bins=50, alpha=0.5, label='Train', density=True)
    axes[1, 0].hist(val_stats['mins'], bins=50, alpha=0.5, label='Val', density=True)
    axes[1, 0].hist(test_stats['mins'], bins=50, alpha=0.5, label='Test', density=True)
    axes[1, 0].set_xlabel('Min Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Min Value Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Max values
    axes[1, 1].hist(train_stats['maxs'], bins=50, alpha=0.5, label='Train', density=True)
    axes[1, 1].hist(val_stats['maxs'], bins=50, alpha=0.5, label='Val', density=True)
    axes[1, 1].hist(test_stats['maxs'], bins=50, alpha=0.5, label='Test', density=True)
    axes[1, 1].set_xlabel('Max Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Max Value Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'data_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved data distribution analysis to {save_dir / 'data_distribution_analysis.png'}")
    
    # Print statistics
    print("\nStatistics Summary:")
    print(f"Train - Mean: {train_stats['means'].mean():.4f} ± {train_stats['means'].std():.4f}")
    print(f"Val   - Mean: {val_stats['means'].mean():.4f} ± {val_stats['means'].std():.4f}")
    print(f"Test  - Mean: {test_stats['means'].mean():.4f} ± {test_stats['means'].std():.4f}")


def analyze_model_embeddings(model, train_loader, val_loader, test_loader, device, save_dir):
    """Analyze embedding distributions to check for anomalies"""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    def extract_embeddings(loader, name):
        embeddings = []
        labels = []
        distances_to_anchors = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Extracting {name} embeddings'):
                images = batch['image'].to(device)
                batch_labels = batch['label'].numpy()
                
                outputs = model(images, return_dense=False)
                global_feat = outputs['global_feat'].cpu().numpy()
                global_dist = outputs['global_distances'].cpu().numpy()
                
                embeddings.append(global_feat)
                labels.extend(batch_labels)
                distances_to_anchors.append(global_dist)
        
        embeddings = np.concatenate(embeddings, axis=0)
        distances_to_anchors = np.concatenate(distances_to_anchors, axis=0)
        labels = np.array(labels)
        
        return embeddings, labels, distances_to_anchors
    
    train_emb, train_labels, train_dists = extract_embeddings(train_loader, 'Train')
    val_emb, val_labels, val_dists = extract_embeddings(val_loader, 'Val')
    test_emb, test_labels, test_dists = extract_embeddings(test_loader, 'Test')
    
    # Analyze embedding norms
    train_norms = np.linalg.norm(train_emb, axis=1)
    val_norms = np.linalg.norm(val_emb, axis=1)
    test_norms = np.linalg.norm(test_emb, axis=1)
    
    # Plot embedding norms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(train_norms, bins=50, alpha=0.5, label='Train', density=True)
    axes[0].hist(val_norms, bins=50, alpha=0.5, label='Val', density=True)
    axes[0].hist(test_norms, bins=50, alpha=0.5, label='Test', density=True)
    axes[0].set_xlabel('Embedding Norm')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Embedding L2 Norms')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Min distance to anchors
    train_min_dist = train_dists.min(axis=1)
    val_min_dist = val_dists.min(axis=1)
    test_min_dist = test_dists.min(axis=1)
    
    axes[1].hist(train_min_dist, bins=50, alpha=0.5, label='Train', density=True)
    axes[1].hist(val_min_dist[val_labels==0], bins=50, alpha=0.5, label='Val Normal', density=True)
    axes[1].hist(val_min_dist[val_labels==1], bins=50, alpha=0.5, label='Val Anomaly', density=True)
    axes[1].set_xlabel('Min Distance to Anchor')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Minimum Distance to Any Anchor')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'embedding_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved embedding analysis to {save_dir / 'embedding_analysis.png'}")
    
    # Check if train distances are smaller than test (should be for good anchors)
    print(f"\nDistance to Anchors:")
    print(f"  Train:       {train_min_dist.mean():.4f} ± {train_min_dist.std():.4f}")
    print(f"  Val Normal:  {val_min_dist[val_labels==0].mean():.4f} ± {val_min_dist[val_labels==0].std():.4f}")
    print(f"  Val Anomaly: {val_min_dist[val_labels==1].mean():.4f} ± {val_min_dist[val_labels==1].std():.4f}")
    print(f"  Test Normal: {test_min_dist[test_labels==0].mean():.4f} ± {test_min_dist[test_labels==0].std():.4f}")
    print(f"  Test Anomaly:{test_min_dist[test_labels==1].mean():.4f} ± {test_min_dist[test_labels==1].std():.4f}")
    
    # Compute AUROC on each split
    val_auroc = roc_auc_score(val_labels, val_min_dist)
    test_auroc = roc_auc_score(test_labels, test_min_dist)
    
    print(f"\nAUROC using min distance:")
    print(f"  Val:  {val_auroc:.4f}")
    print(f"  Test: {test_auroc:.4f}")
    
    if abs(val_auroc - test_auroc) > 0.1:
        print("\n⚠️  WARNING: Large AUROC gap between val and test!")
        print("   This suggests potential distribution shift or data issues.")


def check_val_test_overlap(val_paths, test_paths):
    """Check if there's any overlap between val and test sets"""
    print("\n" + "="*80)
    print("VAL/TEST OVERLAP CHECK")
    print("="*80)
    
    val_names = set([Path(p).name for p in val_paths])
    test_names = set([Path(p).name for p in test_paths])
    
    overlap = val_names.intersection(test_names)
    
    if overlap:
        print(f"⚠️  WARNING: Found {len(overlap)} overlapping files between val and test!")
        print(f"   Examples: {list(overlap)[:5]}")
    else:
        print("✓ No overlap between val and test sets")


def analyze_anchor_quality(model, device, save_dir):
    """Analyze anchor embeddings"""
    print("\n" + "="*80)
    print("ANCHOR QUALITY ANALYSIS")
    print("="*80)
    
    save_dir = Path(save_dir)
    
    # Get anchor embeddings
    anchor_global, anchor_dense = model._get_projected_anchors()
    anchor_global = anchor_global.cpu().numpy()
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    
    pairwise_dists = squareform(pdist(anchor_global, metric='euclidean'))
    
    # Remove diagonal
    np.fill_diagonal(pairwise_dists, np.nan)
    
    print(f"\nAnchor Statistics:")
    print(f"  Number of anchors: {len(anchor_global)}")
    print(f"  Embedding dimension: {anchor_global.shape[1]}")
    print(f"  Min pairwise distance: {np.nanmin(pairwise_dists):.4f}")
    print(f"  Mean pairwise distance: {np.nanmean(pairwise_dists):.4f}")
    print(f"  Max pairwise distance: {np.nanmax(pairwise_dists):.4f}")
    
    # Plot pairwise distance heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pairwise_dists, annot=True, fmt='.3f', cmap='coolwarm', 
                cbar_kws={'label': 'Euclidean Distance'})
    plt.title('Anchor Pairwise Distances')
    plt.xlabel('Anchor Index')
    plt.ylabel('Anchor Index')
    plt.tight_layout()
    plt.savefig(save_dir / 'anchor_pairwise_distances.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved anchor analysis to {save_dir / 'anchor_pairwise_distances.png'}")
    
    # Check if anchors are too close (problematic)
    min_dist = np.nanmin(pairwise_dists)
    if min_dist < 0.1:
        print(f"\n⚠️  WARNING: Some anchors are very close (min dist: {min_dist:.4f})")
        print("   This may indicate redundant anchors or poor anchor generation.")


def main(args):
    """Run comprehensive analysis"""
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data_root = config['data'].get('data_root', '../data/BraTS2021_slice')
    train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(data_root)
    
    # Check overlap
    check_val_test_overlap(val_paths, test_paths)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        val_mask_paths=val_mask_paths,
        test_mask_paths=test_mask_paths,
        batch_size=config['training']['batch_size'],
        num_workers=2,  # Reduce for analysis
        target_size=tuple(config['data']['target_size'])
    )
    
    # Output directory
    save_dir = Path(config['output_dir']) / 'analysis'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze data distribution
    analyze_data_distribution(train_loader, val_loader, test_loader, save_dir)
    
    # Load model
    model_path = Path(config['output_dir']) / 'best_model.pth'
    if not model_path.exists():
        print(f"\n⚠️  Model not found at {model_path}")
        print("Please train the model first.")
        return
    
    # Load anchor embeddings
    anchor_path = Path(config['output_dir']) / 'anchor_embeddings.pt'
    anchor_data = torch.load(anchor_path, weights_only=False)
    anchor_global = anchor_data['anchor_global']
    anchor_dense = anchor_data['anchor_dense']
    
    # Create model
    from main import create_model
    model = create_model(config, anchor_global, anchor_dense)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Analyze embeddings
    analyze_model_embeddings(model, train_loader, val_loader, test_loader, device, save_dir)
    
    # Analyze anchors
    analyze_anchor_quality(model, device, save_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze model and data issues')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    main(args)
