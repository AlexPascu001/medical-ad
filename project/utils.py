"""
Utility functions for visualization, metrics, and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import json


def plot_training_history(history_path: str, save_dir: str):
    """
    Plot training history from JSON file
    
    Args:
        history_path: Path to training_history.json
        save_dir: Directory to save plots
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(alpha=0.3)
    
    # Validation loss
    if 'val_loss' in history:
        axes[0, 1].plot(history['val_loss'], linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].grid(alpha=0.3)
    
    # Image AUROC
    if 'val_image_auroc' in history:
        axes[1, 0].plot(history['val_image_auroc'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUROC')
        axes[1, 0].set_title('Validation Image AUROC')
        axes[1, 0].grid(alpha=0.3)
    
    # Pixel AUROC
    if 'val_pixel_auroc' in history and history['val_pixel_auroc']:
        axes[1, 1].plot(history['val_pixel_auroc'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUROC')
        axes[1, 1].set_title('Validation Pixel AUROC')
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training history plot to {save_dir / 'training_history.png'}")


def compare_experiments(experiment_dirs: List[str], metric: str = 'image_auroc', save_path: str = None):
    """
    Compare multiple experiments
    
    Args:
        experiment_dirs: List of paths to experiment directories
        metric: Metric to compare ('image_auroc', 'pixel_auroc')
        save_path: Path to save comparison plot
    """
    results = {}
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        eval_metrics_path = exp_path / 'evaluation' / 'evaluation_metrics.json'
        
        if eval_metrics_path.exists():
            with open(eval_metrics_path, 'r') as f:
                metrics = json.load(f)
            
            exp_name = exp_path.name
            results[exp_name] = metrics.get(metric, 0.0)
    
    if not results:
        print("No results found!")
        return
    
    # Sort by performance
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    values = list(results.values())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Experiment Comparison: {metric.replace("_", " ").title()}', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print results
    print(f"\n{metric.replace('_', ' ').title()} Comparison:")
    print("-" * 60)
    for rank, (name, value) in enumerate(results.items(), 1):
        print(f"{rank}. {name:30s}: {value:.4f}")


def create_summary_table(experiment_dirs: List[str], save_path: str = None):
    """
    Create summary table of all experiments
    
    Args:
        experiment_dirs: List of paths to experiment directories
        save_path: Path to save summary CSV
    """
    import pandas as pd
    
    rows = []
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        
        # Load config
        config_path = exp_path / 'config.yaml'
        eval_path = exp_path / 'evaluation' / 'evaluation_metrics.json'
        
        if not (config_path.exists() and eval_path.exists()):
            continue
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        with open(eval_path, 'r') as f:
            metrics = json.load(f)
        
        row = {
            'experiment': exp_name,
            'n_anchors': config['anchor']['n_anchors'],
            'n_components': config['anchor']['n_components'],
            'backbone': config['model']['backbone'],
            'freeze_backbone': config['model']['freeze_backbone'],
            'margin_attract': config['loss']['margin_attract'],
            'margin_repel': config['loss']['margin_repel'],
            'image_auroc': metrics.get('image_auroc', 0.0),
            'image_aupr': metrics.get('image_aupr', 0.0),
            'pixel_auroc': metrics.get('pixel_auroc', 0.0),
            'pixel_aupr': metrics.get('pixel_aupr', 0.0)
        }
        
        # Add CI if available
        if 'confidence_intervals' in metrics:
            ci = metrics['confidence_intervals']
            row['auroc_ci_lower'] = ci.get('auroc_lower', 0.0)
            row['auroc_ci_upper'] = ci.get('auroc_upper', 0.0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('image_auroc', ascending=False)
    
    # Print table
    print("\nExperiment Summary:")
    print("=" * 120)
    print(df.to_string(index=False))
    
    # Save to CSV
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nSaved summary table to {save_path}")
    
    return df


def visualize_anchor_diversity(anchor_images: np.ndarray, save_path: str):
    """
    Visualize diversity between anchors using pairwise distances
    
    Args:
        anchor_images: (K, H, W) anchor images
        save_path: Path to save visualization
    """
    K = len(anchor_images)
    
    # Flatten and compute pairwise distances
    anchors_flat = anchor_images.reshape(K, -1)
    
    # Cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(anchors_flat)
    distance = 1 - similarity
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance heatmap
    sns.heatmap(distance, annot=True, fmt='.3f', cmap='YlOrRd', 
                square=True, ax=axes[0], cbar_kws={'label': 'Distance'})
    axes[0].set_title('Pairwise Anchor Distances', fontsize=14)
    axes[0].set_xlabel('Anchor ID')
    axes[0].set_ylabel('Anchor ID')
    
    # Distance distribution
    upper_tri_indices = np.triu_indices(K, k=1)
    distances_flat = distance[upper_tri_indices]
    
    axes[1].hist(distances_flat, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(distances_flat.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {distances_flat.mean():.3f}')
    axes[1].set_xlabel('Distance', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Pairwise Distances', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Mean pairwise distance: {distances_flat.mean():.4f} ± {distances_flat.std():.4f}")
    print(f"Min distance: {distances_flat.min():.4f}")
    print(f"Max distance: {distances_flat.max():.4f}")
    print(f"Saved diversity analysis to {save_path}")


def export_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str
):
    """
    Export all predictions to a file for external analysis
    
    Args:
        model: Anomaly detector
        dataloader: Test dataloader
        device: Device
        save_path: Path to save predictions (NPZ file)
    """
    model.eval()
    
    all_paths = []
    all_labels = []
    all_scores = []
    all_assigned_anchors = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            paths = batch['path']
            
            outputs = model.compute_anomaly_scores(images, return_maps=False)
            scores = outputs['image_scores'].cpu().numpy()
            assigned = outputs['assigned_anchors'].cpu().numpy()
            
            all_paths.extend(paths)
            all_labels.append(labels)
            all_scores.append(scores)
            all_assigned_anchors.append(assigned)
    
    # Concatenate
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_assigned_anchors = np.concatenate(all_assigned_anchors)
    
    # Save
    np.savez(
        save_path,
        paths=np.array(all_paths),
        labels=all_labels,
        scores=all_scores,
        assigned_anchors=all_assigned_anchors
    )
    
    print(f"Exported {len(all_paths)} predictions to {save_path}")


def analyze_failure_cases(
    predictions_path: str,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    save_dir: str,
    top_k: int = 10
):
    """
    Analyze top-K false positives and false negatives
    
    Args:
        predictions_path: Path to predictions NPZ file
        dataloader: Test dataloader
        model: Model for generating visualizations
        device: Device
        save_dir: Save directory
        top_k: Number of top failures to analyze
    """
    # Load predictions
    data = np.load(predictions_path, allow_pickle=True)
    paths = data['paths']
    labels = data['labels']
    scores = data['scores']
    
    # Find failures
    # False positives: normal (label=0) with high score
    fp_indices = np.where(labels == 0)[0]
    fp_scores = scores[fp_indices]
    top_fp = fp_indices[np.argsort(fp_scores)[-top_k:]]
    
    # False negatives: anomaly (label=1) with low score
    fn_indices = np.where(labels == 1)[0]
    fn_scores = scores[fn_indices]
    top_fn = fn_indices[np.argsort(fn_scores)[:top_k]]
    
    print(f"\nTop-{top_k} False Positives:")
    for i, idx in enumerate(reversed(top_fp)):
        print(f"  {i+1}. {paths[idx]} - Score: {scores[idx]:.4f}")
    
    print(f"\nTop-{top_k} False Negatives:")
    for i, idx in enumerate(top_fn):
        print(f"  {i+1}. {paths[idx]} - Score: {scores[idx]:.4f}")
    
    # TODO: Add visualization of these cases


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Plot training history:")
        print("    python utils.py plot_history <experiment_dir>")
        print("  Compare experiments:")
        print("    python utils.py compare <exp_dir1> <exp_dir2> ... [--metric image_auroc]")
        print("  Create summary table:")
        print("    python utils.py summary <exp_dir1> <exp_dir2> ... [--output summary.csv]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'plot_history':
        exp_dir = sys.argv[2]
        history_path = Path(exp_dir) / 'training_history.json'
        if history_path.exists():
            plot_training_history(str(history_path), str(Path(exp_dir) / 'plots'))
        else:
            print(f"History file not found: {history_path}")
    
    elif command == 'compare':
        exp_dirs = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
        metric = 'image_auroc'
        
        if '--metric' in sys.argv:
            metric = sys.argv[sys.argv.index('--metric') + 1]
        
        compare_experiments(exp_dirs, metric=metric, save_path='comparison.png')
    
    elif command == 'summary':
        exp_dirs = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
        output = 'summary.csv'
        
        if '--output' in sys.argv:
            output = sys.argv[sys.argv.index('--output') + 1]
        
        create_summary_table(exp_dirs, save_path=output)
    
    else:
        print(f"Unknown command: {command}")