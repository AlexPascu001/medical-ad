"""
Compare Results Across Learnable Anchor Experiments

This script loads results from multiple experiments and creates comparison visualizations:
1. Bar chart comparing Image AUROC across experiments
2. Table with all metrics
3. Statistical analysis

Usage:
    python compare_experiment_results.py --experiments-dir experiments
    python compare_experiment_results.py --experiments bmad_learnable_random_fixed bmad_learnable_kmeans_fixed
"""

import argparse
import torch
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def load_experiment_results(exp_dir: Path) -> Optional[Dict]:
    """
    Load results from a single experiment directory.
    
    Returns dict with:
        - name: experiment name
        - config: configuration dict
        - best_auroc: best validation image AUROC
        - final_auroc: final image AUROC
        - history: training history (if available)
    """
    exp_dir = Path(exp_dir)
    
    # Check required files
    config_path = exp_dir / 'config.yaml'
    model_path = exp_dir / 'best_model.pth'
    
    if not config_path.exists() or not model_path.exists():
        return None
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint for history
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    history = checkpoint.get('history', {})
    
    # Extract metrics
    val_auroc = history.get('val_image_auroc', [])
    val_pixel = history.get('val_pixel_auroc', [])
    
    best_auroc = max(val_auroc) if val_auroc else 0.0
    final_auroc = val_auroc[-1] if val_auroc else 0.0
    
    best_pixel = max([x for x in val_pixel if x > 0]) if val_pixel and any(x > 0 for x in val_pixel) else 0.0
    
    # Parse experiment details from config
    strategy = config.get('anchor', {}).get('strategy', 'unknown')
    learnable = config.get('anchor', {}).get('learnable', False)
    n_anchors = config.get('anchor', {}).get('n_anchors', 8)
    dynamic = config.get('training', {}).get('dynamic_reassignment', False)
    
    return {
        'name': exp_dir.name,
        'path': str(exp_dir),
        'strategy': strategy,
        'learnable': learnable,
        'n_anchors': n_anchors,
        'dynamic': dynamic,
        'best_image_auroc': best_auroc,
        'final_image_auroc': final_auroc,
        'best_pixel_auroc': best_pixel,
        'epochs_trained': len(val_auroc),
        'config': config,
        'history': history
    }


def find_experiments(base_dir: Path, pattern: str = 'bmad_*') -> List[Path]:
    """Find all experiment directories matching pattern."""
    base_dir = Path(base_dir)
    experiments = []
    
    for exp_dir in sorted(base_dir.glob(pattern)):
        if exp_dir.is_dir() and (exp_dir / 'config.yaml').exists():
            experiments.append(exp_dir)
    
    return experiments


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame with comparison metrics."""
    data = []
    
    for r in results:
        data.append({
            'Experiment': r['name'],
            'Strategy': r['strategy'],
            'Learnable': '✓' if r['learnable'] else '-',
            'Dynamic': '✓' if r['dynamic'] else '-',
            'K': r['n_anchors'],
            'Best Image AUROC': f"{r['best_image_auroc']:.4f}",
            'Best Pixel AUROC': f"{r['best_pixel_auroc']:.4f}" if r['best_pixel_auroc'] > 0 else '-',
            'Epochs': r['epochs_trained']
        })
    
    df = pd.DataFrame(data)
    return df


def plot_auroc_comparison(results: List[Dict], save_path: Path):
    """Create bar chart comparing AUROC across experiments."""
    # Sort by AUROC
    results_sorted = sorted(results, key=lambda x: x['best_image_auroc'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    names = [r['name'].replace('bmad_', '') for r in results_sorted]
    aurocs = [r['best_image_auroc'] for r in results_sorted]
    
    # Color by learnable/fixed
    colors = ['steelblue' if r['learnable'] else 'coral' for r in results_sorted]
    
    bars = ax.bar(range(len(names)), aurocs, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Image AUROC', fontsize=12)
    ax.set_title('Image AUROC Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, auroc in zip(bars, aurocs):
        height = bar.get_height()
        ax.annotate(f'{auroc:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='Learnable Anchors'),
        Patch(facecolor='coral', edgecolor='black', label='Fixed Anchors')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_strategy_comparison(results: List[Dict], save_path: Path):
    """Create grouped bar chart comparing strategies."""
    # Group by strategy and learnable
    strategies = ['random', 'kmeans', 'eigenface']
    
    fixed_aurocs = {s: [] for s in strategies}
    learnable_fixed_aurocs = {s: [] for s in strategies}
    learnable_dynamic_aurocs = {s: [] for s in strategies}
    
    for r in results:
        strategy = r['strategy']
        if strategy not in strategies:
            continue
            
        auroc = r['best_image_auroc']
        
        if not r['learnable']:
            fixed_aurocs[strategy].append(auroc)
        elif r['dynamic']:
            learnable_dynamic_aurocs[strategy].append(auroc)
        else:
            learnable_fixed_aurocs[strategy].append(auroc)
    
    # Calculate means
    x = np.arange(len(strategies))
    width = 0.25
    
    fixed_means = [np.mean(fixed_aurocs[s]) if fixed_aurocs[s] else 0 for s in strategies]
    learnable_fixed_means = [np.mean(learnable_fixed_aurocs[s]) if learnable_fixed_aurocs[s] else 0 for s in strategies]
    learnable_dynamic_means = [np.mean(learnable_dynamic_aurocs[s]) if learnable_dynamic_aurocs[s] else 0 for s in strategies]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, fixed_means, width, label='Fixed Anchors', color='coral', edgecolor='black')
    bars2 = ax.bar(x, learnable_fixed_means, width, label='Learnable (Fixed Labels)', color='steelblue', edgecolor='black')
    bars3 = ax.bar(x + width, learnable_dynamic_means, width, label='Learnable (Dynamic Labels)', color='forestgreen', edgecolor='black')
    
    ax.set_ylabel('Image AUROC', fontsize=12)
    ax.set_title('AUROC by Strategy and Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strategies], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_training_comparison(results: List[Dict], save_path: Path, max_experiments: int = 6):
    """Plot training curves for multiple experiments."""
    # Select experiments with training history
    results_with_history = [r for r in results if r['history'].get('val_image_auroc')]
    
    if len(results_with_history) > max_experiments:
        # Sort by AUROC and take top N
        results_with_history = sorted(
            results_with_history, 
            key=lambda x: x['best_image_auroc'], 
            reverse=True
        )[:max_experiments]
    
    if not results_with_history:
        print("No training history available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_with_history)))
    
    for i, r in enumerate(results_with_history):
        history = r['history']
        val_auroc = history.get('val_image_auroc', [])
        
        if val_auroc:
            epochs = range(1, len(val_auroc) + 1)
            label = r['name'].replace('bmad_', '')[:30]  # Truncate long names
            ax.plot(epochs, val_auroc, color=colors[i], linewidth=2, 
                   label=f"{label} ({r['best_image_auroc']:.3f})", alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Image AUROC', fontsize=12)
    ax.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Directory containing experiment folders')
    parser.add_argument('--experiments', nargs='+',
                        help='Specific experiment names to compare')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Directory to save comparison outputs')
    parser.add_argument('--pattern', type=str, default='bmad_*',
                        help='Pattern to match experiment directories')
    
    args = parser.parse_args()
    
    # Find experiments
    base_dir = Path(__file__).parent.parent / args.experiments_dir
    
    if args.experiments:
        exp_dirs = [base_dir / exp for exp in args.experiments]
    else:
        exp_dirs = find_experiments(base_dir, args.pattern)
    
    print(f"Found {len(exp_dirs)} experiments")
    
    # Load results
    results = []
    for exp_dir in exp_dirs:
        result = load_experiment_results(exp_dir)
        if result:
            results.append(result)
            print(f"  Loaded: {result['name']} (AUROC: {result['best_image_auroc']:.4f})")
        else:
            print(f"  Skipped: {exp_dir.name} (missing files)")
    
    if not results:
        print("\nNo valid experiments found!")
        return
    
    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison table
    df = create_comparison_table(results)
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Save table
    df.to_csv(output_dir / 'comparison_table.csv', index=False)
    print(f"\nSaved: {output_dir / 'comparison_table.csv'}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_auroc_comparison(results, output_dir / 'auroc_comparison.png')
    plot_strategy_comparison(results, output_dir / 'strategy_comparison.png')
    plot_training_comparison(results, output_dir / 'training_comparison.png')
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    learnable_results = [r for r in results if r['learnable']]
    fixed_results = [r for r in results if not r['learnable']]
    
    if learnable_results:
        learnable_aurocs = [r['best_image_auroc'] for r in learnable_results]
        print(f"\nLearnable Anchors:")
        print(f"  Count: {len(learnable_results)}")
        print(f"  Mean AUROC: {np.mean(learnable_aurocs):.4f}")
        print(f"  Std AUROC: {np.std(learnable_aurocs):.4f}")
        print(f"  Best: {max(learnable_aurocs):.4f}")
    
    if fixed_results:
        fixed_aurocs = [r['best_image_auroc'] for r in fixed_results]
        print(f"\nFixed Anchors:")
        print(f"  Count: {len(fixed_results)}")
        print(f"  Mean AUROC: {np.mean(fixed_aurocs):.4f}")
        print(f"  Std AUROC: {np.std(fixed_aurocs):.4f}")
        print(f"  Best: {max(fixed_aurocs):.4f}")
    
    # Best overall
    best_result = max(results, key=lambda x: x['best_image_auroc'])
    print(f"\nBest Overall: {best_result['name']}")
    print(f"  Strategy: {best_result['strategy']}")
    print(f"  Learnable: {best_result['learnable']}")
    print(f"  Dynamic: {best_result['dynamic']}")
    print(f"  Image AUROC: {best_result['best_image_auroc']:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
