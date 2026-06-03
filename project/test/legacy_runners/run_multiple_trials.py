"""
Run multiple trials for each anchor strategy to get robust statistics

This script:
- Runs N trials for each anchor strategy (eigenface, kmeans, random)
- Uses different random seeds for each trial
- Saves all models and results
- Computes mean ± std for all metrics
- Generates comparison visualizations
"""

import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import argparse
from typing import Dict, List
import shutil


def run_single_trial(
    strategy: str,
    trial_id: int,
    seed: int,
    base_config: Dict,
    venv_python: Path,
    skip_training: bool = False
) -> Dict:
    """Run a single training trial"""
    
    # Create experiment name
    exp_name = f"bmad_{strategy}_k8_l2_trial{trial_id}"
    exp_dir = Path('experiments') / exp_name
    
    print(f"\n{'='*80}")
    print(f"TRIAL {trial_id}: {strategy.upper()} (seed={seed})")
    print(f"{'='*80}")
    print(f"Experiment: {exp_name}")
    
    # Check if experiment already exists
    if exp_dir.exists():
        eval_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
        if eval_path.exists():
            print(f"\n✓ Experiment already exists and has results!")
            print(f"  Loading existing results from: {eval_path}")
            
            with open(eval_path, 'r') as f:
                metrics = json.load(f)
            
            return {
                'strategy': strategy,
                'trial_id': trial_id,
                'seed': seed,
                'exp_name': exp_name,
                'exp_dir': str(exp_dir),
                'metrics': metrics
            }
        else:
            print(f"\n→ Experiment directory exists but no results found. Re-running...")
    
    # Update config for this trial
    config = base_config.copy()
    config['output_dir'] = f'./experiments/{exp_name}'
    config['seed'] = seed
    config['anchor']['strategy'] = strategy
    
    # Fix data path to be absolute or relative to root directory
    data_root = config['data']['data_root']
    if data_root.startswith('../'):
        # Change from ../data to ./data (since we run from root, not project/)
        config['data']['data_root'] = data_root.replace('../', './')
    
    # Save config
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Config saved: {config_path}")
    
    # Skip training if requested
    if skip_training:
        print(f"\n⊘ Skipping training (--skip-training flag set)")
        eval_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
        if not eval_path.exists():
            print(f"✗ No evaluation results found: {eval_path}")
            return None
    else:
        # Run training
        print(f"\n→ Starting training...")
        print(f"   Command: {' '.join([str(venv_python), 'project/main.py', '--config', str(config_path)])}")
        print(f"   (This will take several minutes per trial...)\n")
    
        cmd = [
            str(venv_python),
            'project/main.py',
            '--config', str(config_path)
        ]
        
        # Run with real-time output
        result = subprocess.run(cmd, text=True)
        
        if result.returncode != 0:
            print(f"\n✗ Training failed with exit code {result.returncode}!")
            return None
    
    print(f"\n✓ Training complete!")
    
    # Load results
    eval_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
    if not eval_path.exists():
        print(f"✗ Evaluation metrics not found: {eval_path}")
        return None
    
    with open(eval_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"\n✓ Results:")
    print(f"  Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"  Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    
    return {
        'strategy': strategy,
        'trial_id': trial_id,
        'seed': seed,
        'exp_name': exp_name,
        'exp_dir': str(exp_dir),
        'metrics': metrics
    }


def aggregate_results(all_results: List[Dict]) -> pd.DataFrame:
    """Aggregate results across all trials"""
    
    # Convert to DataFrame
    rows = []
    for result in all_results:
        if result is None:
            continue
        row = {
            'strategy': result['strategy'],
            'trial_id': result['trial_id'],
            'seed': result['seed'],
            'exp_name': result['exp_name'],
        }
        # Add all metrics
        for key, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                row[key] = value
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def compute_statistics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute mean ± std for each strategy"""
    
    # Get metric columns (exclude metadata)
    metric_cols = [col for col in df.columns 
                   if col not in ['strategy', 'trial_id', 'seed', 'exp_name']]
    
    # Compute statistics
    stats_list = []
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        
        stats = {'strategy': strategy, 'n_trials': len(strategy_df)}
        for col in metric_cols:
            values = strategy_df[col].values
            stats[f'{col}_mean'] = np.mean(values)
            stats[f'{col}_std'] = np.std(values, ddof=1)
            stats[f'{col}_min'] = np.min(values)
            stats[f'{col}_max'] = np.max(values)
        
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    # Save statistics
    stats_path = output_dir / 'statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✓ Statistics saved: {stats_path}")
    
    return stats_df


def find_best_models(df: pd.DataFrame, metric: str = 'image_auroc') -> Dict:
    """Find best model for each strategy based on a metric"""
    
    best_models = {}
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        best_idx = strategy_df[metric].idxmax()
        best_row = strategy_df.loc[best_idx]
        
        best_models[strategy] = {
            'exp_name': best_row['exp_name'],
            'trial_id': best_row['trial_id'],
            'seed': best_row['seed'],
            metric: best_row[metric]
        }
    
    return best_models


def visualize_results(df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations"""
    
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Set style
    sns.set_style("whitegrid")
    colors = {'eigenface': '#FF6B6B', 'kmeans': '#4ECDC4', 'random': '#95E1D3'}
    
    # 1. Image AUROC comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax = axes[0]
    df_sorted = df.sort_values('strategy')
    sns.boxplot(data=df_sorted, x='strategy', y='image_auroc', ax=ax, palette=colors)
    sns.stripplot(data=df_sorted, x='strategy', y='image_auroc', ax=ax, 
                  color='black', alpha=0.5, size=8)
    ax.set_ylabel('Image AUROC', fontsize=12)
    ax.set_xlabel('Anchor Strategy', fontsize=12)
    ax.set_title('Image AUROC Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Bar plot with error bars
    ax = axes[1]
    x_pos = np.arange(len(stats_df))
    means = [stats_df[stats_df['strategy'] == s]['image_auroc_mean'].values[0] 
             for s in ['eigenface', 'kmeans', 'random']]
    stds = [stats_df[stats_df['strategy'] == s]['image_auroc_std'].values[0] 
            for s in ['eigenface', 'kmeans', 'random']]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.8,
                  color=[colors[s] for s in ['eigenface', 'kmeans', 'random']])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Eigenface', 'K-Means', 'Random'])
    ax.set_ylabel('Image AUROC', fontsize=12)
    ax.set_xlabel('Anchor Strategy', fontsize=12)
    ax.set_title('Image AUROC: Mean ± Std', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_path = output_dir / 'image_auroc_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # 2. Pixel AUROC comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax = axes[0]
    sns.boxplot(data=df_sorted, x='strategy', y='pixel_auroc', ax=ax, palette=colors)
    sns.stripplot(data=df_sorted, x='strategy', y='pixel_auroc', ax=ax,
                  color='black', alpha=0.5, size=8)
    ax.set_ylabel('Pixel AUROC', fontsize=12)
    ax.set_xlabel('Anchor Strategy', fontsize=12)
    ax.set_title('Pixel AUROC Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Bar plot with error bars
    ax = axes[1]
    means = [stats_df[stats_df['strategy'] == s]['pixel_auroc_mean'].values[0]
             for s in ['eigenface', 'kmeans', 'random']]
    stds = [stats_df[stats_df['strategy'] == s]['pixel_auroc_std'].values[0]
            for s in ['eigenface', 'kmeans', 'random']]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.8,
                  color=[colors[s] for s in ['eigenface', 'kmeans', 'random']])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Eigenface', 'K-Means', 'Random'])
    ax.set_ylabel('Pixel AUROC', fontsize=12)
    ax.set_xlabel('Anchor Strategy', fontsize=12)
    ax.set_title('Pixel AUROC: Mean ± Std', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_path = output_dir / 'pixel_auroc_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # 3. Combined metrics heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap - only use available metrics
    available_metrics = [col for col in stats_df.columns if col.endswith('_mean') and col != 'n_trials']
    metrics_to_plot = [m for m in ['image_auroc_mean', 'pixel_auroc_mean'] if m in available_metrics]
    
    if metrics_to_plot:
        heatmap_data = stats_df[['strategy'] + metrics_to_plot].set_index('strategy')
        # Create readable column names
        column_mapping = {
            'image_auroc_mean': 'Image AUROC',
            'pixel_auroc_mean': 'Pixel AUROC',
            'image_f1_mean': 'Image F1',
            'pixel_f1_mean': 'Pixel F1'
        }
        heatmap_data.columns = [column_mapping.get(col, col) for col in heatmap_data.columns]
        heatmap_data = heatmap_data.loc[['eigenface', 'kmeans', 'random']]
        heatmap_data.index = ['Eigenface', 'K-Means', 'Random']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('Mean Performance Across Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Anchor Strategy', fontsize=12)
        ax.set_xlabel('Metric', fontsize=12)
        
        plt.tight_layout()
        fig_path = output_dir / 'metrics_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        plt.close()
    else:
        print("⊘ Skipping heatmap - no metrics available")
    
    # 4. Trial-by-trial comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for strategy in ['eigenface', 'kmeans', 'random']:
        strategy_df = df[df['strategy'] == strategy].sort_values('trial_id')
        ax.plot(strategy_df['trial_id'], strategy_df['image_auroc'], 
                marker='o', linewidth=2, markersize=8, label=strategy.capitalize(),
                color=colors[strategy])
    
    ax.set_xlabel('Trial ID', fontsize=12)
    ax.set_ylabel('Image AUROC', fontsize=12)
    ax.set_title('Image AUROC Across Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / 'trials_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


def create_summary_report(df: pd.DataFrame, stats_df: pd.DataFrame, 
                         best_models: Dict, output_dir: Path):
    """Create a comprehensive text summary"""
    
    report_path = output_dir / 'RESULTS_SUMMARY.md'
    
    with open(report_path, 'w') as f:
        f.write("# Multiple Trial Experiment Results\n\n")
        f.write(f"**Total Trials:** {len(df)}\n")
        f.write(f"**Strategies:** {', '.join(df['strategy'].unique())}\n")
        f.write(f"**Trials per Strategy:** {len(df) // len(df['strategy'].unique())}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("### Image-Level AUROC\n\n")
        f.write("| Strategy | Mean | Std | Min | Max |\n")
        f.write("|----------|------|-----|-----|-----|\n")
        for _, row in stats_df.iterrows():
            f.write(f"| {row['strategy'].capitalize():10s} | "
                   f"{row['image_auroc_mean']:.4f} | "
                   f"{row['image_auroc_std']:.4f} | "
                   f"{row['image_auroc_min']:.4f} | "
                   f"{row['image_auroc_max']:.4f} |\n")
        
        f.write("\n### Pixel-Level AUROC\n\n")
        f.write("| Strategy | Mean | Std | Min | Max |\n")
        f.write("|----------|------|-----|-----|-----|\n")
        for _, row in stats_df.iterrows():
            f.write(f"| {row['strategy'].capitalize():10s} | "
                   f"{row['pixel_auroc_mean']:.4f} | "
                   f"{row['pixel_auroc_std']:.4f} | "
                   f"{row['pixel_auroc_min']:.4f} | "
                   f"{row['pixel_auroc_max']:.4f} |\n")
        
        f.write("\n## Best Models (by Image AUROC)\n\n")
        for strategy, info in best_models.items():
            f.write(f"### {strategy.capitalize()}\n")
            f.write(f"- **Experiment:** `{info['exp_name']}`\n")
            f.write(f"- **Trial ID:** {info['trial_id']}\n")
            f.write(f"- **Seed:** {info['seed']}\n")
            f.write(f"- **Image AUROC:** {info['image_auroc']:.4f}\n\n")
        
        f.write("\n## Individual Trial Results\n\n")
        f.write("| Strategy | Trial | Seed | Image AUROC | Pixel AUROC |\n")
        f.write("|----------|-------|------|-------------|-------------|\n")
        for _, row in df.sort_values(['strategy', 'trial_id']).iterrows():
            f.write(f"| {row['strategy']:10s} | "
                   f"{row['trial_id']:5d} | "
                   f"{row['seed']:5d} | "
                   f"{row['image_auroc']:.4f} | "
                   f"{row['pixel_auroc']:.4f} |\n")
    
    print(f"\n✓ Summary report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run multiple trials for anchor strategies')
    parser.add_argument('--n-trials', type=int, default=5,
                       help='Number of trials per strategy')
    parser.add_argument('--strategies', type=str, nargs='+',
                       default=['eigenface', 'kmeans', 'random'],
                       help='Anchor strategies to test')
    parser.add_argument('--base-config', type=str,
                       default='project/configs/default.yaml',
                       help='Base configuration file')
    parser.add_argument('--output-dir', type=str,
                       default='experiments/multi_trial_results',
                       help='Directory to save aggregated results')
    parser.add_argument('--start-seed', type=int, default=42,
                       help='Starting random seed (will use seed, seed+1, seed+2, ...)')
    parser.add_argument('--venv-python', type=str,
                       default='venv/Scripts/python.exe',
                       help='Path to venv Python executable')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments that already have results')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training phase and only analyze existing experiments')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("MULTIPLE TRIAL EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Trials per strategy: {args.n_trials}")
    print(f"Total experiments: {len(args.strategies) * args.n_trials}")
    print(f"Starting seed: {args.start_seed}")
    
    # Load base config
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    venv_python = Path(args.venv_python)
    if not venv_python.exists():
        print(f"\n✗ Python executable not found: {venv_python}")
        print("Please activate your virtual environment or specify correct path")
        return
    
    # Run all trials
    all_results = []
    seed = args.start_seed
    
    for strategy in args.strategies:
        for trial_id in range(args.n_trials):
            result = run_single_trial(
                strategy=strategy,
                trial_id=trial_id,
                seed=seed,
                base_config=base_config,
                venv_python=venv_python,
                skip_training=args.skip_training
            )
            
            if result is not None:
                all_results.append(result)
            
            seed += 1
    
    # Check if we have results
    if len(all_results) == 0:
        print("\n✗ No successful trials!")
        return
    
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}")
    print(f"Successful trials: {len(all_results)} / {len(args.strategies) * args.n_trials}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results
    df = aggregate_results(all_results)
    
    # Save raw results
    df.to_csv(output_dir / 'all_trials.csv', index=False)
    print(f"✓ All trials saved: {output_dir / 'all_trials.csv'}")
    
    # Compute statistics
    stats_df = compute_statistics(df, output_dir)
    
    # Print statistics
    print(f"\n{'='*80}")
    print("STATISTICS SUMMARY")
    print(f"{'='*80}")
    for _, row in stats_df.iterrows():
        print(f"\n{row['strategy'].upper()}: (n={int(row['n_trials'])})")
        print(f"  Image AUROC: {row['image_auroc_mean']:.4f} ± {row['image_auroc_std']:.4f}")
        print(f"  Pixel AUROC: {row['pixel_auroc_mean']:.4f} ± {row['pixel_auroc_std']:.4f}")
    
    # Find best models
    best_models = find_best_models(df, metric='image_auroc')
    print(f"\n{'='*80}")
    print("BEST MODELS (by Image AUROC)")
    print(f"{'='*80}")
    for strategy, info in best_models.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Experiment: {info['exp_name']}")
        print(f"  Image AUROC: {info['image_auroc']:.4f}")
    
    # Copy best models to a dedicated directory
    best_models_dir = output_dir / 'best_models'
    best_models_dir.mkdir(exist_ok=True)
    
    for strategy, info in best_models.items():
        src_dir = Path('experiments') / info['exp_name']
        dst_dir = best_models_dir / strategy
        
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        
        shutil.copytree(src_dir, dst_dir)
        print(f"✓ Copied best {strategy} model to: {dst_dir}")
    
    # Generate visualizations
    visualize_results(df, stats_df, output_dir)
    
    # Create summary report
    create_summary_report(df, stats_df, best_models, output_dir)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Results directory: {output_dir}")
    print(f"Best models directory: {best_models_dir}")
    print(f"\nGenerated files:")
    print(f"  - all_trials.csv: All trial results")
    print(f"  - statistics.csv: Mean ± std for each strategy")
    print(f"  - RESULTS_SUMMARY.md: Comprehensive text report")
    print(f"  - *.png: Visualization plots")
    print(f"  - best_models/: Best models for each strategy")


if __name__ == '__main__':
    main()
