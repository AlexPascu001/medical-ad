"""
Run experiments with different anchor strategies and numbers
"""

import subprocess
import yaml
from pathlib import Path
import argparse


def create_config_variant(base_config: str, strategy: str, n_anchors: int, distance_metric: str, output_path: str):
    """Create a config file variant with specified strategy, n_anchors, and distance metric"""
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update anchor config
    config['anchor']['strategy'] = strategy
    config['anchor']['n_anchors'] = n_anchors
    
    # Update distance metric
    config['loss']['distance_metric'] = distance_metric
    
    # Adjust margin for cosine (typically smaller)
    if distance_metric == 'cosine':
        config['loss']['margin'] = 0.5
    else:
        config['loss']['margin'] = 1.0
    
    # Save variant
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Created config: {output_path}")
    return output_path


def run_experiment(config_path: str, skip_anchors: bool = False):
    """Run a single experiment"""
    # Use venv Python on Windows
    venv_python = Path(__file__).parent.parent / 'venv' / 'Scripts' / 'python.exe'
    
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = 'python'  # Fallback to system python
    
    cmd = [python_cmd, 'main.py', '--config', config_path, '--auto-name']
    if skip_anchors:
        cmd.append('--skip-anchors')
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("="*80)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run anchor strategy experiments')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['random', 'kmeans', 'eigenface'],
                       default=['random', 'kmeans', 'eigenface'],
                       help='Anchor strategies to test')
    parser.add_argument('--n-anchors', nargs='+', type=int,
                       default=[4, 8, 16],
                       help='Numbers of anchors to test')
    parser.add_argument('--distance-metrics', nargs='+',
                       choices=['cosine', 'euclidean'],
                       default=['cosine', 'euclidean'],
                       help='Distance metrics to test')
    parser.add_argument('--base-config', type=str,
                       default='configs/default.yaml',
                       help='Base configuration file')
    parser.add_argument('--skip-anchors', action='store_true',
                       help='Skip anchor generation if exists')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANCHOR STRATEGY COMPARISON EXPERIMENTS")
    print("="*80)
    print(f"Strategies: {args.strategies}")
    print(f"Number of anchors: {args.n_anchors}")
    print(f"Distance metrics: {args.distance_metrics}")
    print(f"Base config: {args.base_config}")
    print("="*80)
    
    # Create temporary configs directory
    temp_dir = Path('configs/temp')
    temp_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Run experiments for each combination
    for strategy in args.strategies:
        for n_anchors in args.n_anchors:
            for distance_metric in args.distance_metrics:
                print(f"\n{'='*80}")
                print(f"EXPERIMENT: {strategy.upper()} K={n_anchors} Distance={distance_metric.upper()}")
                print(f"{'='*80}")
                
                # Create config variant
                dist_abbrev = 'cos' if distance_metric == 'cosine' else 'l2'
                config_name = f"temp_{strategy}_k{n_anchors}_{dist_abbrev}.yaml"
                config_path = temp_dir / config_name
                
                create_config_variant(
                    base_config=args.base_config,
                    strategy=strategy,
                    n_anchors=n_anchors,
                    distance_metric=distance_metric,
                    output_path=str(config_path)
                )
                
                # Run experiment
                returncode = run_experiment(str(config_path), args.skip_anchors)
                
                results.append({
                    'strategy': strategy,
                    'n_anchors': n_anchors,
                    'distance_metric': distance_metric,
                    'status': 'SUCCESS' if returncode == 0 else 'FAILED'
                })
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for result in results:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        dist_abbrev = 'cos' if result['distance_metric'] == 'cosine' else 'l2'
        print(f"{status_symbol} {result['strategy']:10s} K={result['n_anchors']:2d} {dist_abbrev:3s}: {result['status']}")
    print("="*80)
    
    # Print where to find results
    print("\nResults saved in:")
    print("  experiments/bmad_random_k8_cos/")
    print("  experiments/bmad_random_k8_l2/")
    print("  experiments/bmad_kmeans_k8_cos/")
    print("  etc.")


if __name__ == '__main__':
    main()
