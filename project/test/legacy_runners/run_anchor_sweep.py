"""
Run experiments sweeping over number of anchors for different strategies.
"""

import subprocess
import yaml
from pathlib import Path
import argparse
import sys


def run_anchor_sweep_experiment(
    strategy: str,
    n_anchors: int,
    base_config_path: str,
    venv_python: Path,
    seed: int = 42
):
    """Run a single experiment with specific strategy and number of anchors"""
    
    exp_name = f"bmad_{strategy}_k{n_anchors}_l2_hyperparam_sweep"
    exp_dir = Path('experiments') / exp_name
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {strategy.upper()} with K={n_anchors}")
    print(f"{'='*80}")
    print(f"Output: {exp_dir}")
    
    # Check if experiment already exists
    if exp_dir.exists():
        eval_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
        if eval_path.exists():
            print(f"\n✓ Experiment already exists with results!")
            print(f"  Skipping to avoid overwriting existing results.")
            return True
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for this experiment
    config['output_dir'] = f'./experiments/{exp_name}'
    config['seed'] = seed
    config['anchor']['strategy'] = strategy
    config['anchor']['n_anchors'] = n_anchors
    
    # Ensure data path is correct
    if config['data']['data_root'].startswith('../'):
        config['data']['data_root'] = config['data']['data_root'].replace('../', './')
    
    # Save config
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Config saved: {config_path}")
    
    # Run training
    print(f"\n→ Starting training...")
    print(f"   Strategy: {strategy}")
    print(f"   Anchors: {n_anchors}")
    print(f"   Seed: {seed}")
    
    cmd = [
        str(venv_python),
        'project/main.py',
        '--config', str(config_path)
    ]
    
    print(f"   Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print(f"\n✗ Training failed with exit code {result.returncode}!")
        return False
    
    print(f"\n✓ Training complete!")
    
    # Check evaluation results
    eval_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
    if not eval_path.exists():
        print(f"✗ Evaluation metrics not found: {eval_path}")
        return False
    
    print(f"✓ Evaluation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run anchor number sweep experiments')
    parser.add_argument('--strategies', type=str, nargs='+',
                       default=['random', 'kmeans'],
                       help='Anchor strategies to test')
    parser.add_argument('--n-anchors', type=int, nargs='+',
                       default=[2, 4, 8],
                       help='Number of anchors to test')
    parser.add_argument('--base-config', type=str,
                       default='project/configs/default.yaml',
                       help='Base configuration file')
    parser.add_argument('--venv-python', type=str,
                       default='venv/Scripts/python.exe',
                       help='Path to venv Python executable')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments that already have results')
    
    args = parser.parse_args()
    
    venv_python = Path(args.venv_python)
    if not venv_python.exists():
        print(f"\n✗ Python executable not found: {venv_python}")
        print("Please activate your virtual environment or specify correct path")
        return
    
    print(f"{'='*80}")
    print("ANCHOR NUMBER SWEEP EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Number of anchors: {', '.join(map(str, args.n_anchors))}")
    print(f"Total experiments: {len(args.strategies) * len(args.n_anchors)}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}")
    
    # Run all experiments
    results = []
    total = len(args.strategies) * len(args.n_anchors)
    completed = 0
    
    for strategy in args.strategies:
        for n_anchors in args.n_anchors:
            success = run_anchor_sweep_experiment(
                strategy=strategy,
                n_anchors=n_anchors,
                base_config_path=args.base_config,
                venv_python=venv_python,
                seed=args.seed
            )
            
            if success:
                completed += 1
            
            results.append({
                'strategy': strategy,
                'n_anchors': n_anchors,
                'success': success
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Successful experiments: {completed} / {total}")
    
    if completed < total:
        print("\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  - {r['strategy']} K={r['n_anchors']}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Group by strategy
    for strategy in args.strategies:
        print(f"\n{strategy.upper()}:")
        strategy_results = [r for r in results if r['strategy'] == strategy and r['success']]
        if strategy_results:
            for r in strategy_results:
                exp_name = f"bmad_{r['strategy']}_k{r['n_anchors']}_l2"
                print(f"  K={r['n_anchors']:2d}: experiments/{exp_name}")
        else:
            print("  No successful experiments")
    
    print(f"\n{'='*80}")
    print("Next steps:")
    print("1. Compare results across different K values")
    print("2. Visualize anchor utilization for each configuration")
    print("3. Analyze which K gives best performance per strategy")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
