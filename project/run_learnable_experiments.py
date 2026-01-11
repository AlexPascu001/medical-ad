"""
Run Learnable Anchor Experiments

This script runs a comprehensive set of experiments to compare:
1. Fixed vs Learnable anchors
2. Different anchor initialization strategies (random, kmeans, eigenface)
3. Fixed vs Dynamic pseudo-label assignment

Experiment Matrix:
┌──────────────┬─────────────────────┬─────────────────────┐
│ Init Strategy│ Fixed Anchors       │ Learnable Anchors   │
│              │                     │ Fixed / Dynamic     │
├──────────────┼─────────────────────┼─────────────────────┤
│ Random       │ fixed_random        │ learnable_random_*  │
│ K-Means      │ fixed_kmeans        │ learnable_kmeans_*  │
│ Eigenface    │ fixed_eigenface     │ learnable_eigen_*   │
└──────────────┴─────────────────────┴─────────────────────┘

Usage:
    python run_learnable_experiments.py --all
    python run_learnable_experiments.py --strategies random kmeans
    python run_learnable_experiments.py --learnable-only
    python run_learnable_experiments.py --fixed-only
"""

import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys


# Define experiment configurations
EXPERIMENTS = {
    # Baseline: Fixed anchors (not learnable)
    'fixed_random': {
        'config': 'configs/fixed_random.yaml',
        'description': 'Fixed anchors, random initialization',
        'learnable': False,
        'strategy': 'random',
        'dynamic': False
    },
    'fixed_kmeans': {
        'config': 'configs/fixed_kmeans.yaml',
        'description': 'Fixed anchors, k-means initialization',
        'learnable': False,
        'strategy': 'kmeans',
        'dynamic': False
    },
    'fixed_eigenface': {
        'config': 'configs/fixed_eigenface.yaml',
        'description': 'Fixed anchors, eigenface initialization',
        'learnable': False,
        'strategy': 'eigenface',
        'dynamic': False
    },
    
    # Learnable anchors with fixed pseudo-labels
    'learnable_random_fixed': {
        'config': 'configs/learnable_random_fixed.yaml',
        'description': 'Learnable anchors, random init, fixed labels',
        'learnable': True,
        'strategy': 'random',
        'dynamic': False
    },
    'learnable_kmeans_fixed': {
        'config': 'configs/learnable_kmeans_fixed.yaml',
        'description': 'Learnable anchors, k-means init, fixed labels',
        'learnable': True,
        'strategy': 'kmeans',
        'dynamic': False
    },
    'learnable_eigenface_fixed': {
        'config': 'configs/learnable_eigenface_fixed.yaml',
        'description': 'Learnable anchors, eigenface init, fixed labels',
        'learnable': True,
        'strategy': 'eigenface',
        'dynamic': False
    },
    
    # Learnable anchors with dynamic pseudo-labels
    'learnable_random_dynamic': {
        'config': 'configs/learnable_random_dynamic.yaml',
        'description': 'Learnable anchors, random init, dynamic labels',
        'learnable': True,
        'strategy': 'random',
        'dynamic': True
    },
    'learnable_kmeans_dynamic': {
        'config': 'configs/learnable_kmeans_dynamic.yaml',
        'description': 'Learnable anchors, k-means init, dynamic labels',
        'learnable': True,
        'strategy': 'kmeans',
        'dynamic': True
    },
    'learnable_eigenface_dynamic': {
        'config': 'configs/learnable_eigenface_dynamic.yaml',
        'description': 'Learnable anchors, eigenface init, dynamic labels',
        'learnable': True,
        'strategy': 'eigenface',
        'dynamic': True
    },
}


def get_python_executable():
    """Get the path to the Python executable in the virtual environment."""
    venv_python = Path(__file__).parent.parent / 'venv' / 'Scripts' / 'python.exe'
    if venv_python.exists():
        return str(venv_python)
    
    # Try alternative paths
    alternatives = [
        Path(__file__).parent / 'venv' / 'Scripts' / 'python.exe',
        Path(__file__).parent / '.venv' / 'Scripts' / 'python.exe',
    ]
    
    for alt in alternatives:
        if alt.exists():
            return str(alt)
    
    return sys.executable  # Fallback to current Python


def run_experiment(exp_name: str, exp_config: dict, skip_anchors: bool = False) -> dict:
    """
    Run a single experiment.
    
    Returns:
        dict with experiment results and status
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  {exp_config['description']}")
    print(f"  Config: {exp_config['config']}")
    print(f"{'='*80}")
    
    python_cmd = get_python_executable()
    
    cmd = [
        python_cmd, 'main.py',
        '--config', exp_config['config'],
        '--auto-name'
    ]
    
    if skip_anchors:
        cmd.append('--skip-anchors')
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            'experiment': exp_name,
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
            'returncode': result.returncode,
            'duration_seconds': duration,
            'config': exp_config
        }
        
    except Exception as e:
        return {
            'experiment': exp_name,
            'status': 'ERROR',
            'error': str(e),
            'config': exp_config
        }


def filter_experiments(
    experiments: dict,
    strategies: list = None,
    learnable_only: bool = False,
    fixed_only: bool = False,
    dynamic_only: bool = False,
    static_only: bool = False
) -> dict:
    """Filter experiments based on criteria."""
    filtered = {}
    
    for name, config in experiments.items():
        # Filter by strategy
        if strategies and config['strategy'] not in strategies:
            continue
        
        # Filter by learnable
        if learnable_only and not config['learnable']:
            continue
        if fixed_only and config['learnable']:
            continue
        
        # Filter by dynamic
        if dynamic_only and not config['dynamic']:
            continue
        if static_only and config['dynamic']:
            continue
        
        filtered[name] = config
    
    return filtered


def print_summary(results: list):
    """Print a summary of all experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Group by status
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']
    
    print(f"\nTotal Experiments: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    # Print table
    print("\n" + "-" * 80)
    print(f"{'Experiment':<35} {'Strategy':<12} {'Learnable':<10} {'Dynamic':<10} {'Status':<10}")
    print("-" * 80)
    
    for r in results:
        config = r['config']
        learnable_str = '✓' if config['learnable'] else '-'
        dynamic_str = '✓' if config['dynamic'] else '-'
        status_str = '✓' if r['status'] == 'SUCCESS' else '✗'
        
        print(f"{r['experiment']:<35} {config['strategy']:<12} {learnable_str:<10} {dynamic_str:<10} {status_str:<10}")
    
    print("-" * 80)
    
    # Timing summary for successful experiments
    if successful:
        durations = [r.get('duration_seconds', 0) for r in successful]
        total_time = sum(durations)
        avg_time = total_time / len(durations)
        
        print(f"\nTiming (successful experiments):")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Average per experiment: {avg_time/60:.1f} minutes")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run learnable anchor experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all experiments:
    python run_learnable_experiments.py --all
    
  Run only learnable anchor experiments:
    python run_learnable_experiments.py --learnable-only
    
  Run specific strategies:
    python run_learnable_experiments.py --strategies random kmeans
    
  Run only dynamic label experiments:
    python run_learnable_experiments.py --dynamic-only
    
  Run specific experiments by name:
    python run_learnable_experiments.py --experiments learnable_random_fixed learnable_kmeans_fixed
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--strategies', nargs='+',
                        choices=['random', 'kmeans', 'eigenface'],
                        help='Filter by anchor initialization strategy')
    parser.add_argument('--learnable-only', action='store_true',
                        help='Run only learnable anchor experiments')
    parser.add_argument('--fixed-only', action='store_true',
                        help='Run only fixed anchor experiments (baselines)')
    parser.add_argument('--dynamic-only', action='store_true',
                        help='Run only dynamic label reassignment experiments')
    parser.add_argument('--static-only', action='store_true',
                        help='Run only fixed label experiments')
    parser.add_argument('--experiments', nargs='+',
                        help='Run specific experiments by name')
    parser.add_argument('--skip-anchors', action='store_true',
                        help='Skip anchor generation if already exists')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments without running')
    parser.add_argument('--save-results', type=str, default='experiment_results.json',
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Filter experiments
    if args.experiments:
        # Run specific experiments
        experiments = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
    elif args.all:
        experiments = EXPERIMENTS
    else:
        experiments = filter_experiments(
            EXPERIMENTS,
            strategies=args.strategies,
            learnable_only=args.learnable_only,
            fixed_only=args.fixed_only,
            dynamic_only=args.dynamic_only,
            static_only=args.static_only
        )
    
    if not experiments:
        print("No experiments match the specified criteria.")
        print("\nAvailable experiments:")
        for name, config in EXPERIMENTS.items():
            print(f"  {name}: {config['description']}")
        return
    
    # List mode
    if args.list:
        print("\nExperiments to run:")
        print("-" * 80)
        for name, config in experiments.items():
            learnable = '✓' if config['learnable'] else '-'
            dynamic = '✓' if config['dynamic'] else '-'
            print(f"  {name:<35} [{config['strategy']:<10}] L:{learnable} D:{dynamic}")
            print(f"    {config['description']}")
        print("-" * 80)
        print(f"\nTotal: {len(experiments)} experiments")
        return
    
    # Run experiments
    print("\n" + "=" * 80)
    print("LEARNABLE ANCHOR EXPERIMENTS")
    print("=" * 80)
    print(f"Running {len(experiments)} experiments...")
    print("Experiments:", list(experiments.keys()))
    print("=" * 80)
    
    results = []
    
    for exp_name, exp_config in experiments.items():
        result = run_experiment(exp_name, exp_config, skip_anchors=args.skip_anchors)
        results.append(result)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.save_results:
        results_path = Path(__file__).parent / args.save_results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
