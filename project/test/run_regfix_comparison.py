"""
Run register-fix comparison experiments.

Runs 6 experiments sequentially (K=1/512/1024 × ±stage2) and collects
results into a summary table at the end.

Usage:
    python run_regfix_comparison.py [--dry-run] [--only NAMES...]

Examples:
    python run_regfix_comparison.py                     # run all 6
    python run_regfix_comparison.py --only regfix_k1    # run just K=1
    python run_regfix_comparison.py --dry-run            # print commands only
"""

import subprocess
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta


EXPERIMENTS = [
    {
        "name": "regfix_k1",
        "config": "project/configs/regfix_k1.yaml",
        "compare_against": "reproject_k1_early_trainable_backbone",
        "old_auroc": 0.8346,
    },
    {
        "name": "regfix_k512",
        "config": "project/configs/regfix_k512.yaml",
        "compare_against": "reproject_k512_early_trainable_backbone",
        "old_auroc": 0.7942,
    },
    {
        "name": "regfix_k1024",
        "config": "project/configs/regfix_k1024.yaml",
        "compare_against": "reproject_k1024_early_trainable_backbone",
        "old_auroc": 0.7604,
    },
    {
        "name": "regfix_k1_stage2",
        "config": "project/configs/regfix_k1_stage2.yaml",
        "compare_against": "reproject_k1_early_trainable_backbone_stage2_3",
        "old_auroc": 0.8532,
    },
    {
        "name": "regfix_k512_stage2",
        "config": "project/configs/regfix_k512_stage2.yaml",
        "compare_against": "reproject_k512_early_trainable_backbone_stage2",
        "old_auroc": 0.7728,
    },
    {
        "name": "regfix_k1024_stage2",
        "config": "project/configs/regfix_k1024_stage2.yaml",
        "compare_against": None,
        "old_auroc": None,
    },
]

PYTHON = r".\venv\Scripts\python.exe"
MAIN = r".\project\main.py"


def run_experiment(exp: dict, dry_run: bool = False) -> dict:
    """Run a single experiment and return result info."""
    name = exp["name"]
    config = exp["config"]

    cmd = [PYTHON, MAIN, "--config", config]
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Config:     {config}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    if dry_run:
        return {"name": name, "status": "dry-run", "duration": 0}

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
        )
        status = "success"
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Experiment {name} failed with exit code {e.returncode}")
        status = "failed"
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user during {name}")
        status = "interrupted"

    duration = time.time() - t0
    print(f"\n  Finished {name} in {timedelta(seconds=int(duration))}, status={status}")

    return {"name": name, "status": status, "duration": duration}


def read_metrics(exp_name: str) -> dict:
    """Read evaluation metrics from an experiment directory."""
    # Check common metric file locations
    exp_dir = Path("experiments") / exp_name
    for candidate in [
        exp_dir / "evaluation" / "evaluation_metrics.json",
        exp_dir / "evaluation_metrics.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)

    # Try to read from training summary
    summary = exp_dir / "training_summary.json"
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)
            return {"image_auroc": data.get("best_val_auroc")}

    return {}


def print_summary(experiments: list, results: list):
    """Print comparison table."""
    print(f"\n\n{'='*100}")
    print("  REGISTER-FIX COMPARISON RESULTS")
    print(f"{'='*100}")
    print(f"{'Experiment':<28} {'Status':<10} {'Duration':<12} {'New AUROC':<12} {'Old AUROC':<12} {'Delta':<10}")
    print(f"{'-'*28} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for exp, res in zip(experiments, results):
        status = res["status"]
        dur = str(timedelta(seconds=int(res["duration"]))) if res["duration"] else "-"

        metrics = {}
        if status == "success":
            metrics = read_metrics(exp["name"])

        new_auroc = metrics.get("image_auroc")
        old_auroc = exp["old_auroc"]

        new_str = f"{new_auroc:.4f}" if new_auroc else "-"
        old_str = f"{old_auroc:.4f}" if old_auroc else "-"

        if new_auroc and old_auroc:
            delta = new_auroc - old_auroc
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "-"

        print(f"{exp['name']:<28} {status:<10} {dur:<12} {new_str:<12} {old_str:<12} {delta_str:<10}")

    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Run register-fix comparison experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Run only these experiments")
    args = parser.parse_args()

    to_run = EXPERIMENTS
    if args.only:
        to_run = [e for e in EXPERIMENTS if e["name"] in args.only]
        if not to_run:
            print(f"No matching experiments. Available: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)

    print(f"Register-fix comparison: {len(to_run)} experiments")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print("(DRY RUN — no experiments will be executed)\n")

    results = []
    for exp in to_run:
        res = run_experiment(exp, dry_run=args.dry_run)
        results.append(res)
        if res["status"] == "interrupted":
            break

    print_summary(to_run, results)

    # Save results to JSON
    out_path = Path("experiments") / "regfix_comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        summary = []
        for exp, res in zip(to_run, results):
            metrics = read_metrics(exp["name"]) if res["status"] == "success" else {}
            summary.append({
                "name": exp["name"],
                "config": exp["config"],
                "compare_against": exp["compare_against"],
                "old_auroc": exp["old_auroc"],
                "new_auroc": metrics.get("image_auroc"),
                "pixel_auroc": metrics.get("pixel_auroc"),
                "status": res["status"],
                "duration_s": res["duration"],
            })
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
