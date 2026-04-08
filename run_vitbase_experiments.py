"""
ViT-Base DINOv3 experiments runner.

Runs K=1, K=512, K=1024 with vit_base_patch16_dinov3 (trainable backbone, stage 1+2).
Reports stage 1 and stage 2 AUROC and compares against vit_small regfix baselines.

Usage:
    python run_vitbase_experiments.py              # run all 3
    python run_vitbase_experiments.py --dry-run     # print commands only
    python run_vitbase_experiments.py --analyze-only # skip training, read existing results
    python run_vitbase_experiments.py --only vitbase_k1  # run one experiment
"""

import subprocess
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

PYTHON = r".\venv\Scripts\python.exe"
MAIN = r".\project\main.py"

# ViT-Small regfix baselines for comparison
REGFIX_BASELINES = {
    1:    {"stage1": 0.8539, "stage2": 0.8395},
    512:  {"stage1": 0.7878, "stage2": 0.7958},
    1024: {"stage1": 0.7877, "stage2": 0.7857},
}

EXPERIMENTS = [
    {"name": "vitbase_k1",    "config": "project/configs/vitbase_k1.yaml",    "k": 1},
    {"name": "vitbase_k512",  "config": "project/configs/vitbase_k512.yaml",  "k": 512},
    {"name": "vitbase_k1024", "config": "project/configs/vitbase_k1024.yaml", "k": 1024},
]


def run_experiment(exp: dict, dry_run: bool = False) -> str:
    cmd = [PYTHON, MAIN, "--config", exp["config"]]
    print(f"\n{'='*70}")
    print(f"  {exp['name']} (K={exp['k']})")
    print(f"  Config: {exp['config']}")
    print(f"  Time:   {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return "dry-run"

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, text=True)
        elapsed = time.time() - t0
        print(f"  Done in {timedelta(seconds=int(elapsed))}")
        return "success"
    except subprocess.CalledProcessError as e:
        print(f"  FAILED (exit code {e.returncode})")
        return "failed"
    except KeyboardInterrupt:
        print(f"  Interrupted")
        return "interrupted"


def _find_experiment_dir(base_name: str) -> Path | None:
    """Find experiment directory, handling _1, _2 suffixes from main.py."""
    exps = Path("experiments")
    # Try exact name first, then suffixed versions (pick latest)
    candidates = sorted(exps.glob(f"{base_name}*"))
    # Filter to only dirs that have evaluation results
    with_eval = [c for c in candidates if (c / "evaluation").exists() or (c / "evaluation_stage1").exists()]
    return with_eval[-1] if with_eval else None


def read_auroc(exp_dir: Path, stage: str) -> float | None:
    """Read image AUROC from metrics JSON or CSV."""
    if stage == "stage1":
        metrics_path = exp_dir / "evaluation_stage1" / "evaluation_metrics.json"
        csv_path = exp_dir / "evaluation_stage1" / "evaluation_image_scores.csv"
    else:
        metrics_path = exp_dir / "evaluation" / "evaluation_metrics.json"
        csv_path = exp_dir / "evaluation" / "evaluation_image_scores.csv"

    # Try metrics JSON first
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
            if "image_auroc" in data:
                return data["image_auroc"]

    # Fall back to CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'label' in df.columns and 'anchor_score' in df.columns:
            return roc_auc_score(df['label'].values, df['anchor_score'].values)

    return None


def read_pixel_auroc(exp_dir: Path) -> float | None:
    """Read pixel AUROC from stage 2 evaluation."""
    metrics_path = exp_dir / "evaluation" / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
            return data.get("pixel_auroc")
    return None


def analyze_results():
    print(f"\n{'='*110}")
    print("  VIT-BASE vs VIT-SMALL COMPARISON")
    print(f"{'='*110}")
    print(f"{'Experiment':<18} {'K':<6} {'Stage1':<10} {'Stage2':<10} {'Pixel':<10} "
          f"{'S1 vs Small':<14} {'S2 vs Small':<14} {'S1→S2 Δ':<10}")
    print(f"{'-'*18} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*14} {'-'*10}")

    results_data = []

    for exp in EXPERIMENTS:
        k = exp["k"]
        exp_dir = _find_experiment_dir(exp["name"])

        s1_auroc = read_auroc(exp_dir, "stage1") if exp_dir else None
        s2_auroc = read_auroc(exp_dir, "stage2") if exp_dir else None
        px_auroc = read_pixel_auroc(exp_dir) if exp_dir else None

        baseline = REGFIX_BASELINES[k]
        s1_str = f"{s1_auroc:.4f}" if s1_auroc else "missing"
        s2_str = f"{s2_auroc:.4f}" if s2_auroc else "missing"
        px_str = f"{px_auroc:.4f}" if px_auroc else "-"

        s1_delta = ""
        if s1_auroc and baseline["stage1"]:
            d = s1_auroc - baseline["stage1"]
            s1_delta = f"{d:+.4f}"

        s2_delta = ""
        if s2_auroc and baseline["stage2"]:
            d = s2_auroc - baseline["stage2"]
            s2_delta = f"{d:+.4f}"

        stage_delta = ""
        if s1_auroc and s2_auroc:
            d = s2_auroc - s1_auroc
            stage_delta = f"{d:+.4f}"

        print(f"{exp['name']:<18} {k:<6} {s1_str:<10} {s2_str:<10} {px_str:<10} "
              f"{s1_delta:<14} {s2_delta:<14} {stage_delta:<10}")

        results_data.append({
            "name": exp["name"],
            "k": k,
            "backbone": "vit_base_patch16_dinov3",
            "stage1_auroc": s1_auroc,
            "stage2_auroc": s2_auroc,
            "pixel_auroc": px_auroc,
            "regfix_stage1": baseline["stage1"],
            "regfix_stage2": baseline["stage2"],
            "exp_dir": str(exp_dir) if exp_dir else None,
        })

    # Summary
    print(f"\n  ViT-Small (regfix) baselines:")
    for k, bl in REGFIX_BASELINES.items():
        print(f"    K={k:<5} stage1={bl['stage1']:.4f}  stage2={bl['stage2']:.4f}")

    # Save
    out_path = Path("experiments/vitbase_comparison_results.json")
    with open(out_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ViT-Base DINOv3 experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    args = parser.parse_args()

    to_run = EXPERIMENTS
    if args.only:
        to_run = [e for e in EXPERIMENTS if e["name"] in args.only]
        if not to_run:
            print(f"No match. Available: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)

    if not args.analyze_only:
        print(f"ViT-Base experiments: {len(to_run)} configs")
        print(f"Backbone: vit_base_patch16_dinov3.lvd1689m (768D, ~86M params)")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.dry_run:
            print("(DRY RUN)\n")

        for exp in to_run:
            status = run_experiment(exp, dry_run=args.dry_run)
            if status == "interrupted":
                break

        print(f"\nAll experiments finished at {datetime.now().strftime('%H:%M:%S')}")

    if args.dry_run:
        return

    analyze_results()


if __name__ == "__main__":
    main()
