"""
Seed-varied comparison: Anchor-only (stage 1) vs Fusion (stage 2).

Runs K=1 experiments with multiple seeds for both stage1-only and stage1+stage2,
then computes mean ± std of image AUROC to determine if fusion consistently helps.

Fusion uses fixed weights: anchor=0.72, pixel=0.12, reconstruction included if available.
Also reports per-run optimized fusion as a reference.

Usage:
    python run_seed_comparison.py                 # run all 8 experiments
    python run_seed_comparison.py --dry-run        # print commands only
    python run_seed_comparison.py --analyze-only   # skip training, just read existing results
    python run_seed_comparison.py --seeds 42 123   # custom seeds
"""

import subprocess
import sys
import json
import time
import argparse
import yaml
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

PYTHON = r".\venv\Scripts\python.exe"
MAIN = r".\project\main.py"

SEEDS = [42, 123, 456, 789]

# Base configs to template from
STAGE1_BASE = "project/configs/regfix_k1.yaml"
STAGE2_BASE = "project/configs/regfix_k1_stage2.yaml"

# Fixed fusion weights (anchor, divergence, pixel)
FUSION_WA = 0.72
FUSION_WD = 0.16
FUSION_WP = 0.12


def make_config(base_path: str, seed: int, output_dir: str) -> str:
    """Create a temporary config with overridden seed and output_dir."""
    with open(base_path) as f:
        config = yaml.safe_load(f)

    config['seed'] = seed
    config['output_dir'] = output_dir

    tmp = Path(output_dir) / "config.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(tmp)


def run_experiment(config_path: str, name: str, dry_run: bool = False) -> str:
    """Run a single experiment. Returns status string."""
    cmd = [PYTHON, MAIN, "--config", config_path]
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Config: {config_path}")
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


def minmax_normalize(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def compute_fusion_auroc(df, wa=FUSION_WA, wd=FUSION_WD, wp=FUSION_WP):
    """Compute fused AUROC from stage2 CSV using available signals."""
    labels = df['label'].values
    anchor = minmax_normalize(df['anchor_score'].values)

    signals = {'anchor': (anchor, wa)}
    used_weights = wa

    # Pixel
    if 'pixel_aggregated_score' in df.columns:
        pix = minmax_normalize(df['pixel_aggregated_score'].values)
        pix_auroc = roc_auc_score(labels, pix)
        if pix_auroc >= 0.5:
            signals['pixel'] = (pix, wp)
            used_weights += wp

    # Reconstruction
    if 'reconstruction_score' in df.columns:
        rec = minmax_normalize(df['reconstruction_score'].values)
        rec_auroc = roc_auc_score(labels, rec)
        if rec_auroc >= 0.5:
            # Give reconstruction the divergence weight slot since no divergence exists
            signals['reconstruction'] = (rec, wd)
            used_weights += wd

    # Divergence (if present)
    if 'bottleneck_divergence' in df.columns:
        div = minmax_normalize(df['bottleneck_divergence'].values)
        div_auroc = roc_auc_score(labels, div)
        if div_auroc >= 0.5:
            signals['divergence'] = (div, wd)
            if 'reconstruction' not in signals:
                used_weights += wd

    fused = sum(w * s for s, w in signals.values()) / used_weights
    return roc_auc_score(labels, fused), list(signals.keys())


def optimize_fusion(df):
    """Quick grid search for best fusion weights on this run's data."""
    labels = df['label'].values
    cols = {}

    if 'anchor_score' in df.columns:
        cols['anchor'] = minmax_normalize(df['anchor_score'].values)
    if 'pixel_aggregated_score' in df.columns:
        pix = minmax_normalize(df['pixel_aggregated_score'].values)
        if roc_auc_score(labels, pix) >= 0.5:
            cols['pixel'] = pix
    if 'reconstruction_score' in df.columns:
        rec = minmax_normalize(df['reconstruction_score'].values)
        if roc_auc_score(labels, rec) >= 0.5:
            cols['reconstruction'] = rec

    if len(cols) <= 1:
        auroc = roc_auc_score(labels, cols.get('anchor', np.zeros(len(labels))))
        return auroc, {k: 1.0 for k in cols}

    keys = list(cols.keys())
    best_auroc = -1
    best_w = None
    step = 0.05
    grid = np.arange(0, 1.01, step)

    if len(keys) == 2:
        for w0 in grid:
            w1 = 1.0 - w0
            if w1 < -0.01:
                continue
            fused = w0 * cols[keys[0]] + w1 * cols[keys[1]]
            auroc = roc_auc_score(labels, fused)
            if auroc > best_auroc:
                best_auroc = auroc
                best_w = {keys[0]: w0, keys[1]: w1}
    elif len(keys) == 3:
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -0.01:
                    continue
                fused = w0 * cols[keys[0]] + w1 * cols[keys[1]] + w2 * cols[keys[2]]
                auroc = roc_auc_score(labels, fused)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_w = {keys[0]: w0, keys[1]: w1, keys[2]: w2}

    return best_auroc, best_w


def _find_eval_csv(base_name: str) -> Path | None:
    """Find the evaluation CSV for an experiment, handling _1, _2, etc. suffixes.
    
    main.py appends _N when output_dir already exists, so we search for the
    highest-numbered variant that contains an evaluation CSV.
    """
    exps = Path("experiments")
    candidates = sorted(exps.glob(f"{base_name}*/evaluation/evaluation_image_scores.csv"))
    return candidates[-1] if candidates else None


def analyze_results(seeds):
    """Read all experiment CSVs and compute comparison statistics."""
    stage1_aurocs = []
    stage2_anchor_aurocs = []
    stage2_fusion_aurocs = []
    stage2_optimized_aurocs = []

    print(f"\n{'='*90}")
    print("  PER-RUN RESULTS")
    print(f"{'='*90}")
    print(f"{'Seed':<8} {'Stage1 Anchor':<16} {'Stage2 Anchor':<16} {'Stage2 Fusion':<16} {'Stage2 Optimal':<16} {'Signals'}")
    print(f"{'-'*8} {'-'*16} {'-'*16} {'-'*16} {'-'*16} {'-'*20}")

    for seed in seeds:
        s1_csv = _find_eval_csv(f"seed_cmp_stage1_s{seed}")
        s2_csv = _find_eval_csv(f"seed_cmp_stage2_s{seed}")

        s1_auroc = None
        s2_anchor = None
        s2_fusion = None
        s2_optimal = None
        signals_used = ""

        if s1_csv and s1_csv.exists():
            df1 = pd.read_csv(s1_csv)
            s1_auroc = roc_auc_score(df1['label'].values, df1['anchor_score'].values)
            stage1_aurocs.append(s1_auroc)

        if s2_csv and s2_csv.exists():
            df2 = pd.read_csv(s2_csv)
            s2_anchor = roc_auc_score(df2['label'].values, df2['anchor_score'].values)
            stage2_anchor_aurocs.append(s2_anchor)

            s2_fusion, sigs = compute_fusion_auroc(df2)
            stage2_fusion_aurocs.append(s2_fusion)
            signals_used = "+".join(sigs)

            s2_optimal, opt_w = optimize_fusion(df2)
            stage2_optimized_aurocs.append(s2_optimal)
            opt_str = ", ".join(f"{k}={v:.2f}" for k, v in opt_w.items())
            signals_used += f" | opt: {opt_str}"

        s1_str = f"{s1_auroc:.4f}" if s1_auroc is not None else "missing"
        s2a_str = f"{s2_anchor:.4f}" if s2_anchor is not None else "missing"
        s2f_str = f"{s2_fusion:.4f}" if s2_fusion is not None else "missing"
        s2o_str = f"{s2_optimal:.4f}" if s2_optimal is not None else "missing"
        print(f"{seed:<8} {s1_str:<16} {s2a_str:<16} {s2f_str:<16} {s2o_str:<16} {signals_used}")

    print(f"\n{'='*90}")
    print("  SUMMARY (mean ± std)")
    print(f"{'='*90}")

    def fmt(vals, label):
        if not vals:
            return f"  {label}: no data"
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"  {label}: {mean:.4f} ± {std:.4f}  (n={len(vals)}, range=[{min(vals):.4f}, {max(vals):.4f}])"

    print(fmt(stage1_aurocs, "Stage1 anchor-only   "))
    print(fmt(stage2_anchor_aurocs, "Stage2 anchor-only   "))
    print(fmt(stage2_fusion_aurocs, "Stage2 fixed fusion  "))
    print(fmt(stage2_optimized_aurocs, "Stage2 optimal fusion"))

    if stage1_aurocs and stage2_fusion_aurocs:
        delta = np.mean(stage2_fusion_aurocs) - np.mean(stage1_aurocs)
        print(f"\n  Fusion vs Stage1 anchor: {delta:+.4f} ({'fusion better' if delta > 0 else 'anchor better'})")

    if stage1_aurocs and stage2_anchor_aurocs:
        delta = np.mean(stage2_anchor_aurocs) - np.mean(stage1_aurocs)
        print(f"  Stage2 anchor vs Stage1 anchor: {delta:+.4f}")

    # Save results
    results = {
        "seeds": seeds,
        "stage1_anchor_aurocs": stage1_aurocs,
        "stage2_anchor_aurocs": stage2_anchor_aurocs,
        "stage2_fusion_aurocs": stage2_fusion_aurocs,
        "stage2_optimized_aurocs": stage2_optimized_aurocs,
        "summary": {
            "stage1_anchor_mean": float(np.mean(stage1_aurocs)) if stage1_aurocs else None,
            "stage1_anchor_std": float(np.std(stage1_aurocs, ddof=1)) if len(stage1_aurocs) > 1 else None,
            "stage2_fusion_mean": float(np.mean(stage2_fusion_aurocs)) if stage2_fusion_aurocs else None,
            "stage2_fusion_std": float(np.std(stage2_fusion_aurocs, ddof=1)) if len(stage2_fusion_aurocs) > 1 else None,
        },
        "fusion_weights": {"anchor": FUSION_WA, "divergence": FUSION_WD, "pixel": FUSION_WP},
    }
    out_path = Path("experiments/seed_comparison_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Seed-varied: anchor-only vs fusion comparison")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--analyze-only", action="store_true", help="Skip training, analyze existing results")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Seeds to use (default: 42 123 456 789)")
    args = parser.parse_args()

    seeds = args.seeds

    if not args.analyze_only:
        print(f"Seed comparison: {len(seeds)} seeds × 2 configs = {len(seeds)*2} experiments")
        print(f"Seeds: {seeds}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.dry_run:
            print("(DRY RUN)\n")

        statuses = []
        for seed in seeds:
            # Stage 1
            s1_out = f"./experiments/seed_cmp_stage1_s{seed}"
            s1_cfg = make_config(STAGE1_BASE, seed, s1_out)
            status = run_experiment(s1_cfg, f"Stage1 K=1 seed={seed}", dry_run=args.dry_run)
            statuses.append(("stage1", seed, status))
            if status == "interrupted":
                break

            # Stage 2
            s2_out = f"./experiments/seed_cmp_stage2_s{seed}"
            s2_cfg = make_config(STAGE2_BASE, seed, s2_out)
            status = run_experiment(s2_cfg, f"Stage2 K=1 seed={seed}", dry_run=args.dry_run)
            statuses.append(("stage2", seed, status))
            if status == "interrupted":
                break

        print(f"\nAll experiments finished at {datetime.now().strftime('%H:%M:%S')}")
        failed = [s for s in statuses if s[2] == "failed"]
        if failed:
            print(f"  WARNING: {len(failed)} experiment(s) failed: {failed}")

    if args.dry_run:
        return

    analyze_results(seeds)


if __name__ == "__main__":
    main()
