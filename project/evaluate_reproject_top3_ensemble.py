"""
Evaluate normalized score-level ensemble across multiple experiment outputs.

Expected input per experiment:
    experiments/<exp_name>/evaluation/evaluation_image_scores.csv

Required columns in CSV:
    - path
    - label
    - image_score (or another selected score column)

Example:
    python project/evaluate_reproject_top3_ensemble.py \
        --experiments reproject_k1024_early reproject_k256_early solution_a_reproject_k1_1 \
        --normalization minmax \
        --score-column image_score \
        --weights 1 1 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def normalize_scores(scores: np.ndarray, method: str) -> np.ndarray:
    scores = scores.astype(np.float64)

    if method == 'none':
        return scores

    if method == 'minmax':
        min_val = float(np.min(scores))
        max_val = float(np.max(scores))
        denom = max(max_val - min_val, 1e-12)
        return (scores - min_val) / denom

    if method == 'zscore':
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std < 1e-12:
            import warnings
            warnings.warn('zscore normalization: constant signal (std≈0), returning zeros')
            return np.zeros_like(scores)
        return (scores - mean) / std

    if method == 'robust':
        median = float(np.median(scores))
        q1 = float(np.quantile(scores, 0.25))
        q3 = float(np.quantile(scores, 0.75))
        iqr = max(q3 - q1, 1e-12)
        return (scores - median) / iqr

    if method == 'rank':
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(scores), dtype=np.float64)
        if len(scores) <= 1:
            return np.zeros_like(scores)
        return ranks / (len(scores) - 1)

    raise ValueError(f"Unsupported normalization method: {method}")


def load_experiment_scores(experiment_dir: Path, score_column: str) -> pd.DataFrame:
    csv_path = experiment_dir / 'evaluation' / 'evaluation_image_scores.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing score file: {csv_path}. "
            f"Run evaluation with updated code to export per-sample scores."
        )

    df = pd.read_csv(csv_path)
    required_cols = {'path', 'label', score_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    return df[['path', 'label', score_column]].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate normalized top-k reproject ensemble.')
    parser.add_argument(
        '--root',
        type=str,
        default='experiments',
        help='Root folder containing experiment directories.'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        required=True,
        help='Experiment directory names under --root.'
    )
    parser.add_argument(
        '--score-column',
        type=str,
        default='image_score',
        help='Score column to ensemble from each experiment CSV.'
    )
    parser.add_argument(
        '--normalization',
        type=str,
        default='minmax',
        choices=['none', 'minmax', 'zscore', 'robust', 'rank'],
        help='Per-model normalization before aggregation.'
    )
    parser.add_argument(
        '--weights',
        nargs='*',
        type=float,
        default=None,
        help='Optional per-model weights (same length as experiments).'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/ensemble_reproject_top3',
        help='Output folder for ensemble artifacts.'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.weights is None or len(args.weights) == 0:
        weights = np.ones(len(args.experiments), dtype=np.float64)
    else:
        if len(args.weights) != len(args.experiments):
            raise ValueError('Number of weights must match number of experiments.')
        weights = np.array(args.weights, dtype=np.float64)

    if np.allclose(weights.sum(), 0.0):
        raise ValueError('Weights sum to zero.')

    weights = weights / weights.sum()

    merged_df = None
    model_columns: List[str] = []

    print('============================================================')
    print('Normalized Ensemble Evaluation')
    print('============================================================')
    print(f'Experiments: {args.experiments}')
    print(f'Score column: {args.score_column}')
    print(f'Normalization: {args.normalization}')
    print(f'Weights: {weights.tolist()}')

    for exp_name in args.experiments:
        exp_dir = root / exp_name
        df = load_experiment_scores(exp_dir, args.score_column)

        raw_col = f'{exp_name}__raw'
        norm_col = f'{exp_name}__norm'

        df = df.rename(columns={args.score_column: raw_col})

        # Validate raw scores
        raw_scores = df[raw_col].to_numpy()
        if not np.all(np.isfinite(raw_scores)):
            n_bad = (~np.isfinite(raw_scores)).sum()
            raise ValueError(f'[{exp_name}] {n_bad} non-finite raw scores detected (NaN/Inf)')

        df[norm_col] = normalize_scores(raw_scores, args.normalization)

        # Validate normalized scores
        norm_scores = df[norm_col]
        if not np.all(np.isfinite(norm_scores)):
            n_bad = (~np.isfinite(norm_scores)).sum()
            raise ValueError(f'[{exp_name}] {n_bad} non-finite normalized scores after {args.normalization}')

        # Per-model AUROC check
        _labels = df['label'].to_numpy(dtype=np.int64)
        _model_auroc = roc_auc_score(_labels, raw_scores)
        if _model_auroc < 0.6:
            import warnings
            warnings.warn(f'[{exp_name}] individual AUROC={_model_auroc:.4f} < 0.6 — '
                          f'this model may hurt ensemble performance')
        print(f'[{exp_name}] individual AUROC={_model_auroc:.4f}')

        model_columns.append(norm_col)

        if merged_df is None:
            merged_df = df[['path', 'label', raw_col, norm_col]].copy()
        else:
            merged_df = merged_df.merge(
                df[['path', 'label', raw_col, norm_col]],
                on=['path', 'label'],
                how='inner'
            )

        print(
            f'[{exp_name}] raw range=({df[raw_col].min():.6f}, {df[raw_col].max():.6f}) | '
            f'norm range=({df[norm_col].min():.6f}, {df[norm_col].max():.6f})'
        )

    if merged_df is None or len(merged_df) == 0:
        raise RuntimeError('No samples available after merge.')

    score_matrix = merged_df[model_columns].to_numpy(dtype=np.float64)
    ensemble_scores = score_matrix.dot(weights)

    labels = merged_df['label'].to_numpy(dtype=np.int64)
    auroc = roc_auc_score(labels, ensemble_scores)
    aupr = average_precision_score(labels, ensemble_scores)

    merged_df['ensemble_score'] = ensemble_scores
    merged_df.to_csv(output_dir / 'ensemble_scores.csv', index=False)

    summary = {
        'experiments': args.experiments,
        'score_column': args.score_column,
        'normalization': args.normalization,
        'weights': weights.tolist(),
        'num_samples': int(len(merged_df)),
        'num_normal': int((labels == 0).sum()),
        'num_anomaly': int((labels == 1).sum()),
        'ensemble_auroc': float(auroc),
        'ensemble_aupr': float(aupr)
    }

    import json
    with open(output_dir / 'ensemble_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('------------------------------------------------------------')
    print(f"Samples used: {len(merged_df)}")
    print(f"Ensemble AUROC: {auroc:.4f}")
    print(f"Ensemble AUPR:  {aupr:.4f}")
    print(f"Saved: {output_dir / 'ensemble_scores.csv'}")
    print(f"Saved: {output_dir / 'ensemble_metrics.json'}")


if __name__ == '__main__':
    main()
