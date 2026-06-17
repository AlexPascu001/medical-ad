from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "fusion_candidate_experiments"


@dataclass(frozen=True)
class Candidate:
    key: str
    role: str
    csv_path: Path
    metrics_path: Path
    primary: bool
    note: str


@dataclass(frozen=True)
class SignalSpec:
    candidate_key: str
    signal: str


@dataclass(frozen=True)
class FusionExperiment:
    name: str
    group: str
    kind: str
    specs: tuple[SignalSpec, ...]
    note: str


CANDIDATES = {
    "global_k1_clean": Candidate(
        key="global_k1_clean",
        role="Clean global one-anchor",
        csv_path=ROOT / "runs/full_redesign_stage2e70_k1/full_redesign_stage2e70_k1/evaluation/evaluation_image_scores.csv",
        metrics_path=ROOT / "runs/full_redesign_stage2e70_k1/full_redesign_stage2e70_k1/evaluation/evaluation_metrics.json",
        primary=True,
        note="Primary CLS/image-level K=1 candidate.",
    ),
    "global_k1024_clean": Candidate(
        key="global_k1024_clean",
        role="Clean global multi-anchor",
        csv_path=ROOT / "runs/full_redesign_stage2e70_k1024/full_redesign_stage2e70_k1024/evaluation/evaluation_image_scores.csv",
        metrics_path=ROOT / "runs/full_redesign_stage2e70_k1024/full_redesign_stage2e70_k1024/evaluation/evaluation_metrics.json",
        primary=True,
        note="Primary global multi-anchor candidate.",
    ),
    "patch_k32_clean": Candidate(
        key="patch_k32_clean",
        role="Clean patch multi-anchor",
        csv_path=ROOT / "runs/patch_stage2e70_k32/patch_stage2e70_k32/evaluation/evaluation_image_scores.csv",
        metrics_path=ROOT / "runs/patch_stage2e70_k32/patch_stage2e70_k32/evaluation/evaluation_metrics.json",
        primary=True,
        note="Primary patch-level candidate.",
    ),
    "patch_loc_cos_k32": Candidate(
        key="patch_loc_cos_k32",
        role="Location-aware patch",
        csv_path=ROOT / "runs/patch_location_kmeans_stage2recon_cosine_k32/patch_location_kmeans_stage2recon_cosine_k32/evaluation/evaluation_image_scores.csv",
        metrics_path=ROOT / "runs/patch_location_kmeans_stage2recon_cosine_k32/patch_location_kmeans_stage2recon_cosine_k32/evaluation/evaluation_metrics.json",
        primary=False,
        note="Secondary local patch-bank variant.",
    ),
    "global_k1_nofuser": Candidate(
        key="global_k1_nofuser",
        role="Global K=1 ablation",
        csv_path=ROOT / "runs/nofuser_k1/nofuser_k1/evaluation/evaluation_image_scores.csv",
        metrics_path=ROOT / "runs/nofuser_k1/nofuser_k1/evaluation/evaluation_metrics.json",
        primary=False,
        note="Higher raw K=1 ablation; not the clean headline run.",
    ),
    "dual_bottleneck_k1_hist": Candidate(
        key="dual_bottleneck_k1_hist",
        role="Historical fused K=1",
        csv_path=ROOT / "experiments/dual_bottleneck_k1/evaluation/evaluation_image_scores.csv",
        metrics_path=ROOT / "experiments/dual_bottleneck_k1/evaluation/evaluation_metrics.json",
        primary=False,
        note="Historical upper-bound/reference; less cleanly comparable.",
    ),
    "patchcore_baseline": Candidate(
        key="patchcore_baseline",
        role="PatchCore baseline",
        csv_path=ROOT / "experiments/patchcore_dinov3_vitsmall_2/evaluation/evaluation_image_scores.csv",
        metrics_path=ROOT / "experiments/patchcore_dinov3_vitsmall_2/evaluation/evaluation_metrics.json",
        primary=False,
        note="External frozen-feature baseline; not fused into CAM-anchor candidates.",
    ),
}


EXPERIMENTS = (
    FusionExperiment(
        name="cls_patch_clean_image",
        group="image_cls_plus_patch",
        kind="primary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("patch_k32_clean", "image_score"),
        ),
        note="Cleanest CLS/image-level vs patch-level pairing.",
    ),
    FusionExperiment(
        name="cls_patch_clean_patch_recomputed_fused",
        group="image_cls_plus_patch",
        kind="primary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("patch_k32_clean", "recomputed_fused"),
        ),
        note="Tests patch auxiliary signals beyond patch anchor distance.",
    ),
    FusionExperiment(
        name="cls_patch_location_image",
        group="image_cls_plus_patch",
        kind="secondary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("patch_loc_cos_k32", "image_score"),
        ),
        note="Tests global CLS against spatially constrained patch references.",
    ),
    FusionExperiment(
        name="cls_patch_location_recomputed_fused",
        group="image_cls_plus_patch",
        kind="secondary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("patch_loc_cos_k32", "recomputed_fused"),
        ),
        note="Tests global CLS with the location-aware patch run's recomputed fused signal.",
    ),
    FusionExperiment(
        name="patch_k32_plus_location_recomputed_fused",
        group="patch_plus_location_patch",
        kind="secondary",
        specs=(
            SignalSpec("patch_k32_clean", "recomputed_fused"),
            SignalSpec("patch_loc_cos_k32", "recomputed_fused"),
        ),
        note="Tests whether shared patch K=32 and location-aware patch K=32 are complementary.",
    ),
    FusionExperiment(
        name="cls_patch_ablation_nofuser_patch_image",
        group="image_cls_plus_patch",
        kind="ablation",
        specs=(
            SignalSpec("global_k1_nofuser", "image_score"),
            SignalSpec("patch_k32_clean", "image_score"),
        ),
        note="Performance-oriented K=1 ablation paired with clean patch K=32.",
    ),
    FusionExperiment(
        name="one_multi_clean",
        group="one_anchor_plus_multi_anchor",
        kind="primary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("global_k1024_clean", "recomputed_fused"),
        ),
        note="Clean one-anchor raw distance plus large-K fused signal.",
    ),
    FusionExperiment(
        name="one_multi_ablation_nofuser",
        group="one_anchor_plus_multi_anchor",
        kind="ablation",
        specs=(
            SignalSpec("global_k1_nofuser", "image_score"),
            SignalSpec("global_k1024_clean", "recomputed_fused"),
        ),
        note="Higher-performing K=1 ablation plus clean large-K fused signal.",
    ),
    FusionExperiment(
        name="three_way_clean_global_patch",
        group="one_anchor_multi_anchor_patch",
        kind="primary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("global_k1024_clean", "recomputed_fused"),
            SignalSpec("patch_k32_clean", "recomputed_fused"),
        ),
        note="Three-way clean grid: K=1 global, K=1024 global, and patch K=32.",
    ),
    FusionExperiment(
        name="three_way_clean_global_location_patch",
        group="one_anchor_multi_anchor_patch",
        kind="secondary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("global_k1024_clean", "recomputed_fused"),
            SignalSpec("patch_loc_cos_k32", "recomputed_fused"),
        ),
        note="Three-way clean grid using the location-aware patch K=32 fused signal.",
    ),
    FusionExperiment(
        name="four_way_clean_global_patch_and_location",
        group="one_anchor_multi_anchor_patch",
        kind="secondary",
        specs=(
            SignalSpec("global_k1_clean", "image_score"),
            SignalSpec("global_k1024_clean", "recomputed_fused"),
            SignalSpec("patch_k32_clean", "recomputed_fused"),
            SignalSpec("patch_loc_cos_k32", "recomputed_fused"),
        ),
        note="Four-way clean grid that includes both shared patch K=32 and location-aware patch K=32.",
    ),
    FusionExperiment(
        name="three_way_ablation_global_patch",
        group="one_anchor_multi_anchor_patch",
        kind="ablation",
        specs=(
            SignalSpec("global_k1_nofuser", "image_score"),
            SignalSpec("global_k1024_clean", "recomputed_fused"),
            SignalSpec("patch_k32_clean", "recomputed_fused"),
        ),
        note="Three-way performance-oriented grid using the no-fuser K=1 ablation.",
    ),
)


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scores(candidate: Candidate) -> pd.DataFrame:
    df = pd.read_csv(candidate.csv_path)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{candidate.csv_path} must contain path and label columns")
    df = df.copy()
    df["path_norm"] = df["path"].astype(str).str.lower()
    df["label"] = df["label"].astype(int)
    return df


def normalize(values: np.ndarray, mode: str = "minmax") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mode == "zscore":
        std = values.std()
        if std < 1e-12:
            return np.zeros_like(values)
        return (values - values.mean()) / std
    if mode == "robust":
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        iqr = max(q75 - q25, 1e-12)
        return (values - q50) / iqr
    if mode == "rank":
        from scipy.stats import rankdata

        return rankdata(values) / len(values)
    lo, hi = values.min(), values.max()
    return (values - lo) / max(hi - lo, 1e-12)


def safe_auroc(labels: np.ndarray, scores: np.ndarray | None) -> float | None:
    if scores is None:
        return None
    return float(roc_auc_score(labels, scores))


def safe_aupr(labels: np.ndarray, scores: np.ndarray | None) -> float | None:
    if scores is None:
        return None
    return float(average_precision_score(labels, scores))


def optional_column(df: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in df.columns:
        return None
    return df[name].to_numpy(dtype=np.float64)


def choose_divergence(
    labels: np.ndarray,
    bottleneck: np.ndarray | None,
    patch: np.ndarray | None,
) -> tuple[str, float | None, np.ndarray | None]:
    bottleneck_auroc = safe_auroc(labels, bottleneck)
    patch_auroc = safe_auroc(labels, patch)
    bottleneck_ok = bottleneck is not None and bottleneck_auroc is not None and bottleneck_auroc >= 0.5
    patch_ok = patch is not None and patch_auroc is not None and patch_auroc >= 0.5

    if bottleneck_ok and patch_ok:
        if bottleneck_auroc >= patch_auroc:
            return "bottleneck_divergence", bottleneck_auroc, bottleneck
        return "patch_divergence_aggregated", patch_auroc, patch
    if bottleneck_ok:
        return "bottleneck_divergence", bottleneck_auroc, bottleneck
    if patch_ok:
        return "patch_divergence_aggregated", patch_auroc, patch
    return "dropped", None, None


def recompute_model_fused_signal(
    df: pd.DataFrame,
    normalization: str = "minmax",
    anchor_weight: float = 0.4,
    divergence_weight: float = 0.3,
    pixel_weight: float = 0.3,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = df["label"].to_numpy(dtype=np.int64)
    anchor = df["anchor_score"].to_numpy(dtype=np.float64)
    pixel = optional_column(df, "pixel_aggregated_score")
    bottleneck = optional_column(df, "bottleneck_divergence")
    patch = optional_column(df, "patch_divergence_aggregated")

    div_name, div_auroc, div_scores = choose_divergence(labels, bottleneck, patch)
    anchor_norm = normalize(anchor, normalization)

    components: list[tuple[float, np.ndarray]] = [(anchor_weight, anchor_norm)]
    if div_scores is not None:
        components.append((divergence_weight, normalize(div_scores, normalization)))
    if pixel is not None:
        pixel_auroc = roc_auc_score(labels, pixel)
        if pixel_auroc >= 0.5:
            components.append((pixel_weight, normalize(pixel, normalization)))

    total_weight = sum(weight for weight, _ in components)
    fused = sum(weight * values for weight, values in components) / total_weight
    diagnostics = {
        "selected_divergence_signal": div_name,
        "selected_divergence_auroc": div_auroc,
        "active_weights_sum": float(total_weight),
    }
    return fused, diagnostics


def get_signal(df: pd.DataFrame, signal_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    if signal_name == "recomputed_fused":
        return recompute_model_fused_signal(df)
    if signal_name not in df.columns:
        raise ValueError(f"Missing signal column {signal_name}")
    return df[signal_name].to_numpy(dtype=np.float64), {}


def candidate_summary(candidates: dict[str, Candidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates.values():
        metrics = load_metrics(candidate.metrics_path)
        rows.append(
            {
                "candidate": candidate.key,
                "role": candidate.role,
                "primary": candidate.primary,
                "image_auroc": metrics.get("image_auroc"),
                "fused_auroc": metrics.get("fused_image_auroc"),
                "pixel_auroc": metrics.get("pixel_auroc"),
                "num_normal": metrics.get("num_normal"),
                "num_anomaly": metrics.get("num_anomaly"),
                "csv_path": rel(candidate.csv_path),
                "metrics_path": rel(candidate.metrics_path),
                "note": candidate.note,
            }
        )
    return rows


def signal_column_name(spec: SignalSpec) -> str:
    return f"{spec.candidate_key}__{spec.signal}"


def build_aligned_frame(experiment: FusionExperiment) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    aligned: pd.DataFrame | None = None
    signal_rows: list[dict[str, Any]] = []

    for spec in experiment.specs:
        candidate = CANDIDATES[spec.candidate_key]
        df = load_scores(candidate)
        signal, diagnostics = get_signal(df, spec.signal)
        column = signal_column_name(spec)
        subset = pd.DataFrame(
            {
                "path_norm": df["path_norm"],
                "path": df["path"],
                "label": df["label"],
                column: signal,
            }
        )
        aligned = subset if aligned is None else aligned.merge(subset.drop(columns=["path"]), on=["path_norm", "label"])
        labels = df["label"].to_numpy(dtype=np.int64)
        signal_rows.append(
            {
                "candidate": spec.candidate_key,
                "signal": spec.signal,
                "column": column,
                "individual_auroc": safe_auroc(labels, signal),
                "individual_aupr": safe_aupr(labels, signal),
                **diagnostics,
            }
        )

    if aligned is None:
        raise ValueError(f"No signals configured for {experiment.name}")
    if aligned["path_norm"].duplicated().any():
        raise ValueError(f"Duplicate paths after alignment for {experiment.name}")
    return aligned, signal_rows


def grid_weights(n_signals: int, step: float) -> list[tuple[float, ...]]:
    if n_signals < 2:
        raise ValueError(f"At least two signals are required, got {n_signals}")

    denominator = int(round(1.0 / step))
    if denominator <= 0 or not np.isclose(denominator * step, 1.0):
        raise ValueError(f"Grid step must evenly divide 1.0, got {step}")

    weights: list[tuple[float, ...]] = []

    def build(prefix: tuple[int, ...], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append(tuple(value / denominator for value in (*prefix, remaining)))
            return
        for value in range(remaining + 1):
            build((*prefix, value), remaining - value, slots - 1)

    build((), denominator, n_signals)
    return weights


def evaluate_experiment(experiment: FusionExperiment, step: float) -> dict[str, Any]:
    aligned, signal_rows = build_aligned_frame(experiment)
    labels = aligned["label"].to_numpy(dtype=np.int64)
    columns = [signal_column_name(spec) for spec in experiment.specs]
    normalized = [normalize(aligned[column].to_numpy(dtype=np.float64)) for column in columns]

    best: dict[str, Any] | None = None
    for weights in grid_weights(len(columns), step):
        fused = np.zeros_like(normalized[0])
        for weight, values in zip(weights, normalized):
            fused = fused + weight * values
        auroc = float(roc_auc_score(labels, fused))
        aupr = float(average_precision_score(labels, fused))
        if best is None or auroc > best["fusion_auroc"]:
            best = {
                "weights": weights,
                "fusion_auroc": auroc,
                "fusion_aupr": aupr,
            }

    if best is None:
        raise ValueError(f"No grid weights evaluated for {experiment.name}")

    strongest_component = max(signal_rows, key=lambda row: row["individual_auroc"] or -1.0)
    return {
        "name": experiment.name,
        "group": experiment.group,
        "kind": experiment.kind,
        "note": experiment.note,
        "n_samples": int(len(aligned)),
        "num_normal": int((labels == 0).sum()),
        "num_anomaly": int((labels == 1).sum()),
        "signals": signal_rows,
        "best_weights": {
            signal_rows[index]["column"]: float(weight)
            for index, weight in enumerate(best["weights"])
        },
        "fusion_auroc": best["fusion_auroc"],
        "fusion_aupr": best["fusion_aupr"],
        "strongest_component": strongest_component["column"],
        "strongest_component_auroc": strongest_component["individual_auroc"],
        "delta_vs_strongest_component": best["fusion_auroc"] - strongest_component["individual_auroc"],
    }


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_experiment_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        weight_text = ";".join(f"{key}={value:.2f}" for key, value in result["best_weights"].items())
        signal_text = ";".join(
            f"{row['column']}:{row['individual_auroc']:.4f}"
            for row in result["signals"]
        )
        rows.append(
            {
                "name": result["name"],
                "group": result["group"],
                "kind": result["kind"],
                "n_samples": result["n_samples"],
                "num_normal": result["num_normal"],
                "num_anomaly": result["num_anomaly"],
                "fusion_auroc": result["fusion_auroc"],
                "fusion_aupr": result["fusion_aupr"],
                "strongest_component": result["strongest_component"],
                "strongest_component_auroc": result["strongest_component_auroc"],
                "delta_vs_strongest_component": result["delta_vs_strongest_component"],
                "best_weights": weight_text,
                "signals": signal_text,
                "note": result["note"],
            }
        )
    return rows


def write_report(output_dir: Path, candidates: list[dict[str, Any]], results: list[dict[str, Any]], step: float) -> None:
    def append_candidate_table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                "",
                title,
                "",
                "| Candidate | Role | Image AUROC | Fused AUROC | Pixel AUROC | Note |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            fused = "n/a" if row["fused_auroc"] is None else f"{row['fused_auroc']:.4f}"
            pixel = "n/a" if row["pixel_auroc"] is None else f"{row['pixel_auroc']:.4f}"
            lines.append(
                f"| `{row['candidate']}` | {row['role']} | {row['image_auroc']:.4f} | {fused} | {pixel} | {row['note']} |"
            )

    def append_result_table(title: str, rows: list[dict[str, Any]]) -> None:
        if title:
            lines.extend(["", title])
        lines.extend(
            [
                "",
                "| Experiment | Group | Kind | Fusion AUROC | Strongest Component AUROC | Delta | Best Weights |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for result in sorted(rows, key=lambda item: item["fusion_auroc"], reverse=True):
            weights = ", ".join(f"`{key}`={value:.2f}" for key, value in result["best_weights"].items())
            lines.append(
                f"| `{result['name']}` | {result['group']} | {result['kind']} | "
                f"{result['fusion_auroc']:.4f} | {result['strongest_component_auroc']:.4f} | "
                f"{result['delta_vs_strongest_component']:+.4f} | {weights} |"
            )

    lines = [
        "# Fusion Candidate Experiment Results",
        "",
        "These are post-hoc exploratory fusions on existing test-set score CSVs. "
        "They are useful for candidate selection, but final fusion weights should be selected on validation data.",
        "",
        f"Grid step: `{step}`.",
    ]

    clean_candidates = [
        row
        for row in candidates
        if row["primary"] or row["candidate"] in {"patch_loc_cos_k32", "patchcore_baseline", "dual_bottleneck_k1_hist"}
    ]
    ablation_candidates = [row for row in candidates if row["candidate"] == "global_k1_nofuser"]
    append_candidate_table("## Clean Candidate Metrics", clean_candidates)
    append_candidate_table("## Ablation Candidate Metrics", ablation_candidates)

    clean_results = [row for row in results if row["kind"] != "ablation"]
    ablation_results = [row for row in results if row["kind"] == "ablation"]
    append_result_table("## Clean Fusion Results", clean_results)
    lines.extend(
        [
            "",
            "## Ablation Fusion Results",
            "",
            "These include `nofuser_k1`, which is a useful architectural ablation but a messier comparison than the clean Stage2E70 family.",
        ]
    )
    append_result_table("", ablation_results)

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Rows are aligned by lowercase-normalized image path and label.",
            "- `recomputed_fused` uses anchor + best non-anticorrelated divergence + pixel aggregation, matching the evaluation policy.",
            "- PatchCore is reported as a baseline only and is not included in the CAM-anchor fusion grids.",
            "- `nofuser_k1` is separated as an ablation because it removes anchor-conditioned reconstruction and also comes from a slightly different run family.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path, step: float) -> dict[str, Any]:
    for candidate in CANDIDATES.values():
        if not candidate.csv_path.exists():
            raise FileNotFoundError(candidate.csv_path)
        if not candidate.metrics_path.exists():
            raise FileNotFoundError(candidate.metrics_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = candidate_summary(CANDIDATES)
    results = [evaluate_experiment(experiment, step) for experiment in EXPERIMENTS]

    write_csv(
        output_dir / "candidate_summary.csv",
        candidate_rows,
        [
            "candidate",
            "role",
            "primary",
            "image_auroc",
            "fused_auroc",
            "pixel_auroc",
            "num_normal",
            "num_anomaly",
            "csv_path",
            "metrics_path",
            "note",
        ],
    )
    write_csv(
        output_dir / "fusion_experiments.csv",
        flatten_experiment_rows(results),
        [
            "name",
            "group",
            "kind",
            "n_samples",
            "num_normal",
            "num_anomaly",
            "fusion_auroc",
            "fusion_aupr",
            "strongest_component",
            "strongest_component_auroc",
            "delta_vs_strongest_component",
            "best_weights",
            "signals",
            "note",
        ],
    )

    payload = {
        "description": "Post-hoc fusion candidate experiments for CLS/image-level, patch-level, one-anchor, and multi-anchor signals.",
        "warning": "These weights are optimized on existing test-set scores and should be treated as exploratory. Final weights should be selected on validation data.",
        "grid_step": step,
        "candidates": candidate_rows,
        "experiments": results,
    }
    with (output_dir / "fusion_experiments.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_report(output_dir, candidate_rows, results, step)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc fusion experiments for selected compatible candidates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step", type=float, default=0.05, help="Grid step for fusion weights.")
    args = parser.parse_args()

    payload = run(args.output_dir, args.step)
    print(f"Wrote fusion candidate outputs to {args.output_dir}")
    for result in sorted(payload["experiments"], key=lambda item: item["fusion_auroc"], reverse=True):
        print(
            f"{result['name']}: AUROC={result['fusion_auroc']:.4f}, "
            f"delta={result['delta_vs_strongest_component']:+.4f}"
        )


if __name__ == "__main__":
    main()
