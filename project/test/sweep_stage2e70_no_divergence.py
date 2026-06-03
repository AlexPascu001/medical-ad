from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
OUTPUT_JSON = ROOT / "experiments" / "stage2e70_no_divergence_sweep.json"
OUTPUT_CSV = ROOT / "experiments" / "stage2e70_no_divergence_sweep.csv"

PATCH_PREFIX = "patch_stage2e70"
REDESIGN_PREFIX = "full_redesign_stage2e70"


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    scale = max(maximum - minimum, 1e-12)
    return (values - minimum) / scale


def safe_auroc(labels: np.ndarray, scores: np.ndarray | None) -> float | None:
    if scores is None:
        return None
    return float(roc_auc_score(labels, scores))


def load_image_scores(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header in {csv_path}")
        rows = list(reader)

    columns: dict[str, np.ndarray] = {}
    for field in reader.fieldnames:
        values = [row[field] for row in rows]
        if field == "path":
            continue
        if field == "label":
            columns[field] = np.asarray([int(float(value)) for value in values], dtype=np.int64)
        else:
            columns[field] = np.asarray([float(value) for value in values], dtype=np.float64)
    return columns


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def classify_bucket(run_name: str) -> str:
    if run_name.startswith(PATCH_PREFIX):
        return "original_patch" if "_vit" not in run_name else "new_patch"
    if run_name.startswith(REDESIGN_PREFIX):
        return "original_redesign" if "_vit" not in run_name else "new_redesign"
    raise ValueError(f"Unsupported run name: {run_name}")


def extract_k(run_name: str) -> int:
    match = re.search(r"_k(\d+)$", run_name)
    if not match:
        raise ValueError(f"Could not parse K from run name: {run_name}")
    return int(match.group(1))


def stable_sort_key(run_name: str) -> tuple[int, int, str]:
    bucket_order = {
        "original_redesign": 0,
        "new_redesign": 1,
        "original_patch": 2,
        "new_patch": 3,
    }
    bucket = classify_bucket(run_name)
    return bucket_order[bucket], extract_k(run_name), run_name


def choose_divergence_signal(
    bottleneck_auroc: float | None,
    patch_auroc: float | None,
    bottleneck_scores: np.ndarray | None,
    patch_scores: np.ndarray | None,
) -> tuple[str, float | None, np.ndarray | None]:
    bottleneck_valid = bottleneck_auroc is not None and bottleneck_auroc >= 0.5 and bottleneck_scores is not None
    patch_valid = patch_auroc is not None and patch_auroc >= 0.5 and patch_scores is not None

    if bottleneck_valid and patch_valid:
        if bottleneck_auroc >= patch_auroc:
            return "cls_bottleneck", bottleneck_auroc, bottleneck_scores
        return "patch_divergence", patch_auroc, patch_scores
    if bottleneck_valid:
        return "cls_bottleneck", bottleneck_auroc, bottleneck_scores
    if patch_valid:
        return "patch_divergence", patch_auroc, patch_scores
    return "dropped", None, None


def compute_current_fused_auroc(
    labels: np.ndarray,
    anchor_scores: np.ndarray,
    pixel_scores: np.ndarray | None,
    bottleneck_scores: np.ndarray | None,
    patch_scores: np.ndarray | None,
    bottleneck_auroc: float | None,
    patch_auroc: float | None,
) -> tuple[float, str, float | None]:
    anchor_norm = minmax_normalize(anchor_scores)
    choice, selected_auroc, selected_scores = choose_divergence_signal(
        bottleneck_auroc=bottleneck_auroc,
        patch_auroc=patch_auroc,
        bottleneck_scores=bottleneck_scores,
        patch_scores=patch_scores,
    )

    if selected_scores is not None and pixel_scores is not None:
        fused = 0.4 * anchor_norm + 0.3 * minmax_normalize(selected_scores) + 0.3 * minmax_normalize(pixel_scores)
    elif selected_scores is not None:
        total = 0.4 + 0.3
        fused = (0.4 / total) * anchor_norm + (0.3 / total) * minmax_normalize(selected_scores)
    elif pixel_scores is not None:
        total = 0.4 + 0.3
        fused = (0.4 / total) * anchor_norm + (0.3 / total) * minmax_normalize(pixel_scores)
    else:
        fused = anchor_norm

    return float(roc_auc_score(labels, fused)), choice, selected_auroc


def compute_no_divergence_auroc(labels: np.ndarray, anchor_scores: np.ndarray, pixel_scores: np.ndarray | None) -> float:
    anchor_norm = minmax_normalize(anchor_scores)
    if pixel_scores is None:
        return float(roc_auc_score(labels, anchor_norm))
    fused = 0.5 * anchor_norm + 0.5 * minmax_normalize(pixel_scores)
    return float(roc_auc_score(labels, fused))


def discover_csv_paths() -> list[Path]:
    discovered: list[Path] = []
    for csv_path in RUNS_DIR.rglob("evaluation_image_scores.csv"):
        if csv_path.parent.name != "evaluation":
            continue
        run_name = csv_path.parent.parent.name
        if run_name.startswith(PATCH_PREFIX) or run_name.startswith(REDESIGN_PREFIX):
            discovered.append(csv_path)
    return sorted(discovered, key=lambda path: stable_sort_key(path.parent.parent.name))


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for family in ("redesign", "patch"):
        family_rows = [row for row in rows if row["family"] == family]
        improved = [row for row in family_rows if row["no_divergence_anchor_pixel_50_50_image_auroc"] > row["stored_fused_image_auroc"]]
        families[family] = {
            "count": len(family_rows),
            "best_no_divergence_run": max(
                family_rows,
                key=lambda row: row["no_divergence_anchor_pixel_50_50_image_auroc"],
            )["run_name"],
            "best_no_divergence_auroc": max(row["no_divergence_anchor_pixel_50_50_image_auroc"] for row in family_rows),
            "mean_delta_vs_stored_fused": float(
                np.mean([row["no_divergence_minus_stored_fused"] for row in family_rows])
            ),
            "num_improved_vs_stored_fused": len(improved),
            "selected_divergence_counts": {
                choice: sum(1 for row in family_rows if row["selected_divergence_signal"] == choice)
                for choice in ("cls_bottleneck", "patch_divergence", "dropped")
            },
        }
    return {
        "run_count": len(rows),
        "families": families,
    }


def build_row(csv_path: Path) -> dict[str, Any]:
    run_name = csv_path.parent.parent.name
    metrics_path = csv_path.parent / "evaluation_metrics.json"

    image_scores = load_image_scores(csv_path)
    metrics = load_metrics(metrics_path)
    labels = image_scores["label"]

    anchor_scores = image_scores["anchor_score"]
    reconstruction_scores = image_scores.get("reconstruction_score")
    bottleneck_scores = image_scores.get("bottleneck_divergence")
    pixel_scores = image_scores.get("pixel_aggregated_score")
    patch_scores = image_scores.get("patch_divergence_aggregated")

    anchor_auroc = safe_auroc(labels, anchor_scores)
    reconstruction_auroc = safe_auroc(labels, reconstruction_scores)
    bottleneck_auroc = safe_auroc(labels, bottleneck_scores)
    pixel_auroc = safe_auroc(labels, pixel_scores)
    patch_auroc = safe_auroc(labels, patch_scores)

    recomputed_fused_auroc, selected_divergence_signal, selected_divergence_auroc = compute_current_fused_auroc(
        labels=labels,
        anchor_scores=anchor_scores,
        pixel_scores=pixel_scores,
        bottleneck_scores=bottleneck_scores,
        patch_scores=patch_scores,
        bottleneck_auroc=bottleneck_auroc,
        patch_auroc=patch_auroc,
    )
    no_divergence_auroc = compute_no_divergence_auroc(labels, anchor_scores, pixel_scores)

    family = "patch" if run_name.startswith(PATCH_PREFIX) else "redesign"

    return {
        "run_name": run_name,
        "family": family,
        "bucket": classify_bucket(run_name),
        "k": extract_k(run_name),
        "anchor_image_auroc": anchor_auroc,
        "reconstruction_image_auroc": reconstruction_auroc,
        "cls_bottleneck_divergence_image_auroc": bottleneck_auroc,
        "patch_divergence_aggregated_image_auroc": patch_auroc,
        "pixel_aggregated_image_auroc": pixel_auroc,
        "stored_image_auroc": float(metrics["image_auroc"]),
        "stored_fused_image_auroc": float(metrics["fused_image_auroc"]),
        "recomputed_current_fused_image_auroc": recomputed_fused_auroc,
        "current_fused_diff_vs_stored": recomputed_fused_auroc - float(metrics["fused_image_auroc"]),
        "selected_divergence_signal": selected_divergence_signal,
        "selected_divergence_auroc": selected_divergence_auroc,
        "no_divergence_anchor_pixel_50_50_image_auroc": no_divergence_auroc,
        "no_divergence_minus_stored_fused": no_divergence_auroc - float(metrics["fused_image_auroc"]),
        "csv_path": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "metrics_path": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "family",
        "bucket",
        "k",
        "stored_image_auroc",
        "stored_fused_image_auroc",
        "anchor_image_auroc",
        "reconstruction_image_auroc",
        "cls_bottleneck_divergence_image_auroc",
        "patch_divergence_aggregated_image_auroc",
        "pixel_aggregated_image_auroc",
        "selected_divergence_signal",
        "selected_divergence_auroc",
        "recomputed_current_fused_image_auroc",
        "current_fused_diff_vs_stored",
        "no_divergence_anchor_pixel_50_50_image_auroc",
        "no_divergence_minus_stored_fused",
        "csv_path",
        "metrics_path",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Post-hoc Stage2E70 sweep with divergence omitted from fusion and anchor/pixel weights set to 0.5/0.5.",
        "weights": {
            "anchor": 0.5,
            "divergence": 0.0,
            "pixel": 0.5,
        },
        "rows": rows,
        "summary": summarize_results(rows),
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def print_summary(rows: list[dict[str, Any]]) -> None:
    summary = summarize_results(rows)
    print(f"Processed {summary['run_count']} Stage2E70 runs")
    for family, family_summary in summary["families"].items():
        print(
            f"{family}: best no-div run={family_summary['best_no_divergence_run']} "
            f"AUROC={family_summary['best_no_divergence_auroc']:.4f}, "
            f"mean delta vs stored fused={family_summary['mean_delta_vs_stored_fused']:.4f}, "
            f"improved={family_summary['num_improved_vs_stored_fused']}/{family_summary['count']}"
        )


def main() -> None:
    csv_paths = discover_csv_paths()
    rows = [build_row(csv_path) for csv_path in csv_paths]
    write_csv(rows)
    write_json(rows)
    print_summary(rows)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()