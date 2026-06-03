"""
Post-hoc cluster diagnostics for finished experiments.

This script loads one or more experiment folders, rebuilds the model from the
saved config + anchors, loads a checkpoint, and computes assignment / distance
diagnostics on validation and/or test splits.

It is intended to answer questions such as:
- How many anchors are effectively used?
- Are normals concentrated around a few anchors?
- What are the nearest / second-nearest distance margins?
- How imbalanced are anchor assignments?

Outputs for each experiment/checkpoint/split:
- JSON summary with aggregate diagnostics
- CSV with per-anchor counts / proportions / mean distances
- CSV with per-sample assignments and nearest-anchor statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

from data import create_dataloaders
from main import load_dataset_paths
from model import AnomalyDetector, DINOv3Backbone


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_anchor_data(anchor_path: Path, device: torch.device) -> dict:
    raw = torch.load(anchor_path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        return {
            "anchor_global": raw,
            "anchor_dense": None,
            "anchor_semantic": None,
            "anchor_geometric": None,
            "anchor_metadata": {},
        }

    return {
        "anchor_global": raw.get("anchor_global", raw.get("global")),
        "anchor_dense": raw.get("anchor_dense", raw.get("dense")),
        "anchor_semantic": raw.get("anchor_semantic"),
        "anchor_geometric": raw.get("anchor_geometric"),
        "anchor_metadata": raw.get("anchor_metadata", {}),
    }


def _build_model(config: dict, anchor_data: dict, checkpoint_name: str, device: torch.device) -> AnomalyDetector:
    use_pixel_decoder = config["model"].get("use_pixel_decoder", False)
    multi_scale_indices = config["model"].get("multi_scale_indices", [2, 5, 8, 11])
    projection_hidden_dims = config["model"].get("projection_hidden_dims")
    projection_dim = (
        projection_hidden_dims[-1]
        if projection_hidden_dims is not None
        else config["model"].get("projection_dim")
    )

    backbone = DINOv3Backbone(
        model_name=config["model"]["backbone"],
        freeze_backbone=config["model"].get("freeze_backbone", True),
        projection_dim=projection_dim,
        pretrained=True,
        multi_scale_indices=multi_scale_indices if use_pixel_decoder else None,
        projection_hidden_dims=projection_hidden_dims,
    ).to(device)

    target_size = tuple(config["data"]["target_size"])
    learnable_anchors = config["anchor"].get("learnable", False)
    use_embedding_space = config["anchor"].get("use_embedding_space", False)
    reproject_anchors = config["anchor"].get("reproject_anchors", False)
    use_decoupled = (
        use_embedding_space
        and (not reproject_anchors)
        and (anchor_data.get("anchor_semantic") is not None)
        and (anchor_data.get("anchor_geometric") is not None)
    )

    anchor_global = anchor_data["anchor_global"]
    anchor_dense = anchor_data["anchor_dense"]

    if use_decoupled:
        model = AnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_global,
            anchor_dense_embeddings=anchor_dense,
            distance_metric=config["loss"]["distance_metric"],
            learnable_anchors=learnable_anchors,
            use_pixel_decoder=use_pixel_decoder,
            decoder_hidden_dim=config["model"].get("decoder_hidden_dim", 256),
            target_size=target_size,
            anchor_semantic_embeddings=anchor_data["anchor_semantic"],
            anchor_geometric_targets=anchor_data["anchor_geometric"],
            use_decoupled_anchors=True,
        ).to(device)
    else:
        anchors_already_projected = False if use_embedding_space else (projection_dim is not None)
        model = AnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_global,
            anchor_dense_embeddings=anchor_dense,
            distance_metric=config["loss"]["distance_metric"],
            learnable_anchors=learnable_anchors,
            use_pixel_decoder=use_pixel_decoder,
            decoder_hidden_dim=config["model"].get("decoder_hidden_dim", 256),
            target_size=target_size,
            anchors_already_projected=anchors_already_projected,
        ).to(device)

    if "stage2" in checkpoint_name and config.get("stage2", {}).get("enabled", False):
        stage2_cfg = config["stage2"]
        pixel_map_cfg = stage2_cfg.get("pixel_map", {})
        recon_proj_dim = config.get("model", {}).get("projection_dim_recon")
        model.enable_reconstruction_branch(
            freeze_anchor_target=stage2_cfg.get("freeze_anchor_target", True),
            out_channels=3,
            pixel_map_enabled=pixel_map_cfg.get("enabled", True),
            pixel_map_type=pixel_map_cfg.get("type", "reconstruction_l2"),
            use_frozen_bottleneck=stage2_cfg.get("frozen_bottleneck", True),
            recon_projection_dim=recon_proj_dim,
            no_fuser=stage2_cfg.get("no_fuser", False),
        )

        pixel_agg_cfg = stage2_cfg.get("pixel_aggregation", {})
        model.configure_pixel_aggregation(
            method=pixel_agg_cfg.get("method", "top_k_percentile"),
            percentile=pixel_agg_cfg.get("percentile", 95),
            threshold=pixel_agg_cfg.get("threshold_n_std") if pixel_agg_cfg.get("method") == "threshold_ratio" else None,
        )

        fusion_cfg = stage2_cfg.get("score_fusion", {})
        model.configure_score_fusion(
            enabled=fusion_cfg.get("enabled", False),
            normalization=fusion_cfg.get("normalization", "minmax"),
            anchor_weight=fusion_cfg.get("anchor_weight", 0.4),
            divergence_weight=fusion_cfg.get("divergence_weight", 0.3),
            pixel_weight=fusion_cfg.get("pixel_weight", 0.3),
            drop_anticorrelated=fusion_cfg.get("drop_anticorrelated", True),
        )

    return model


def _resolve_checkpoints(experiment_dir: Path, requested: Optional[List[str]]) -> List[Path]:
    if requested:
        checkpoints = [experiment_dir / name for name in requested if (experiment_dir / name).exists()]
        if not checkpoints:
            raise FileNotFoundError(f"None of the requested checkpoints exist in {experiment_dir}")
        return checkpoints

    best_stage2 = experiment_dir / "best_stage2_model.pth"
    if best_stage2.exists():
        return [best_stage2]

    best_stage1 = experiment_dir / "best_model.pth"
    if best_stage1.exists():
        return [best_stage1]

    raise FileNotFoundError(f"No supported checkpoint found in {experiment_dir}")


def _entropy_from_counts(counts: np.ndarray) -> Dict[str, float]:
    total = float(counts.sum())
    if total <= 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0}

    probs = counts[counts > 0] / total
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    normalized = float(entropy / max(max_entropy, 1e-12))
    return {"entropy": entropy, "normalized_entropy": normalized}


def _describe_subset(name: str, assigned: np.ndarray, nearest: np.ndarray, second: np.ndarray, margin: np.ndarray, ratio: np.ndarray, n_anchors: int) -> Dict[str, float]:
    counts = np.bincount(assigned, minlength=n_anchors).astype(np.int64)
    nonzero = counts[counts > 0]
    largest_share = float(counts.max() / max(counts.sum(), 1))
    result = {
        "subset": name,
        "n_samples": int(len(assigned)),
        "effective_anchors_used": int((counts > 0).sum()),
        "largest_anchor_share": largest_share,
        "max_count": int(counts.max()) if len(counts) else 0,
        "min_nonzero_count": int(nonzero.min()) if len(nonzero) else 0,
        "max_min_nonzero_ratio": float(nonzero.max() / max(nonzero.min(), 1)) if len(nonzero) else 0.0,
        "mean_nearest_distance": float(np.mean(nearest)) if len(nearest) else 0.0,
        "std_nearest_distance": float(np.std(nearest)) if len(nearest) else 0.0,
        "median_nearest_distance": float(np.median(nearest)) if len(nearest) else 0.0,
    }
    result.update(_entropy_from_counts(counts))

    valid_second = second[~np.isnan(second)]
    valid_margin = margin[~np.isnan(margin)]
    valid_ratio = ratio[~np.isnan(ratio)]
    result["mean_second_nearest_distance"] = float(np.mean(valid_second)) if len(valid_second) else None
    result["mean_margin_d2_minus_d1"] = float(np.mean(valid_margin)) if len(valid_margin) else None
    result["median_margin_d2_minus_d1"] = float(np.median(valid_margin)) if len(valid_margin) else None
    result["mean_ratio_d1_over_d2"] = float(np.mean(valid_ratio)) if len(valid_ratio) else None
    result["median_ratio_d1_over_d2"] = float(np.median(valid_ratio)) if len(valid_ratio) else None
    return result


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_split(model: AnomalyDetector, dataloader, device: torch.device, split_name: str, experiment_dir: Path, checkpoint_name: str, config: dict, anchor_metadata: dict) -> Dict[str, object]:
    model.eval()
    n_anchors = int(model.n_anchors)

    all_paths: List[str] = []
    all_labels: List[int] = []
    all_assigned: List[np.ndarray] = []
    all_distances: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = model.compute_anomaly_scores(images, return_maps=False)

            all_paths.extend(batch["path"])
            all_labels.append(batch["label"].cpu().numpy())
            all_assigned.append(outputs["assigned_anchors"].cpu().numpy())
            all_distances.append(outputs["all_distances"].cpu().numpy())

    labels = np.concatenate(all_labels)
    assigned = np.concatenate(all_assigned)
    distances = np.concatenate(all_distances)
    sorted_distances = np.sort(distances, axis=1)
    nearest = sorted_distances[:, 0]

    if distances.shape[1] > 1:
        second = sorted_distances[:, 1]
        margin = second - nearest
        ratio = nearest / np.maximum(second, 1e-12)
    else:
        second = np.full_like(nearest, np.nan)
        margin = np.full_like(nearest, np.nan)
        ratio = np.full_like(nearest, np.nan)

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    per_anchor_rows = []
    for anchor_idx in range(n_anchors):
        anchor_mask = assigned == anchor_idx
        anchor_normal_mask = anchor_mask & normal_mask
        anchor_anomaly_mask = anchor_mask & anomaly_mask
        count_all = int(anchor_mask.sum())
        count_normal = int(anchor_normal_mask.sum())
        count_anomaly = int(anchor_anomaly_mask.sum())

        per_anchor_rows.append({
            "anchor_idx": anchor_idx,
            "count_all": count_all,
            "count_normal": count_normal,
            "count_anomaly": count_anomaly,
            "frac_all": float(count_all / max(len(labels), 1)),
            "frac_normal": float(count_normal / max(int(normal_mask.sum()), 1)),
            "frac_anomaly": float(count_anomaly / max(int(anomaly_mask.sum()), 1)),
            "mean_nearest_distance_assigned": float(np.mean(nearest[anchor_mask])) if count_all else None,
            "mean_margin_assigned": float(np.mean(margin[anchor_mask])) if count_all and distances.shape[1] > 1 else None,
            "mean_ratio_assigned": float(np.mean(ratio[anchor_mask])) if count_all and distances.shape[1] > 1 else None,
        })

    sample_rows = []
    for index, path in enumerate(all_paths):
        sample_rows.append({
            "path": path,
            "label": int(labels[index]),
            "assigned_anchor": int(assigned[index]),
            "nearest_distance": float(nearest[index]),
            "second_nearest_distance": None if np.isnan(second[index]) else float(second[index]),
            "margin_d2_minus_d1": None if np.isnan(margin[index]) else float(margin[index]),
            "ratio_d1_over_d2": None if np.isnan(ratio[index]) else float(ratio[index]),
        })

    summary = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": checkpoint_name,
        "split": split_name,
        "requested_k": int(config["anchor"]["n_anchors"]),
        "effective_k": int(anchor_metadata.get("effective_k", n_anchors)),
        "initial_k": int(anchor_metadata.get("initial_k", anchor_metadata.get("effective_k", n_anchors))),
        "discarded_clusters": int(len(anchor_metadata.get("discarded_cluster_ids", []))),
        "representation": anchor_metadata.get("representation", config["anchor"].get("representation")),
        "all_samples": _describe_subset("all", assigned, nearest, second, margin, ratio, n_anchors),
        "normal_samples": _describe_subset("normal", assigned[normal_mask], nearest[normal_mask], second[normal_mask], margin[normal_mask], ratio[normal_mask], n_anchors),
        "anomaly_samples": _describe_subset("anomaly", assigned[anomaly_mask], nearest[anomaly_mask], second[anomaly_mask], margin[anomaly_mask], ratio[anomaly_mask], n_anchors),
    }

    output_dir = experiment_dir / "evaluation" / "cluster_diagnostics" / checkpoint_name.replace(".pth", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{split_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _write_csv(
        output_dir / f"{split_name}_per_anchor.csv",
        per_anchor_rows,
        [
            "anchor_idx",
            "count_all",
            "count_normal",
            "count_anomaly",
            "frac_all",
            "frac_normal",
            "frac_anomaly",
            "mean_nearest_distance_assigned",
            "mean_margin_assigned",
            "mean_ratio_assigned",
        ],
    )
    _write_csv(
        output_dir / f"{split_name}_per_sample.csv",
        sample_rows,
        [
            "path",
            "label",
            "assigned_anchor",
            "nearest_distance",
            "second_nearest_distance",
            "margin_d2_minus_d1",
            "ratio_d1_over_d2",
        ],
    )

    return summary


def analyze_experiment(experiment_dir: Path, checkpoints: Optional[List[str]], split_mode: str, device: torch.device) -> List[Dict[str, object]]:
    config = _load_config(experiment_dir / "config.yaml")
    anchor_data = _load_anchor_data(experiment_dir / "anchor_embeddings.pt", device)
    anchor_metadata = anchor_data.get("anchor_metadata", {})

    train_paths, val_paths, val_labels, val_masks, test_paths, test_labels, test_masks = load_dataset_paths(config["data"]["data_root"])
    _, val_loader, test_loader = create_dataloaders(
        train_paths,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        val_mask_paths=val_masks,
        test_mask_paths=test_masks,
        batch_size=config["training"].get("batch_size", 64),
        num_workers=config["training"].get("num_workers", 4),
        target_size=tuple(config["data"]["target_size"]),
        normalize_mode=config["data"].get("normalization", "zscore_only"),
    )

    summaries: List[Dict[str, object]] = []
    split_loaders = []
    if split_mode in {"val", "both"}:
        split_loaders.append(("val", val_loader))
    if split_mode in {"test", "both"}:
        split_loaders.append(("test", test_loader))

    for checkpoint_path in _resolve_checkpoints(experiment_dir, checkpoints):
        model = _build_model(config, anchor_data, checkpoint_path.name, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        for split_name, loader in split_loaders:
            summary = analyze_split(model, loader, device, split_name, experiment_dir, checkpoint_path.name, config, anchor_metadata)
            summaries.append(summary)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc cluster diagnostics for finished experiments")
    parser.add_argument("--experiment", nargs="+", required=True, help="One or more experiment directories")
    parser.add_argument("--checkpoint", nargs="*", default=None, help="Checkpoint file names inside each experiment dir (default: best_stage2_model.pth if present else best_model.pth)")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    all_summaries: List[Dict[str, object]] = []

    for experiment in args.experiment:
        experiment_dir = Path(experiment)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        print(f"\nAnalyzing {experiment_dir} ...")
        summaries = analyze_experiment(experiment_dir, args.checkpoint, args.split, device)
        all_summaries.extend(summaries)
        for summary in summaries:
            normal = summary["normal_samples"]
            print(
                f"  {summary['checkpoint']} [{summary['split']}] | "
                f"effective_k={summary['effective_k']} | "
                f"normal_mean_d1={normal['mean_nearest_distance']:.4f} | "
                f"normal_entropy={normal['normalized_entropy']:.4f} | "
                f"normal_used={normal['effective_anchors_used']}"
            )

    print(f"\nDone. Wrote diagnostics under each experiment's evaluation/cluster_diagnostics folder.")


if __name__ == "__main__":
    main()