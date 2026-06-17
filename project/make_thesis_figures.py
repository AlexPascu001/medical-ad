"""Generate polished thesis figures from saved experiment artifacts.

The existing evaluation figures are intentionally diagnostic. This script creates
compact, caption-friendly figures for the dissertation using the same metrics,
score CSVs, checkpoints, and test images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import patches
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore", category=FutureWarning)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data import BMADPreprocessor  # noqa: E402
from main import _load_model_checkpoint, create_model  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    path: Path
    color: str


RUNS = {
    "global": RunSpec(
        "global",
        "Global centroid",
        REPO_ROOT / "runs" / "full_redesign_stage2e70_k1" / "full_redesign_stage2e70_k1",
        "#4C78A8",
    ),
    "shared": RunSpec(
        "shared",
        "Shared patch",
        REPO_ROOT / "runs" / "patch_stage2e70_k32" / "patch_stage2e70_k32",
        "#F58518",
    ),
    "location": RunSpec(
        "location",
        "Location-aware",
        REPO_ROOT
        / "runs"
        / "patch_location_kmeans_stage2recon_cosine_k32"
        / "patch_location_kmeans_stage2recon_cosine_k32",
        "#54A24B",
    ),
    "patchcore": RunSpec(
        "patchcore",
        "PatchCore",
        REPO_ROOT / "experiments" / "patchcore_dinov3_vitsmall_2",
        "#B279A2",
    ),
}

FIGURE_NAMES = [
    "fig-method-evolution.png",
    "fig-family-performance.png",
    "fig-score-signals.png",
    "fig-reconstruction-localization.png",
    "fig-qualitative-comparison.png",
    "fig-anchor-space.png",
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalize01(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [low, high])
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def load_scores(run: RunSpec) -> pd.DataFrame:
    return pd.read_csv(run.path / "evaluation" / "evaluation_image_scores.csv")


def load_metrics(run: RunSpec) -> dict[str, Any]:
    return load_json(run.path / "evaluation" / "evaluation_metrics.json")


def metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else float(value)


def make_method_evolution(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.set_axis_off()

    boxes = [
        ("Global\ncentroid anchors", "CLS token\nnearest anchor", RUNS["global"].color),
        ("Shared\npatch anchors", "Dense tokens\nshared bank", RUNS["shared"].color),
        ("Location-aware\npatch centroids", "Dense tokens\nsame-location bank", RUNS["location"].color),
        ("PatchCore\nreference", "Frozen dense tokens\nmemory bank", RUNS["patchcore"].color),
    ]

    for i, (title, body, color) in enumerate(boxes):
        x = 0.03 + i * 0.245
        rect = patches.FancyBboxPatch(
            (x, 0.28),
            0.19,
            0.44,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
        ax.add_patch(rect)
        ax.text(x + 0.095, 0.59, title, ha="center", va="center", weight="bold", color=color)
        ax.text(x + 0.095, 0.42, body, ha="center", va="center", fontsize=8, color="#333333")
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x + 0.225, 0.50),
                xytext=(x + 0.19, 0.50),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"),
            )

    ax.text(
        0.5,
        0.10,
        "Increasing spatial specificity: image-level prototypes to local normal references",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    path = out_dir / "fig-method-evolution.png"
    save_figure(fig, path)
    return path


def make_family_performance(out_dir: Path) -> Path:
    rows = []
    for key in ["global", "shared", "location", "patchcore"]:
        run = RUNS[key]
        metrics = load_metrics(run)
        rows.append(
            {
                "family": run.label,
                "color": run.color,
                "Image AUROC": metric(metrics, "image_auroc"),
                "Fused AUROC": metric(metrics, "fused_image_auroc"),
                "Pixel AUROC": metric(metrics, "pixel_auroc"),
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    metric_names = ["Image AUROC", "Fused AUROC", "Pixel AUROC"]
    y = np.arange(len(rows))
    for ax, metric_name in zip(axes, metric_names):
        for idx, row in enumerate(rows):
            value = row[metric_name]
            if value is None:
                ax.text(0.83, idx, "n/a", ha="center", va="center", color="#777777", fontsize=8)
                continue
            ax.scatter(value, idx, s=56, color=row["color"], edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(value + 0.006, idx, f"{value:.3f}", va="center", fontsize=7.5, color="#333333")
        ax.set_title(metric_name, fontsize=9, weight="bold")
        ax.set_xlim(0.68, 0.98)
        ax.set_xlabel("AUROC")
        ax.set_yticks(y)
        ax.invert_yaxis()

    axes[0].set_yticklabels([row["family"] for row in rows])
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout(w_pad=1.0)
    path = out_dir / "fig-family-performance.png"
    save_figure(fig, path)
    return path


def make_score_signals(out_dir: Path) -> Path:
    run = RUNS["location"]
    scores = load_scores(run)
    labels = scores["label"].to_numpy()
    panels = [
        ("anchor_score", "Anchor distance"),
        ("reconstruction_score", "Reconstruction error"),
        ("bottleneck_divergence", "Divergence"),
        ("fused_score", "Fused score"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25))
    for ax, (col, title) in zip(axes, panels):
        values = scores[col].to_numpy(dtype=np.float64)
        lo, hi = np.percentile(values, [0.5, 99.5])
        bins = np.linspace(lo, hi, 36)
        normal = values[labels == 0]
        anomaly = values[labels == 1]
        ax.hist(normal, bins=bins, density=True, color="#4C78A8", alpha=0.52, label="Normal")
        ax.hist(anomaly, bins=bins, density=True, color="#E45756", alpha=0.48, label="Anomaly")
        ax.set_title(title, fontsize=8.5, weight="bold")
        ax.set_xlabel("Score", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=0.9)
    path = out_dir / "fig-score-signals.png"
    save_figure(fig, path)
    return path


def make_anchor_space(out_dir: Path) -> Path:
    all_points = []
    labels = []
    colors = []
    markers = []

    for key in ["global", "shared", "location"]:
        run = RUNS[key]
        artifact = torch.load(run.path / "anchor_embeddings.pt", map_location="cpu", weights_only=False)
        anchors = artifact.get("anchor_global")
        if anchors is None:
            dense = artifact.get("anchor_dense")
            anchors = dense.mean(dim=(1, 2))
        anchors_np = anchors.detach().cpu().float().numpy()
        all_points.append(anchors_np)
        labels.extend([run.label] * len(anchors_np))
        colors.extend([run.color] * len(anchors_np))
        markers.extend(["D" if key == "global" else ("o" if key == "shared" else "^")] * len(anchors_np))

    points = np.vstack(all_points)
    if points.shape[0] < 3:
        coords = np.zeros((points.shape[0], 2), dtype=np.float32)
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(points)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for label in ["Global centroid", "Shared patch", "Location-aware"]:
        mask = np.array(labels) == label
        marker = np.array(markers)[mask][0]
        color = np.array(colors)[mask][0]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=62 if label == "Global centroid" else 28,
            marker=marker,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
    ax.axhline(0, color="#999999", lw=0.7, alpha=0.35)
    ax.axvline(0, color="#999999", lw=0.7, alpha=0.35)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig-anchor-space.png"
    save_figure(fig, path)
    return path


def select_representative_anomalies(run: RunSpec, n: int = 3) -> list[str]:
    scores = load_scores(run)
    anomalies = scores[scores["label"] == 1].copy()
    primary_col = "pixel_aggregated_score" if "pixel_aggregated_score" in anomalies.columns else "fused_score"
    anomalies = anomalies.sort_values(primary_col)

    candidates = []
    for quantile in [0.92, 0.62, 0.35]:
        target = anomalies[primary_col].quantile(quantile)
        row = anomalies.iloc[(anomalies[primary_col] - target).abs().argsort().iloc[0]]
        candidates.append(str(row["path"]))

    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    if len(unique) < n:
        for path in anomalies.sort_values(primary_col, ascending=False)["path"].astype(str):
            if path not in unique:
                unique.append(path)
            if len(unique) >= n:
                break
    return unique[:n]


def mask_path_for_image(image_path: str | Path) -> Path | None:
    path = Path(image_path)
    parts = list(path.parts)
    try:
        img_idx = parts.index("img")
        parts[img_idx] = "label"
        candidate = Path(*parts)
        return candidate if candidate.exists() else None
    except ValueError:
        return None


def load_image_tensor(path: str | Path, config: dict[str, Any]) -> tuple[torch.Tensor, np.ndarray, np.ndarray | None]:
    path = Path(path)
    target_size = tuple(config["data"].get("target_size", [240, 240]))
    norm_mode = config["data"].get("normalization", "zscore_only")
    preprocessor = BMADPreprocessor(target_size=target_size, normalize_mode=norm_mode)

    raw = np.load(str(path)) if str(path).endswith(".npy") else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = preprocessor.preprocess(raw)
    display = image.copy()
    if norm_mode == "minmax_imagenet":
        tensor_np = np.stack([image, image, image], axis=0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        tensor_np = (tensor_np - mean) / std
    else:
        tensor_np = np.stack([image, image, image], axis=0)

    mask = None
    mpath = mask_path_for_image(path)
    if mpath is not None:
        mask_raw = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask_raw is not None:
            if mask_raw.shape != target_size:
                mask_raw = cv2.resize(mask_raw, target_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask_raw > 0).astype(np.float32)

    return torch.from_numpy(tensor_np).float(), display, mask


def configure_stage2(model: torch.nn.Module, config: dict[str, Any]) -> None:
    stage2_cfg = config.get("stage2", {})
    if not stage2_cfg.get("enabled", False):
        return
    pixel_map_cfg = stage2_cfg.get("pixel_map", {})
    if not getattr(model, "reconstruction_enabled", False):
        model.enable_reconstruction_branch(
            freeze_anchor_target=stage2_cfg.get("freeze_anchor_target", True),
            out_channels=3,
            pixel_map_enabled=pixel_map_cfg.get("enabled", True),
            pixel_map_type=pixel_map_cfg.get("type", "reconstruction_l2"),
            use_frozen_bottleneck=stage2_cfg.get("frozen_bottleneck", False),
            recon_projection_dim=config.get("model", {}).get("projection_dim_recon", None),
            no_fuser=stage2_cfg.get("no_fuser", False),
        )
    pix_agg_cfg = stage2_cfg.get("pixel_aggregation", {})
    agg_method = pix_agg_cfg.get("method", "top_k_percentile")
    agg_threshold = pix_agg_cfg.get("threshold_n_std", 2.0) if agg_method == "threshold_ratio" else None
    model.configure_pixel_aggregation(
        method=agg_method,
        percentile=pix_agg_cfg.get("percentile", 95),
        threshold=agg_threshold,
    )
    fusion_cfg = stage2_cfg.get("score_fusion", {})
    model.configure_score_fusion(
        enabled=fusion_cfg.get("enabled", False),
        normalization=fusion_cfg.get("normalization", "minmax"),
        anchor_weight=fusion_cfg.get("anchor_weight", 0.4),
        divergence_weight=fusion_cfg.get("divergence_weight", 0.3),
        pixel_weight=fusion_cfg.get("pixel_weight", 0.3),
    )
    score_cfg = stage2_cfg.get("score_combination", {})
    model.configure_score_combination(
        enabled=score_cfg.get("enabled", False),
        alpha=score_cfg.get("alpha", 0.5),
        normalization=score_cfg.get("normalization", "minmax"),
    )


def load_model_for_run(run: RunSpec, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    with (run.path / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    anchors = torch.load(run.path / "anchor_embeddings.pt", map_location="cpu", weights_only=False)
    model = create_model(config, anchors.get("anchor_global"), anchors.get("anchor_dense"))
    configure_stage2(model, config)
    model = model.to(device)
    checkpoint = run.path / "best_stage2_model.pth"
    if not checkpoint.exists():
        checkpoint = run.path / "best_model.pth"
    _load_model_checkpoint(model, checkpoint, device, strict=True)
    model.eval()
    return model, config


def infer_maps(run: RunSpec, paths: list[str], device: torch.device) -> dict[str, Any]:
    model, config = load_model_for_run(run, device)
    tensors, displays, masks = [], [], []
    for path in paths:
        tensor, display, mask = load_image_tensor(path, config)
        tensors.append(tensor)
        displays.append(display)
        masks.append(mask)

    batch = torch.stack(tensors, dim=0).to(device)
    target_size = tuple(config["data"].get("target_size", [240, 240]))
    with torch.no_grad():
        outputs = model.compute_anomaly_scores(batch, return_maps=True, target_size=target_size)

    recon = outputs.get("reconstruction")
    if recon is not None:
        recon_np = recon.detach().cpu().numpy()
    else:
        recon_np = None
    pixel = outputs.get("pixel_scores")
    pixel_np = pixel.detach().cpu().numpy() if pixel is not None else None

    return {
        "display": displays,
        "mask": masks,
        "input_tensor": batch.detach().cpu().numpy(),
        "reconstruction": recon_np,
        "pixel": pixel_np,
    }


def fallback_crop_grid(source: Path, dest: Path, rows: int = 3) -> Path:
    img = plt.imread(source)
    h = img.shape[0]
    crop = img[: int(h * rows / 8.0), :, :]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.imshow(crop)
    ax.axis("off")
    save_figure(fig, dest)
    return dest


def make_reconstruction_localization(out_dir: Path, selected_paths: list[str], device: torch.device) -> tuple[Path, str]:
    path = out_dir / "fig-reconstruction-localization.png"
    run = RUNS["location"]
    try:
        data = infer_maps(run, selected_paths, device)
        recon = data["reconstruction"]
        if recon is None:
            raise RuntimeError("Reconstruction output was not available.")

        inputs = data["input_tensor"]
        diffs = np.abs(recon - inputs).mean(axis=1)
        vmax = float(np.percentile(np.concatenate([d.ravel() for d in diffs]), 99.0))
        vmax = max(vmax, 1e-6)

        fig, axes = plt.subplots(len(selected_paths), 4, figsize=(6.8, 5.3))
        headers = ["MRI", "Reconstruction", "Error", "Mask + error"]
        for col, header in enumerate(headers):
            axes[0, col].set_title(header, fontsize=8.5, weight="bold")

        mappable = None
        for row in range(len(selected_paths)):
            img = data["display"][row]
            rec = recon[row, 0]
            diff = diffs[row]
            mask = data["mask"][row]

            axes[row, 0].imshow(normalize01(img), cmap="gray")
            axes[row, 1].imshow(normalize01(rec), cmap="gray")
            mappable = axes[row, 2].imshow(diff, cmap="magma", vmin=0, vmax=vmax)

            axes[row, 3].imshow(normalize01(img), cmap="gray")
            axes[row, 3].imshow(diff, cmap="magma", alpha=0.58, vmin=0, vmax=vmax)
            if mask is not None:
                axes[row, 3].contour(mask, levels=[0.5], colors=["#00D5FF"], linewidths=0.9)

            for col in range(4):
                axes[row, col].axis("off")
                axes[row, col].text(
                    0.02,
                    0.06,
                    f"({chr(ord('a') + row * 4 + col)})",
                    transform=axes[row, col].transAxes,
                    color="white",
                    fontsize=7,
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5),
                )

        fig.subplots_adjust(left=0.02, right=0.92, top=0.94, bottom=0.03, wspace=0.015, hspace=0.015)
        if mappable is not None:
            cax = fig.add_axes([0.935, 0.18, 0.012, 0.64])
            cbar = fig.colorbar(mappable, cax=cax)
            cbar.ax.tick_params(labelsize=7)
        save_figure(fig, path)
        return path, "checkpoint_inference"
    except Exception as exc:
        print(f"[warn] Reconstruction inference failed, using compact diagnostic crop: {exc}")
        return fallback_crop_grid(run.path / "evaluation" / "reconstruction_anomalies.png", path, rows=3), "fallback_crop"


def make_qualitative_comparison(out_dir: Path, selected_paths: list[str], device: torch.device) -> tuple[Path, str]:
    path = out_dir / "fig-qualitative-comparison.png"
    try:
        global_maps = infer_maps(RUNS["global"], selected_paths, device)
        shared_maps = infer_maps(RUNS["shared"], selected_paths, device)
        location_maps = infer_maps(RUNS["location"], selected_paths, device)

        map_sets = [
            ("Global", global_maps["pixel"]),
            ("Shared patch", shared_maps["pixel"]),
            ("Location-aware", location_maps["pixel"]),
        ]
        if any(maps is None for _, maps in map_sets):
            raise RuntimeError("One or more models did not return pixel maps.")

        all_maps = np.concatenate([maps.reshape(-1) for _, maps in map_sets if maps is not None])
        vmax = float(np.percentile(all_maps, 99.0))
        vmax = max(vmax, 1e-6)
        base = location_maps

        fig, axes = plt.subplots(len(selected_paths), 4, figsize=(6.9, 5.2))
        headers = ["MRI + mask", "Global", "Shared patch", "Location-aware"]
        for col, header in enumerate(headers):
            axes[0, col].set_title(header, fontsize=8.5, weight="bold")

        for row in range(len(selected_paths)):
            img = base["display"][row]
            mask = base["mask"][row]
            axes[row, 0].imshow(normalize01(img), cmap="gray")
            if mask is not None:
                axes[row, 0].contour(mask, levels=[0.5], colors=["#00D5FF"], linewidths=1.0)
            for col, (_, maps) in enumerate(map_sets, start=1):
                axes[row, col].imshow(normalize01(img), cmap="gray")
                axes[row, col].imshow(maps[row], cmap="magma", alpha=0.60, vmin=0, vmax=vmax)
                if mask is not None:
                    axes[row, col].contour(mask, levels=[0.5], colors=["#00D5FF"], linewidths=0.8)
            for col in range(4):
                axes[row, col].axis("off")
        fig.tight_layout(w_pad=0.15, h_pad=0.25)
        save_figure(fig, path)
        return path, "checkpoint_inference"
    except Exception as exc:
        print(f"[warn] Qualitative comparison inference failed, using compact diagnostic crop: {exc}")
        return fallback_crop_grid(RUNS["location"].path / "evaluation" / "pixel_anomaly_overlay.png", path, rows=3), "fallback_crop"


def copy_to_thesis(paths: list[Path], thesis_dir: Path | None) -> list[str]:
    copied = []
    if thesis_dir is None:
        return copied
    thesis_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = thesis_dir / path.name
        shutil.copy2(path, target)
        copied.append(str(target))
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polished figures for the thesis.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "thesis_figures")
    parser.add_argument("--copy-to-thesis", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_style()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    generated: list[Path] = []
    generated.append(make_method_evolution(out_dir))
    generated.append(make_family_performance(out_dir))
    generated.append(make_score_signals(out_dir))
    generated.append(make_anchor_space(out_dir))

    selected_paths = select_representative_anomalies(RUNS["location"], n=3)
    recon_path, recon_mode = make_reconstruction_localization(out_dir, selected_paths, device)
    qual_path, qual_mode = make_qualitative_comparison(out_dir, selected_paths, device)
    generated.extend([recon_path, qual_path])

    copied = copy_to_thesis(generated, args.copy_to_thesis)

    manifest = {
        "source_runs": {key: str(spec.path) for key, spec in RUNS.items()},
        "selected_images": selected_paths,
        "outputs": [str(path) for path in generated],
        "copied_to_thesis": copied,
        "qualitative_modes": {
            "reconstruction": recon_mode,
            "qualitative_comparison": qual_mode,
        },
        "metrics": {
            key: load_metrics(spec)
            for key, spec in RUNS.items()
            if (spec.path / "evaluation" / "evaluation_metrics.json").exists()
        },
    }
    with (out_dir / "figure_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Generated thesis figures:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
