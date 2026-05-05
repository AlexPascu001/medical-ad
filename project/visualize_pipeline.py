"""
Unified Pipeline Visualization for BMAD Anomaly Detection.

Generates comprehensive per-step diagnostic visualizations covering the entire
pipeline from raw input to final fused score. Replaces the functionality of
visualize_embeddings.py, visualize_from_checkpoint.py, and plot_from_checkpoint.py.

Usage (standalone):
    python visualize_pipeline.py --experiment <experiment_dir>

Usage (programmatic):
    from visualize_pipeline import generate_full_pipeline_visualization
    generate_full_pipeline_visualization(model, dataloader, device, save_dir)
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize as mplNormalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_samples(dataloader, n_normal: int = 5, n_anomaly: int = 5, seed: int = 42):
    """Deterministically select samples for consistent visualisation across steps.

    Returns a list of dicts with keys: image, label, mask, index.
    """
    rng = np.random.RandomState(seed)

    normal_indices: List[int] = []
    anomaly_indices: List[int] = []

    # Gather all indices by class
    offset = 0
    for batch in dataloader:
        labels = batch['label'].numpy()
        for i, lab in enumerate(labels):
            if lab == 0:
                normal_indices.append(offset + i)
            else:
                anomaly_indices.append(offset + i)
        offset += len(labels)

    # Random subset
    rng.shuffle(normal_indices)
    rng.shuffle(anomaly_indices)
    chosen_normal = sorted(normal_indices[:n_normal])
    chosen_anomaly = sorted(anomaly_indices[:n_anomaly])
    chosen = chosen_normal + chosen_anomaly

    # Second pass: collect actual data
    samples = {idx: None for idx in chosen}
    offset = 0
    for batch in dataloader:
        bs = batch['image'].shape[0]
        for i in range(bs):
            global_idx = offset + i
            if global_idx in samples:
                samples[global_idx] = {
                    'image': batch['image'][i],          # (C, H, W) tensor
                    'label': int(batch['label'][i]),
                    'mask': batch['mask'][i].numpy() if 'mask' in batch else None,
                    'index': global_idx,
                }
        offset += bs

    ordered = [samples[idx] for idx in chosen if samples[idx] is not None]
    return ordered


def _to_displayable(img_tensor):
    """Convert a (C,H,W) or (H,W) tensor/ndarray to (H,W) grayscale numpy for display."""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = np.array(img_tensor)
    if img.ndim == 3:
        # Take first channel (all 3 are identical for grayscale replicated)
        img = img[0]
    return img


def _norm01(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _make_grid_figure(samples, n_cols, row_height=2.5, col_width=2.5):
    n = len(samples)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, row_height * n_rows))
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    return fig, axes


# ---------------------------------------------------------------------------
# Step 1: Raw input images
# ---------------------------------------------------------------------------

def viz_raw_input(samples: list, save_dir: Path):
    """Show raw input images with intensity histograms."""
    n = len(samples)
    fig = plt.figure(figsize=(3.5 * n, 7))
    gs = gridspec.GridSpec(2, n, hspace=0.35, wspace=0.3)

    for i, s in enumerate(samples):
        img = _to_displayable(s['image'])
        label_str = 'Normal' if s['label'] == 0 else 'Anomaly'
        color = 'green' if s['label'] == 0 else 'red'

        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f'{label_str} #{s["index"]}', fontsize=9, color=color)
        ax_img.axis('off')

        ax_hist = fig.add_subplot(gs[1, i])
        ax_hist.hist(img.ravel(), bins=60, color=color, alpha=0.7, density=True)
        ax_hist.set_xlabel('Intensity', fontsize=7)
        ax_hist.tick_params(labelsize=6)

    fig.suptitle('Step 1: Input Images (after preprocessing)', fontsize=13, y=1.01)
    fig.savefig(save_dir / 'step01_raw_input.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 2: Preprocessed images (same as raw for us since dataloader returns preprocessed)
# ---------------------------------------------------------------------------

def viz_preprocessed(samples: list, save_dir: Path):
    """Show preprocessed images with normalization statistics."""
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for i, s in enumerate(samples):
        img = _to_displayable(s['image'])
        label_str = 'Normal' if s['label'] == 0 else 'Anomaly'
        color = 'green' if s['label'] == 0 else 'red'

        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'{label_str}\nμ={img.mean():.2f} σ={img.std():.2f}', fontsize=8, color=color)
        axes[0, i].axis('off')

        # Show mask if available
        if s['mask'] is not None:
            mask = s['mask'] if s['mask'].ndim == 2 else s['mask'][0]
            axes[1, i].imshow(mask, cmap='Reds', vmin=0, vmax=1)
            axes[1, i].set_title(f'GT mask ({mask.sum():.0f} px)', fontsize=8)
        else:
            axes[1, i].imshow(np.zeros_like(img), cmap='gray')
            axes[1, i].set_title('No mask', fontsize=8)
        axes[1, i].axis('off')

    fig.suptitle('Step 2: Preprocessed Images & Ground-Truth Masks', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / 'step02_preprocessed.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3: Backbone features (PCA of patch tokens)
# ---------------------------------------------------------------------------

def viz_backbone_features(samples: list, features_list: list, save_dir: Path):
    """PCA of patch tokens → RGB visualization + CLS norm bar chart.

    features_list: list of dicts with 'patch_raw' (N_patches, D) and 'global_raw' (D,).
    """
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(-1, 1)

    # Gather all patch tokens for a single shared PCA
    all_patches = []
    patch_counts = []
    for f in features_list:
        p = f['patch_raw']  # (N_patches, D) numpy
        all_patches.append(p)
        patch_counts.append(p.shape[0])
    all_patches_np = np.concatenate(all_patches, axis=0)

    pca = PCA(n_components=3)
    projected = pca.fit_transform(all_patches_np)
    # Normalise each component independently for display
    for c in range(3):
        lo, hi = projected[:, c].min(), projected[:, c].max()
        if hi - lo > 1e-12:
            projected[:, c] = (projected[:, c] - lo) / (hi - lo)

    offset = 0
    for i, s in enumerate(samples):
        nc = patch_counts[i]
        patch_pca = projected[offset:offset + nc]
        offset += nc
        h = w = int(np.sqrt(nc))
        rgb = patch_pca.reshape(h, w, 3)

        label_str = 'Normal' if s['label'] == 0 else 'Anomaly'
        color = 'green' if s['label'] == 0 else 'red'

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f'{label_str} #{s["index"]}', fontsize=8, color=color)
        axes[0, i].axis('off')

        # CLS token norm
        cls_norm = float(np.linalg.norm(features_list[i]['global_raw']))
        axes[1, i].barh(['CLS norm'], [cls_norm], color=color, height=0.4)
        axes[1, i].set_xlim(0, max(cls_norm * 1.3, 1))
        axes[1, i].set_title(f'||CLS|| = {cls_norm:.2f}', fontsize=8)
        axes[1, i].tick_params(labelsize=7)

    fig.suptitle('Step 3: Backbone Features (PCA of patch tokens → RGB)', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / 'step03_backbone_features.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 4: Projected embeddings + anchor positions (t-SNE / PCA)
# ---------------------------------------------------------------------------

def viz_projected_embeddings(
    all_global_feats: np.ndarray,
    all_labels: np.ndarray,
    anchor_embeddings: np.ndarray,
    assigned_anchors: np.ndarray,
    save_dir: Path,
    tracked_indices: Optional[List[int]] = None,
):
    """t-SNE and PCA of projected embeddings with anchors."""
    K = anchor_embeddings.shape[0]
    N = all_global_feats.shape[0]

    combined = np.vstack([all_global_feats, anchor_embeddings])  # (N+K, D)

    # t-SNE
    perp = min(30, combined.shape[0] - 1)
    import sklearn
    _tsne_kwargs = {'n_components': 2, 'perplexity': perp, 'random_state': 42}
    # 'n_iter' was renamed to 'max_iter' in scikit-learn 1.5
    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 5):
        _tsne_kwargs['max_iter'] = 1000
    else:
        _tsne_kwargs['n_iter'] = 1000
    tsne = TSNE(**_tsne_kwargs)
    coords_2d = tsne.fit_transform(combined)

    # PCA
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(combined)

    for method_name, emb in [('t-SNE', coords_2d), ('PCA', pca_2d)]:
        data_coords = emb[:N]
        anchor_coords = emb[N:]

        fig, ax = plt.subplots(figsize=(8, 7))

        # Scatter: normal vs anomaly
        normal_mask = all_labels == 0
        anomaly_mask = all_labels == 1
        ax.scatter(data_coords[normal_mask, 0], data_coords[normal_mask, 1],
                   c='steelblue', alpha=0.3, s=8, label='Normal')
        ax.scatter(data_coords[anomaly_mask, 0], data_coords[anomaly_mask, 1],
                   c='salmon', alpha=0.3, s=8, label='Anomaly')

        # Anchors
        cmap_anchors = plt.cm.Set1(np.linspace(0, 1, max(K, 2)))
        for k in range(K):
            ax.scatter(anchor_coords[k, 0], anchor_coords[k, 1],
                       marker='*', s=300, edgecolors='black', linewidths=1.0,
                       c=[cmap_anchors[k]], zorder=5, label=f'Anchor {k}')

        # Highlight tracked samples
        if tracked_indices is not None:
            for idx in tracked_indices:
                if idx < N:
                    ax.scatter(data_coords[idx, 0], data_coords[idx, 1],
                               marker='D', s=60, edgecolors='black', linewidths=1.2,
                               c='gold', zorder=4)

        ax.set_title(f'Step 4: Projected Embeddings ({method_name})', fontsize=13)
        ax.legend(fontsize=7, loc='best', markerscale=0.8, ncol=2)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        tag = method_name.lower().replace('-', '')
        fig.savefig(save_dir / f'step04_embeddings_{tag}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        counts = np.bincount(assigned_anchors, minlength=K)
        used_anchors = int((counts > 0).sum())
        nonzero_counts = counts[counts > 0]
        probs = nonzero_counts / max(nonzero_counts.sum(), 1) if len(nonzero_counts) > 0 else np.array([])
        entropy = float(-(probs * np.log(probs)).sum()) if len(probs) > 0 else 0.0
        max_entropy = np.log(K) if K > 1 else 1.0
        entropy_normalized = float(entropy / max(max_entropy, 1e-12))

        fig, ax = plt.subplots(figsize=(8, 7))
        if K <= 32:
            cluster_colors = plt.cm.get_cmap('tab20', max(K, 2))(np.arange(max(K, 2)))
            for k in range(K):
                color = cluster_colors[k % len(cluster_colors)]
                normal_cluster_mask = (assigned_anchors == k) & normal_mask
                anomaly_cluster_mask = (assigned_anchors == k) & anomaly_mask
                if normal_cluster_mask.any():
                    ax.scatter(
                        data_coords[normal_cluster_mask, 0],
                        data_coords[normal_cluster_mask, 1],
                        c=[color],
                        alpha=0.45,
                        s=12,
                        marker='o',
                    )
                if anomaly_cluster_mask.any():
                    ax.scatter(
                        data_coords[anomaly_cluster_mask, 0],
                        data_coords[anomaly_cluster_mask, 1],
                        c=[color],
                        alpha=0.8,
                        s=28,
                        marker='x',
                    )
                ax.scatter(
                    anchor_coords[k, 0],
                    anchor_coords[k, 1],
                    marker='*',
                    s=260,
                    edgecolors='black',
                    linewidths=1.0,
                    c=[color],
                    zorder=5,
                )
                if counts[k] > 0:
                    ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]), fontsize=8, ha='center', va='center', zorder=6)
        else:
            top_anchor_k = min(24, K)
            top_anchor_ids = np.argsort(counts)[::-1][:top_anchor_k]
            top_colors = plt.cm.get_cmap('tab20', max(top_anchor_k, 2))(np.arange(max(top_anchor_k, 2)))
            top_mask = np.isin(assigned_anchors, top_anchor_ids)
            if (~top_mask).any():
                ax.scatter(
                    data_coords[~top_mask, 0],
                    data_coords[~top_mask, 1],
                    c='lightgray',
                    alpha=0.2,
                    s=8,
                    marker='o',
                    label='Other assigned anchors',
                )
            for rank, k in enumerate(top_anchor_ids):
                color = top_colors[rank % len(top_colors)]
                normal_cluster_mask = (assigned_anchors == k) & normal_mask
                anomaly_cluster_mask = (assigned_anchors == k) & anomaly_mask
                if normal_cluster_mask.any():
                    ax.scatter(
                        data_coords[normal_cluster_mask, 0],
                        data_coords[normal_cluster_mask, 1],
                        c=[color],
                        alpha=0.5,
                        s=14,
                        marker='o',
                        label=f'A{k} normal ({counts[k]})',
                    )
                if anomaly_cluster_mask.any():
                    ax.scatter(
                        data_coords[anomaly_cluster_mask, 0],
                        data_coords[anomaly_cluster_mask, 1],
                        c=[color],
                        alpha=0.9,
                        s=32,
                        marker='x',
                    )
                ax.scatter(
                    anchor_coords[k, 0],
                    anchor_coords[k, 1],
                    marker='*',
                    s=280,
                    edgecolors='black',
                    linewidths=1.0,
                    c=[color],
                    zorder=5,
                )
                ax.annotate(f'A{k}', (anchor_coords[k, 0], anchor_coords[k, 1]), fontsize=8, ha='center', va='center', zorder=6)

        if tracked_indices is not None:
            for idx in tracked_indices:
                if idx < N:
                    ax.scatter(
                        data_coords[idx, 0],
                        data_coords[idx, 1],
                        marker='D',
                        s=58,
                        edgecolors='black',
                        linewidths=1.0,
                        c='gold',
                        zorder=6,
                    )

        ax.set_title(
            f'Step 4B: Embeddings by Assigned Anchor ({method_name})\n'
            f'used={used_anchors}/{K}, entropy={entropy_normalized:.3f}, anomalies=x',
            fontsize=12,
        )
        ax.grid(alpha=0.2)
        if K > 32:
            ax.legend(fontsize=7, loc='best', ncol=2)
        fig.tight_layout()
        fig.savefig(save_dir / f'step04_clusters_{tag}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Step 5: Anchor distances
# ---------------------------------------------------------------------------

def viz_anchor_distances(
    all_distances: np.ndarray,   # (N, K)
    all_labels: np.ndarray,
    anchor_embeddings: np.ndarray,
    save_dir: Path,
):
    """Distance histograms, boxplots, assignment counts, pairwise heatmap."""
    K = all_distances.shape[1]
    min_dists = all_distances.min(axis=1)
    assigned = all_distances.argmin(axis=1)

    normal_mask = all_labels == 0
    anomaly_mask = all_labels == 1

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # (a) Histogram of min-distance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(min_dists[normal_mask], bins=50, alpha=0.6, color='steelblue', density=True, label='Normal')
    ax1.hist(min_dists[anomaly_mask], bins=50, alpha=0.6, color='salmon', density=True, label='Anomaly')
    ax1.set_xlabel('Min distance to nearest anchor')
    ax1.set_ylabel('Density')
    ax1.set_title('Distance to Nearest Anchor')
    ax1.legend(fontsize=8)

    # (b) Boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    data_for_box = [min_dists[normal_mask], min_dists[anomaly_mask]]
    bp = ax2.boxplot(data_for_box, labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('salmon')
    ax2.set_ylabel('Min distance')
    ax2.set_title('Distance Distribution')

    # (c) Per-anchor assignment count
    ax3 = fig.add_subplot(gs[0, 2])
    counts = np.bincount(assigned, minlength=K)
    ax3.bar(range(K), counts, color='teal')
    ax3.set_xlabel('Anchor index')
    ax3.set_ylabel('Count')
    ax3.set_title('Samples per Anchor')

    # (d) Per-anchor distance distributions
    ax4 = fig.add_subplot(gs[1, 0])
    per_anchor_normal = []
    per_anchor_anomaly = []
    for k in range(K):
        per_anchor_normal.append(all_distances[normal_mask, k])
        per_anchor_anomaly.append(all_distances[anomaly_mask, k])
    positions = np.arange(K)
    bp_n = ax4.boxplot(per_anchor_normal, positions=positions - 0.15, widths=0.25,
                       patch_artist=True, manage_ticks=False)
    bp_a = ax4.boxplot(per_anchor_anomaly, positions=positions + 0.15, widths=0.25,
                       patch_artist=True, manage_ticks=False)
    for b in bp_n['boxes']:
        b.set_facecolor('steelblue')
        b.set_alpha(0.7)
    for b in bp_a['boxes']:
        b.set_facecolor('salmon')
        b.set_alpha(0.7)
    ax4.set_xticks(positions)
    ax4.set_xticklabels([f'A{k}' for k in range(K)], fontsize=7)
    ax4.set_title('Per-Anchor Distances (blue=normal, red=anomaly)')
    ax4.set_ylabel('Distance')

    # (e) Pairwise anchor separation heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    if K > 1:
        from scipy.spatial.distance import pdist, squareform
        pw = squareform(pdist(anchor_embeddings, metric='euclidean'))
        im = ax5.imshow(pw, cmap='YlOrRd')
        fig.colorbar(im, ax=ax5, fraction=0.046)
        ax5.set_xticks(range(K))
        ax5.set_yticks(range(K))
        ax5.set_title('Pairwise Anchor Distance')
    else:
        ax5.text(0.5, 0.5, 'K=1\n(single anchor)', ha='center', va='center', fontsize=12)
        ax5.set_title('Pairwise Anchor Distance')

    # (f) Normal vs anomaly AUROC from anchor score alone
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        fpr, tpr, _ = roc_curve(all_labels, min_dists)
        auroc = roc_auc_score(all_labels, min_dists)
        ax6.plot(fpr, tpr, linewidth=2, label=f'AUROC={auroc:.4f}')
        ax6.plot([0, 1], [0, 1], 'k--', lw=1)
        ax6.set_xlabel('FPR')
        ax6.set_ylabel('TPR')
        ax6.set_title('Anchor Score ROC')
        ax6.legend(fontsize=9)
    except Exception:
        ax6.text(0.5, 0.5, 'ROC N/A', ha='center', va='center')

    fig.suptitle('Step 5: Anchor Distance Analysis', fontsize=14, y=1.01)
    fig.savefig(save_dir / 'step05_anchor_distances.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 6: Reconstruction (stage-2)
# ---------------------------------------------------------------------------

def viz_reconstruction(samples: list, outputs_list: list, save_dir: Path):
    """Original | Reconstruction | |Error| for tracked samples."""
    n = len(samples)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for i, (s, out) in enumerate(zip(samples, outputs_list)):
        img = _to_displayable(s['image'])
        label_str = 'Normal' if s['label'] == 0 else 'Anomaly'
        color = 'green' if s['label'] == 0 else 'red'

        recon = out.get('reconstruction')
        if recon is None:
            for row in range(3):
                axes[row, i].text(0.5, 0.5, 'No recon', ha='center', va='center')
                axes[row, i].axis('off')
            continue

        recon_img = _to_displayable(recon)
        error = np.abs(img - recon_img)
        mse = float(np.mean((img - recon_img) ** 2))

        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'{label_str} #{s["index"]}', fontsize=8, color=color)
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title(f'Reconstruction', fontsize=8)
        axes[1, i].axis('off')

        axes[2, i].imshow(error, cmap='hot')
        axes[2, i].set_title(f'|Error| MSE={mse:.4f}', fontsize=8)
        axes[2, i].axis('off')

    fig.suptitle('Step 6: Reconstruction (Original → Reconstructed → |Error|)', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / 'step06_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 7: Pixel anomaly maps
# ---------------------------------------------------------------------------

def viz_pixel_anomaly_maps(samples: list, outputs_list: list, save_dir: Path):
    """Original | GT Mask | Recon pixel map | Anchor pixel map."""
    n = len(samples)
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n, figsize=(3 * n, 3 * n_rows))
    if n == 1:
        axes = axes.reshape(-1, 1)

    # Find global vmax for consistent colormap across samples
    recon_maps = []
    anchor_maps = []
    for out in outputs_list:
        r = out.get('reconstruction_pixel_scores')
        a = out.get('anchor_pixel_scores')
        if r is not None:
            recon_maps.append(r)
        if a is not None:
            anchor_maps.append(a)
    recon_vmax = max(m.max() for m in recon_maps) if recon_maps else 1.0
    anchor_vmax = max(m.max() for m in anchor_maps) if anchor_maps else 1.0

    for i, (s, out) in enumerate(zip(samples, outputs_list)):
        img = _to_displayable(s['image'])
        label_str = 'Normal' if s['label'] == 0 else 'Anomaly'
        color = 'green' if s['label'] == 0 else 'red'

        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'{label_str} #{s["index"]}', fontsize=8, color=color)
        axes[0, i].axis('off')

        # GT mask
        if s['mask'] is not None:
            mask = s['mask'] if s['mask'].ndim == 2 else s['mask'][0]
            axes[1, i].imshow(mask, cmap='Reds', vmin=0, vmax=1)
        else:
            axes[1, i].imshow(np.zeros_like(img), cmap='gray')
        axes[1, i].set_title('GT Mask', fontsize=8)
        axes[1, i].axis('off')

        # Reconstruction pixel map
        r = out.get('reconstruction_pixel_scores')
        if r is not None:
            axes[2, i].imshow(r, cmap='hot', vmin=0, vmax=recon_vmax)
        else:
            axes[2, i].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
        axes[2, i].set_title('Recon pixel map', fontsize=8)
        axes[2, i].axis('off')

        # Anchor pixel map
        a = out.get('anchor_pixel_scores')
        if a is not None:
            axes[3, i].imshow(a, cmap='hot', vmin=0, vmax=anchor_vmax)
        else:
            axes[3, i].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
        axes[3, i].set_title('Anchor pixel map', fontsize=8)
        axes[3, i].axis('off')

    fig.suptitle('Step 7: Pixel-Level Anomaly Maps', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / 'step07_pixel_anomaly_maps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 8: Divergence maps
# ---------------------------------------------------------------------------

def viz_divergence_maps(
    samples: list,
    outputs_list: list,
    all_divergence: Optional[np.ndarray],
    all_labels: np.ndarray,
    save_dir: Path,
):
    """CLS-level divergence histogram + spatial patch divergence maps."""
    n = len(samples)

    # Determine layout
    has_maps = any(out.get('patch_divergence_map') is not None for out in outputs_list)
    n_rows = 2 if has_maps else 1

    fig = plt.figure(figsize=(max(3 * n, 8), 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, max(n, 2), hspace=0.35, wspace=0.3)

    # Row 0: CLS-level divergence histogram (spans full width)
    ax_hist = fig.add_subplot(gs[0, :])
    if all_divergence is not None and len(all_divergence) > 0:
        normal_mask = all_labels == 0
        anomaly_mask = all_labels == 1
        ax_hist.hist(all_divergence[normal_mask], bins=50, alpha=0.6, color='steelblue', density=True, label='Normal')
        ax_hist.hist(all_divergence[anomaly_mask], bins=50, alpha=0.6, color='salmon', density=True, label='Anomaly')
        try:
            auroc = roc_auc_score(all_labels, all_divergence)
            ax_hist.set_title(f'Bottleneck Divergence Distribution (AUROC={auroc:.4f})', fontsize=11)
        except Exception:
            ax_hist.set_title('Bottleneck Divergence Distribution', fontsize=11)
        ax_hist.set_xlabel('Divergence (1 - cos_sim)')
        ax_hist.set_ylabel('Density')
        ax_hist.legend(fontsize=8)
    else:
        ax_hist.text(0.5, 0.5, 'Divergence not computed\n(frozen_bottleneck disabled)',
                     ha='center', va='center', fontsize=12, color='gray')
        ax_hist.set_title('Bottleneck Divergence — Not Available', fontsize=11)

    # Row 1: Spatial patch divergence maps for tracked samples
    if has_maps:
        divmaps = [out.get('patch_divergence_map') for out in outputs_list]
        vmax = max((m.max() for m in divmaps if m is not None), default=1.0)
        for i in range(n):
            ax = fig.add_subplot(gs[1, i])
            m = divmaps[i]
            label_str = 'Normal' if samples[i]['label'] == 0 else 'Anomaly'
            color = 'green' if samples[i]['label'] == 0 else 'red'
            if m is not None:
                ax.imshow(m, cmap='hot', vmin=0, vmax=vmax)
                ax.set_title(f'{label_str} #{samples[i]["index"]}', fontsize=8, color=color)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            ax.axis('off')

    fig.suptitle('Step 8: Bottleneck Divergence', fontsize=13, y=1.01)
    fig.savefig(save_dir / 'step08_divergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 9: Per-signal score distributions
# ---------------------------------------------------------------------------

def viz_score_distributions(
    scores_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    save_dir: Path,
):
    """Per-signal histograms, per-signal ROC, and cross-signal correlation."""
    signal_names = [k for k in scores_dict if scores_dict[k] is not None and len(scores_dict[k]) > 0]
    n_signals = len(signal_names)
    if n_signals == 0:
        return

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    fig = plt.figure(figsize=(5 * min(n_signals, 3), 10))
    gs = gridspec.GridSpec(3, max(n_signals, 2), hspace=0.4, wspace=0.35)

    # Row 0: Histograms
    for j, name in enumerate(signal_names):
        ax = fig.add_subplot(gs[0, j])
        vals = scores_dict[name]
        ax.hist(vals[normal_mask], bins=50, alpha=0.6, color='steelblue', density=True, label='Normal')
        ax.hist(vals[anomaly_mask], bins=50, alpha=0.6, color='salmon', density=True, label='Anomaly')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Score', fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    # Row 1: ROC curves
    ax_roc = fig.add_subplot(gs[1, :])
    for name in signal_names:
        vals = scores_dict[name]
        try:
            fpr, tpr, _ = roc_curve(labels, vals)
            auroc = roc_auc_score(labels, vals)
            ax_roc.plot(fpr, tpr, linewidth=1.5, label=f'{name} (AUROC={auroc:.4f})')
        except Exception:
            pass
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR')
    ax_roc.set_title('Per-Signal ROC Curves')
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.2)

    # Row 2: Correlation matrix
    ax_corr = fig.add_subplot(gs[2, :])
    if n_signals >= 2:
        arr = np.column_stack([scores_dict[n] for n in signal_names])
        corr = np.corrcoef(arr.T)
        im = ax_corr.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax_corr, fraction=0.046)
        ax_corr.set_xticks(range(n_signals))
        ax_corr.set_yticks(range(n_signals))
        ax_corr.set_xticklabels(signal_names, fontsize=7, rotation=30, ha='right')
        ax_corr.set_yticklabels(signal_names, fontsize=7)
        for r in range(n_signals):
            for c in range(n_signals):
                ax_corr.text(c, r, f'{corr[r, c]:.2f}', ha='center', va='center', fontsize=8)
        ax_corr.set_title('Cross-Signal Correlation')
    else:
        ax_corr.text(0.5, 0.5, 'Single signal — no correlation', ha='center', va='center')

    fig.suptitle('Step 9: Score Distributions & Signal Analysis', fontsize=13, y=1.01)
    fig.savefig(save_dir / 'step09_score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 10: Fused score + error analysis
# ---------------------------------------------------------------------------

def viz_fused_scores(
    scores_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    samples: list,
    outputs_list: list,
    save_dir: Path,
):
    """Final fused/best score: ROC, PR, histogram, top FP/FN."""
    # Pick fused if available, else anchor
    if 'fused' in scores_dict and scores_dict['fused'] is not None:
        final_scores = scores_dict['fused']
        final_name = 'Fused Score'
    elif 'anchor' in scores_dict and scores_dict['anchor'] is not None:
        final_scores = scores_dict['anchor']
        final_name = 'Anchor Score'
    else:
        # Fallback: first available
        for k in scores_dict:
            if scores_dict[k] is not None:
                final_scores = scores_dict[k]
                final_name = k
                break
        else:
            return

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # (a) ROC
    ax1 = fig.add_subplot(gs[0, 0])
    try:
        fpr, tpr, thresholds = roc_curve(labels, final_scores)
        auroc = roc_auc_score(labels, final_scores)
        ax1.plot(fpr, tpr, linewidth=2, label=f'AUROC={auroc:.4f}')
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title(f'{final_name} — ROC Curve')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.2)
    except Exception:
        ax1.text(0.5, 0.5, 'ROC N/A', ha='center', va='center')

    # (b) Precision-Recall
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        prec, rec, _ = precision_recall_curve(labels, final_scores)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(labels, final_scores)
        ax2.plot(rec, prec, linewidth=2, label=f'AP={ap:.4f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{final_name} — PR Curve')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.2)
    except Exception:
        ax2.text(0.5, 0.5, 'PR N/A', ha='center', va='center')

    # (c) Histogram with threshold
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(final_scores[normal_mask], bins=50, alpha=0.6, color='steelblue', density=True, label='Normal')
    ax3.hist(final_scores[anomaly_mask], bins=50, alpha=0.6, color='salmon', density=True, label='Anomaly')
    # Optimal threshold (Youden's J)
    try:
        fpr_t, tpr_t, thr_t = roc_curve(labels, final_scores)
        j_scores = tpr_t - fpr_t
        best_idx = np.argmax(j_scores)
        best_thr = thr_t[best_idx]
        ax3.axvline(best_thr, color='black', linestyle='--', linewidth=1.5, label=f'Threshold={best_thr:.4f}')
    except Exception:
        pass
    ax3.set_title(f'{final_name} — Score Distribution')
    ax3.set_xlabel('Score')
    ax3.legend(fontsize=7)

    # (d) Confusion matrix at optimal threshold
    ax4 = fig.add_subplot(gs[1, 0])
    try:
        preds = (final_scores >= best_thr).astype(int)
        cm = confusion_matrix(labels, preds)
        im = ax4.imshow(cm, cmap='Blues')
        fig.colorbar(im, ax=ax4, fraction=0.046)
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Normal', 'Anomaly'])
        ax4.set_yticklabels(['Normal', 'Anomaly'])
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        for r in range(2):
            for c in range(2):
                ax4.text(c, r, str(cm[r, c]), ha='center', va='center',
                         fontsize=14, color='white' if cm[r, c] > cm.max() / 2 else 'black')
        ax4.set_title(f'Confusion Matrix (thr={best_thr:.4f})')
    except Exception:
        ax4.text(0.5, 0.5, 'CM N/A', ha='center', va='center')

    # (e) Top-5 False Positives
    ax5 = fig.add_subplot(gs[1, 1])
    normal_scores = final_scores[normal_mask]
    normal_indices_sorted = np.where(normal_mask)[0][np.argsort(normal_scores)[::-1]]
    fp_text = "Top-5 False Positives (normals with highest score):\n"
    for rank, idx in enumerate(normal_indices_sorted[:5]):
        fp_text += f"  {rank+1}. idx={idx}, score={final_scores[idx]:.4f}\n"
    ax5.text(0.05, 0.95, fp_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    ax5.set_title('Top False Positives')
    ax5.axis('off')

    # (f) Top-5 False Negatives
    ax6 = fig.add_subplot(gs[1, 2])
    anomaly_scores = final_scores[anomaly_mask]
    anomaly_indices_sorted = np.where(anomaly_mask)[0][np.argsort(anomaly_scores)]
    fn_text = "Top-5 False Negatives (anomalies with lowest score):\n"
    for rank, idx in enumerate(anomaly_indices_sorted[:5]):
        fn_text += f"  {rank+1}. idx={idx}, score={final_scores[idx]:.4f}\n"
    ax6.text(0.05, 0.95, fn_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Top False Negatives')
    ax6.axis('off')

    fig.suptitle('Step 10: Final Score Analysis', fontsize=14, y=1.01)
    fig.savefig(save_dir / 'step10_fused_scores.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bonus: Training curves (stage-1 + stage-2)
# ---------------------------------------------------------------------------

def viz_training_curves(experiment_dir: Path, save_dir: Path):
    """Plot stage-1 and stage-2 training curves from training_history.json."""
    history_path = experiment_dir / 'training_history.json'
    if not history_path.exists():
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Detect stage-2 keys
    stage2_keys = [k for k in history if k.startswith('stage2_')]
    has_stage2 = len(stage2_keys) > 0

    n_cols = 2
    n_rows = 2 + (2 if has_stage2 else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))

    def _plot(ax, key, ylabel, title, color='tab:blue'):
        if key in history and history[key]:
            ax.plot(history[key], linewidth=1.5, color=color)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.2)
        else:
            ax.text(0.5, 0.5, f'{key}\nnot available', ha='center', va='center', fontsize=10, color='gray')
            ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch', fontsize=9)

    # Stage-1
    _plot(axes[0, 0], 'train_loss', 'Loss', 'Stage-1: Training Loss')
    _plot(axes[0, 1], 'val_loss', 'Loss', 'Stage-1: Validation Loss', 'tab:orange')
    _plot(axes[1, 0], 'val_image_auroc', 'AUROC', 'Stage-1: Image AUROC', 'tab:green')
    _plot(axes[1, 1], 'val_pixel_auroc', 'AUROC', 'Stage-1: Pixel AUROC', 'tab:purple')

    # Stage-2
    if has_stage2:
        _plot(axes[2, 0], 'stage2_train_loss', 'Loss', 'Stage-2: Training Loss')
        _plot(axes[2, 1], 'stage2_val_recon_auroc', 'AUROC', 'Stage-2: Reconstruction AUROC', 'tab:orange')
        _plot(axes[3, 0], 'stage2_val_pixel_agg_auroc', 'AUROC', 'Stage-2: Pixel-Aggregated AUROC', 'tab:green')
        _plot(axes[3, 1], 'stage2_val_divergence_auroc', 'AUROC', 'Stage-2: Divergence AUROC', 'tab:purple')

    fig.suptitle('Training Curves', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Orchestrator
# ===========================================================================

def generate_full_pipeline_visualization(
    model,
    dataloader,
    device: torch.device,
    save_dir,
    n_normal: int = 5,
    n_anomaly: int = 5,
    seed: int = 42,
    target_size: Tuple[int, int] = (240, 240),
    experiment_dir=None,
):
    """Generate all 10 pipeline visualisation steps + training curves.

    Args:
        model: AnomalyDetector (already loaded & on device).
        dataloader: Test DataLoader.
        device: torch device.
        save_dir: Directory to save outputs (will create pipeline_viz/ inside).
        n_normal, n_anomaly: How many tracked samples per class.
        seed: For deterministic sample selection.
        target_size: For pixel-map upsampling.
        experiment_dir: Path to experiment root (for training curves). If None,
                        derived from save_dir.
    """
    save_dir = Path(save_dir)
    viz_dir = save_dir / 'pipeline_viz'
    viz_dir.mkdir(parents=True, exist_ok=True)

    if experiment_dir is None:
        # Heuristic: save_dir is often <exp>/evaluation, so parent is experiment root
        experiment_dir = save_dir.parent if save_dir.name == 'evaluation' else save_dir

    print(f"\n{'='*60}")
    print("GENERATING FULL PIPELINE VISUALIZATION")
    print(f"{'='*60}")
    print(f"  Output: {viz_dir}")

    model.eval()

    # ------------------------------------------------------------------
    # 1. Select tracked samples
    # ------------------------------------------------------------------
    print("  Selecting tracked samples...")
    samples = _select_samples(dataloader, n_normal=n_normal, n_anomaly=n_anomaly, seed=seed)
    tracked_global_indices = [s['index'] for s in samples]
    print(f"    Selected {len(samples)} samples (indices: {tracked_global_indices})")

    # ------------------------------------------------------------------
    # 2. Run model on tracked samples to get per-sample outputs
    # ------------------------------------------------------------------
    print("  Running model on tracked samples...")
    tracked_outputs = []
    tracked_features = []
    with torch.no_grad():
        for s in samples:
            img = s['image'].unsqueeze(0).to(device)  # (1, C, H, W)

            # Full forward for intermediates
            features = model.backbone(img, return_multi_scale=getattr(model, 'use_pixel_decoder', False))
            tracked_features.append({
                'global_raw': features['global_raw'][0].detach().cpu().numpy(),
                'patch_raw': features['patch_raw'][0].detach().cpu().numpy(),
            })

            # Anomaly scores (with maps)
            outputs = model.compute_anomaly_scores(img, return_maps=True, target_size=target_size)
            out_cpu = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    out_cpu[k] = v[0].detach().cpu().numpy()
                else:
                    out_cpu[k] = v
            tracked_outputs.append(out_cpu)

    # ------------------------------------------------------------------
    # 3. Run model on FULL dataset for dataset-level plots
    # ------------------------------------------------------------------
    print("  Running model on full dataset...")
    all_image_scores = []
    all_anchor_scores = []
    all_recon_scores = []
    all_divergence_scores = []
    all_pixel_agg_scores = []
    all_patch_div_agg_scores = []
    all_global_feats = []
    all_distances = []
    all_labels = []
    all_assigned = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels_batch = batch['label'].numpy()

            outputs = model.compute_anomaly_scores(images, return_maps=False, target_size=target_size)

            all_image_scores.append(outputs['image_scores'].detach().cpu().numpy())
            all_labels.append(labels_batch)
            all_distances.append(outputs['all_distances'].detach().cpu().numpy())
            all_assigned.append(outputs['assigned_anchors'].detach().cpu().numpy())

            if 'anchor_scores' in outputs:
                all_anchor_scores.append(outputs['anchor_scores'].detach().cpu().numpy())
            if 'reconstruction_scores' in outputs:
                all_recon_scores.append(outputs['reconstruction_scores'].detach().cpu().numpy())
            if 'bottleneck_divergence' in outputs and outputs['bottleneck_divergence'] is not None:
                all_divergence_scores.append(outputs['bottleneck_divergence'].detach().cpu().numpy())
            if 'pixel_aggregated_score' in outputs:
                all_pixel_agg_scores.append(outputs['pixel_aggregated_score'].detach().cpu().numpy())
            if 'patch_divergence_aggregated_score' in outputs:
                all_patch_div_agg_scores.append(outputs['patch_divergence_aggregated_score'].detach().cpu().numpy())

            # Extract global features for embedding viz
            features = model.backbone(images, return_multi_scale=False)
            gf = features['global']
            if model.backbone.projection is not None:
                gf = model.backbone.projection(features['global_raw'])
                gf = F.normalize(gf, dim=1)
            all_global_feats.append(gf.detach().cpu().numpy())

    all_labels_np = np.concatenate(all_labels)
    all_distances_np = np.concatenate(all_distances)
    all_assigned_np = np.concatenate(all_assigned)
    all_global_feats_np = np.concatenate(all_global_feats)
    all_image_scores_np = np.concatenate(all_image_scores)

    anchor_scores_np = np.concatenate(all_anchor_scores) if all_anchor_scores else all_image_scores_np
    recon_scores_np = np.concatenate(all_recon_scores) if all_recon_scores else None
    divergence_np = np.concatenate(all_divergence_scores) if all_divergence_scores else None
    pixel_agg_np = np.concatenate(all_pixel_agg_scores) if all_pixel_agg_scores else None
    patch_div_agg_np = np.concatenate(all_patch_div_agg_scores) if all_patch_div_agg_scores else None

    # Get anchor embeddings for plots
    anchor_global, _ = model._get_projected_anchors()
    anchor_emb_np = anchor_global.detach().cpu().numpy()

    # Build fused score (dataset-level normalization)
    fused_np = None
    if getattr(model, 'score_fusion_enabled', False):
        norm_mode = getattr(model, 'score_fusion_normalization', 'minmax')
        w_a = getattr(model, 'score_fusion_anchor_weight', 0.4)
        w_d = getattr(model, 'score_fusion_divergence_weight', 0.3)
        w_p = getattr(model, 'score_fusion_pixel_weight', 0.3)

        def _nf(v):
            if norm_mode == 'zscore':
                std = v.std()
                return (v - v.mean()) / max(std, 1e-12)
            elif norm_mode == 'robust':
                q25, q50, q75 = np.percentile(v, [25, 50, 75])
                return (v - q50) / max(q75 - q25, 1e-12)
            else:
                lo, hi = v.min(), v.max()
                return (v - lo) / max(hi - lo, 1e-12)

        a_n = _nf(anchor_scores_np)
        # Pick best divergence signal
        div_signal = None
        if divergence_np is not None and patch_div_agg_np is not None:
            try:
                div_auroc = roc_auc_score(all_labels_np, divergence_np)
            except Exception:
                div_auroc = 0.5
            try:
                pdiv_auroc = roc_auc_score(all_labels_np, patch_div_agg_np)
            except Exception:
                pdiv_auroc = 0.5
            div_signal = divergence_np if div_auroc >= pdiv_auroc else patch_div_agg_np
        elif divergence_np is not None:
            div_signal = divergence_np
        elif patch_div_agg_np is not None:
            div_signal = patch_div_agg_np

        if div_signal is not None and pixel_agg_np is not None:
            fused_np = w_a * a_n + w_d * _nf(div_signal) + w_p * _nf(pixel_agg_np)
        elif div_signal is not None:
            tw = w_a + w_d
            fused_np = (w_a / tw) * a_n + (w_d / tw) * _nf(div_signal)
        elif pixel_agg_np is not None:
            tw = w_a + w_p
            fused_np = (w_a / tw) * a_n + (w_p / tw) * _nf(pixel_agg_np)
        else:
            fused_np = a_n

    # ------------------------------------------------------------------
    # Generate all visualisations
    # ------------------------------------------------------------------
    print("  Generating step visualizations...")

    # Step 1
    print("    Step 1: Raw input images")
    viz_raw_input(samples, viz_dir)

    # Step 2
    print("    Step 2: Preprocessed images + GT masks")
    viz_preprocessed(samples, viz_dir)

    # Step 3
    print("    Step 3: Backbone features")
    viz_backbone_features(samples, tracked_features, viz_dir)

    # Step 4
    print("    Step 4: Projected embeddings")
    viz_projected_embeddings(
        all_global_feats_np, all_labels_np, anchor_emb_np,
        all_assigned_np, viz_dir, tracked_indices=tracked_global_indices,
    )

    # Step 5
    print("    Step 5: Anchor distances")
    viz_anchor_distances(all_distances_np, all_labels_np, anchor_emb_np, viz_dir)

    # Step 6
    print("    Step 6: Reconstruction")
    viz_reconstruction(samples, tracked_outputs, viz_dir)

    # Step 7
    print("    Step 7: Pixel anomaly maps")
    viz_pixel_anomaly_maps(samples, tracked_outputs, viz_dir)

    # Step 8
    print("    Step 8: Divergence maps")
    viz_divergence_maps(samples, tracked_outputs, divergence_np, all_labels_np, viz_dir)

    # Step 9
    print("    Step 9: Score distributions")
    scores_dict = {
        'anchor': anchor_scores_np,
        'reconstruction': recon_scores_np,
        'divergence': divergence_np,
        'pixel_aggregated': pixel_agg_np,
        'patch_div_aggregated': patch_div_agg_np,
    }
    if fused_np is not None:
        scores_dict['fused'] = fused_np
    viz_score_distributions(scores_dict, all_labels_np, viz_dir)

    # Step 10
    print("    Step 10: Final score analysis")
    viz_fused_scores(scores_dict, all_labels_np, samples, tracked_outputs, viz_dir)

    # Training curves
    print("    Training curves")
    viz_training_curves(Path(experiment_dir), viz_dir)

    print(f"  Pipeline visualization complete → {viz_dir}")
    print(f"{'='*60}\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate full pipeline visualization for an experiment')
    parser.add_argument('--experiment', '-e', required=True, help='Path to experiment directory')
    parser.add_argument('--n-normal', type=int, default=5, help='Number of normal samples to track')
    parser.add_argument('--n-anomaly', type=int, default=5, help='Number of anomaly samples to track')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint to load')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    import yaml
    from data import BMADPreprocessor, create_dataloaders
    from main import load_config, load_dataset_paths, create_model, prepare_anchors_in_embedding_space
    from model import DINOv3Backbone

    exp_dir = Path(args.experiment)
    config_path = exp_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    device = torch.device(args.device)

    # Load data
    data_root = config['data']['data_root']
    target_size = tuple(config['data']['target_size'])
    train_paths, train_labels, val_paths, val_labels, val_masks, test_paths, test_labels, test_masks = \
        load_dataset_paths(data_root)

    _, _, test_loader = create_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels, val_masks,
        test_paths, test_labels, test_masks,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        target_size=target_size,
        normalize_mode=config['data'].get('normalization', 'zscore_only'),
    )

    # Build model
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model'].get('freeze_backbone', True),
        projection_dim=config['model'].get('projection_dim', 128),
        projection_hidden_dims=config['model'].get('projection_hidden_dims', None),
    ).to(device)

    # Load anchor embeddings
    anchor_path = exp_dir / 'anchor_embeddings.pt'
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchors not found: {anchor_path}")
    anchor_data = torch.load(anchor_path, map_location=device, weights_only=False)
    anchor_embeddings = anchor_data['embeddings']

    model = create_model(config, backbone, anchor_embeddings, device)

    # Enable stage-2 if configured
    stage2_cfg = config.get('stage2', {})
    if stage2_cfg.get('enabled', False):
        pm_cfg = stage2_cfg.get('pixel_map', {})
        recon_proj_dim = config.get('model', {}).get('projection_dim_recon', None)
        model.enable_reconstruction_branch(
            freeze_anchor_target=stage2_cfg.get('freeze_anchor_target', True),
            out_channels=3,
            pixel_map_enabled=pm_cfg.get('enabled', True),
            pixel_map_type=pm_cfg.get('type', 'reconstruction_l2'),
            use_frozen_bottleneck=stage2_cfg.get('frozen_bottleneck', True),
            recon_projection_dim=recon_proj_dim,
        )
        pix_agg_cfg = stage2_cfg.get('pixel_aggregation', {})
        model.configure_pixel_aggregation(
            method=pix_agg_cfg.get('method', 'top_k_percentile'),
            percentile=pix_agg_cfg.get('percentile', 95),
        )
        fusion_cfg = stage2_cfg.get('score_fusion', {})
        model.configure_score_fusion(
            enabled=fusion_cfg.get('enabled', False),
            normalization=fusion_cfg.get('normalization', 'minmax'),
            anchor_weight=fusion_cfg.get('anchor_weight', 0.4),
            divergence_weight=fusion_cfg.get('divergence_weight', 0.3),
            pixel_weight=fusion_cfg.get('pixel_weight', 0.3),
        )

    # Load checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        best_s2 = exp_dir / 'best_stage2_model.pth'
        best_s1 = exp_dir / 'best_model.pth'
        ckpt_path = best_s2 if best_s2.exists() else best_s1

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    model.to(device)

    eval_dir = exp_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)

    generate_full_pipeline_visualization(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir,
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly,
        seed=args.seed,
        target_size=target_size,
        experiment_dir=exp_dir,
    )


if __name__ == '__main__':
    main()
