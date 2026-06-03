"""
Main training script for BMAD Brain MRI Anomaly Detection
"""

import argparse
import os
import warnings
import yaml
import torch
import numpy as np
import copy
from pathlib import Path
import random

# Suppress pydantic warnings from timm library (internal compatibility issue)
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

from data import BMADPreprocessor, create_dataloaders
from anchors import AnchorGenerator, compute_anchor_embeddings, visualize_anchors
from model import DINOv3Backbone, AnomalyDetector
from loss import AnchorMarginLoss, DenseAnchorMarginLoss, CombinedAnchorLoss
from contrastive_loss import CenterLoss, InfoNCEAnchorLoss, HybridAnchorLoss, CombinedContrastiveLoss
from patch_mode import create_patch_criterion, create_patch_detector, get_patch_variant, prepare_location_kmeans_anchors
from train import Trainer
from eval import evaluate_comprehensive, visualize_predictions, analyze_anchor_assignments


def _merge_nested_defaults(target: dict, defaults: dict) -> None:
    """Recursively fill missing config keys without overwriting user values."""
    for key, value in defaults.items():
        if isinstance(value, dict):
            target.setdefault(key, {})
            _merge_nested_defaults(target[key], value)
        else:
            target.setdefault(key, value)


def _get_anchor_mode(config: dict) -> str:
    """Return the configured anchor mode with backward-compatible defaulting."""
    return str(config.get('anchor', {}).get('mode', 'global')).lower()


def _get_model_anchor_mode(model: torch.nn.Module) -> str:
    """Return the model anchor mode with backward-compatible defaulting."""
    return str(getattr(model, 'anchor_mode', 'global')).lower()


def _validate_checkpoint_anchor_mode(checkpoint: dict, expected_anchor_mode: str, checkpoint_path: Path) -> None:
    """Reject checkpoints that were saved from a different anchor mode."""
    checkpoint_anchor_mode = str(checkpoint.get('anchor_mode', 'global')).lower()
    if checkpoint_anchor_mode != expected_anchor_mode:
        raise ValueError(
            f"Checkpoint anchor_mode mismatch for {checkpoint_path}: "
            f"expected '{expected_anchor_mode}', found '{checkpoint_anchor_mode}'."
        )


def _load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict: bool = True
) -> dict:
    """Load a checkpoint after validating that its anchor mode matches the model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _validate_checkpoint_anchor_mode(
        checkpoint,
        expected_anchor_mode=_get_model_anchor_mode(model),
        checkpoint_path=checkpoint_path
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return checkpoint


def _resolve_kept_kmeans_clusters(labels: np.ndarray, n_clusters: int, prune_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cluster sizes, kept ids, and discarded ids after optional pruning."""
    cluster_sizes = np.bincount(labels, minlength=n_clusters).astype(np.int64)
    min_k = int(prune_cfg.get('min_k', 10))
    discard_ratio = float(prune_cfg.get('discard_ratio', 0.0))

    if discard_ratio <= 0.0 or n_clusters < min_k:
        kept_cluster_ids = np.arange(n_clusters, dtype=np.int64)
        discarded_cluster_ids = np.array([], dtype=np.int64)
        return cluster_sizes, kept_cluster_ids, discarded_cluster_ids

    n_discard = int(np.floor(discard_ratio * n_clusters))
    n_discard = max(0, min(n_discard, n_clusters - 1))
    if n_discard == 0:
        kept_cluster_ids = np.arange(n_clusters, dtype=np.int64)
        discarded_cluster_ids = np.array([], dtype=np.int64)
        return cluster_sizes, kept_cluster_ids, discarded_cluster_ids

    # Deterministic tie-break: smaller cluster first, then lower cluster index.
    sorted_cluster_ids = np.lexsort((np.arange(n_clusters), cluster_sizes))
    discarded_cluster_ids = np.sort(sorted_cluster_ids[:n_discard]).astype(np.int64)
    kept_cluster_ids = np.array([idx for idx in range(n_clusters) if idx not in set(discarded_cluster_ids.tolist())], dtype=np.int64)
    return cluster_sizes, kept_cluster_ids, discarded_cluster_ids


def _reassign_to_kept_centroids(
    embeddings_384d: torch.Tensor,
    initial_labels: np.ndarray,
    centroids_384d: np.ndarray,
    kept_cluster_ids: np.ndarray,
    discarded_cluster_ids: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Map initial K-means labels to the kept effective-K labels."""
    if len(discarded_cluster_ids) == 0:
        mapping = {int(cluster_id): new_idx for new_idx, cluster_id in enumerate(kept_cluster_ids.tolist())}
        final_assignments = np.array([mapping[int(label)] for label in initial_labels], dtype=np.int64)
        return final_assignments, mapping

    kept_centroids = torch.from_numpy(centroids_384d[kept_cluster_ids]).float()
    final_assignments = np.full(len(initial_labels), -1, dtype=np.int64)
    mapping = {int(cluster_id): new_idx for new_idx, cluster_id in enumerate(kept_cluster_ids.tolist())}

    for old_cluster_id, new_cluster_id in mapping.items():
        final_assignments[initial_labels == old_cluster_id] = new_cluster_id

    discarded_mask = np.isin(initial_labels, discarded_cluster_ids)
    if discarded_mask.any():
        discarded_embeddings = embeddings_384d[discarded_mask]
        distances = torch.cdist(discarded_embeddings, kept_centroids, p=2)
        reassigned = distances.argmin(dim=1).cpu().numpy().astype(np.int64)
        final_assignments[discarded_mask] = reassigned

    if (final_assignments < 0).any():
        raise RuntimeError("Failed to assign some samples to kept K-means clusters.")

    return final_assignments, mapping


def _select_kmeans_anchor_artifacts(
    images_np: np.ndarray,
    embeddings_384d: torch.Tensor,
    initial_labels: np.ndarray,
    final_assignments: np.ndarray,
    centroids_384d: np.ndarray,
    kept_cluster_ids: np.ndarray,
    representation: str
) -> tuple[np.ndarray, torch.Tensor, list[int], str]:
    """Build visualization artifacts and semantic anchors for the kept K-means clusters."""
    anchor_images_list = []
    anchor_indices: list[int] = []

    if representation == 'centroids':
        for new_cluster_id, old_cluster_id in enumerate(kept_cluster_ids.tolist()):
            final_mask = final_assignments == new_cluster_id
            if final_mask.any():
                cluster_mean_image = images_np[final_mask].mean(axis=0)
            else:
                cluster_mean_image = images_np.mean(axis=0)
            anchor_images_list.append(cluster_mean_image.astype(np.float32))
            anchor_indices.append(-1)
            print(
                f"   Anchor {new_cluster_id}: source cluster {old_cluster_id:4d}, "
                f"final members {int(final_mask.sum()):4d}, stored semantic anchor = centroid"
            )

        anchor_images = np.array(anchor_images_list, dtype=np.float32)
        anchor_embeddings_384d = torch.from_numpy(centroids_384d[kept_cluster_ids]).float()
        image_source = 'cluster_mean'
        return anchor_images, anchor_embeddings_384d, anchor_indices, image_source

    for new_cluster_id, old_cluster_id in enumerate(kept_cluster_ids.tolist()):
        original_mask = initial_labels == old_cluster_id
        count = int(original_mask.sum())
        if count == 0:
            print(f"   Anchor {new_cluster_id}: kept cluster {old_cluster_id} empty, selecting global nearest")
            dists = np.linalg.norm(embeddings_384d.numpy() - centroids_384d[old_cluster_id], axis=1)
            idx = int(np.argmin(dists))
        else:
            cluster_embeddings = embeddings_384d[original_mask]
            dists = torch.norm(cluster_embeddings - torch.from_numpy(centroids_384d[old_cluster_id]).float(), dim=1)
            local_idx = int(torch.argmin(dists))
            idx = int(np.where(original_mask)[0][local_idx])

        anchor_indices.append(idx)
        anchor_images_list.append(images_np[idx])
        print(
            f"   Anchor {new_cluster_id}: source cluster {old_cluster_id:4d}, "
            f"original members {count:4d}, selected sample {idx}"
        )

    anchor_images = np.array(anchor_images_list)
    anchor_embeddings_384d = embeddings_384d[anchor_indices]
    image_source = 'closest_sample'
    return anchor_images, anchor_embeddings_384d, anchor_indices, image_source


def load_dataset_paths(data_root: str):
    """
    Load image and label paths from BraTS2021_slice dataset structure
    
    Structure:
        data_root/
            train/
                good/
                    *.png
            valid/
                good/
                    img/*.png
                    label/*.png
                Ungood/
                    img/*.png
                    label/*.png
            test/
                good/
                    img/*.png
                    label/*.png
                Ungood/
                    img/*.png
                    label/*.png
    
    Returns:
        train_paths, val_paths, val_labels, val_mask_paths,
        test_paths, test_labels, test_mask_paths
    """
    data_root = Path(data_root)
    
    # Training: only normal images (good/)
    train_dir = data_root / 'train' / 'good'
    train_paths = sorted([str(p) for p in train_dir.glob('*.png')])
    
    # Validation: good + Ungood
    val_paths = []
    val_labels = []
    val_mask_paths = []
    
    # Val - good (label 0, no anomaly)
    val_good_img_dir = data_root / 'valid' / 'good' / 'img'
    val_good_label_dir = data_root / 'valid' / 'good' / 'label'
    val_good_imgs = sorted([str(p) for p in val_good_img_dir.glob('*.png')])
    for img_path in val_good_imgs:
        img_name = Path(img_path).name
        label_path = val_good_label_dir / img_name
        val_paths.append(img_path)
        val_labels.append(0)
        val_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    # Val - Ungood (label 1, anomaly)
    val_ungood_img_dir = data_root / 'valid' / 'Ungood' / 'img'
    val_ungood_label_dir = data_root / 'valid' / 'Ungood' / 'label'
    val_ungood_imgs = sorted([str(p) for p in val_ungood_img_dir.glob('*.png')])
    for img_path in val_ungood_imgs:
        img_name = Path(img_path).name
        label_path = val_ungood_label_dir / img_name
        val_paths.append(img_path)
        val_labels.append(1)
        val_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    # Test: good + Ungood
    test_paths = []
    test_labels = []
    test_mask_paths = []
    
    # Test - good (label 0, no anomaly)
    test_good_img_dir = data_root / 'test' / 'good' / 'img'
    test_good_label_dir = data_root / 'test' / 'good' / 'label'
    test_good_imgs = sorted([str(p) for p in test_good_img_dir.glob('*.png')])
    for img_path in test_good_imgs:
        img_name = Path(img_path).name
        label_path = test_good_label_dir / img_name
        test_paths.append(img_path)
        test_labels.append(0)
        test_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    # Test - Ungood (label 1, anomaly)
    test_ungood_img_dir = data_root / 'test' / 'Ungood' / 'img'
    test_ungood_label_dir = data_root / 'test' / 'Ungood' / 'label'
    test_ungood_imgs = sorted([str(p) for p in test_ungood_img_dir.glob('*.png')])
    for img_path in test_ungood_imgs:
        img_name = Path(img_path).name
        label_path = test_ungood_label_dir / img_name
        test_paths.append(img_path)
        test_labels.append(1)
        test_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    print(f"Loaded dataset from {data_root}:")
    print(f"  Train: {len(train_paths)} normal images")
    print(f"  Val: {len(val_paths)} images ({val_labels.count(0)} normal, {val_labels.count(1)} anomaly)")
    print(f"  Test: {len(test_paths)} images ({test_labels.count(0)} normal, {test_labels.count(1)} anomaly)")
    
    return train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve data_root relative to config file so the script works from any CWD.
    # Absolute paths are left unchanged.
    data_root = config.get('data', {}).get('data_root', './data/BraTS2021_slice')
    if not os.path.isabs(data_root):
        config.setdefault('data', {})['data_root'] = os.path.normpath(
            os.path.join(config_dir, '../..', data_root)
        )

    # Resolve output_dir relative to workspace root (configs/ is two levels deep)
    output_dir = config.get('output_dir', './runs/bmad_baseline')
    if not os.path.isabs(output_dir):
        config['output_dir'] = os.path.normpath(
            os.path.join(config_dir, '../..', output_dir)
        )

    anchor_defaults = {
        'mode': 'global',
        'representation': 'closest_samples',
        'patch': {
            'variant': 'legacy',
            'score_reduction': 'mean',
            'local_distance_metric': 'euclidean',
            'local_score_reduction': 'percentile',
            'local_score_percentile': 95.0
        },
        'prune': {
            'min_k': 10,
            'discard_ratio': 0.0
        }
    }
    data_defaults = {
        'train_augment_mode': 'full'
    }
    training_defaults = {
        'fixed_pseudo_labels': True,
        'dynamic_reassignment': False,
        'reassignment_interval': 5,
        'pseudo_label_assignment': 'nearest',
        'capacity_multiplier': 2.0
    }
    stage2_defaults = {
        'enabled': False,
        'epochs': 20,
        'lr': 0.0001,
        'weight_decay': 0.000001,
        'freeze_encoder': True,
        'freeze_anchors': True,
        'freeze_anchor_target': True,
        'recon_loss': 'mse',
        'recon_weight': 1.0,
        'consistency_loss': 'cosine',
        'consistency_weight': 0.1,
        'frozen_bottleneck': True,
        'alignment_weight': 0.1,
        'alignment_target': 'sample',
        'freeze_encoder_mode': 'full',
        'unfreeze_last_n_blocks': 2,
        'unfreeze_lr_multiplier': 0.1,
        'early_stopping_patience': 10,
        'early_stopping_metric': 'pixel_aggregated_image_auroc',
        'pixel_map': {
            'enabled': True,
            'type': 'reconstruction_l2'
        },
        'pixel_metrics': {
            'enabled': True
        },
        'score_combination': {
            'enabled': False,
            'alpha': 0.5,
            'normalization': 'minmax'
        },
        'pixel_aggregation': {
            'method': 'top_k_percentile',
            'percentile': 95,
            'threshold_n_std': 2.0
        },
        'score_fusion': {
            'enabled': False,
            'normalization': 'minmax',
            'anchor_weight': 0.4,
            'divergence_weight': 0.3,
            'pixel_weight': 0.3
        }
    }

    _merge_nested_defaults(config.setdefault('data', {}), data_defaults)
    _merge_nested_defaults(config.setdefault('anchor', {}), anchor_defaults)
    _merge_nested_defaults(config.setdefault('training', {}), training_defaults)
    _merge_nested_defaults(config.setdefault('stage2', {}), stage2_defaults)

    return config


def _validate_anchors(anchor_embeddings: torch.Tensor, margin: float = 1.0) -> None:
    """Validate anchor quality: separation, duplicates, coverage."""
    import warnings
    K = anchor_embeddings.shape[0]
    if K < 2:
        return

    dists = torch.cdist(anchor_embeddings.float(), anchor_embeddings.float(), p=2)
    # Mask diagonal
    off_diag = dists + torch.eye(K) * 1e12
    min_sep = off_diag.min().item()
    mean_sep = off_diag[off_diag < 1e11].mean().item()

    print(f"\n  Anchor quality check (K={K}):")
    print(f"    Min pairwise distance : {min_sep:.4f}")
    print(f"    Mean pairwise distance: {mean_sep:.4f}")
    print(f"    Margin (2m target)    : {2 * margin:.4f}")

    if min_sep < 1e-4:
        warnings.warn(f"DUPLICATE ANCHORS detected: min distance = {min_sep:.6f}. "
                       "Two or more anchors are nearly identical.")
    elif min_sep < margin:
        warnings.warn(f"Low anchor separation: {min_sep:.4f} < margin {margin:.4f}. "
                       "Repeller loss will need to push these apart.")


def prepare_anchors_in_embedding_space(
    train_images: list,
    preprocessor: BMADPreprocessor,
    config: dict,
    save_dir: Path,
    backbone: DINOv3Backbone,
    device: torch.device
) -> tuple:
    """
    Generate anchors in DINOv3 embedding space (SOLUTION A).
    
    This is the CORRECT approach:
    1. Extract DINOv3 embeddings for all training images
    2. Run k-means in semantic space (not random pixel space)
    3. Select anchor images corresponding to cluster centers
    4. Store anchors in RAW embedding space (NOT projected)
    5. Anchors will be re-projected each forward pass (moving with projection head)
    
    This ensures:
    - Pseudo-labels based on semantic similarity (frozen DINOv3)
    - Anchors and samples always in same embedding space
    - No mismatch between anchor positions and projection learning
    """
    embed_dim = backbone.embed_dim
    print("\n" + "="*80)
    print(f"ANCHOR GENERATION IN {embed_dim}D DINOV3 EMBEDDING SPACE (SOLUTION A)")
    print("="*80)
    print("Strategy: Semantic clustering in frozen DINOv3 space")
    print(f"  1. Extract {embed_dim}D embeddings for training images")
    print(f"  2. Run k-means/eigenface in {embed_dim}D space")
    print("  3. Build final effective anchors after optional pruning")
    print(f"  4. Store in RAW {embed_dim}D (re-project each forward)")
    
    # Load training images
    max_images = config['anchor'].get('max_images_for_pca', 5000)
    if max_images is None:
        max_images = len(train_images)
        print(f"\nLoading ALL {len(train_images)} training images...")
    else:
        print(f"\nLoading {min(len(train_images), max_images)} training images...")
    
    import cv2
    images_list = []
    for img_path in train_images[:max_images]:
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = preprocessor.preprocess(img)
        images_list.append(img)
    
    images_np = np.array(images_list)
    print(f"Loaded {len(images_np)} images, shape: {images_np.shape}")
    
    # Extract DINOv3 embeddings (frozen backbone, NO projection)
    print(f"\nExtracting {embed_dim}D DINOv3 embeddings...")
    backbone.eval()
    embeddings_384d_list = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(images_np), batch_size):
            batch_imgs = images_np[i:i+batch_size]
            
            # Convert to 3-channel tensor (must match BMADDataset preprocessing)
            from anchors import _grayscale_batch_to_tensor
            _norm_mode = config['data'].get('normalization', 'zscore_only')
            batch_tensor = _grayscale_batch_to_tensor(batch_imgs, device,
                                                       apply_imagenet_norm=(_norm_mode == 'minmax_imagenet'))
            
            # Extract RAW DINOv3 features (CLS token)
            features = backbone.backbone.forward_features(batch_tensor)
            cls_tokens = features[:, 0]  # (B, embed_dim)
            
            embeddings_384d_list.append(cls_tokens.cpu())
    
    embeddings_384d = torch.cat(embeddings_384d_list, dim=0)  # (N, embed_dim)
    print(f"✓ Extracted embeddings: {embeddings_384d.shape}")
    
    # Run clustering in embedding space
    strategy_name = config['anchor']['strategy']
    n_anchors = config['anchor']['n_anchors']
    print(f"\nClustering in {embed_dim}D space: strategy={strategy_name}, n_anchors={n_anchors}")
    
    if strategy_name == 'kmeans':
        from sklearn.cluster import KMeans
        representation = config['anchor'].get('representation', 'closest_samples')
        if representation not in {'closest_samples', 'centroids'}:
            raise ValueError(f"Unsupported anchor.representation: {representation}")

        kmeans = KMeans(n_clusters=n_anchors, random_state=config['seed'], n_init=10)
        labels = kmeans.fit_predict(embeddings_384d.numpy())
        centroids_384d = kmeans.cluster_centers_  # (K, embed_dim)
        prune_cfg = config['anchor'].get('prune', {})
        cluster_sizes, kept_cluster_ids, discarded_cluster_ids = _resolve_kept_kmeans_clusters(labels, n_anchors, prune_cfg)

        print("  Initial cluster sizes:")
        for cluster_id, cluster_size in enumerate(cluster_sizes.tolist()):
            print(f"    Cluster {cluster_id:4d}: {cluster_size:5d} samples")

        print(
            f"  Effective K after pruning: {len(kept_cluster_ids)} "
            f"(discarded {len(discarded_cluster_ids)} / {n_anchors})"
        )
        if len(discarded_cluster_ids) > 0:
            print(f"  Discarded clusters: {discarded_cluster_ids.tolist()}")

        final_assignments, kept_mapping = _reassign_to_kept_centroids(
            embeddings_384d=embeddings_384d,
            initial_labels=labels,
            centroids_384d=centroids_384d,
            kept_cluster_ids=kept_cluster_ids,
            discarded_cluster_ids=discarded_cluster_ids
        )
        final_cluster_sizes = np.bincount(final_assignments, minlength=len(kept_cluster_ids)).astype(np.int64)
        moved_from_discarded = int(np.isin(labels, discarded_cluster_ids).sum())

        anchor_images, anchor_embeddings_384d, anchor_indices, anchor_image_source = _select_kmeans_anchor_artifacts(
            images_np=images_np,
            embeddings_384d=embeddings_384d,
            initial_labels=labels,
            final_assignments=final_assignments,
            centroids_384d=centroids_384d,
            kept_cluster_ids=kept_cluster_ids,
            representation=representation
        )

        anchor_metadata = {
            'representation': representation,
            'initial_k': int(n_anchors),
            'effective_k': int(len(kept_cluster_ids)),
            'cluster_sizes_initial': cluster_sizes.tolist(),
            'cluster_sizes_effective': final_cluster_sizes.tolist(),
            'kept_cluster_ids': kept_cluster_ids.tolist(),
            'discarded_cluster_ids': discarded_cluster_ids.tolist(),
            'moved_from_discarded': moved_from_discarded,
            'anchor_image_source': anchor_image_source,
            'anchor_image_indices': anchor_indices,
            'subset_assignments_effective': final_assignments.tolist(),
            'kept_cluster_id_to_effective_index': {str(key): int(value) for key, value in kept_mapping.items()}
        }
        
    elif strategy_name == 'eigenface':
        from sklearn.decomposition import PCA
        
        n_components = config['anchor'].get('n_components', 50)
        print(f"  Running PCA: {embed_dim}D -> {n_components}D...")
        pca = PCA(n_components=n_components, random_state=config['seed'])
        embeddings_pca = pca.fit_transform(embeddings_384d.numpy())
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # K-means in PCA space
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_anchors, random_state=config['seed'], n_init=10)
        labels = kmeans.fit_predict(embeddings_pca)
        
        # Select images closest to each centroid in PCA space
        anchor_indices = []
        anchor_images_list = []
        for k in range(n_anchors):
            mask = labels == k
            count = mask.sum()
            if count == 0:
                dists = np.linalg.norm(embeddings_pca - kmeans.cluster_centers_[k], axis=1)
                idx = int(np.argmin(dists))
            else:
                cluster_embeddings = embeddings_pca[mask]
                dists = np.linalg.norm(cluster_embeddings - kmeans.cluster_centers_[k], axis=1)
                local_idx = int(np.argmin(dists))
                idx = np.where(mask)[0][local_idx]
            
            anchor_indices.append(idx)
            anchor_images_list.append(images_np[idx])
            print(f"   Anchor {k}: {count:4d} images in cluster, selected sample {idx}")
        
        anchor_images = np.array(anchor_images_list)
        anchor_embeddings_384d = embeddings_384d[anchor_indices]
        anchor_metadata = {
            'representation': 'closest_samples',
            'initial_k': int(n_anchors),
            'effective_k': int(n_anchors),
            'cluster_sizes_initial': np.bincount(labels, minlength=n_anchors).astype(np.int64).tolist(),
            'cluster_sizes_effective': np.bincount(labels, minlength=n_anchors).astype(np.int64).tolist(),
            'kept_cluster_ids': list(range(n_anchors)),
            'discarded_cluster_ids': [],
            'moved_from_discarded': 0,
            'anchor_image_source': 'closest_sample',
            'anchor_image_indices': [int(idx) for idx in anchor_indices]
        }
    
    else:
        raise ValueError(f"Strategy {strategy_name} not supported in embedding space generation")

    print("\nComputing dense anchor maps for selected embedding-space anchors...")
    _, anchor_dense_384d = compute_anchor_embeddings(
        anchor_images=anchor_images,
        backbone_model=backbone,
        device=device,
        batch_size=8,
        return_projected=False,
        apply_imagenet_norm=(config['data'].get('normalization', 'zscore_only') == 'minmax_imagenet')
    )
    print(f"  Dense anchor maps: {anchor_dense_384d.shape}")
    
    # EXPERT'S APPROACH: Create FIXED geometric targets in 128D projection space
    # SOLUTION A: Skip this step - will re-project anchors each forward instead
    reproject_anchors = config['anchor'].get('reproject_anchors', False)

    # --- Anchor Quality Validation ---
    _validate_anchors(anchor_embeddings_384d, margin=config['loss'].get('margin', 1.0))

    projection_hidden_dims = config['model'].get('projection_hidden_dims', None)
    projection_dim = projection_hidden_dims[-1] if projection_hidden_dims is not None else config['model'].get('projection_dim', None)
    
    effective_n_anchors = int(anchor_embeddings_384d.shape[0])

    if projection_dim and not reproject_anchors:
        print(f"\n{'='*80}")
        print("CREATING FIXED GEOMETRIC TARGETS (Expert's Approach)")
        print(f"{'='*80}")
        
        init_method = config['anchor'].get('geometric_init', 'random_orthogonal')
        
        if init_method == 'random_orthogonal':
            # Option A: Random orthogonal normalized vectors
            print(f"  Method: Random orthogonal vectors in {projection_dim}D space")
            geometric_targets = torch.randn(effective_n_anchors, projection_dim)
            geometric_targets = torch.nn.functional.normalize(geometric_targets, dim=1)
            print(f"  ✓ Generated {effective_n_anchors} random orthogonal targets")
            
        elif init_method == 'project_once':
            # Option B: Project semantic anchors ONCE through random projection head and detach
            print(f"  Method: Project semantic anchors once through random projection head")
            backbone.eval()
            with torch.no_grad():
                geometric_targets = backbone.projection(anchor_embeddings_384d.to(device))
                geometric_targets = torch.nn.functional.normalize(geometric_targets, dim=1)
            geometric_targets = geometric_targets.cpu().detach()
            print(f"  ✓ Projected {effective_n_anchors} anchors through random projection head (detached)")
            
        else:
            raise ValueError(f"Unknown geometric_init method: {init_method}")
        
        print(f"  Shape: {geometric_targets.shape}")
        print(f"  These targets are FIXED and will NEVER change during training")
        print(f"  Projection head learns to map samples with Label_K → Target_K")
    elif reproject_anchors:
        print(f"\n{'='*80}")
        print("SOLUTION A: Anchors will be RE-PROJECTED each forward pass")
        print(f"{'='*80}")
        print(f"  Semantic anchors ({embed_dim}D): {anchor_embeddings_384d.shape}")
        print(f"  These will be re-projected through projection head EVERY forward pass")
        print(f"  Anchors 'move' with projection head during training")
        print(f"  Requires diversity loss (delta={config['loss'].get('delta', 0.0)}) to prevent collapse")
        geometric_targets = None  # No geometric targets for Solution A
        
    else:
        # No projection head - geometric targets same as semantic anchors
        geometric_targets = anchor_embeddings_384d
        print(f"\n  No projection head - using semantic anchors as geometric targets")
    
    # Save anchor data with BOTH semantic and geometric anchors
    torch.save({
        'anchor_images': anchor_images,
        'anchor_semantic': anchor_embeddings_384d,  # For pseudo-label computation (embed_dim D)
        'anchor_geometric': geometric_targets,       # For training targets (projection_dim D, FIXED)
        'anchor_global': anchor_embeddings_384d,     # Legacy compatibility
        'anchor_dense': anchor_dense_384d,
        'embedding_dim': embed_dim,
        'projection_dim': projection_dim,
        'is_projected': False,
        'use_decoupled': geometric_targets is not None,
        'generation_method': 'decoupled_semantic_geometric' if geometric_targets is not None else 'embedding_space_reproject',
        'anchor_metadata': anchor_metadata
    }, save_dir / 'anchor_embeddings.pt')
    
    # Visualize anchors
    visualize_anchors(anchor_images, save_dir / 'anchor_images.png')
    
    print(f"\n{'='*80}")
    if geometric_targets is not None:
        print(f"✓ Generated {effective_n_anchors} DECOUPLED anchors (Expert's Approach)")
        print(f"{'='*80}")
        print(f"  Semantic anchors ({embed_dim}D): {anchor_embeddings_384d.shape} [for pseudo-labels]")
        print(f"  Geometric targets ({projection_dim}D): {geometric_targets.shape} [FIXED training targets]")
        print(f"  Decoupling prevents moving target problem and collapse!")
    else:
        print(f"✓ Generated {n_anchors} semantic anchors (Solution A: Re-projection)")
        print(f"{'='*80}")
        print(f"  Semantic anchors ({embed_dim}D): {anchor_embeddings_384d.shape} [for pseudo-labels AND re-projection]")
        print(f"  These will be RE-PROJECTED through projection head each forward pass")
    print(f"  Dense anchors ({embed_dim}D patch grid): {anchor_dense_384d.shape}")
    
    return anchor_images, anchor_embeddings_384d, anchor_dense_384d, geometric_targets


def prepare_anchors(
    train_images: list,
    preprocessor: BMADPreprocessor,
    config: dict,
    save_dir: Path,
    backbone_for_projection: DINOv3Backbone = None,
    device: torch.device = None
) -> tuple:
    """
    Prepare anchor images and embeddings.
    
    CRITICAL: If backbone_for_projection is provided, anchors are projected through
    that SPECIFIC backbone's projection head. This ensures anchors and samples use
    the SAME projection weights. The anchors are then stored in PROJECTED space
    and NOT re-projected during training (acting as fixed targets).
    
    Args:
        train_images: List of training image paths
        preprocessor: Image preprocessor
        config: Configuration dictionary
        save_dir: Directory to save anchor data
        backbone_for_projection: If provided, project anchors through THIS backbone
        device: Device for computation
    
    Returns:
        anchor_images, anchor_global_embeddings (PROJECTED if backbone provided), anchor_dense_embeddings
    """
    print("\n" + "="*80)
    print("ANCHOR GENERATION")
    print("="*80)
    
    # Load and preprocess training images
    print("Loading training images...")
    images = []
    for img_path in train_images[:config['anchor']['max_images_for_pca']]:
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = preprocessor.preprocess(img)
        images.append(img)
    
    images = np.array(images)
    print(f"Loaded {len(images)} images, shape: {images.shape}")
    
    # Generate anchors with selected strategy
    strategy_name = config['anchor'].get('strategy', 'eigenface')
    
    # Build kwargs for AnchorGenerator based on strategy
    anchor_gen_kwargs = {
        'strategy': strategy_name,
        'n_anchors': config['anchor']['n_anchors'],
        'random_state': config['seed']
    }
    
    # Only add n_components if the strategy uses it
    if strategy_name in ['eigenface', 'kcenter', 'density', 'gmm', 'stratified']:
        anchor_gen_kwargs['n_components'] = config['anchor'].get('n_components', 50)
    
    anchor_gen = AnchorGenerator(**anchor_gen_kwargs)
    
    anchor_images = anchor_gen.fit(images)
    
    # Save anchor generator
    anchor_gen.save(save_dir / 'anchor_generator.pkl')
    
    # Visualize anchors
    visualize_anchors(anchor_images, save_dir / 'anchor_images.png')
    
    # Compute anchor embeddings
    print("\nComputing anchor embeddings with DINOv3...")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    projection_hidden_dims = config['model'].get('projection_hidden_dims', None)
    projection_dim = projection_hidden_dims[-1] if projection_hidden_dims is not None else config['model'].get('projection_dim', None)

    if backbone_for_projection is not None:
        # Use the PROVIDED backbone (same one the model will use)
        # This ensures anchors and samples use the SAME projection weights
        print(f"  Using provided backbone for projection (ensuring same weights as model)")
        backbone = backbone_for_projection
        use_model_backbone = True
    else:
        # Create a temporary backbone (legacy behavior - NOT recommended with projection)
        print(f"  Creating temporary backbone for embedding extraction")
        backbone = DINOv3Backbone(
            model_name=config['model']['backbone'],
            freeze_backbone=True,
            projection_dim=projection_dim,
            pretrained=True,
            projection_hidden_dims=projection_hidden_dims
        )
        backbone = backbone.to(device)
        use_model_backbone = False
    
    backbone.eval()
    
    # Extract embeddings - use return_projected=True if we have a projection head
    has_projection = backbone.projection is not None
    anchor_global, anchor_dense = compute_anchor_embeddings(
        anchor_images=anchor_images,
        backbone_model=backbone,
        device=device,
        batch_size=8,
        return_projected=has_projection,
        apply_imagenet_norm=(config['data'].get('normalization', 'zscore_only') == 'minmax_imagenet')
    )
    
    # Save embeddings
    torch.save({
        'anchor_images': anchor_images,
        'anchor_global': anchor_global,
        'anchor_dense': anchor_dense,
        'projection_dim': projection_dim if has_projection else None,
        'is_projected': has_projection,
        'used_model_backbone': use_model_backbone
    }, save_dir / 'anchor_embeddings.pt')
    
    print(f"\nAnchor preparation complete!")
    print(f"  Global embeddings: {anchor_global.shape}")
    if has_projection:
        print(f"  Anchors are in PROJECTED space ({anchor_global.shape[1]}D)")
        if use_model_backbone:
            print(f"  ✓ Projected through MODEL's backbone - same weights as training!")
        else:
            print(f"  ⚠ Projected through TEMPORARY backbone - weights differ from model!")
    else:
        print(f"  Anchors are in RAW space ({backbone.embed_dim}D) - no projection head")
    print(f"  Dense embeddings: {anchor_dense.shape}")
    
    return anchor_images, anchor_global, anchor_dense


def create_model(config: dict, anchor_global: torch.Tensor, anchor_dense: torch.Tensor) -> AnomalyDetector:
    """Create anomaly detector model"""
    anchor_mode = _get_anchor_mode(config)
    if anchor_mode == 'patch':
        print("\nPATCH MODE: creating dense-anchor detector")
        return create_patch_detector(config, anchor_global, anchor_dense)

    print("\n" + "="*80)
    print("MODEL CREATION")
    print("="*80)
    
    # Check if pixel decoder is requested
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    multi_scale_indices = config['model'].get('multi_scale_indices', [2, 5, 8, 11])
    projection_hidden_dims = config['model'].get('projection_hidden_dims', None)
    projection_dim = projection_hidden_dims[-1] if projection_hidden_dims is not None else config['model'].get('projection_dim', None)
    
    # Create backbone with multi-scale support if pixel decoder is enabled
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=projection_dim,
        pretrained=True,
        multi_scale_indices=multi_scale_indices if use_pixel_decoder else None,
        projection_hidden_dims=projection_hidden_dims
    )
    
    # Check if anchors should be learnable
    learnable_anchors = config['anchor'].get('learnable', False)
    
    # CRITICAL: Anchors are already in PROJECTED space (from prepare_anchors).
    # They were projected ONCE through a fresh projection head and stored.
    # The model will use them as FIXED targets - NOT re-project them.
    # This prevents collapse: the projection head learns to map samples TO
    # these fixed anchor locations, rather than collapsing everything together.
    if projection_dim:
        print(f"\nAnchors are in PROJECTED space: {anchor_global.shape}")
        print(f"  They are FIXED targets - will NOT be re-projected during training")
        print(f"  Projection head learns to map samples TO these fixed anchors")
    else:
        print(f"\nAnchors are in RAW space: {anchor_global.shape}")
        print(f"  No projection head configured")
    
    # Get target size from config
    target_size = tuple(config['data']['target_size'])
    
    # Create detector
    # SOLUTION A: anchors_already_projected=False so anchors are re-projected each forward
    use_embedding_space = config['anchor'].get('use_embedding_space', True)
    detector = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_global,
        anchor_dense_embeddings=anchor_dense,
        distance_metric=config['loss']['distance_metric'],
        learnable_anchors=learnable_anchors,
        use_pixel_decoder=use_pixel_decoder,
        decoder_hidden_dim=config['model'].get('decoder_hidden_dim', 256),
        target_size=target_size,
        anchors_already_projected=False if use_embedding_space else (projection_dim is not None)
    )
    
    return detector


def create_criterion(config: dict):
    """
    Create loss function based on config.
    
    Supports:
    - 'cam': Class Anchor Margin Loss (original, attractor + repeller + min-norm)
    - 'center': Center Loss (pull samples + anchors toward each other)
    - 'infonce': InfoNCE contrastive loss (soft assignments with temperature)
    - 'hybrid': Hybrid of Center + InfoNCE (best of both)
    
    For learnable anchors, 'center', 'infonce', or 'hybrid' are recommended.
    """
    anchor_mode = _get_anchor_mode(config)
    if anchor_mode == 'patch':
        return create_patch_criterion(config)

    loss_type = config['loss'].get('type', 'cam')  # Default to CAM loss for backward compatibility
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    
    print(f"\nCreating loss function: {loss_type}")
    if use_pixel_decoder:
        print(f"  Pixel decoder enabled: dense loss will be computed")
    
    if loss_type == 'cam':
        # Original CAM loss with diversity regularization
        global_loss = AnchorMarginLoss(
            margin=config['loss']['margin'],
            alpha=config['loss']['alpha'],
            beta=config['loss']['beta'],
            gamma=config['loss'].get('gamma', 0.0),
            delta=config['loss'].get('delta', 0.0),
            min_norm=config['loss'].get('min_norm', 0.5),
            diversity_temperature=config['loss'].get('diversity_temperature', 0.1),
            distance_metric=config['loss']['distance_metric']
        )

        # Create dense loss if pixel decoder is enabled
        dense_loss = None
        if use_pixel_decoder:
            dense_loss = DenseAnchorMarginLoss(
                margin=config['loss']['margin'],
                alpha=config['loss']['alpha'],
                distance_metric=config['loss']['distance_metric'],
                spatial_reduction='mean'
            )
            print(f"  Dense loss: DenseAnchorMarginLoss (alpha={config['loss']['alpha']})")
        
        # Combined loss
        criterion = CombinedAnchorLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    elif loss_type == 'center':
        # Center Loss (dense branch disabled)
        global_loss = CenterLoss(
            distance_metric=config['loss']['distance_metric'],
            lambda_center=config['loss'].get('lambda_center', 1.0),
            lambda_repel=config['loss'].get('lambda_repel', 0.1),
            margin=config['loss']['margin']
        )

        dense_loss = None
        config['loss']['use_dense'] = False

        criterion = CombinedContrastiveLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    elif loss_type == 'infonce':
        # InfoNCE Loss (dense branch disabled)
        global_loss = InfoNCEAnchorLoss(
            temperature=config['loss'].get('temperature', 0.07),
            lambda_repel=config['loss'].get('lambda_repel', 0.1),
            margin=config['loss']['margin'],
            distance_metric=config['loss']['distance_metric']
        )

        dense_loss = None
        config['loss']['use_dense'] = False

        criterion = CombinedContrastiveLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    elif loss_type == 'hybrid':
        # Hybrid: Center + InfoNCE (dense branch disabled)
        global_loss = HybridAnchorLoss(
            lambda_center=config['loss'].get('lambda_center', 1.0),
            lambda_infonce=config['loss'].get('lambda_infonce', 0.5),
            lambda_repel=config['loss'].get('lambda_repel', 0.1),
            temperature=config['loss'].get('temperature', 0.07),
            margin=config['loss']['margin'],
            distance_metric=config['loss']['distance_metric']
        )

        dense_loss = None
        config['loss']['use_dense'] = False

        criterion = CombinedContrastiveLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: cam, center, infonce, hybrid")
    
    print(f"  ✓ Loss type: {loss_type}")
    return criterion


def generate_experiment_name(config: dict, base_dir: str = './runs') -> str:
    """
    Generate experiment name based on anchor and distance configuration
    
    Format: <base_name>_<strategy>_k<num_anchors>_<distance>
    Example: bmad_eigenface_k8_cosine, bmad_random_k16_l2, bmad_kmeans_k4_cosine
    """
    base_name = Path(base_dir).name if '/' in base_dir or '\\' in base_dir else 'bmad'
    strategy = config['anchor']['strategy']
    n_anchors = config['anchor']['n_anchors']
    distance = config['loss']['distance_metric']
    
    # Abbreviate distance metric
    dist_abbrev = 'cos' if distance == 'cosine' else 'l2'
    
    exp_name = f"{base_name}_{strategy}_k{n_anchors}_{dist_abbrev}"
    
    return exp_name


def make_unique_dir(base: Path) -> Path:
    """Create a unique directory by adding numeric suffix if needed."""
    if not base.exists():
        return base
    idx = 1
    while True:
        cand = base.parent / f"{base.name}_{idx}"
        if not cand.exists():
            return cand
        idx += 1


def main(args):
    """Main training pipeline"""
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Auto/explicit experiment naming and uniqueness
    if args.exp_name:
        save_dir = Path(config['output_dir']) / args.exp_name
    elif args.auto_name or config['output_dir'] in ('./experiments/bmad_baseline', './runs/bmad_baseline'):
        base_output = Path(config['output_dir']).parent
        exp_name = generate_experiment_name(config, str(base_output))
        save_dir = base_output / exp_name
    else:
        save_dir = Path(config['output_dir'])

    # Avoid overwrite by uniquifying when directory exists, except eval-only mode
    # where we intentionally reuse an existing experiment directory.
    if args.eval_only:
        if not save_dir.exists():
            raise FileNotFoundError(
                f"Eval-only mode expects an existing output directory, but not found: {save_dir}"
            )
    else:
        save_dir = make_unique_dir(save_dir)
    config['output_dir'] = str(save_dir)
    
    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("="*80)
    print("BMAD BRAIN MRI ANOMALY DETECTION")
    print("="*80)
    print(f"Output directory: {save_dir}")
    print(f"Config: {args.config}")
    print(f"Anchor strategy: {config['anchor']['strategy']}")
    print(f"Number of anchors: {config['anchor']['n_anchors']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ===== STAGE 1: Data Preparation =====
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    # Load dataset paths from BraTS2021_slice structure
    data_root = config['data'].get('data_root', './data/BraTS2021_slice')
    normalize_mode = config['data'].get('normalization', 'zscore_only')
    
    train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(data_root)
    
    if not train_paths:
        print("\nWARNING: No data found!")
        print(f"Please check that the dataset exists at: {data_root}")
        print("Expected structure: train/good/*.png, valid/good/img/*.png, etc.")
        return

    # Optional: limit training set size (for quick tests / ablations)
    max_train = config['data'].get('max_train_samples', None)
    if max_train is not None and max_train < len(train_paths):
        import random as _random
        _rng = _random.Random(config.get('seed', 42))
        train_paths = _rng.sample(train_paths, max_train)
        train_paths = sorted(train_paths)
        print(f"  [quicktest] Limiting to {max_train} training images")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        val_mask_paths=val_mask_paths,
        test_mask_paths=test_mask_paths,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        target_size=tuple(config['data']['target_size']),
        normalize_mode=normalize_mode,
        train_augment_mode=config['data'].get('train_augment_mode', 'full')
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ===== STAGE 2: Create Model Backbone FIRST =====
    # We need to create the backbone first so we can project anchors through
    # the SAME projection head that will be used during training
    print("\n" + "="*80)
    print("CREATING BACKBONE")
    print("="*80)
    
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    multi_scale_indices = config['model'].get('multi_scale_indices', [2, 5, 8, 11])
    projection_hidden_dims = config['model'].get('projection_hidden_dims', None)
    projection_dim = projection_hidden_dims[-1] if projection_hidden_dims is not None else config['model'].get('projection_dim', None)
    
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=projection_dim,
        pretrained=True,
        multi_scale_indices=multi_scale_indices if use_pixel_decoder else None,
        projection_hidden_dims=projection_hidden_dims
    )
    backbone = backbone.to(device)
    
    # ===== STAGE 2.5: Pre-Train Projection Head + Generate Anchors (if enabled) =====
    from pretrain import pretrain_projection_head
    
    # Create cache directory for pre-trained weights
    cache_dir = Path('./cache/pretrained_projections')
    
    # Pre-train and get anchors (or skip pre-training and generate anchors normally)
    pretrain_anchors = pretrain_projection_head(
        backbone=backbone,
        train_paths=train_paths,
        preprocessor=BMADPreprocessor(target_size=tuple(config['data']['target_size']), normalize_mode=normalize_mode),
        config=config,
        device=device,
        cache_dir=cache_dir,
        force_retrain=False
    )
    
    # ===== STAGE 3: Anchor Setup =====
    preprocessor = BMADPreprocessor(target_size=tuple(config['data']['target_size']), normalize_mode=normalize_mode)
    
    # Use pre-trained anchors if available, otherwise generate new ones
    if pretrain_anchors is not None:
        # Use anchors from pre-training (perfect alignment!)
        anchor_images, anchor_global = pretrain_anchors
        anchor_dense = None
        
        print(f"\n✓ Using anchors from pre-training: {anchor_global.shape}")
        print(f"✓ Perfect alignment between pre-training and main training!")
        
        # Save to current experiment directory
        torch.save({
            'anchor_images': anchor_images,
            'anchor_global': anchor_global,
            'anchor_dense': anchor_dense,
            'source': 'pretraining'
        }, save_dir / 'anchor_embeddings.pt')
        
    elif config['anchor'].get('init_from', None) is not None:
        # Load anchors from another experiment (for learnable anchors)
        init_from = config['anchor']['init_from']
        print(f"\nLoading anchors from: {init_from}")
        init_anchor_path = Path(init_from) / 'anchor_embeddings.pt'
        if not init_anchor_path.exists():
            raise FileNotFoundError(f"Cannot initialize anchors from {init_from}: anchor_embeddings.pt not found")
        
        anchor_data = torch.load(init_anchor_path, weights_only=False)
        if isinstance(anchor_data, dict):
            anchor_global = anchor_data.get('anchor_global', anchor_data.get('global'))
            anchor_dense = anchor_data.get('anchor_dense', anchor_data.get('dense'))
            anchor_images = anchor_data.get('anchor_images', None)
        else:
            anchor_global = anchor_data
            anchor_dense = None
            anchor_images = None
        
        print(f"✓ Loaded anchors: {anchor_global.shape}")
        
        # Save to current experiment directory
        torch.save({
            'anchor_images': anchor_images,
            'anchor_global': anchor_global,
            'anchor_dense': anchor_dense,
            'anchor_metadata': anchor_data.get('anchor_metadata', {}),
            'initialized_from': str(init_from)
        }, save_dir / 'anchor_embeddings.pt')
        
    elif args.skip_anchors and (save_dir / 'anchor_embeddings.pt').exists():
        print("\nLoading existing anchors...")
        anchor_data = torch.load(save_dir / 'anchor_embeddings.pt', weights_only=False)
        anchor_images = anchor_data['anchor_images']
        anchor_global = anchor_data.get('anchor_global', None)
        anchor_dense = anchor_data.get('anchor_dense', None)
        
        # Check if decoupled anchors exist
        if anchor_data.get('anchor_semantic') is not None and anchor_data.get('anchor_geometric') is not None:
            anchor_semantic = anchor_data['anchor_semantic']
            anchor_geometric = anchor_data['anchor_geometric']
            print(f"  \u2713 Loaded decoupled anchors:")
            print(f"    - Semantic ({anchor_semantic.shape[1]}D): {anchor_semantic.shape}")
            print(f"    - Geometric ({anchor_geometric.shape[1]}D): {anchor_geometric.shape}")
        else:
            print(f"  \u2713 Loaded legacy anchors: {anchor_global.shape}")
        
    if pretrain_anchors is None and not (args.skip_anchors and (save_dir / 'anchor_embeddings.pt').exists()) and config['anchor'].get('init_from', None) is None:
        # Generate new anchors only if we didn't get them from pre-training
        # SOLUTION A: Use embedding space generation (semantic clustering)
        use_embedding_space = config['anchor'].get('use_embedding_space', True)
        patch_variant = get_patch_variant(config) if _get_anchor_mode(config) == 'patch' else 'legacy'
        
        if use_embedding_space:
            if _get_anchor_mode(config) == 'patch' and patch_variant == 'location_kmeans':
                print("\n[PATCH/LOCATION_KMEANS] Building same-location centroid bank in frozen DINOv3 patch space")
                anchor_images, anchor_global, anchor_dense = prepare_location_kmeans_anchors(
                    train_images=train_paths,
                    preprocessor=preprocessor,
                    config=config,
                    save_dir=save_dir,
                    backbone=backbone,
                    device=device
                )
                anchor_geometric = None
            else:
                print("\n[EMBEDDING-SPACE] Using semantic anchors generated in frozen DINOv3 space")
                anchor_images, anchor_semantic, anchor_dense, anchor_geometric = prepare_anchors_in_embedding_space(
                    train_images=train_paths,
                    preprocessor=preprocessor,
                    config=config,
                    save_dir=save_dir,
                    backbone=backbone,
                    device=device
                )
                # For compatibility, set anchor_global to semantic anchors
                anchor_global = anchor_semantic
        else:
            print("\n[LEGACY] Using pixel space anchor generation")
            anchor_images, anchor_global, anchor_dense = prepare_anchors(
                train_images=train_paths,
                preprocessor=preprocessor,
                config=config,
                save_dir=save_dir,
                backbone_for_projection=backbone,  # Use MODEL's backbone!
                device=device
            )
    
    # ===== STAGE 4: Complete Model Creation =====
    # Now create the full detector with the backbone and anchors
    learnable_anchors = config['anchor'].get('learnable', False)
    target_size = tuple(config['data']['target_size'])
    anchor_mode = _get_anchor_mode(config)
    
    print("\n" + "="*80)
    print("MODEL CREATION")
    print("="*80)
    
    # EXPERT'S APPROACH: Use decoupled semantic/geometric anchors
    # SOLUTION A: Use legacy re-projection path (anchors_already_projected=False)
    use_embedding_space = config['anchor'].get('use_embedding_space', False)
    reproject_anchors = config['anchor'].get('reproject_anchors', False)
    use_decoupled = use_embedding_space and not reproject_anchors  # Only use decoupled if NOT Solution A
    
    if anchor_mode == 'patch':
        print("PATCH MODE: using separate dense-anchor detector")
        model = create_patch_detector(
            config=config,
            anchor_global=anchor_global,
            anchor_dense=anchor_dense,
            backbone=backbone,
        )
        model = model.to(device)

    elif use_decoupled and 'anchor_geometric' in locals() and anchor_geometric is not None:
        print(f"EXPERT'S APPROACH: Decoupled anchors")
        print(f"  Semantic anchors ({anchor_semantic.shape[1]}D): {anchor_semantic.shape} [for pseudo-labels]")
        print(f"  Geometric targets ({anchor_geometric.shape[1]}D): {anchor_geometric.shape} [FIXED training targets]")
        
        model = AnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_global,  # Legacy compatibility
            anchor_dense_embeddings=anchor_dense,
            distance_metric=config['loss']['distance_metric'],
            learnable_anchors=learnable_anchors,
            use_pixel_decoder=use_pixel_decoder,
            decoder_hidden_dim=config['model'].get('decoder_hidden_dim', 256),
            target_size=target_size,
            anchor_semantic_embeddings=anchor_semantic,
            anchor_geometric_targets=anchor_geometric,
            use_decoupled_anchors=True
        )
        model = model.to(device)
        
    else:
        # LEGACY PATHS: Define anchors_already_projected for non-decoupled approaches
        if use_embedding_space:
            print(f"Anchors in RAW {backbone.embed_dim}D space: {anchor_global.shape}")
            print(f"  ✓ SOLUTION A: Anchors will be RE-PROJECTED each forward pass")
            print(f"  ✓ Anchors move with projection head (semantic clustering preserved)")
            anchors_already_projected = False
        elif projection_dim:
            print(f"Anchors are in PROJECTED space: {anchor_global.shape}")
            print(f"  Projected through MODEL's backbone - same weights as training!")
            print(f"  They are FIXED targets - will NOT be re-projected")
            anchors_already_projected = True
        else:
            print(f"Anchors are in RAW space: {anchor_global.shape}")
            anchors_already_projected = False
        
        model = AnomalyDetector(
            backbone=backbone,
            anchor_global_embeddings=anchor_global,
            anchor_dense_embeddings=anchor_dense,
            distance_metric=config['loss']['distance_metric'],
            learnable_anchors=learnable_anchors,
            use_pixel_decoder=use_pixel_decoder,
            decoder_hidden_dim=config['model'].get('decoder_hidden_dim', 256),
            target_size=target_size,
            anchors_already_projected=anchors_already_projected
        )
        model = model.to(device)
    
    # ===== STAGE 5: Training Setup =====
    criterion = create_criterion(config)
    
    # Get trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # Optimizer
    if len(trainable_params) > 0:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['lr'] * 0.01
        )
    else:
        print("\nNote: No trainable parameters (backbone is frozen, no projection head).")
        print("Skipping training and proceeding directly to evaluation...")
        optimizer = None
        scheduler = None
    
    # ===== STAGE 5: Training =====
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    trainer = None

    if optimizer is not None and not args.eval_only:
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=save_dir,
            use_amp=config['training']['use_amp'],
            log_interval=config['training']['log_interval'],
            val_interval=config['training']['val_interval'],
            fixed_pseudo_labels=config['training'].get('fixed_pseudo_labels', False),
            pseudo_label_assignment=config['training'].get('pseudo_label_assignment', 'nearest'),
            capacity_multiplier=config['training'].get('capacity_multiplier', 2.0),
            dynamic_reassignment=config['training'].get('dynamic_reassignment', False),
            reassignment_interval=config['training'].get('reassignment_interval', 5),
            save_checkpoints=config['training'].get('save_checkpoints', True)
        )
        
        trainer.train(
            num_epochs=config['training']['epochs'],
            scheduler=scheduler,
            early_stopping_patience=config['training']['early_stopping_patience'],
            min_epochs_before_early_stopping=config['training'].get('min_epochs_before_early_stopping', 0)
        )

        if config.get('stage2', {}).get('enabled', False):
            # ---- Intermediate evaluation: report stage 1 performance ----
            print("\n" + "="*80)
            print("STAGE 1 EVALUATION (before stage 2)")
            print("="*80)

            best_model_path = save_dir / 'best_model.pth'
            if best_model_path.exists():
                _load_model_checkpoint(model, best_model_path, device)
                print("Loaded best stage-1 checkpoint for intermediate evaluation")

            stage1_eval_dir = save_dir / 'evaluation_stage1'
            stage1_eval_dir.mkdir(exist_ok=True)
            stage1_results = evaluate_comprehensive(
                model=model,
                dataloader=test_loader,
                device=device,
                save_dir=stage1_eval_dir,
                compute_pixel=False,
                target_size=tuple(config['data']['target_size'])
            )
            print(f"Stage 1 Image AUROC: {stage1_results['image_auroc']:.4f}")

            # ---- Stage 2 training ----
            print("\n" + "="*80)
            print("STAGE 2: RECONSTRUCTION TRAINING")
            print("="*80)

            # Reload best stage-1 checkpoint for stage-2 initialization
            if best_model_path.exists():
                _load_model_checkpoint(model, best_model_path, device)
                print("Loaded best stage-1 checkpoint for stage-2 initialization")

            stage2_cfg = config['stage2']
            pixel_map_cfg = stage2_cfg.get('pixel_map', {})
            score_comb_cfg = stage2_cfg.get('score_combination', {})
            use_frozen_bottleneck = stage2_cfg.get('frozen_bottleneck', False)
            recon_proj_dim = config.get('model', {}).get('projection_dim_recon', None)

            model.enable_reconstruction_branch(
                freeze_anchor_target=stage2_cfg.get('freeze_anchor_target', True),
                out_channels=3,
                pixel_map_enabled=pixel_map_cfg.get('enabled', True),
                pixel_map_type=pixel_map_cfg.get('type', 'reconstruction_l2'),
                use_frozen_bottleneck=use_frozen_bottleneck,
                recon_projection_dim=recon_proj_dim,
                no_fuser=stage2_cfg.get('no_fuser', False)
            )
            model.configure_score_combination(
                enabled=score_comb_cfg.get('enabled', False),
                alpha=score_comb_cfg.get('alpha', 0.5),
                normalization=score_comb_cfg.get('normalization', 'minmax')
            )

            # Configure pixel aggregation
            pix_agg_cfg = stage2_cfg.get('pixel_aggregation', {})
            agg_method = pix_agg_cfg.get('method', 'top_k_percentile')
            agg_threshold = pix_agg_cfg.get('threshold_n_std', 2.0) if agg_method == 'threshold_ratio' else None
            model.configure_pixel_aggregation(
                method=agg_method,
                percentile=pix_agg_cfg.get('percentile', 95),
                threshold=agg_threshold
            )

            # Configure three-signal score fusion
            fusion_cfg = stage2_cfg.get('score_fusion', {})
            model.configure_score_fusion(
                enabled=fusion_cfg.get('enabled', False),
                normalization=fusion_cfg.get('normalization', 'minmax'),
                anchor_weight=fusion_cfg.get('anchor_weight', 0.4),
                divergence_weight=fusion_cfg.get('divergence_weight', 0.3),
                pixel_weight=fusion_cfg.get('pixel_weight', 0.3),
                drop_anticorrelated=fusion_cfg.get('drop_anticorrelated', True)
            )

            # Prepare stage-2 training with configurable encoder freezing
            freeze_encoder_mode = stage2_cfg.get('freeze_encoder_mode', 'full')
            # Backward compat: if old freeze_encoder=true is set, treat as 'full'
            if stage2_cfg.get('freeze_encoder', True) and freeze_encoder_mode == 'full':
                freeze_encoder_mode = 'full'
            elif not stage2_cfg.get('freeze_encoder', True):
                freeze_encoder_mode = 'none'

            model.prepare_stage2_training(
                freeze_encoder=True,  # always pass True; mode handles the rest
                freeze_anchor_parameters=stage2_cfg.get('freeze_anchors', True),
                freeze_encoder_mode=freeze_encoder_mode,
                unfreeze_last_n_blocks=stage2_cfg.get('unfreeze_last_n_blocks', 2),
                unfreeze_lr_multiplier=stage2_cfg.get('unfreeze_lr_multiplier', 0.1)
            )

            # Build optimizer with param groups (different LR for unfrozen encoder)
            stage2_base_lr = stage2_cfg.get('lr', config['training']['lr'])
            stage2_wd = stage2_cfg.get('weight_decay', config['training']['weight_decay'])
            stage2_param_groups = model.get_stage2_param_groups(
                base_lr=stage2_base_lr,
                weight_decay=stage2_wd
            )

            if len(stage2_param_groups) == 0 or all(len(g['params']) == 0 for g in stage2_param_groups):
                raise RuntimeError("Stage-2 enabled but no stage-2 trainable parameters were found.")

            stage2_optimizer = torch.optim.AdamW(
                stage2_param_groups
            )

            stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                stage2_optimizer,
                T_max=stage2_cfg.get('epochs', 20),
                eta_min=stage2_cfg.get('lr', config['training']['lr']) * 0.01
            )

            stage2_trainer = Trainer(
                model=model,
                criterion=None,
                optimizer=stage2_optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                save_dir=save_dir,
                use_amp=config['training']['use_amp'],
                log_interval=config['training']['log_interval'],
                val_interval=config['training']['val_interval'],
                fixed_pseudo_labels=config['training'].get('fixed_pseudo_labels', False),
                pseudo_label_assignment=config['training'].get('pseudo_label_assignment', 'nearest'),
                capacity_multiplier=config['training'].get('capacity_multiplier', 2.0),
                dynamic_reassignment=False,
                reassignment_interval=stage2_cfg.get('reassignment_interval', 5),
                save_checkpoints=config['training'].get('save_checkpoints', True),
                stage2_mode=True,
                stage2_config=stage2_cfg
            )

            # Preserve stage-1 history and append stage-2 metrics on top
            if trainer is not None:
                stage2_trainer.history = copy.deepcopy(trainer.history)

            stage2_trainer.train_stage2(
                num_epochs=stage2_cfg.get('epochs', 20),
                scheduler=stage2_scheduler,
                early_stopping_patience=stage2_cfg.get('early_stopping_patience', 10)
            )

            # Prefer best stage-2 model for final evaluation when available
            best_stage2_path = save_dir / 'best_stage2_model.pth'
            if best_stage2_path.exists():
                _load_model_checkpoint(model, best_stage2_path, device)
                print("Loaded best stage-2 checkpoint for final evaluation")
            trainer = stage2_trainer
    else:
        if optimizer is None:
            print("Skipping training (no trainable parameters)")
        else:
            print("Skipping training (--eval-only flag)")
        
        # Save model anyway for evaluation
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'anchor_mode': _get_model_anchor_mode(model)
        }, save_dir / 'best_model.pth')
    
    # ===== STAGE 6: Evaluation =====
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # When eval-only and stage-2 is configured, enable reconstruction branch
    # before loading the checkpoint (the checkpoint has stage-2 keys)
    stage2_enabled = config.get('stage2', {}).get('enabled', False)
    best_stage2_path = save_dir / 'best_stage2_model.pth'
    if args.eval_only and stage2_enabled and best_stage2_path.exists():
        s2_cfg = config['stage2']
        pm_cfg = s2_cfg.get('pixel_map', {})
        if not model.reconstruction_enabled:
            recon_proj_dim = config.get('model', {}).get('projection_dim_recon', None)
            model.enable_reconstruction_branch(
                freeze_anchor_target=s2_cfg.get('freeze_anchor_target', True),
                out_channels=3,
                pixel_map_enabled=pm_cfg.get('enabled', True),
                pixel_map_type=pm_cfg.get('type', 'reconstruction_l2'),
                use_frozen_bottleneck=s2_cfg.get('frozen_bottleneck', False),
                recon_projection_dim=recon_proj_dim,
                no_fuser=s2_cfg.get('no_fuser', False)
            )
        pix_agg_cfg = s2_cfg.get('pixel_aggregation', {})
        agg_method = pix_agg_cfg.get('method', 'top_k_percentile')
        agg_threshold = pix_agg_cfg.get('threshold_n_std', 2.0) if agg_method == 'threshold_ratio' else None
        model.configure_pixel_aggregation(
            method=agg_method,
            percentile=pix_agg_cfg.get('percentile', 95),
            threshold=agg_threshold
        )
        fusion_cfg = s2_cfg.get('score_fusion', {})
        model.configure_score_fusion(
            enabled=fusion_cfg.get('enabled', False),
            normalization=fusion_cfg.get('normalization', 'minmax'),
            anchor_weight=fusion_cfg.get('anchor_weight', 0.4),
            divergence_weight=fusion_cfg.get('divergence_weight', 0.3),
            pixel_weight=fusion_cfg.get('pixel_weight', 0.3)
        )
        sc_cfg = s2_cfg.get('score_combination', {})
        model.configure_score_combination(
            enabled=sc_cfg.get('enabled', False),
            alpha=sc_cfg.get('alpha', 0.5),
            normalization=sc_cfg.get('normalization', 'minmax')
        )
        print("  Stage-2 reconstruction branch enabled for eval-only loading")

    # Load best model
    if getattr(args, 'checkpoint', None):
        best_model_path = Path(args.checkpoint)
    else:
        best_model_path = best_stage2_path if (stage2_enabled and best_stage2_path.exists()) else save_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = _load_model_checkpoint(model, best_model_path, device)
        if 'epoch' in checkpoint:
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print(f"Loaded model checkpoint")
    
    # Comprehensive evaluation
    eval_dir = save_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    
    compute_pixel = config.get('eval', {}).get('compute_pixel', False)
    results = evaluate_comprehensive(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir,
        compute_pixel=compute_pixel,
        target_size=tuple(config['data']['target_size'])
    )
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir,
        num_samples=16,
        target_size=tuple(config['data']['target_size'])
    )
    
    analyze_anchor_assignments(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir
    )

    # Generate train/test visualization snapshots using best_model.pth explicitly
    print("\nGenerating best-model visualizations...")
    if trainer is not None:
        best_stage1_path = save_dir / 'best_model.pth'
        if best_stage1_path.exists():
            _load_model_checkpoint(model, best_stage1_path, device, strict=False)
            print("Loaded best_model.pth for visualization snapshots")

        trainer._visualize_training_samples(epoch=0, save_name='train_best_model')
        trainer._visualize_test_samples(test_loader=test_loader, save_name='test_best_model')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Results saved to: {save_dir}")
    print(f"Best Image AUROC: {results['image_auroc']:.4f}")
    if 'pixel_auroc' in results:
        print(f"Best Pixel AUROC: {results['pixel_auroc']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BMAD anomaly detector')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--skip-anchors', action='store_true',
                        help='Skip anchor generation if already exists')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on existing model')
    parser.add_argument('--auto-name', action='store_true',
                        help='Auto-generate experiment name from anchor config (strategy_k<n_anchors>)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Explicit experiment name subfolder (overrides auto-name)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path for eval-only mode (absolute or relative path)')
    
    args = parser.parse_args()
    main(args)