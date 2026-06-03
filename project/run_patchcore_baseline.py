"""Standalone runner for the frozen DINOv3 PatchCore baseline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import BMADDataset, BMADPreprocessor, create_dataloaders
from eval import evaluate_comprehensive
from main import load_config, load_dataset_paths, make_unique_dir, set_seed
from model import DINOv3Backbone
from patchcore_baseline import PatchCoreBaseline


def _build_memory_loader(
    train_paths: list[str],
    *,
    target_size: tuple[int, int],
    normalize_mode: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Create a non-augmented, non-dropping loader for the PatchCore bank."""
    preprocessor = BMADPreprocessor(target_size=target_size, normalize_mode=normalize_mode)
    dataset = BMADDataset(
        image_paths=train_paths,
        labels=None,
        preprocessor=preprocessor,
        augment=False,
        is_training=False,
        normalize_mode=normalize_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _save_json(path: Path, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    set_seed(config['seed'])

    save_dir = make_unique_dir(Path(config['output_dir']))
    config['output_dir'] = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'config.yaml', 'w', encoding='utf-8') as handle:
        yaml.dump(config, handle)

    device = _resolve_device(args.device)
    target_size = tuple(config['data']['target_size'])
    normalize_mode = config['data'].get('normalization', 'zscore_only')

    print('=' * 80)
    print('FROZEN DINOV3 PATCHCORE BASELINE')
    print('=' * 80)
    print(f'Output directory: {save_dir}')
    print(f'Config: {args.config}')
    print(f'Device: {device}')

    train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(
        config['data'].get('data_root', './data/BraTS2021_slice')
    )

    if not train_paths:
        raise FileNotFoundError('No training images found for PatchCore baseline.')

    max_train = config['data'].get('max_train_samples')
    if max_train is not None and max_train < len(train_paths):
        rng = random.Random(config.get('seed', 42))
        train_paths = sorted(rng.sample(train_paths, max_train))
        print(f'Limiting memory-bank training set to {len(train_paths)} images')

    batch_size = int(config['training'].get('batch_size', 64))
    num_workers = int(config['training'].get('num_workers', 4))

    _, val_loader, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        val_mask_paths=val_mask_paths,
        test_mask_paths=test_mask_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=target_size,
        normalize_mode=normalize_mode,
        train_augment_mode=config['data'].get('train_augment_mode', 'full'),
    )
    memory_loader = _build_memory_loader(
        train_paths,
        target_size=target_size,
        normalize_mode=normalize_mode,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=True,
        projection_dim=None,
        pretrained=True,
        multi_scale_indices=None,
    ).to(device)

    patchcore_cfg = config.get('patchcore', {})
    memory_cfg = patchcore_cfg.get('memory_bank', {})
    score_cfg = patchcore_cfg.get('image_score', {})

    model = PatchCoreBaseline(
        backbone=backbone,
        n_neighbors=int(patchcore_cfg.get('n_neighbors', 1)),
        distance_metric=patchcore_cfg.get('distance_metric', 'euclidean'),
        patches_per_image=memory_cfg.get('patches_per_image'),
        max_memory_bank_size=memory_cfg.get('max_features'),
        query_batch_size=int(patchcore_cfg.get('query_batch_size', 4096)),
        aggregation_method=score_cfg.get('method', 'top_k_percentile'),
        aggregation_percentile=float(score_cfg.get('percentile', 95.0)),
        aggregation_threshold=score_cfg.get('threshold'),
        neighbor_reduction=patchcore_cfg.get('neighbor_reduction', 'mean'),
        random_state=int(config.get('seed', 42)),
    ).to(device)

    fit_summary = model.fit(
        memory_loader,
        device=device,
        max_images=args.memory_max_images or memory_cfg.get('max_images'),
    )
    _save_json(save_dir / 'patchcore_fit_summary.json', asdict(fit_summary))

    if patchcore_cfg.get('save_memory_bank', True):
        torch.save(
            {
                'memory_bank': model.memory_bank,
                'fit_summary': asdict(fit_summary),
                'distance_metric': model.distance_metric,
                'n_neighbors': model.n_neighbors,
            },
            save_dir / 'patchcore_memory_bank.pt',
        )

    compute_pixel = bool(config.get('eval', {}).get('compute_pixel', True))
    print('\nEvaluating on validation split...')
    evaluate_comprehensive(
        model,
        val_loader,
        device,
        str(save_dir / 'validation'),
        compute_pixel=compute_pixel,
        target_size=target_size,
    )

    print('\nEvaluating on test split...')
    evaluate_comprehensive(
        model,
        test_loader,
        device,
        str(save_dir / 'evaluation'),
        compute_pixel=compute_pixel,
        target_size=target_size,
    )

    return save_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the frozen DINOv3 PatchCore baseline.')
    parser.add_argument('--config', required=True, help='Path to the baseline YAML config.')
    parser.add_argument('--device', default=None, help='Optional torch device override, e.g. cuda or cpu.')
    parser.add_argument(
        '--memory-max-images',
        type=int,
        default=None,
        help='Optional override for the number of train images used to fit the memory bank.',
    )
    return parser


if __name__ == '__main__':
    run(build_arg_parser().parse_args())