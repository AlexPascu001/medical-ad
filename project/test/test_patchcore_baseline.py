"""Contract smoke tests for the standalone PatchCore baseline."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from patchcore_baseline import PatchCoreBaseline


class DummyDenseBackbone(nn.Module):
    """Small deterministic backbone for interface tests."""

    def __init__(self, feature_dim: int = 8):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor):
        pooled = torch.nn.functional.interpolate(x, size=(4, 4), mode='bilinear', align_corners=False)
        dense = pooled[:, :1].permute(0, 2, 3, 1).repeat(1, 1, 1, self.feature_dim)
        dense = dense + torch.linspace(0.0, 0.07, self.feature_dim, device=x.device).view(1, 1, 1, -1)
        return {'dense': dense}


class DummyImageDataset(Dataset):
    def __init__(self, length: int = 5):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return {
            'image': torch.randn(3, 32, 32),
            'label': torch.tensor(0),
            'path': f'dummy_{idx}.png',
        }


def test_patchcore_baseline_contract():
    backbone = DummyDenseBackbone(feature_dim=8)
    baseline = PatchCoreBaseline(
        backbone,
        n_neighbors=1,
        aggregation_method='top_k_percentile',
        aggregation_percentile=95.0,
        query_batch_size=8,
    )

    memory_bank = torch.randn(24, 8)
    baseline.set_memory_bank(memory_bank)

    images = torch.randn(2, 3, 32, 32)
    outputs = baseline.compute_anomaly_scores(images, return_maps=True, target_size=(32, 32))

    assert outputs['image_scores'].shape == (2,)
    assert outputs['anchor_scores'].shape == (2,)
    assert outputs['pixel_aggregated_score'].shape == (2,)
    assert outputs['pixel_scores'].shape == (2, 32, 32)
    assert outputs['anchor_pixel_scores'].shape == (2, 32, 32)
    assert outputs['pixel_scores_source'] == 'patchcore_patch_knn'


def test_patchcore_fit_smoke():
    backbone = DummyDenseBackbone(feature_dim=8)
    baseline = PatchCoreBaseline(
        backbone,
        n_neighbors=1,
        patches_per_image=4,
        max_memory_bank_size=12,
        query_batch_size=8,
    )

    loader = DataLoader(DummyImageDataset(length=5), batch_size=2, shuffle=False)
    summary = baseline.fit(loader, device=torch.device('cpu'))

    assert summary.num_images == 5
    assert summary.feature_dim == 8
    assert summary.num_features_after_cap <= 12
    assert baseline.memory_bank is not None
    assert baseline.memory_bank.shape[1] == 8


if __name__ == '__main__':
    test_patchcore_baseline_contract()
    test_patchcore_fit_smoke()
    print('PatchCore baseline contract smoke test passed.')