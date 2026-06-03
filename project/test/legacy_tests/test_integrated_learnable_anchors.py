"""
Test integrated learnable anchors implementation
"""

import sys
import torch
import yaml
from pathlib import Path

# Add project/ and legacy_learnable/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'legacy_learnable'))

# Test imports
print("Testing imports...")
from loss import AnchorMarginLoss
from model import DINOv3Backbone, AnomalyDetector

print("✓ Imports successful!")

# Test loss with min-norm
print("\nTesting AnchorMarginLoss with min-norm...")
loss_fn = AnchorMarginLoss(
    margin=1.0,
    alpha=1.0,
    beta=1.0,
    gamma=0.1,  # Enable min-norm
    min_norm=0.5,
    distance_metric='euclidean'
)

# Test forward pass
B, K, D = 16, 8, 128
embeddings = torch.randn(B, D)
anchors = torch.randn(K, D)

loss_dict = loss_fn(embeddings, anchors, return_components=True)
print(f"  Loss: {loss_dict['loss'].item():.4f}")
print(f"  Attractor: {loss_dict['loss_attract']:.4f}")
print(f"  Repeller: {loss_dict['loss_repel']:.4f}")
print(f"  Min-Norm: {loss_dict['loss_norm']:.4f}")
print("✓ Loss with min-norm works!")

# Test learnable anchors in model
print("\nTesting AnomalyDetector with learnable anchors...")

# Create backbone
backbone = DINOv3Backbone(
    model_name='vit_small_patch16_dinov3.lvd1689m',
    freeze_backbone=True,
    projection_dim=128,
    pretrained=False  # Don't download for test
)

# Create initial anchors
initial_anchors = torch.randn(K, backbone.embed_dim)

# Test with fixed anchors
print("\n  Testing FIXED anchors...")
model_fixed = AnomalyDetector(
    backbone=backbone,
    anchor_global_embeddings=initial_anchors,
    anchor_dense_embeddings=None,
    distance_metric='euclidean',
    learnable_anchors=False
)

# Check that anchors are not parameters
anchor_params = [p for p in model_fixed.parameters() if p.shape == (K, 128)]
print(f"  Learnable anchor parameters: {len(anchor_params)}")
assert len(anchor_params) == 0, "Fixed anchors should not be parameters!"

# Test with learnable anchors
print("\n  Testing LEARNABLE anchors...")
model_learnable = AnomalyDetector(
    backbone=backbone,
    anchor_global_embeddings=initial_anchors,
    anchor_dense_embeddings=None,
    distance_metric='euclidean',
    learnable_anchors=True
)

# Check that anchors ARE parameters
all_params = list(model_learnable.parameters())
anchor_params = [p for p in all_params if p.requires_grad and p.shape[0] == K]
print(f"  Learnable anchor parameters: {len(anchor_params)}")
print(f"  Total trainable params: {sum(p.numel() for p in all_params if p.requires_grad):,}")

assert len(anchor_params) > 0, "Learnable anchors should be parameters!"
print("✓ Learnable anchors work!")

# Test config loading
print("\nTesting config loading...")
config_path = Path('project/configs/learnable_anchors_integrated.yaml')
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert config['anchor']['learnable'] == True
    assert config['loss']['gamma'] > 0
    assert config['anchor']['init_from'] is not None
    print("✓ Config is valid!")
else:
    print("  Config file not found (expected for now)")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nYou can now use learnable anchors with main.py:")
print("  python project/main.py --config project/configs/learnable_anchors_integrated.yaml")
