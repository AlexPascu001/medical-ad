"""
Debug script to investigate why anchors appear far from samples in embedding space.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Load experiment
exp_dir = Path("experiments/bmad_fixed_random/test_viz")

# Load anchor embeddings
anchor_data = torch.load(exp_dir / "anchor_embeddings.pt", weights_only=False)
print("=== Loaded anchor_embeddings.pt ===")
print(f"anchor_images shape: {anchor_data['anchor_images'].shape}")
print(f"anchor_global shape: {anchor_data['anchor_global'].shape}")
print(f"anchor_dense shape: {anchor_data['anchor_dense'].shape}")

# Check anchor norms
anchor_global_raw = anchor_data['anchor_global']
print(f"\n=== Raw anchor embeddings (from file) ===")
print(f"Shape: {anchor_global_raw.shape}")
print(f"Norms: {torch.norm(anchor_global_raw, dim=1)}")
print(f"Mean norm: {torch.norm(anchor_global_raw, dim=1).mean():.4f}")

# Load best model checkpoint
checkpoint = torch.load(exp_dir / "best_model.pth", map_location='cpu', weights_only=False)
print(f"\n=== Checkpoint keys: {checkpoint.keys()} ===")

# Check model state dict for anchor embeddings
state_dict = checkpoint['model_state_dict']
anchor_keys = [k for k in state_dict.keys() if 'anchor' in k.lower()]
print(f"\nAnchor-related keys in state_dict: {anchor_keys}")

for key in anchor_keys:
    tensor = state_dict[key]
    print(f"\n{key}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Norms: {torch.norm(tensor, dim=-1) if tensor.dim() > 1 else tensor.norm()}")

# Check distances between anchors
if 'anchor_global' in state_dict:
    anchors = state_dict['anchor_global']
elif 'detector.anchor_global' in state_dict:
    anchors = state_dict['detector.anchor_global']
else:
    # Find any anchor-like key
    for k in state_dict:
        if 'anchor' in k and 'global' in k:
            anchors = state_dict[k]
            break
    else:
        anchors = anchor_global_raw

print(f"\n=== Anchor-to-Anchor distances ===")
for i in range(anchors.shape[0]):
    for j in range(i+1, anchors.shape[0]):
        dist = torch.norm(anchors[i] - anchors[j])
        cosine_sim = torch.dot(anchors[i], anchors[j]) / (anchors[i].norm() * anchors[j].norm())
        print(f"Anchor {i} <-> {j}: L2={dist:.4f}, Cosine={cosine_sim:.4f}")

# Now load a few training samples and compare
print("\n\n=== Loading model and computing sample embeddings ===")

# Import model components
from model import DINOv3Backbone, AnomalyDetector
from data import BMADPreprocessor, create_dataloaders
import yaml

# Load config
with open(exp_dir / "config.yaml") as f:
    config = yaml.safe_load(f)

# Create backbone
backbone = DINOv3Backbone(
    model_name=config['model']['backbone'],
    freeze_backbone=True,
    projection_dim=config['model'].get('projection_dim', None),
    pretrained=True
)
backbone.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = backbone.to(device)

# Load a few training samples
data_dir = config['data']['dataset_dir']
_, train_loader, _, _ = create_dataloaders(
    data_dir,
    data_dir,  
    data_dir,
    batch_size=64,
    num_workers=0
)

# Get one batch
batch = next(iter(train_loader))
images = batch['image'].to(device)

with torch.no_grad():
    outputs = backbone(images)
    sample_embeddings = outputs['global']

print(f"\n=== Sample embeddings ===")
print(f"Shape: {sample_embeddings.shape}")
print(f"Norms: min={sample_embeddings.norm(dim=1).min():.4f}, max={sample_embeddings.norm(dim=1).max():.4f}, mean={sample_embeddings.norm(dim=1).mean():.4f}")

# Check if anchors were projected
anchors_in_model = anchors.to(device)
print(f"\n=== Anchors in model ===")
print(f"Shape: {anchors_in_model.shape}")
print(f"Norms: {anchors_in_model.norm(dim=1)}")

# Compare dimensions
print(f"\n=== Dimension check ===")
print(f"Sample dim: {sample_embeddings.shape[1]}")
print(f"Anchor dim: {anchors_in_model.shape[1]}")

if sample_embeddings.shape[1] != anchors_in_model.shape[1]:
    print("ERROR: Dimension mismatch!")
else:
    # Compute distances from samples to anchors
    print(f"\n=== Sample-to-Anchor distances ===")
    distances = torch.cdist(sample_embeddings, anchors_in_model)
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Min distance per sample: {distances.min(dim=1)[0].mean():.4f}")
    print(f"Max distance per sample: {distances.max(dim=1)[0].mean():.4f}")
    print(f"Mean distance: {distances.mean():.4f}")
    
    # Check anchor assignments
    assignments = distances.argmin(dim=1)
    print(f"\n=== Anchor assignments (first batch) ===")
    for i in range(8):
        count = (assignments == i).sum().item()
        print(f"Anchor {i}: {count} samples ({count/len(assignments)*100:.1f}%)")

# Now compare with projection applied manually
print(f"\n\n=== Testing projection manually ===")
if backbone.projection is not None:
    # Project raw anchors
    raw_anchors = anchor_data['anchor_global'].to(device)
    print(f"Raw anchor dim: {raw_anchors.shape}")
    
    with torch.no_grad():
        projected_anchors = backbone.projection(raw_anchors)
        projected_anchors = torch.nn.functional.normalize(projected_anchors, dim=1)
    
    print(f"Projected anchor dim: {projected_anchors.shape}")
    print(f"Projected anchor norms: {projected_anchors.norm(dim=1)}")
    
    # Compare with stored anchors
    print(f"\n=== Comparison: stored vs freshly projected ===")
    diff = (anchors_in_model - projected_anchors).norm()
    print(f"L2 difference: {diff:.6f}")
    
    # Distances with projected
    distances_proj = torch.cdist(sample_embeddings, projected_anchors)
    print(f"\nDistances to freshly projected anchors:")
    print(f"Mean: {distances_proj.mean():.4f}")
else:
    print("No projection layer found")

# Final check: are anchors normalized?
print(f"\n=== Final normalization check ===")
print(f"Sample norms (should be ~1.0): {sample_embeddings.norm(dim=1)[:5]}")
print(f"Anchor norms (should be ~1.0): {anchors_in_model.norm(dim=1)}")
