"""
Test script to debug pixel AUROC computation during validation
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import AnomalyDetector
from data import create_dataloaders
from eval import evaluate_model
from glob import glob
import yaml

# Load config
config_path = Path('experiments/bmad_fixed/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load validation data
val_dir = Path('../data/BraTS2021_slice/valid')
good_imgs = sorted(glob(str(val_dir / 'good/img/*.png')))
good_masks = sorted(glob(str(val_dir / 'good/label/*.png')))
good_labels = [0] * len(good_imgs)

ungood_imgs = sorted(glob(str(val_dir / 'Ungood/img/*.png')))
ungood_masks = sorted(glob(str(val_dir / 'Ungood/label/*.png')))
ungood_labels = [1] * len(ungood_imgs)

val_imgs = good_imgs + ungood_imgs
val_masks = good_masks + ungood_masks
val_labels = good_labels + ungood_labels

print(f"Validation set: {len(val_imgs)} images ({len(good_imgs)} normal, {len(ungood_imgs)} anomalous)")
print(f"Masks: {len(val_masks)}")

# Create minimal dataloaders (just need val)
train_imgs = good_imgs[:10]  # Dummy
_, val_loader, _ = create_dataloaders(
    train_paths=train_imgs,
    val_paths=val_imgs,
    val_labels=val_labels,
    test_paths=val_imgs[:5],
    test_labels=val_labels[:5],
    val_mask_paths=val_masks,
    test_mask_paths=val_masks[:5],
    batch_size=16,
    num_workers=0,
    target_size=(240, 240)
)

# Load model - use main.py logic
from main import setup_experiment
import argparse

checkpoint_path = Path('experiments/bmad_fixed/best_model.pth')

# Create args namespace
args = argparse.Namespace(
    config=str(config_path),
    resume=str(checkpoint_path),
    eval_only=False
)

# Setup will load the model with anchors
model, _, _, _, _, config = setup_experiment(args)
device = model.backbone.patch_embed.proj.weight.device  # Get device from model
model.eval()

print(f"\nLoaded model from checkpoint")
print(f"Model has {model.n_anchors} anchors")
print(f"Device: {device}")

# Test evaluation
print("\n" + "="*60)
print("Running evaluation with pixel AUROC computation...")
print("="*60)

metrics = evaluate_model(
    model=model,
    dataloader=val_loader,
    device=device,
    compute_pixel_auroc=True,
    target_size=(240, 240)
)

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

if 'pixel_auroc' in metrics:
    print("\n✓ Pixel AUROC successfully computed!")
else:
    print("\n✗ Pixel AUROC NOT computed - check debug output above")
