"""
Debug script to check if validation masks are being loaded correctly
"""
import torch
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data import create_dataloaders
from glob import glob

# Load validation paths
val_dir = Path('../data/BraTS2021_slice/valid')

# Good samples (normal)
good_imgs = sorted(glob(str(val_dir / 'good/img/*.png')))
good_masks = sorted(glob(str(val_dir / 'good/label/*.png')))
good_labels = [0] * len(good_imgs)

# Ungood samples (anomalies)
ungood_imgs = sorted(glob(str(val_dir / 'Ungood/img/*.png')))
ungood_masks = sorted(glob(str(val_dir / 'Ungood/label/*.png')))
ungood_labels = [1] * len(ungood_imgs)

# Combine
val_imgs = good_imgs + ungood_imgs
val_masks = good_masks + ungood_masks
val_labels = good_labels + ungood_labels

print(f"Validation set:")
print(f"  Total images: {len(val_imgs)}")
print(f"  Total masks: {len(val_masks)}")
print(f"  Normal samples: {sum(1 for l in val_labels if l == 0)}")
print(f"  Anomaly samples: {sum(1 for l in val_labels if l == 1)}")
print()

# Create dummy dataloaders (just create validation dataset directly)
from data import BMADDataset, BMADPreprocessor

preprocessor = BMADPreprocessor()
val_dataset = BMADDataset(
    image_paths=val_imgs,
    labels=val_labels,
    mask_paths=val_masks,
    preprocessor=preprocessor,
    augment=False,
    is_training=False
)

from torch.utils.data import DataLoader
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Check first few batches
print("Checking validation batches:")
masks_with_anomalies = 0
total_masks = 0
batches_checked = 0

for batch_idx, batch in enumerate(val_loader):
    if batch_idx >= 3:  # Check first 3 batches
        break
    
    batches_checked += 1
    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Labels: {batch['label'].tolist()}")
    
    if 'mask' in batch:
        masks = batch['mask']
        print(f"  Masks shape: {masks.shape}")
        
        for i, mask in enumerate(masks):
            total_masks += 1
            num_anomaly_pixels = (mask > 0).sum().item()
            if num_anomaly_pixels > 0:
                masks_with_anomalies += 1
                print(f"    Sample {i}: {num_anomaly_pixels} anomaly pixels (label={batch['label'][i].item()})")
            else:
                print(f"    Sample {i}: 0 anomaly pixels (label={batch['label'][i].item()})")
    else:
        print("  NO MASKS IN BATCH!")

print(f"\n" + "="*60)
print(f"Summary: {masks_with_anomalies}/{total_masks} masks have anomalous pixels")
print(f"Batches checked: {batches_checked}")
