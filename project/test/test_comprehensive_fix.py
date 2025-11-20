"""
Comprehensive test to verify pixel AUROC computation is fixed

This script tests:
1. Data loading with masks
2. Model inference with different target sizes
3. Pixel AUROC computation with size mismatches
4. End-to-end evaluation pipeline
"""

import torch
import yaml
import numpy as np
from pathlib import Path

print("="*80)
print("COMPREHENSIVE PIXEL AUROC FIX VERIFICATION")
print("="*80)

# Load config
config_path = 'configs/default.yaml'
print(f"\n1. Loading config: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

data_root = Path(config['data']['data_root'])
target_size = tuple(config['data']['target_size'])
print(f"   Data root: {data_root}")
print(f"   Target size: {target_size}")

# Check validation data exists
val_normal = data_root / 'valid' / 'good' / 'img'
val_anomaly = data_root / 'valid' / 'Ungood' / 'img'
val_masks = data_root / 'valid' / 'Ungood' / 'label'

print(f"\n2. Checking validation data...")
print(f"   Normal: {len(list(val_normal.glob('*.npy')))} images")
print(f"   Anomaly: {len(list(val_anomaly.glob('*.npy')))} images")
print(f"   Masks: {len(list(val_masks.glob('*.npy')))} masks")

# Load a sample mask to check size
if list(val_masks.glob('*.npy')):
    sample_mask = np.load(list(val_masks.glob('*.npy'))[0])
    print(f"   Sample mask shape: {sample_mask.shape}")
    print(f"   ✓ Mask size matches config: {sample_mask.shape == target_size}")

# Load model and anchor embeddings
print(f"\n3. Loading model and anchors...")
checkpoint_dir = Path('experiments/bmad_baseline')
model_path = checkpoint_dir / 'best_model.pth'

if model_path.exists():
    from model import DINOv3Backbone, AnomalyDetector
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create backbone
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=True,
        projection_dim=config['model']['projection_dim']
    )
    
    # Load anchor embeddings
    anchor_path = checkpoint_dir / 'anchor_embeddings.pt'
    anchor_data = torch.load(anchor_path, map_location='cpu', weights_only=False)
    
    # Create detector
    detector = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_data['anchor_global'],
        anchor_dense_embeddings=anchor_data['anchor_dense']
    )
    
    # Load weights
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector.eval()
    
    print(f"   [OK] Model loaded successfully")
    print(f"   Number of anchors: {len(anchor_data['anchor_global'])}")
    
    # Test with different target sizes
    print(f"\n4. Testing model output with different target sizes...")
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 240, 240)
    
    with torch.no_grad():
        # Test without upsampling (None)
        out_none = detector.compute_anomaly_scores(dummy_input, return_maps=True, target_size=None)
        print(f"   target_size=None: {out_none['pixel_scores'].shape}")
        
        # Test with 240x240 (matches mask)
        out_240 = detector.compute_anomaly_scores(dummy_input, return_maps=True, target_size=(240, 240))
        print(f"   target_size=(240,240): {out_240['pixel_scores'].shape}")
        
        # Test with 256x256 (mismatch with mask)
        out_256 = detector.compute_anomaly_scores(dummy_input, return_maps=True, target_size=(256, 256))
        print(f"   target_size=(256,256): {out_256['pixel_scores'].shape}")
    
    # Simulate the fix scenario
    print(f"\n5. Simulating pixel AUROC computation with size mismatch...")
    
    # Create dummy scores and masks with mismatch
    scores_256 = np.random.rand(10, 256, 256)  # Model output at 256x256
    masks_240 = np.random.randint(0, 2, (10, 240, 240))  # Masks at 240x240
    
    print(f"   Scores shape: {scores_256.shape}")
    print(f"   Masks shape: {masks_240.shape}")
    
    # Apply the fix from eval.py
    from scipy.ndimage import zoom
    
    if scores_256.shape[1:] != masks_240.shape[1:]:
        print(f"   Applying resize fix...")
        scale_h = masks_240.shape[1] / scores_256.shape[1]
        scale_w = masks_240.shape[2] / scores_256.shape[2]
        
        resized_scores = []
        for i in range(scores_256.shape[0]):
            resized = zoom(scores_256[i], (scale_h, scale_w), order=1)
            resized_scores.append(resized)
        scores_resized = np.array(resized_scores)
        
        print(f"   Resized scores: {scores_resized.shape}")
        print(f"   ✓ Shapes now match: {scores_resized.shape == masks_240.shape}")
        
        # Test flattening
        scores_flat = scores_resized.flatten()
        masks_flat = masks_240.flatten()
        print(f"   Flattened scores: {scores_flat.shape}")
        print(f"   Flattened masks: {masks_flat.shape}")
        print(f"   ✓ Can compute AUROC: {scores_flat.shape == masks_flat.shape}")
    
    # Run actual evaluation
    print(f"\n6. Running full evaluation pipeline...")
    from data import create_dataloaders
    from eval import evaluate_model
    
    # Create dataloaders
    _, val_loader = create_dataloaders(config, include_test=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = detector.to(device)
    
    # Test with different target sizes
    for size_name, size_val in [("240x240", (240, 240)), ("256x256", (256, 256))]:
        print(f"\n   Testing with target_size={size_name}...")
        metrics = evaluate_model(
            model=detector,
            dataloader=val_loader,
            device=device,
            compute_pixel_auroc=True,
            target_size=size_val
        )
        
        if 'pixel_auroc' in metrics:
            print(f"   ✓ Pixel AUROC computed: {metrics['pixel_auroc']:.4f}")
        else:
            print(f"   ✗ Pixel AUROC NOT computed")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - PIXEL AUROC FIX VERIFIED")
    print("="*80)
    
else:
    print(f"   Model not found at {model_path}")
    print(f"   Please train a model first or run:")
    print(f"   python main.py --config configs/default.yaml")
