"""
Quick validation test - run one validation pass to verify pixel AUROC computation
"""
import torch
import yaml
from pathlib import Path
from glob import glob

# Import project modules
from model import AnomalyDetector, DINOv3Backbone
from data import create_dataloaders
from eval import evaluate_model

def main():
    # Paths
    experiment_dir = Path('experiments/bmad_fixed')
    config_path = experiment_dir / 'config.yaml'
    checkpoint_path = experiment_dir / 'best_model.pth'
    anchor_path = experiment_dir / 'anchor_embeddings.pt'
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Prepare validation data
    val_dir = Path('../data/BraTS2021_slice/valid')
    good_imgs = sorted(glob(str(val_dir / 'good/img/*.png')))
    good_masks = sorted(glob(str(val_dir / 'good/label/*.png')))
    ungood_imgs = sorted(glob(str(val_dir / 'Ungood/img/*.png')))
    ungood_masks = sorted(glob(str(val_dir / 'Ungood/label/*.png')))
    
    val_imgs = good_imgs + ungood_imgs
    val_masks = good_masks + ungood_masks
    val_labels = [0] * len(good_imgs) + [1] * len(ungood_imgs)
    
    print(f"Validation: {len(val_imgs)} images ({len(good_imgs)} normal, {len(ungood_imgs)} anomaly)")
    
    # Create dataloaders (use dummy train/test)
    _, val_loader, _ = create_dataloaders(
        train_paths=good_imgs[:10],
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
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load anchors
    anchor_data = torch.load(anchor_path, weights_only=False)
    
    # Create backbone
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=config['model']['projection_dim']
    )
    
    # Create model
    model = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_data['anchor_global'],
        anchor_dense_embeddings=anchor_data['anchor_dense']
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n{'='*60}")
    print("RUNNING VALIDATION WITH PIXEL AUROC")
    print('='*60)
    
    # Run evaluation
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        compute_pixel_auroc=True,
        target_size=(240, 240)
    )
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"Image AUPR: {metrics['image_aupr']:.4f}")
    
    if 'pixel_auroc' in metrics:
        print(f"Pixel AUROC: {metrics['pixel_auroc']:.4f}")
        print(f"Pixel AUPR: {metrics['pixel_aupr']:.4f}")
        print(f"\n✓ SUCCESS: Pixel AUROC computed correctly!")
    else:
        print(f"\n✗ FAILURE: Pixel AUROC not computed!")
        print("Check the debug output above for details.")
    
    print(f"\nAll metrics: {list(metrics.keys())}")

if __name__ == '__main__':
    main()
