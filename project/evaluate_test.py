"""
Evaluate a trained model on the test set
"""

import argparse
import torch
import yaml
from pathlib import Path

from main import load_dataset_paths, create_model
from data import create_dataloaders
from eval import evaluate_comprehensive


def main(args):
    """Evaluate model on test set"""
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output dir if specified
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(config['output_dir'])
    
    print("="*80)
    print("MODEL EVALUATION ON TEST SET")
    print("="*80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    data_root = config['data'].get('data_root', '../data/BraTS2021_slice')
    train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(data_root)
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        val_mask_paths=val_mask_paths,
        test_mask_paths=test_mask_paths,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        target_size=tuple(config['data']['target_size'])
    )
    
    print(f"Test batches: {len(test_loader)}")
    print(f"Test samples: {len(test_paths)} ({test_labels.count(0)} normal, {test_labels.count(1)} anomaly)")
    
    # Load anchor embeddings
    anchor_path = checkpoint_dir / 'anchor_embeddings.pt'
    if not anchor_path.exists():
        print(f"Error: Anchor embeddings not found at {anchor_path}")
        return
    
    print(f"\nLoading anchor embeddings from {anchor_path}")
    anchor_data = torch.load(anchor_path, weights_only=False)
    anchor_global = anchor_data['anchor_global']
    anchor_dense = anchor_data['anchor_dense']
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, anchor_global, anchor_dense)
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint == 'best':
        checkpoint_path = checkpoint_dir / 'best_model.pth'
    elif args.checkpoint == 'final':
        checkpoint_path = checkpoint_dir / 'final_model.pth'
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch {checkpoint['epoch']}")
    if 'best_val_auroc' in checkpoint:
        print(f"Best validation AUROC: {checkpoint['best_val_auroc']:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("RUNNING EVALUATION ON TEST SET")
    print("="*80)
    print(f"Computing pixel-level metrics: {config['eval']['compute_pixel']}")
    print(f"Target size for pixel maps: {tuple(config['data']['target_size'])}")
    
    eval_dir = checkpoint_dir / 'test_evaluation'
    eval_dir.mkdir(exist_ok=True)
    
    results = evaluate_comprehensive(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir,
        compute_pixel=config['eval']['compute_pixel'],
        target_size=tuple(config['data']['target_size'])
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(f"Image AUROC: {results['image_auroc']:.4f}")
    print(f"Image AUPR:  {results['image_aupr']:.4f}")
    if 'pixel_auroc' in results:
        print(f"Pixel AUROC: {results['pixel_auroc']:.4f}")
        print(f"Pixel AUPR:  {results['pixel_aupr']:.4f}")
    
    print(f"\nDetailed results saved to: {eval_dir}")
    print(f"  - evaluation_metrics.json")
    print(f"  - roc_curve.png (Image-level)")
    if 'pixel_auroc' in results:
        print(f"  - pixel_roc_curve.png (Pixel-level)")
    print(f"  - score_distributions.png")
    print(f"  - normal_samples.png")
    print(f"  - anomaly_samples.png")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--config', type=str, default='configs/recommended.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to checkpoint directory (overrides config output_dir)')
    parser.add_argument('--checkpoint', type=str, default='best',
                        choices=['best', 'final'],
                        help='Which checkpoint to use (best or final)')
    
    args = parser.parse_args()
    main(args)
