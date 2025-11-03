"""
Generate and Visualize Anchors

This script generates anchor embeddings from training data and visualizes them.
Useful for understanding anchor quality before training.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from glob import glob

from model import DINOv3Backbone
from data import BMADPreprocessor
from anchors import AnchorGenerator, compute_anchor_embeddings, visualize_anchors
from PIL import Image
import cv2


def load_training_images(data_root: str, max_images: int = 5000):
    """Load training images"""
    train_dir = Path(data_root) / 'train' / 'good'
    
    # Try different file patterns
    image_files = sorted(glob(str(train_dir / '*.png')))
    if not image_files:
        image_files = sorted(glob(str(train_dir / '*.jpg')))
    if not image_files:
        image_files = sorted(glob(str(train_dir / '*.npy')))
    
    if not image_files:
        raise FileNotFoundError(f"No training images found in {train_dir}")
    
    # Limit number of images
    if len(image_files) > max_images:
        # Sample evenly
        step = len(image_files) // max_images
        image_files = image_files[::step][:max_images]
    
    print(f"Loading {len(image_files)} training images from {train_dir}")
    
    # Load and preprocess
    preprocessor = BMADPreprocessor(target_size=(240, 240))
    images = []
    
    for img_path in image_files:
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        img = preprocessor.preprocess(img)
        images.append(img)
    
    images = np.array(images)
    print(f"Loaded {len(images)} images with shape {images.shape}")
    
    return images, image_files


def main(args):
    """Generate and visualize anchors"""
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("ANCHOR GENERATION AND VISUALIZATION")
    print("="*60)
    
    # Setup output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training images
    data_root = config['data']['data_root']
    max_images = config['anchor'].get('max_images_for_pca', 5000)
    
    images, image_paths = load_training_images(data_root, max_images)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Generate anchor images using AnchorGenerator
    print(f"\nGenerating {config['anchor']['n_anchors']} anchors using {config['anchor']['strategy']} strategy...")
    anchor_gen = AnchorGenerator(
        strategy=config['anchor'].get('strategy', 'eigenface'),
        n_components=config['anchor']['n_components'],
        n_anchors=config['anchor']['n_anchors'],
        random_state=config.get('seed', 42)
    )
    
    anchor_images = anchor_gen.fit(images)
    print(f"Generated anchor images with shape: {anchor_images.shape}")
    
    # Save anchor generator
    anchor_gen_path = output_dir / 'anchor_generator.pkl'
    anchor_gen.save(anchor_gen_path)
    print(f"Saved anchor generator to: {anchor_gen_path}")
    
    # Load backbone
    print(f"\nLoading backbone: {config['model']['backbone']}")
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=True,
        pretrained=True
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # Compute anchor embeddings
    print("\nComputing anchor embeddings with DINOv3...")
    anchor_global, anchor_dense = compute_anchor_embeddings(
        anchor_images=anchor_images,
        backbone_model=backbone,
        device=device,
        batch_size=8
    )
    
    print(f"\nAnchor embeddings computed:")
    print(f"  Global: {anchor_global.shape}")
    if anchor_dense is not None:
        print(f"  Dense: {anchor_dense.shape}")
    
    # Save anchors
    anchor_path = output_dir / 'anchor_embeddings.pt'
    torch.save({
        'anchor_images': anchor_images,
        'anchor_global': anchor_global,
        'anchor_dense': anchor_dense,
        'strategy': config['anchor']['strategy'],
        'n_anchors': config['anchor']['n_anchors']
    }, anchor_path)
    print(f"\nSaved anchors to: {anchor_path}")
    
    # Visualize anchors
    if args.visualize:
        print("\nGenerating anchor visualizations...")
        
        # Visualize anchor images
        viz_path = output_dir / 'anchor_visualization.png'
        visualize_anchors(
            anchor_images=anchor_images,
            save_path=str(viz_path)
        )
        print(f"Saved visualization to: {viz_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("ANCHOR STATISTICS")
    print("="*60)
    
    # Compute some basic stats
    anchor_norms = torch.norm(anchor_global, dim=1)
    print(f"Anchor L2 norms:")
    print(f"  Mean: {anchor_norms.mean():.4f}")
    print(f"  Std: {anchor_norms.std():.4f}")
    print(f"  Min: {anchor_norms.min():.4f}")
    print(f"  Max: {anchor_norms.max():.4f}")
    
    # Compute pairwise distances
    distances = torch.cdist(anchor_global, anchor_global, p=2)
    # Zero out diagonal
    mask = torch.eye(distances.shape[0], dtype=torch.bool)
    distances = distances[~mask]
    
    print(f"\nPairwise anchor distances:")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Std: {distances.std():.4f}")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")
    
    print("\n✓ Anchor generation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review visualization: {viz_path if args.visualize else 'N/A'}")
    print(f"  2. Start training: python main.py --config {args.config} --skip-anchors")
    print(f"  3. Or analyze anchors: python analyze_issues.py --experiment-dir {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and visualize anchors')
    parser.add_argument('--config', type=str, default='configs/recommended.yaml',
                        help='Path to config file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization (slower)')
    
    args = parser.parse_args()
    main(args)
