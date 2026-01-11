"""
Main training script for BMAD Brain MRI Anomaly Detection
"""

import argparse
import warnings
import yaml
import torch
import numpy as np
from pathlib import Path
import random

# Suppress pydantic warnings from timm library (internal compatibility issue)
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

from data import BMADPreprocessor, create_dataloaders
from anchors import AnchorGenerator, compute_anchor_embeddings, visualize_anchors
from model import DINOv3Backbone, AnomalyDetector
from loss import AnchorMarginLoss, DenseAnchorMarginLoss, CombinedAnchorLoss
from contrastive_loss import CenterLoss, InfoNCEAnchorLoss, HybridAnchorLoss, CombinedContrastiveLoss
from train import Trainer
from eval import evaluate_comprehensive, visualize_predictions, analyze_anchor_assignments


def load_dataset_paths(data_root: str):
    """
    Load image and label paths from BraTS2021_slice dataset structure
    
    Structure:
        data_root/
            train/
                good/
                    *.png
            valid/
                good/
                    img/*.png
                    label/*.png
                Ungood/
                    img/*.png
                    label/*.png
            test/
                good/
                    img/*.png
                    label/*.png
                Ungood/
                    img/*.png
                    label/*.png
    
    Returns:
        train_paths, val_paths, val_labels, val_mask_paths,
        test_paths, test_labels, test_mask_paths
    """
    data_root = Path(data_root)
    
    # Training: only normal images (good/)
    train_dir = data_root / 'train' / 'good'
    train_paths = sorted([str(p) for p in train_dir.glob('*.png')])
    
    # Validation: good + Ungood
    val_paths = []
    val_labels = []
    val_mask_paths = []
    
    # Val - good (label 0, no anomaly)
    val_good_img_dir = data_root / 'valid' / 'good' / 'img'
    val_good_label_dir = data_root / 'valid' / 'good' / 'label'
    val_good_imgs = sorted([str(p) for p in val_good_img_dir.glob('*.png')])
    for img_path in val_good_imgs:
        img_name = Path(img_path).name
        label_path = val_good_label_dir / img_name
        val_paths.append(img_path)
        val_labels.append(0)
        val_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    # Val - Ungood (label 1, anomaly)
    val_ungood_img_dir = data_root / 'valid' / 'Ungood' / 'img'
    val_ungood_label_dir = data_root / 'valid' / 'Ungood' / 'label'
    val_ungood_imgs = sorted([str(p) for p in val_ungood_img_dir.glob('*.png')])
    for img_path in val_ungood_imgs:
        img_name = Path(img_path).name
        label_path = val_ungood_label_dir / img_name
        val_paths.append(img_path)
        val_labels.append(1)
        val_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    # Test: good + Ungood
    test_paths = []
    test_labels = []
    test_mask_paths = []
    
    # Test - good (label 0, no anomaly)
    test_good_img_dir = data_root / 'test' / 'good' / 'img'
    test_good_label_dir = data_root / 'test' / 'good' / 'label'
    test_good_imgs = sorted([str(p) for p in test_good_img_dir.glob('*.png')])
    for img_path in test_good_imgs:
        img_name = Path(img_path).name
        label_path = test_good_label_dir / img_name
        test_paths.append(img_path)
        test_labels.append(0)
        test_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    # Test - Ungood (label 1, anomaly)
    test_ungood_img_dir = data_root / 'test' / 'Ungood' / 'img'
    test_ungood_label_dir = data_root / 'test' / 'Ungood' / 'label'
    test_ungood_imgs = sorted([str(p) for p in test_ungood_img_dir.glob('*.png')])
    for img_path in test_ungood_imgs:
        img_name = Path(img_path).name
        label_path = test_ungood_label_dir / img_name
        test_paths.append(img_path)
        test_labels.append(1)
        test_mask_paths.append(str(label_path) if label_path.exists() else None)
    
    print(f"Loaded dataset from {data_root}:")
    print(f"  Train: {len(train_paths)} normal images")
    print(f"  Val: {len(val_paths)} images ({val_labels.count(0)} normal, {val_labels.count(1)} anomaly)")
    print(f"  Test: {len(test_paths)} images ({test_labels.count(0)} normal, {test_labels.count(1)} anomaly)")
    
    return train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_anchors(
    train_images: list,
    preprocessor: BMADPreprocessor,
    config: dict,
    save_dir: Path,
    backbone_for_projection: DINOv3Backbone = None,
    device: torch.device = None
) -> tuple:
    """
    Prepare anchor images and embeddings.
    
    CRITICAL: If backbone_for_projection is provided, anchors are projected through
    that SPECIFIC backbone's projection head. This ensures anchors and samples use
    the SAME projection weights. The anchors are then stored in PROJECTED space
    and NOT re-projected during training (acting as fixed targets).
    
    Args:
        train_images: List of training image paths
        preprocessor: Image preprocessor
        config: Configuration dictionary
        save_dir: Directory to save anchor data
        backbone_for_projection: If provided, project anchors through THIS backbone
        device: Device for computation
    
    Returns:
        anchor_images, anchor_global_embeddings (PROJECTED if backbone provided), anchor_dense_embeddings
    """
    print("\n" + "="*80)
    print("ANCHOR GENERATION")
    print("="*80)
    
    # Load and preprocess training images
    print("Loading training images...")
    images = []
    for img_path in train_images[:config['anchor']['max_images_for_pca']]:
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = preprocessor.preprocess(img)
        images.append(img)
    
    images = np.array(images)
    print(f"Loaded {len(images)} images, shape: {images.shape}")
    
    # Generate anchors with selected strategy
    anchor_gen = AnchorGenerator(
        strategy=config['anchor'].get('strategy', 'eigenface'),
        n_components=config['anchor']['n_components'],
        n_anchors=config['anchor']['n_anchors'],
        random_state=config['seed']
    )
    
    anchor_images = anchor_gen.fit(images)
    
    # Save anchor generator
    anchor_gen.save(save_dir / 'anchor_generator.pkl')
    
    # Visualize anchors
    visualize_anchors(anchor_images, save_dir / 'anchor_images.png')
    
    # Compute anchor embeddings
    print("\nComputing anchor embeddings with DINOv3...")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    projection_dim = config['model'].get('projection_dim', None)
    
    if backbone_for_projection is not None:
        # Use the PROVIDED backbone (same one the model will use)
        # This ensures anchors and samples use the SAME projection weights
        print(f"  Using provided backbone for projection (ensuring same weights as model)")
        backbone = backbone_for_projection
        use_model_backbone = True
    else:
        # Create a temporary backbone (legacy behavior - NOT recommended with projection)
        print(f"  Creating temporary backbone for embedding extraction")
        backbone = DINOv3Backbone(
            model_name=config['model']['backbone'],
            freeze_backbone=True,
            projection_dim=projection_dim,
            pretrained=True
        )
        backbone = backbone.to(device)
        use_model_backbone = False
    
    backbone.eval()
    
    # Extract embeddings - use return_projected=True if we have a projection head
    has_projection = backbone.projection is not None
    anchor_global, anchor_dense = compute_anchor_embeddings(
        anchor_images=anchor_images,
        backbone_model=backbone,
        device=device,
        batch_size=8,
        return_projected=has_projection  # Get projected embeddings if projection exists
    )
    
    # Save embeddings
    torch.save({
        'anchor_images': anchor_images,
        'anchor_global': anchor_global,
        'anchor_dense': anchor_dense,
        'projection_dim': projection_dim if has_projection else None,
        'is_projected': has_projection,
        'used_model_backbone': use_model_backbone
    }, save_dir / 'anchor_embeddings.pt')
    
    print(f"\nAnchor preparation complete!")
    print(f"  Global embeddings: {anchor_global.shape}")
    if has_projection:
        print(f"  Anchors are in PROJECTED space ({anchor_global.shape[1]}D)")
        if use_model_backbone:
            print(f"  ✓ Projected through MODEL's backbone - same weights as training!")
        else:
            print(f"  ⚠ Projected through TEMPORARY backbone - weights differ from model!")
    else:
        print(f"  Anchors are in RAW space (384D) - no projection head")
    print(f"  Dense embeddings: {anchor_dense.shape}")
    
    return anchor_images, anchor_global, anchor_dense


def create_model(config: dict, anchor_global: torch.Tensor, anchor_dense: torch.Tensor) -> AnomalyDetector:
    """Create anomaly detector model"""
    print("\n" + "="*80)
    print("MODEL CREATION")
    print("="*80)
    
    # Check if pixel decoder is requested
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    multi_scale_indices = config['model'].get('multi_scale_indices', [2, 5, 8, 11])
    
    # Create backbone with multi-scale support if pixel decoder is enabled
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=config['model'].get('projection_dim', None),
        pretrained=True,
        multi_scale_indices=multi_scale_indices if use_pixel_decoder else None
    )
    
    # Check if anchors should be learnable
    learnable_anchors = config['anchor'].get('learnable', False)
    
    # CRITICAL: Anchors are already in PROJECTED space (from prepare_anchors).
    # They were projected ONCE through a fresh projection head and stored.
    # The model will use them as FIXED targets - NOT re-project them.
    # This prevents collapse: the projection head learns to map samples TO
    # these fixed anchor locations, rather than collapsing everything together.
    projection_dim = config['model'].get('projection_dim', None)
    if projection_dim:
        print(f"\nAnchors are in PROJECTED space: {anchor_global.shape}")
        print(f"  They are FIXED targets - will NOT be re-projected during training")
        print(f"  Projection head learns to map samples TO these fixed anchors")
    else:
        print(f"\nAnchors are in RAW space: {anchor_global.shape}")
        print(f"  No projection head configured")
    
    # Get target size from config
    target_size = tuple(config['data']['target_size'])
    
    # Create detector with anchors_already_projected=True
    detector = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_global,
        anchor_dense_embeddings=anchor_dense,
        distance_metric=config['loss']['distance_metric'],
        learnable_anchors=learnable_anchors,
        use_pixel_decoder=use_pixel_decoder,
        decoder_hidden_dim=config['model'].get('decoder_hidden_dim', 256),
        target_size=target_size,
        anchors_already_projected=projection_dim is not None  # NEW: Tell model anchors are pre-projected
    )
    
    return detector


def create_criterion(config: dict):
    """
    Create loss function based on config.
    
    Supports:
    - 'cam': Class Anchor Margin Loss (original, attractor + repeller + min-norm)
    - 'center': Center Loss (pull samples + anchors toward each other)
    - 'infonce': InfoNCE contrastive loss (soft assignments with temperature)
    - 'hybrid': Hybrid of Center + InfoNCE (best of both)
    
    For learnable anchors, 'center', 'infonce', or 'hybrid' are recommended.
    """
    loss_type = config['loss'].get('type', 'cam')  # Default to CAM loss for backward compatibility
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    
    print(f"\nCreating loss function: {loss_type}")
    if use_pixel_decoder:
        print(f"  Pixel decoder enabled: dense loss will be computed")
    
    if loss_type == 'cam':
        # Original CAM loss
        global_loss = AnchorMarginLoss(
            margin=config['loss']['margin'],
            alpha=config['loss']['alpha'],
            beta=config['loss']['beta'],
            gamma=config['loss'].get('gamma', 0.0),
            min_norm=config['loss'].get('min_norm', 0.5),
            distance_metric=config['loss']['distance_metric']
        )

        # Create dense loss if pixel decoder is enabled
        dense_loss = None
        if use_pixel_decoder:
            dense_loss = DenseAnchorMarginLoss(
                margin=config['loss']['margin'],
                alpha=config['loss']['alpha'],
                distance_metric=config['loss']['distance_metric'],
                spatial_reduction='mean'
            )
            print(f"  Dense loss: DenseAnchorMarginLoss (alpha={config['loss']['alpha']})")
        
        # Combined loss
        criterion = CombinedAnchorLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    elif loss_type == 'center':
        # Center Loss (dense branch disabled)
        global_loss = CenterLoss(
            distance_metric=config['loss']['distance_metric'],
            lambda_center=config['loss'].get('lambda_center', 1.0),
            lambda_repel=config['loss'].get('lambda_repel', 0.1),
            margin=config['loss']['margin']
        )

        dense_loss = None
        config['loss']['use_dense'] = False

        criterion = CombinedContrastiveLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    elif loss_type == 'infonce':
        # InfoNCE Loss (dense branch disabled)
        global_loss = InfoNCEAnchorLoss(
            temperature=config['loss'].get('temperature', 0.07),
            lambda_repel=config['loss'].get('lambda_repel', 0.1),
            margin=config['loss']['margin'],
            distance_metric=config['loss']['distance_metric']
        )

        dense_loss = None
        config['loss']['use_dense'] = False

        criterion = CombinedContrastiveLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    elif loss_type == 'hybrid':
        # Hybrid: Center + InfoNCE (dense branch disabled)
        global_loss = HybridAnchorLoss(
            lambda_center=config['loss'].get('lambda_center', 1.0),
            lambda_infonce=config['loss'].get('lambda_infonce', 0.5),
            lambda_repel=config['loss'].get('lambda_repel', 0.1),
            temperature=config['loss'].get('temperature', 0.07),
            margin=config['loss']['margin'],
            distance_metric=config['loss']['distance_metric']
        )

        dense_loss = None
        config['loss']['use_dense'] = False

        criterion = CombinedContrastiveLoss(
            global_loss=global_loss,
            dense_loss=dense_loss,
            global_weight=config['loss']['global_weight'],
            dense_weight=config['loss']['dense_weight']
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: cam, center, infonce, hybrid")
    
    print(f"  ✓ Loss type: {loss_type}")
    return criterion


def generate_experiment_name(config: dict, base_dir: str = './experiments') -> str:
    """
    Generate experiment name based on anchor and distance configuration
    
    Format: <base_name>_<strategy>_k<num_anchors>_<distance>
    Example: bmad_eigenface_k8_cosine, bmad_random_k16_l2, bmad_kmeans_k4_cosine
    """
    base_name = Path(base_dir).name if '/' in base_dir or '\\' in base_dir else 'bmad'
    strategy = config['anchor']['strategy']
    n_anchors = config['anchor']['n_anchors']
    distance = config['loss']['distance_metric']
    
    # Abbreviate distance metric
    dist_abbrev = 'cos' if distance == 'cosine' else 'l2'
    
    exp_name = f"{base_name}_{strategy}_k{n_anchors}_{dist_abbrev}"
    
    return exp_name


def make_unique_dir(base: Path) -> Path:
    """Create a unique directory by adding numeric suffix if needed."""
    if not base.exists():
        return base
    idx = 1
    while True:
        cand = base.parent / f"{base.name}_{idx}"
        if not cand.exists():
            return cand
        idx += 1


def main(args):
    """Main training pipeline"""
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Auto/explicit experiment naming and uniqueness
    if args.exp_name:
        save_dir = Path(config['output_dir']) / args.exp_name
    elif args.auto_name or config['output_dir'] == './experiments/bmad_baseline':
        base_output = Path(config['output_dir']).parent
        exp_name = generate_experiment_name(config, str(base_output))
        save_dir = base_output / exp_name
    else:
        save_dir = Path(config['output_dir'])

    # Avoid overwrite by uniquifying when directory exists
    save_dir = make_unique_dir(save_dir)
    config['output_dir'] = str(save_dir)
    
    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("="*80)
    print("BMAD BRAIN MRI ANOMALY DETECTION")
    print("="*80)
    print(f"Output directory: {save_dir}")
    print(f"Config: {args.config}")
    print(f"Anchor strategy: {config['anchor']['strategy']}")
    print(f"Number of anchors: {config['anchor']['n_anchors']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ===== STAGE 1: Data Preparation =====
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    # Load dataset paths from BraTS2021_slice structure
    data_root = config['data'].get('data_root', './data/BraTS2021_slice')
    
    train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(data_root)
    
    if not train_paths:
        print("\nWARNING: No data found!")
        print(f"Please check that the dataset exists at: {data_root}")
        print("Expected structure: train/good/*.png, valid/good/img/*.png, etc.")
        return
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
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
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ===== STAGE 2: Create Model Backbone FIRST =====
    # We need to create the backbone first so we can project anchors through
    # the SAME projection head that will be used during training
    print("\n" + "="*80)
    print("CREATING BACKBONE")
    print("="*80)
    
    use_pixel_decoder = config['model'].get('use_pixel_decoder', False)
    multi_scale_indices = config['model'].get('multi_scale_indices', [2, 5, 8, 11])
    projection_dim = config['model'].get('projection_dim', None)
    
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=config['model']['freeze_backbone'],
        projection_dim=projection_dim,
        pretrained=True,
        multi_scale_indices=multi_scale_indices if use_pixel_decoder else None
    )
    backbone = backbone.to(device)
    
    # ===== STAGE 3: Anchor Generation (using model's backbone) =====
    preprocessor = BMADPreprocessor(target_size=tuple(config['data']['target_size']))
    
    # Check if we should load anchors from another experiment (for learnable anchors)
    init_from = config['anchor'].get('init_from', None)
    
    if init_from is not None:
        print(f"\nLoading anchors from: {init_from}")
        init_anchor_path = Path(init_from) / 'anchor_embeddings.pt'
        if not init_anchor_path.exists():
            raise FileNotFoundError(f"Cannot initialize anchors from {init_from}: anchor_embeddings.pt not found")
        
        anchor_data = torch.load(init_anchor_path, weights_only=False)
        if isinstance(anchor_data, dict):
            anchor_global = anchor_data.get('anchor_global', anchor_data.get('global'))
            anchor_dense = anchor_data.get('anchor_dense', anchor_data.get('dense'))
            anchor_images = anchor_data.get('anchor_images', None)
        else:
            anchor_global = anchor_data
            anchor_dense = None
            anchor_images = None
        
        print(f"✓ Loaded anchors: {anchor_global.shape}")
        
        # Save to current experiment directory
        torch.save({
            'anchor_images': anchor_images,
            'anchor_global': anchor_global,
            'anchor_dense': anchor_dense,
            'initialized_from': str(init_from)
        }, save_dir / 'anchor_embeddings.pt')
        
    elif args.skip_anchors and (save_dir / 'anchor_embeddings.pt').exists():
        print("\nLoading existing anchors...")
        anchor_data = torch.load(save_dir / 'anchor_embeddings.pt', weights_only=False)
        anchor_images = anchor_data['anchor_images']
        anchor_global = anchor_data['anchor_global']
        anchor_dense = anchor_data['anchor_dense']
    else:
        # CRITICAL: Pass the MODEL's backbone so anchors use the SAME projection weights
        anchor_images, anchor_global, anchor_dense = prepare_anchors(
            train_images=train_paths,
            preprocessor=preprocessor,
            config=config,
            save_dir=save_dir,
            backbone_for_projection=backbone,  # Use MODEL's backbone!
            device=device
        )
    
    # ===== STAGE 4: Complete Model Creation =====
    # Now create the full detector with the backbone and anchors
    learnable_anchors = config['anchor'].get('learnable', False)
    target_size = tuple(config['data']['target_size'])
    
    print("\n" + "="*80)
    print("MODEL CREATION")
    print("="*80)
    
    # Anchors are already projected through THIS backbone's projection head
    # So they're in the SAME coordinate system as samples will be
    if projection_dim:
        print(f"Anchors are in PROJECTED space: {anchor_global.shape}")
        print(f"  Projected through MODEL's backbone - same weights as training!")
        print(f"  They are FIXED targets - will NOT be re-projected")
    else:
        print(f"Anchors are in RAW space: {anchor_global.shape}")
    
    model = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_global,
        anchor_dense_embeddings=anchor_dense,
        distance_metric=config['loss']['distance_metric'],
        learnable_anchors=learnable_anchors,
        use_pixel_decoder=use_pixel_decoder,
        decoder_hidden_dim=config['model'].get('decoder_hidden_dim', 256),
        target_size=target_size,
        anchors_already_projected=projection_dim is not None
    )
    model = model.to(device)
    
    # ===== STAGE 5: Training Setup =====
    criterion = create_criterion(config)
    
    # Get trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # Optimizer
    if len(trainable_params) > 0:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['lr'] * 0.01
        )
    else:
        print("\nNote: No trainable parameters (backbone is frozen, no projection head).")
        print("Skipping training and proceeding directly to evaluation...")
        optimizer = None
        scheduler = None
    
    # ===== STAGE 5: Training =====
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    if optimizer is not None and not args.eval_only:
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=save_dir,
            use_amp=config['training']['use_amp'],
            log_interval=config['training']['log_interval'],
            val_interval=config['training']['val_interval'],
            fixed_pseudo_labels=config['training'].get('fixed_pseudo_labels', False),
            dynamic_reassignment=config['training'].get('dynamic_reassignment', False),
            reassignment_interval=config['training'].get('reassignment_interval', 5)
        )
        
        trainer.train(
            num_epochs=config['training']['epochs'],
            scheduler=scheduler,
            early_stopping_patience=config['training']['early_stopping_patience']
        )
    else:
        if optimizer is None:
            print("Skipping training (no trainable parameters)")
        else:
            print("Skipping training (--eval-only flag)")
        
        # Save model anyway for evaluation
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, save_dir / 'best_model.pth')
    
    # ===== STAGE 6: Evaluation =====
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    best_model_path = save_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print(f"Loaded model checkpoint")
    
    # Comprehensive evaluation
    eval_dir = save_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    
    compute_pixel = config.get('eval', {}).get('compute_pixel', False)
    results = evaluate_comprehensive(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir,
        compute_pixel=compute_pixel,
        target_size=tuple(config['data']['target_size'])
    )
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir,
        num_samples=16,
        target_size=tuple(config['data']['target_size'])
    )
    
    analyze_anchor_assignments(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=eval_dir
    )
    
    # Generate test sample visualization (normal vs anomaly)
    print("\nGenerating test sample visualization...")
    trainer._visualize_test_samples(test_loader=test_loader, save_name='test_final')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Results saved to: {save_dir}")
    print(f"Best Image AUROC: {results['image_auroc']:.4f}")
    if 'pixel_auroc' in results:
        print(f"Best Pixel AUROC: {results['pixel_auroc']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BMAD anomaly detector')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--skip-anchors', action='store_true',
                        help='Skip anchor generation if already exists')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on existing model')
    parser.add_argument('--auto-name', action='store_true',
                        help='Auto-generate experiment name from anchor config (strategy_k<n_anchors>)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Explicit experiment name subfolder (overrides auto-name)')
    
    args = parser.parse_args()
    main(args)