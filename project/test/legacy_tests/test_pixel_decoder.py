"""Test pixel decoder implementation"""
import torch
from model import DINOv3Backbone, AnomalyDetector
from loss import AnchorMarginLoss, DenseAnchorMarginLoss, CombinedAnchorLoss

def test_pixel_decoder():
    print("="*60)
    print("Testing Pixel Decoder Implementation")
    print("="*60)
    
    # Create model
    print("\n1. Creating backbone with multi-scale extraction...")
    backbone = DINOv3Backbone(
        model_name='vit_small_patch16_dinov3.lvd1689m',
        freeze_backbone=True,
        projection_dim=128,
        pretrained=True,
        multi_scale_indices=[2, 5, 8, 11]
    )
    
    print("\n2. Creating detector with pixel decoder...")
    # Anchors must be in RAW embedding dimension (384) before projection
    anchor_global = torch.randn(8, backbone.embed_dim)  # 384D for ViT-S
    detector = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_global,
        anchor_dense_embeddings=None,
        distance_metric='euclidean',
        learnable_anchors=True,
        use_pixel_decoder=True,
        decoder_hidden_dim=256,
        target_size=(240, 240)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = detector.to(device)
    print(f"Using device: {device}")
    
    # Create loss
    print("\n3. Creating loss functions...")
    global_loss = AnchorMarginLoss(
        margin=1.0, alpha=1.0, beta=1.0, gamma=0.1, 
        min_norm=0.5, distance_metric='euclidean'
    )
    dense_loss = DenseAnchorMarginLoss(
        margin=1.0, alpha=1.0, 
        distance_metric='euclidean', spatial_reduction='mean'
    )
    criterion = CombinedAnchorLoss(global_loss, dense_loss, 1.0, 0.5)
    
    # Forward pass
    print("\n4. Testing forward pass...")
    x = torch.randn(4, 3, 240, 240).to(device)
    outputs = detector(x, return_dense=True)
    
    print(f"   global_feat: {outputs['global_feat'].shape}")
    print(f"   global_distances: {outputs['global_distances'].shape}")
    print(f"   pixel_embeddings: {outputs['pixel_embeddings'].shape}")
    print(f"   pixel_distances: {outputs['pixel_distances'].shape}")
    
    # Compute loss
    print("\n5. Computing loss...")
    anchor_global_proj, _ = detector._get_projected_anchors()
    loss_dict = criterion(outputs, anchor_global_proj)
    
    print(f"   Total loss: {loss_dict['loss'].item():.4f}")
    print(f"   Global attract: {loss_dict['loss_global_attract']:.4f}")
    print(f"   Global repel: {loss_dict['loss_global_repel']:.4f}")
    print(f"   Dense loss: {loss_dict['loss_dense'].item():.4f}")
    print(f"   Dense attract: {loss_dict['loss_dense_attract']:.4f}")
    
    # Backward pass
    print("\n6. Testing backward pass...")
    loss_dict['loss'].backward()
    
    # Check gradients
    grad_norm = 0
    for p in detector.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"   Gradient norm: {grad_norm:.4f}")
    
    # Test compute_anomaly_scores
    print("\n7. Testing compute_anomaly_scores...")
    detector.eval()
    with torch.no_grad():
        scores = detector.compute_anomaly_scores(x[:2], return_maps=True, target_size=(240, 240))
    
    print(f"   image_scores: {scores['image_scores'].shape}")
    print(f"   pixel_scores: {scores['pixel_scores'].shape}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

if __name__ == '__main__':
    test_pixel_decoder()
