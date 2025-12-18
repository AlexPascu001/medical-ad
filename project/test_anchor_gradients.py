"""
Test script to verify that learnable anchors actually receive gradients and update during training.
This is a minimal test to confirm the .detach() bug fix works.
"""

import torch
import yaml
from pathlib import Path
import numpy as np

from model import DINOv3Backbone, AnomalyDetector
from loss import AnchorMarginLoss


def test_anchor_gradients():
    """Test that gradients flow to learnable anchors"""
    
    print("="*80)
    print("TESTING ANCHOR GRADIENT FLOW")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create a small backbone with projection
    backbone = DINOv3Backbone(
        model_name="vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone=True,
        projection_dim=128,
        pretrained=False  # Faster for testing
    ).to(device)
    
    # Create random anchors
    n_anchors = 8
    anchor_dim = 384  # DINOv3 small embedding dimension
    
    # Initialize anchors with random values
    anchor_embeddings = torch.randn(n_anchors, anchor_dim).to(device)
    anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, dim=1)
    
    print(f"\n1. Creating model with LEARNABLE anchors (K={n_anchors})")
    model = AnomalyDetector(
        backbone=backbone,
        anchor_global_embeddings=anchor_embeddings,
        distance_metric='euclidean',
        learnable_anchors=True  # ← Key: make anchors learnable
    ).to(device)
    
    # Verify anchors are parameters
    anchor_is_param = isinstance(model.anchor_global, torch.nn.Parameter)
    print(f"   Anchors are nn.Parameter: {anchor_is_param}")
    print(f"   Anchors require_grad: {model.anchor_global.requires_grad}")
    
    if not anchor_is_param or not model.anchor_global.requires_grad:
        print("   ❌ ERROR: Anchors are not trainable parameters!")
        return False
    
    # Create loss
    loss_fn = AnchorMarginLoss(
        margin=1.0,
        alpha=1.0,  # Attractor
        beta=1.0,   # Repeller
        gamma=0.1,  # Min-norm
        min_norm=0.5,
        distance_metric='euclidean'
    )
    
    # Create fake batch
    batch_size = 16
    fake_images = torch.randn(batch_size, 3, 240, 240).to(device)
    
    print(f"\n2. Running forward pass with batch_size={batch_size}")
    
    # Store initial anchor values
    initial_anchors = model.anchor_global.clone().detach()
    print(f"   Initial anchor norms: {initial_anchors.norm(dim=1).cpu().numpy()}")
    
    # Forward pass
    outputs = model(fake_images, return_dense=False)
    
    # Get projected anchors (same as training does)
    anchor_global, _ = model._get_projected_anchors()
    
    print(f"   Output features shape: {outputs['global_feat'].shape}")
    print(f"   Projected anchors shape: {anchor_global.shape}")
    
    # Compute loss
    print(f"\n3. Computing loss with 3 components:")
    loss_dict = loss_fn(
        outputs['global_feat'],
        anchor_global,
        return_components=True
    )
    
    loss = loss_dict['loss']
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Attractor: {loss_dict['loss_attract']:.4f}")
    print(f"   Repeller: {loss_dict['loss_repel']:.4f}")
    print(f"   Min-norm: {loss_dict['loss_norm']:.4f}")
    
    # Check if anchors have gradients BEFORE backward
    print(f"\n4. Checking gradient status BEFORE backward:")
    print(f"   anchor_global.requires_grad: {model.anchor_global.requires_grad}")
    print(f"   anchor_global.grad is None: {model.anchor_global.grad is None}")
    
    # Backward pass
    print(f"\n5. Running backward pass...")
    loss.backward()
    
    # Check if anchors have gradients AFTER backward
    print(f"\n6. Checking gradient status AFTER backward:")
    has_grad = model.anchor_global.grad is not None
    print(f"   anchor_global.grad is not None: {has_grad}")
    
    if has_grad:
        grad_norm = model.anchor_global.grad.norm().item()
        grad_mean = model.anchor_global.grad.abs().mean().item()
        print(f"   Gradient norm: {grad_norm:.6f}")
        print(f"   Gradient mean (abs): {grad_mean:.6f}")
        
        if grad_norm < 1e-8:
            print(f"   ⚠️  WARNING: Gradient norm is very small! Anchors might not update much.")
        else:
            print(f"   ✅ Gradients look good!")
    else:
        print(f"   ❌ ERROR: No gradients! The .detach() bug still exists.")
        return False
    
    # Simulate optimizer step
    print(f"\n7. Simulating optimizer step (lr=0.001)...")
    with torch.no_grad():
        model.anchor_global -= 0.001 * model.anchor_global.grad
    
    # Check anchor changes
    final_anchors = model.anchor_global.clone().detach()
    anchor_changes = (final_anchors - initial_anchors).norm(dim=1)
    
    print(f"   Per-anchor changes (L2 norm):")
    for i, change in enumerate(anchor_changes.cpu().numpy()):
        print(f"      Anchor {i}: {change:.6f}")
    
    mean_change = anchor_changes.mean().item()
    print(f"   Mean change: {mean_change:.6f}")
    
    if mean_change < 1e-8:
        print(f"   ❌ ERROR: Anchors didn't move! Something is wrong.")
        return False
    else:
        print(f"   ✅ SUCCESS: Anchors moved! Learnable anchors are working correctly.")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETED SUCCESSFULLY!")
    print("='*80}")
    print("\nSummary:")
    print(f"  ✅ Anchors are trainable parameters")
    print(f"  ✅ Gradients flow to anchors during backward pass")
    print(f"  ✅ Anchors update their positions after optimizer step")
    print(f"\nThe .detach() bug has been fixed! Learnable anchors will now work properly.")
    
    return True


if __name__ == '__main__':
    success = test_anchor_gradients()
    
    if not success:
        print("\n❌ TEST FAILED - Learnable anchors are NOT working correctly!")
        exit(1)
    else:
        print("\n✅ TEST PASSED - Learnable anchors are working correctly!")
        exit(0)
