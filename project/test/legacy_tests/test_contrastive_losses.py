"""
Test all contrastive loss functions to verify they work correctly.
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from contrastive_loss import CenterLoss, InfoNCEAnchorLoss, HybridAnchorLoss


def test_loss(loss_fn, loss_name: str):
    """Test a loss function"""
    print(f"\n{'='*60}")
    print(f"Testing: {loss_name}")
    print('='*60)
    
    # Create fake data
    batch_size = 16
    n_anchors = 8
    embed_dim = 128
    
    embeddings = torch.randn(batch_size, embed_dim)
    anchor_embeddings = torch.randn(n_anchors, embed_dim, requires_grad=True)
    
    # Normalize
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    anchor_embeddings.data = torch.nn.functional.normalize(anchor_embeddings.data, dim=1)
    
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Anchors: {anchor_embeddings.shape}")
    
    # Forward pass
    result = loss_fn(embeddings, anchor_embeddings, return_components=True)
    
    loss = result['loss']
    print(f"\n  Loss: {loss.item():.4f}")
    
    # Print loss components
    for key, value in result.items():
        if key.startswith('loss_'):
            print(f"    {key}: {value:.4f}")
    
    # Check gradients
    loss.backward()
    
    has_grad = anchor_embeddings.grad is not None
    print(f"\n  Anchor gradients exist: {has_grad}")
    
    if has_grad:
        grad_norm = anchor_embeddings.grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 1e-8:
            print(f"  ✅ PASS: Gradients flow correctly!")
            return True
        else:
            print(f"  ⚠️  WARNING: Gradient norm is very small")
            return False
    else:
        print(f"  ❌ FAIL: No gradients!")
        return False


def main():
    """Test all loss functions"""
    print("="*60)
    print("CONTRASTIVE LOSS FUNCTION TESTS")
    print("="*60)
    
    results = {}
    
    # Test Center Loss
    center_loss = CenterLoss(
        distance_metric='euclidean',
        lambda_center=1.0,
        lambda_repel=0.1,
        margin=1.0
    )
    results['Center Loss'] = test_loss(center_loss, 'Center Loss')
    
    # Test InfoNCE Loss
    infonce_loss = InfoNCEAnchorLoss(
        temperature=0.07,
        lambda_repel=0.1,
        margin=1.0,
        distance_metric='euclidean'
    )
    results['InfoNCE Loss'] = test_loss(infonce_loss, 'InfoNCE Loss')
    
    # Test Hybrid Loss
    hybrid_loss = HybridAnchorLoss(
        lambda_center=1.0,
        lambda_infonce=0.5,
        lambda_repel=0.1,
        temperature=0.07,
        margin=1.0,
        distance_metric='euclidean'
    )
    results['Hybrid Loss'] = test_loss(hybrid_loss, 'Hybrid Loss')
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print('='*60)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Contrastive losses working correctly!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Check implementation!")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
