"""
Test learnable anchors implementation
"""

import torch
from learnable_anchors import LearnableAnchors, CAMLoss, assign_to_nearest_anchor

def test_learnable_anchors():
    """Test LearnableAnchors class"""
    print("Testing LearnableAnchors...")
    
    # Create initial anchors
    K, D = 8, 128
    initial_anchors = torch.randn(K, D)
    
    # Create learnable anchors
    learnable = LearnableAnchors(initial_anchors, freeze=False)
    
    # Check forward
    anchors = learnable()
    assert anchors.shape == (K, D)
    assert anchors.requires_grad == True
    
    # Check norms
    norms = learnable.get_anchor_norms()
    assert norms.shape == (K,)
    
    # Check distances
    distances = learnable.get_pairwise_distances()
    assert distances.shape == (K, K)
    
    print("✓ LearnableAnchors works!")


def test_cam_loss():
    """Test CAM Loss"""
    print("\nTesting CAM Loss...")
    
    # Setup
    N, K, D = 32, 8, 128
    embeddings = torch.randn(N, D)
    anchors = torch.randn(K, D)
    assignments = torch.randint(0, K, (N,))
    
    # Create loss
    criterion = CAMLoss(
        lambda_attractor=1.0,
        lambda_repeller=1.0,
        lambda_norm=0.1,
        margin=1.0,
        min_norm=0.5,
        distance_metric='euclidean'
    )
    
    # Compute loss
    total_loss, loss_dict = criterion(embeddings, anchors, assignments)
    
    # Check outputs
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.requires_grad == True
    assert 'total' in loss_dict
    assert 'attractor' in loss_dict
    assert 'repeller' in loss_dict
    assert 'norm' in loss_dict
    
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Attractor: {loss_dict['attractor']:.4f}")
    print(f"  Repeller: {loss_dict['repeller']:.4f}")
    print(f"  Norm: {loss_dict['norm']:.4f}")
    
    # Test backward
    total_loss.backward()
    assert embeddings.grad is not None
    assert anchors.grad is not None
    
    print("✓ CAM Loss works!")


def test_assignment():
    """Test anchor assignment"""
    print("\nTesting anchor assignment...")
    
    N, K, D = 32, 8, 128
    embeddings = torch.randn(N, D)
    anchors = torch.randn(K, D)
    
    # Test euclidean
    assignments_l2 = assign_to_nearest_anchor(embeddings, anchors, 'euclidean')
    assert assignments_l2.shape == (N,)
    assert assignments_l2.min() >= 0
    assert assignments_l2.max() < K
    
    # Test cosine
    assignments_cos = assign_to_nearest_anchor(embeddings, anchors, 'cosine')
    assert assignments_cos.shape == (N,)
    
    print(f"  L2 assignments: {assignments_l2[:10].tolist()}")
    print(f"  Cosine assignments: {assignments_cos[:10].tolist()}")
    print("✓ Assignment works!")


def test_gradient_flow():
    """Test that gradients flow through entire pipeline"""
    print("\nTesting gradient flow...")
    
    N, K, D = 16, 4, 64
    
    # Create learnable anchors
    initial_anchors = torch.randn(K, D)
    learnable = LearnableAnchors(initial_anchors, freeze=False)
    
    # Create loss
    criterion = CAMLoss(
        lambda_attractor=1.0,
        lambda_repeller=1.0,
        lambda_norm=0.1,
        margin=1.0,
        min_norm=0.5
    )
    
    # Forward pass
    embeddings = torch.randn(N, D, requires_grad=True)
    anchors = learnable()
    assignments = assign_to_nearest_anchor(embeddings, anchors, 'euclidean')
    loss, _ = criterion(embeddings, anchors, assignments)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert embeddings.grad is not None
    assert learnable.anchors.grad is not None
    
    print(f"  Embedding grad norm: {embeddings.grad.norm().item():.4f}")
    print(f"  Anchor grad norm: {learnable.anchors.grad.norm().item():.4f}")
    print("✓ Gradients flow correctly!")


if __name__ == '__main__':
    print("="*80)
    print("TESTING LEARNABLE ANCHORS IMPLEMENTATION")
    print("="*80)
    
    test_learnable_anchors()
    test_cam_loss()
    test_assignment()
    test_gradient_flow()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
