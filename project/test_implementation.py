"""
Quick test script to verify implementation changes
Tests:
1. Image dimensions (240x240)
2. Eigenface anchor strategy
3. KMeans anchor strategy
4. Conservative augmentations
"""

import sys
sys.path.append('.')

import numpy as np
import cv2
from data import BMADPreprocessor
from anchors import AnchorGenerator, EigenfaceAnchorStrategy, KMeansCentroidAnchorStrategy

def test_image_dimensions():
    """Test that preprocessor outputs correct 240x240 size"""
    print("="*80)
    print("TEST 1: Image Dimensions")
    print("="*80)
    
    preprocessor = BMADPreprocessor(target_size=(240, 240))
    
    # Create dummy image
    dummy_img = np.random.rand(256, 256).astype(np.float32)
    processed = preprocessor.preprocess(dummy_img)
    
    assert processed.shape == (240, 240), f"Expected (240, 240), got {processed.shape}"
    print(f"✓ Preprocessor output shape: {processed.shape}")
    print()

def test_eigenface_strategy():
    """Test eigenface anchor generation"""
    print("="*80)
    print("TEST 2: Eigenface Strategy")
    print("="*80)
    
    # Create dummy dataset (50 images, 240x240)
    N, H, W = 50, 240, 240
    images = np.random.randn(N, H, W).astype(np.float32)
    
    strategy = EigenfaceAnchorStrategy(n_components=10, n_anchors=4, random_state=42)
    anchors = strategy.fit(images)
    
    assert anchors.shape == (4, 240, 240), f"Expected (4, 240, 240), got {anchors.shape}"
    assert strategy.mean_image is not None, "Mean image not computed"
    assert strategy.eigenvectors is not None, "Eigenvectors not computed"
    
    print(f"✓ Generated {len(anchors)} anchors")
    print(f"✓ Eigenvectors shape: {strategy.eigenvectors.shape}")
    print(f"✓ Mean image shape: {strategy.mean_image.shape}")
    print()

def test_kmeans_strategy():
    """Test KMeans centroid anchor generation"""
    print("="*80)
    print("TEST 3: KMeans Strategy")
    print("="*80)
    
    # Create dummy dataset
    N, H, W = 50, 240, 240
    images = np.random.randn(N, H, W).astype(np.float32)
    
    strategy = KMeansCentroidAnchorStrategy(n_anchors=4, random_state=42)
    anchors = strategy.fit(images)
    
    assert anchors.shape == (4, 240, 240), f"Expected (4, 240, 240), got {anchors.shape}"
    assert strategy.kmeans is not None, "KMeans model not created"
    
    print(f"✓ Generated {len(anchors)} anchors")
    print(f"✓ KMeans inertia: {strategy.kmeans.inertia_:.2e}")
    print()

def test_anchor_generator_factory():
    """Test AnchorGenerator factory with both strategies"""
    print("="*80)
    print("TEST 4: AnchorGenerator Factory")
    print("="*80)
    
    # Create dummy dataset
    N, H, W = 50, 240, 240
    images = np.random.randn(N, H, W).astype(np.float32)
    
    # Test eigenface
    gen1 = AnchorGenerator(strategy='eigenface', n_components=10, n_anchors=4)
    anchors1 = gen1.fit(images)
    assert anchors1.shape == (4, 240, 240)
    print(f"✓ Eigenface strategy works: {anchors1.shape}")
    
    # Test kmeans
    gen2 = AnchorGenerator(strategy='kmeans', n_anchors=4)
    anchors2 = gen2.fit(images)
    assert anchors2.shape == (4, 240, 240)
    print(f"✓ KMeans strategy works: {anchors2.shape}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        gen1.save(f.name)
        loaded_gen = AnchorGenerator.load(f.name)
        assert loaded_gen.strategy_name == 'eigenface'
        print(f"✓ Save/load works for {loaded_gen.strategy_name} strategy")
    
    print()

def test_augmentation_bounds():
    """Test that augmentations are within conservative limits"""
    print("="*80)
    print("TEST 5: Conservative Augmentations")
    print("="*80)
    
    from data import BMADDataset
    import albumentations as A
    
    # Check augmentation parameters
    preprocessor = BMADPreprocessor()
    dataset = BMADDataset(
        image_paths=['dummy.png'],  # Won't be loaded in this test
        preprocessor=preprocessor,
        augment=True,
        is_training=True
    )
    
    # Verify transform parameters
    transforms = dataset.transform.transforms
    print("Augmentation pipeline:")
    for t in transforms:
        print(f"  - {t.__class__.__name__}")
        if hasattr(t, 'rotate_limit'):
            # rotate_limit can be a tuple (-limit, limit) or a single value
            if isinstance(t.rotate_limit, tuple):
                limit = abs(t.rotate_limit[0])
            else:
                limit = abs(t.rotate_limit)
            assert limit == 10.0, f"Rotation limit should be 10, got {t.rotate_limit}"
            print(f"    ✓ Rotation limit: ±{limit}°")
        if hasattr(t, 'shift_limit'):
            # shift_limit can be a tuple or single value
            if isinstance(t.shift_limit, tuple):
                limit = abs(t.shift_limit[0])
            else:
                limit = abs(t.shift_limit)
            assert abs(limit - 0.05) < 0.01, f"Shift limit should be 0.05, got {t.shift_limit}"
            print(f"    ✓ Shift limit: ±{limit*100}%")
        if hasattr(t, 'scale_limit'):
            # scale_limit is stored as (min_scale, max_scale), e.g., (0.9, 1.1) for ±10%
            if isinstance(t.scale_limit, tuple):
                # Convert to ±percentage: (0.9, 1.1) means -10% to +10%
                scale_range = t.scale_limit[1] - 1.0  # 1.1 - 1.0 = 0.1
                assert abs(scale_range - 0.1) < 0.01, f"Scale limit should be ±0.1 (±10%), got {t.scale_limit}"
                print(f"    ✓ Scale limit: {t.scale_limit} (±{scale_range*100}%)")
            else:
                limit = abs(t.scale_limit)
                assert abs(limit - 0.1) < 0.01, f"Scale limit should be 0.1, got {t.scale_limit}"
                print(f"    ✓ Scale limit: ±{limit*100}%")
    
    # Verify no ElasticTransform
    transform_names = [t.__class__.__name__ for t in transforms]
    assert 'ElasticTransform' not in transform_names, "ElasticTransform should be removed"
    assert 'VerticalFlip' not in transform_names, "VerticalFlip should not be present"
    print("  ✓ No ElasticTransform")
    print("  ✓ No VerticalFlip")
    print()

if __name__ == '__main__':
    print("\n" + "="*80)
    print("IMPLEMENTATION VERIFICATION TESTS")
    print("="*80 + "\n")
    
    try:
        test_image_dimensions()
        test_eigenface_strategy()
        test_kmeans_strategy()
        test_anchor_generator_factory()
        test_augmentation_bounds()
        
        print("="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
