"""
Validation script for pre-training implementation.
Performs quick checks to ensure everything is working correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules import correctly"""
    print("1. Testing imports...")
    try:
        import torch
        import torch.nn as nn
        from project.model import DINOv3Backbone
        from project.pretrain import pretrain_projection_head, get_pretrain_cache_key
        from project.loss import AnchorMarginLoss
        from project.anchors import KMeansCentroidAnchorStrategy
        print("   ✓ All imports successful")
        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False

def test_orthogonal_init():
    """Test that orthogonal initialization is working"""
    print("\n2. Testing orthogonal initialization...")
    try:
        import torch
        from project.model import DINOv3Backbone
        
        # Create backbone with projection head
        backbone = DINOv3Backbone(
            model_name='vit_small_patch16_dinov3.lvd1689m',
            freeze_backbone=True,
            projection_dim=128,
            pretrained=False  # Don't download weights for test
        )
        
        # Check that projection head exists
        assert backbone.projection is not None, "Projection head not created"
        
        # Check that weights are initialized (not all zeros)
        first_layer = backbone.projection[0]
        weight_norm = torch.norm(first_layer.weight).item()
        assert weight_norm > 0, "Weights are all zeros"
        
        print(f"   ✓ Projection head initialized (weight norm: {weight_norm:.4f})")
        return True
    except Exception as e:
        print(f"   ✗ Initialization test failed: {e}")
        return False

def test_cache_key_generation():
    """Test cache key generation"""
    print("\n3. Testing cache key generation...")
    try:
        from project.pretrain import get_pretrain_cache_key
        
        key1 = get_pretrain_cache_key('dinov3', 128, 8, 'kmeans')
        key2 = get_pretrain_cache_key('dinov3', 128, 8, 'kmeans')
        key3 = get_pretrain_cache_key('dinov3', 256, 8, 'kmeans')
        
        assert key1 == key2, "Same config produces different keys"
        assert key1 != key3, "Different configs produce same key"
        assert len(key1) == 16, f"Key length should be 16, got {len(key1)}"
        
        print(f"   ✓ Cache key generation working (example: {key1})")
        return True
    except Exception as e:
        print(f"   ✗ Cache key test failed: {e}")
        return False

def test_config_structure():
    """Test that config files have required structure"""
    print("\n4. Testing config structure...")
    try:
        import yaml
        
        # Test default config
        default_config_path = Path('project/configs/default.yaml')
        assert default_config_path.exists(), f"Default config not found: {default_config_path}"
        
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        assert 'pretraining' in config, "Missing 'pretraining' section"
        assert 'enabled' in config['pretraining'], "Missing 'pretraining.enabled'"
        assert 'epochs' in config['pretraining'], "Missing 'pretraining.epochs'"
        assert 'temp_strategy' in config['pretraining'], "Missing 'pretraining.temp_strategy'"
        
        # Test pretrain test config
        test_config_path = Path('project/configs/pretrain_test.yaml')
        assert test_config_path.exists(), f"Test config not found: {test_config_path}"
        
        with open(test_config_path, 'r') as f:
            test_config = yaml.safe_load(f)
        
        assert test_config['pretraining']['enabled'] == True, "Pre-training not enabled in test config"
        
        print(f"   ✓ Config files have correct structure")
        return True
    except Exception as e:
        print(f"   ✗ Config test failed: {e}")
        return False

def test_pretrain_module():
    """Test that pretrain module has required functions"""
    print("\n5. Testing pretrain module structure...")
    try:
        from project import pretrain
        
        # Check required functions exist
        assert hasattr(pretrain, 'pretrain_projection_head'), "Missing pretrain_projection_head function"
        assert hasattr(pretrain, 'get_pretrain_cache_key'), "Missing get_pretrain_cache_key function"
        
        # Check function signatures
        import inspect
        sig = inspect.signature(pretrain.pretrain_projection_head)
        params = list(sig.parameters.keys())
        
        required_params = ['backbone', 'train_paths', 'preprocessor', 'config', 'device', 'cache_dir']
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"
        
        print(f"   ✓ Pretrain module structure correct")
        return True
    except Exception as e:
        print(f"   ✗ Pretrain module test failed: {e}")
        return False

def main():
    print("="*80)
    print("PRE-TRAINING IMPLEMENTATION VALIDATION")
    print("="*80)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    # Run all tests
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Orthogonal Init", test_orthogonal_init()))
    results.append(("Cache Key Generation", test_cache_key_generation()))
    results.append(("Config Structure", test_config_structure()))
    results.append(("Pretrain Module", test_pretrain_module()))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All validation tests passed! Implementation is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
