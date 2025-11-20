"""
Test script to verify data loading works correctly with PNG images
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import load_dataset_paths
from data import create_dataloaders

def test_data_loading():
    """Test that data loading works with the BraTS2021_slice structure"""
    
    print("="*80)
    print("TESTING DATA LOADING")
    print("="*80)
    
    # Test data path loading
    data_root = '../data/BraTS2021_slice'
    
    print(f"\nLoading paths from: {data_root}")
    train_paths, val_paths, val_labels, val_mask_paths, test_paths, test_labels, test_mask_paths = load_dataset_paths(data_root)
    
    # Verify we got data
    assert len(train_paths) > 0, "No training images found!"
    assert len(val_paths) > 0, "No validation images found!"
    assert len(test_paths) > 0, "No test images found!"
    
    print("\n✓ Path loading successful!")
    
    # Check that all paths are PNG files
    assert all(p.endswith('.png') for p in train_paths), "Not all training paths are PNG!"
    assert all(p.endswith('.png') for p in val_paths), "Not all validation paths are PNG!"
    assert all(p.endswith('.png') for p in test_paths), "Not all test paths are PNG!"
    
    print("✓ All paths are PNG files")
    
    # Verify label counts
    print(f"\nValidation set:")
    print(f"  Normal (label 0): {val_labels.count(0)}")
    print(f"  Anomaly (label 1): {val_labels.count(1)}")
    
    print(f"\nTest set:")
    print(f"  Normal (label 0): {test_labels.count(0)}")
    print(f"  Anomaly (label 1): {test_labels.count(1)}")
    
    # Test dataloader creation
    print("\n" + "="*80)
    print("TESTING DATALOADER CREATION")
    print("="*80)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        val_mask_paths=val_mask_paths,
        test_mask_paths=test_mask_paths,
        batch_size=8,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        target_size=(256, 256)
    )
    
    print(f"\n✓ Dataloaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\n" + "="*80)
    print("TESTING BATCH LOADING")
    print("="*80)
    
    # Train batch
    train_batch = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  Image shape: {train_batch['image'].shape}")
    print(f"  Label shape: {train_batch['label'].shape}")
    print(f"  Labels: {train_batch['label'].tolist()}")
    assert train_batch['image'].shape[1] == 1, "Expected 1 channel for grayscale!"
    assert train_batch['image'].shape[2] == 256 and train_batch['image'].shape[3] == 256, "Expected 256x256 images!"
    
    # Val batch
    val_batch = next(iter(val_loader))
    print(f"\nValidation batch:")
    print(f"  Image shape: {val_batch['image'].shape}")
    print(f"  Label shape: {val_batch['label'].shape}")
    print(f"  Labels: {val_batch['label'].tolist()}")
    if 'mask' in val_batch:
        print(f"  Mask shape: {val_batch['mask'].shape}")
        print(f"  Mask unique values: {val_batch['mask'].unique().tolist()}")
    
    # Test batch
    test_batch = next(iter(test_loader))
    print(f"\nTest batch:")
    print(f"  Image shape: {test_batch['image'].shape}")
    print(f"  Label shape: {test_batch['label'].shape}")
    print(f"  Labels: {test_batch['label'].tolist()}")
    if 'mask' in test_batch:
        print(f"  Mask shape: {test_batch['mask'].shape}")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print("\nData loading is working correctly with PNG images.")
    print("You can now run the full training pipeline with:")
    print("  python main.py --config configs/default.yaml")

if __name__ == '__main__':
    test_data_loading()
