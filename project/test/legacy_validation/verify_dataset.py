"""
Simple script to verify dataset structure without requiring dependencies
"""

from pathlib import Path

def verify_structure():
    """Verify the BraTS2021_slice dataset structure"""
    
    print("="*80)
    print("VERIFYING DATASET STRUCTURE")
    print("="*80)
    
    data_root = Path('../data/BraTS2021_slice')
    
    if not data_root.exists():
        print(f"\n❌ ERROR: Dataset root not found at {data_root.absolute()}")
        return False
    
    print(f"\n✓ Dataset root found: {data_root.absolute()}")
    
    # Check train structure
    train_dir = data_root / 'train' / 'good'
    if not train_dir.exists():
        print(f"❌ ERROR: Train directory not found: {train_dir}")
        return False
    
    train_files = list(train_dir.glob('*.png'))
    print(f"\n✓ Train directory found: {len(train_files)} PNG files")
    
    # Check valid structure
    val_good_img = data_root / 'valid' / 'good' / 'img'
    val_good_label = data_root / 'valid' / 'good' / 'label'
    val_ungood_img = data_root / 'valid' / 'Ungood' / 'img'
    val_ungood_label = data_root / 'valid' / 'Ungood' / 'label'
    
    for dir_path, name in [
        (val_good_img, 'valid/good/img'),
        (val_good_label, 'valid/good/label'),
        (val_ungood_img, 'valid/Ungood/img'),
        (val_ungood_label, 'valid/Ungood/label')
    ]:
        if not dir_path.exists():
            print(f"❌ ERROR: {name} not found")
            return False
        files = list(dir_path.glob('*.png'))
        print(f"✓ {name}: {len(files)} PNG files")
    
    # Check test structure
    test_good_img = data_root / 'test' / 'good' / 'img'
    test_good_label = data_root / 'test' / 'good' / 'label'
    test_ungood_img = data_root / 'test' / 'Ungood' / 'img'
    test_ungood_label = data_root / 'test' / 'Ungood' / 'label'
    
    for dir_path, name in [
        (test_good_img, 'test/good/img'),
        (test_good_label, 'test/good/label'),
        (test_ungood_img, 'test/Ungood/img'),
        (test_ungood_label, 'test/Ungood/label')
    ]:
        if not dir_path.exists():
            print(f"❌ ERROR: {name} not found")
            return False
        files = list(dir_path.glob('*.png'))
        print(f"✓ {name}: {len(files)} PNG files")
    
    # Verify matching names
    print("\nVerifying image-mask pairing...")
    
    def check_pairing(img_dir, label_dir, name):
        img_files = {f.name for f in img_dir.glob('*.png')}
        label_files = {f.name for f in label_dir.glob('*.png')}
        
        if img_files != label_files:
            missing_in_label = img_files - label_files
            missing_in_img = label_files - img_files
            
            if missing_in_label:
                print(f"  ⚠️  {name}: {len(missing_in_label)} images missing labels")
            if missing_in_img:
                print(f"  ⚠️  {name}: {len(missing_in_img)} labels missing images")
            return False
        else:
            print(f"  ✓ {name}: All {len(img_files)} images have matching masks")
            return True
    
    all_paired = True
    all_paired &= check_pairing(val_good_img, val_good_label, 'valid/good')
    all_paired &= check_pairing(val_ungood_img, val_ungood_label, 'valid/Ungood')
    all_paired &= check_pairing(test_good_img, test_good_label, 'test/good')
    all_paired &= check_pairing(test_ungood_img, test_ungood_label, 'test/Ungood')
    
    print("\n" + "="*80)
    if all_paired:
        print("✓ DATASET STRUCTURE IS CORRECT!")
    else:
        print("⚠️  DATASET STRUCTURE HAS SOME ISSUES (see above)")
    print("="*80)
    
    # Summary
    train_count = len(train_files)
    val_good_count = len(list(val_good_img.glob('*.png')))
    val_ungood_count = len(list(val_ungood_img.glob('*.png')))
    test_good_count = len(list(test_good_img.glob('*.png')))
    test_ungood_count = len(list(test_ungood_img.glob('*.png')))
    
    print("\nDataset Summary:")
    print(f"  Training:   {train_count} normal images")
    print(f"  Validation: {val_good_count} normal + {val_ungood_count} anomaly = {val_good_count + val_ungood_count} total")
    print(f"  Test:       {test_good_count} normal + {test_ungood_count} anomaly = {test_good_count + test_ungood_count} total")
    
    return True

if __name__ == '__main__':
    verify_structure()
