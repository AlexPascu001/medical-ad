"""
BMAD Brain MRI Dataset Loader and Preprocessing
Handles FLAIR slices from BraTS2021 with patient-wise splits
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BMADPreprocessor:
    """Handles intensity normalization and preprocessing for FLAIR slices"""
    
    def __init__(self, target_size: Tuple[int, int] = (240, 240), normalize_mode: str = 'zscore_only'):
        self.target_size = target_size
        self.normalize_mode = normalize_mode
        
    def clip_percentiles(self, img: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
        """Clip intensity to percentile range to remove outliers"""
        p_low = np.percentile(img, lower)
        p_high = np.percentile(img, upper)
        return np.clip(img, p_low, p_high)
    
    def normalize_slice(self, img: np.ndarray) -> np.ndarray:
        """Z-score normalization per slice"""
        mean = img.mean()
        std = img.std()
        if std < 1e-8:  # Handle empty slices
            return np.zeros_like(img)
        return (img - mean) / std
    
    def scale_to_01(self, img: np.ndarray) -> np.ndarray:
        """Min-max scale to [0, 1] range (for use before ImageNet normalization)"""
        mn = img.min()
        mx = img.max()
        if mx - mn < 1e-8:
            return np.zeros_like(img)
        return (img - mn) / (mx - mn)
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline"""
        # Clip outliers
        img = self.clip_percentiles(img)
        # Normalize (mode-dependent)
        if self.normalize_mode == 'minmax_imagenet':
            img = self.scale_to_01(img)
        else:  # zscore_only (default / backward-compatible)
            img = self.normalize_slice(img)
        # Resize
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        return img.astype(np.float32)


class BMADDataset(Dataset):
    """Dataset for BMAD Brain MRI FLAIR slices"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        mask_paths: Optional[List[str]] = None,
        preprocessor: Optional[BMADPreprocessor] = None,
        augment: bool = False,
        is_training: bool = True,
        normalize_mode: str = 'zscore_only',
        augment_mode: str = 'full'
    ):
        """
        Args:
            image_paths: List of paths to FLAIR slice images
            labels: Binary labels (0=normal, 1=anomaly). None for train (all normal)
            mask_paths: Paths to pixel-level masks (for test/val anomalies)
            preprocessor: BMADPreprocessor instance
            augment: Whether to apply data augmentation
            is_training: Training mode flag
            normalize_mode: 'zscore_only' (z-score, no ImageNet norm) or
                            'minmax_imagenet' (min-max [0,1] + ImageNet norm)
            augment_mode: Training augmentation preset: 'full', 'flip_only', or 'none'
        """
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [0] * len(image_paths)
        self.mask_paths = mask_paths
        self.preprocessor = preprocessor or BMADPreprocessor(normalize_mode=normalize_mode)
        self.is_training = is_training
        
        # Build transform list
        augment_ops = []
        if augment and is_training:
            if augment_mode == 'full':
                augment_ops = [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=10,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5
                    ),
                ]
            elif augment_mode == 'flip_only':
                augment_ops = [A.HorizontalFlip(p=0.5)]
            elif augment_mode == 'none':
                augment_ops = []
            else:
                raise ValueError(f"Unsupported augment_mode: {augment_mode}")

        # Only apply ImageNet normalization when preprocessor outputs [0,1] range
        if normalize_mode == 'minmax_imagenet':
            final_ops = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
        else:  # zscore_only — data already z-score normalized, just convert to tensor
            final_ops = [ToTensorV2()]

        self.transform = A.Compose(augment_ops + final_ops)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        # Load image
        img_path = self.image_paths[idx]
        img_path_str = str(img_path)
        img = np.load(img_path_str) if img_path_str.endswith('.npy') else cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess
        img = self.preprocessor.preprocess(img)
        
        # Convert grayscale to 3-channel BEFORE transforms so that
        # A.Normalize applies per-channel ImageNet stats correctly.
        img = np.stack([img, img, img], axis=-1)  # (H, W) -> (H, W, 3)
        
        # Load mask if available
        mask = None
        if self.mask_paths and idx < len(self.mask_paths) and self.mask_paths[idx]:
            mask_path = self.mask_paths[idx]
            if mask_path:  # Check if not None
                mask_path_str = str(mask_path)
                mask = np.load(mask_path_str) if mask_path_str.endswith('.npy') else \
                       cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape != self.preprocessor.target_size:
                        mask = cv2.resize(mask, self.preprocessor.target_size, interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0).astype(np.float32)
        
        # Apply transforms (includes ImageNet normalization + ToTensorV2)
        if mask is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        output = {
            'image': img,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'path': img_path
        }
        
        if mask is not None:
            # Ensure mask has correct shape (H, W) without channel dimension
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
            output['mask'] = mask
        
        return output


def create_dataloaders(
    train_paths: List[str],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
    val_mask_paths: Optional[List[str]] = None,
    test_mask_paths: Optional[List[str]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (256, 256),
    normalize_mode: str = 'zscore_only',
    train_augment_mode: str = 'full'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with proper preprocessing
    
    Args:
        train_paths: Paths to normal training images
        val_paths: Paths to validation images
        val_labels: Binary labels for validation
        test_paths: Paths to test images
        test_labels: Binary labels for test
        val_mask_paths: Optional pixel masks for validation
        test_mask_paths: Optional pixel masks for test
        batch_size: Batch size
        num_workers: Number of dataloader workers
        target_size: Target image size (H, W)
        normalize_mode: 'zscore_only' or 'minmax_imagenet'
        train_augment_mode: Training augmentation preset: 'full', 'flip_only', or 'none'
    
    Returns:
        train_loader, val_loader, test_loader
    """
    preprocessor = BMADPreprocessor(target_size=target_size, normalize_mode=normalize_mode)
    
    # Training dataset (normal only, with augmentation)
    train_dataset = BMADDataset(
        image_paths=train_paths,
        labels=None,  # All normal
        preprocessor=preprocessor,
        augment=True,
        is_training=True,
        normalize_mode=normalize_mode,
        augment_mode=train_augment_mode
    )
    
    # Validation dataset
    val_dataset = BMADDataset(
        image_paths=val_paths,
        labels=val_labels,
        mask_paths=val_mask_paths,
        preprocessor=preprocessor,
        augment=False,
        is_training=False,
        normalize_mode=normalize_mode
    )
    
    # Test dataset
    test_dataset = BMADDataset(
        image_paths=test_paths,
        labels=test_labels,
        mask_paths=test_mask_paths,
        preprocessor=preprocessor,
        augment=False,
        is_training=False,
        normalize_mode=normalize_mode
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader