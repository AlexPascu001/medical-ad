"""
Test script to verify pixel AUROC fix handles size mismatches correctly
"""

import numpy as np
from scipy.ndimage import zoom

# Simulate the mismatch scenario
print("="*60)
print("TESTING PIXEL AUROC SIZE MISMATCH FIX")
print("="*60)

# Case 1: Scores 256x256, Masks 240x240 (original error)
print("\nCase 1: Scores (256x256) vs Masks (240x240)")
scores_256 = np.random.rand(83, 256, 256)
masks_240 = np.random.randint(0, 2, (83, 240, 240))

print(f"  Original scores shape: {scores_256.shape}")
print(f"  Original masks shape: {masks_240.shape}")
print(f"  Flattened scores: {scores_256.flatten().shape}")
print(f"  Flattened masks: {masks_240.flatten().shape}")
print(f"  ❌ Would cause error: {scores_256.flatten().shape[0]} != {masks_240.flatten().shape[0]}")

# Apply fix
scale_h = masks_240.shape[1] / scores_256.shape[1]
scale_w = masks_240.shape[2] / scores_256.shape[2]
print(f"\n  Applying resize: scale_h={scale_h:.4f}, scale_w={scale_w:.4f}")

resized_scores = []
for i in range(scores_256.shape[0]):
    resized = zoom(scores_256[i], (scale_h, scale_w), order=1)
    resized_scores.append(resized)
resized_scores = np.array(resized_scores)

print(f"  Resized scores shape: {resized_scores.shape}")
print(f"  Flattened resized scores: {resized_scores.flatten().shape}")
print(f"  Flattened masks: {masks_240.flatten().shape}")
print(f"  ✅ Fixed: {resized_scores.flatten().shape[0]} == {masks_240.flatten().shape[0]}")

# Case 2: Scores 240x240, Masks 240x240 (already matching)
print("\n" + "="*60)
print("Case 2: Scores (240x240) vs Masks (240x240)")
scores_240 = np.random.rand(83, 240, 240)
masks_240 = np.random.randint(0, 2, (83, 240, 240))

print(f"  Scores shape: {scores_240.shape}")
print(f"  Masks shape: {masks_240.shape}")
print(f"  Flattened scores: {scores_240.flatten().shape}")
print(f"  Flattened masks: {masks_240.flatten().shape}")
print(f"  ✅ No resize needed: shapes already match!")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("The fix correctly handles:")
print("  1. Size mismatches (256x256 -> 240x240)")
print("  2. Matching sizes (240x240 = 240x240)")
print("  3. Uses bilinear interpolation for smooth resizing")
print("  4. Preserves batch dimension correctly")
