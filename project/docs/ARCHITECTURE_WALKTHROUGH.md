# Complete Architecture Walkthrough with Actual Dimensions

## Overview

This document walks through the **entire BMAD (Brain MRI Anomaly Detection)** pipeline, from raw image to final anomaly score, with **actual tensor dimensions** at every step.

**Config used**: `configs/default.yaml` (eigenface strategy, no repeller: `beta=0.0`)

---

## 📊 Key Numbers Summary

| Component | Value |
|-----------|-------|
| **Input Image** | 240×240 grayscale (FLAIR MRI) |
| **DINOv3 Backbone** | vit_small_patch16_dinov3.lvd1689m |
| **Backbone Embedding** | 384 dimensions |
| **Projection Head Output** | 128 dimensions |
| **Patch Size** | 16×16 pixels |
| **Patch Grid** | 15×15 = 225 patches |
| **Number of Anchors (K)** | 8 |
| **Batch Size** | 64 (training), 8 (example) |
| **PCA Components (M)** | 50 |

---

# Part 1: Data Preprocessing

## 1.1 Raw Input
- **Format**: FLAIR MRI slices (grayscale brain scans)
- **Original size**: 240×240 pixels
- **Type**: Single-channel medical images

```python
# Raw image loaded from disk
raw_image: np.ndarray
Shape: (240, 240)
dtype: float32
```

## 1.2 Preprocessing Pipeline (`BMADPreprocessor`)

**Steps**:
1. **Percentile clipping**: Remove outliers (0.5th to 99.5th percentile)
2. **Z-score normalization**: `(img - mean) / std`
3. **Resize**: Already 240×240, no change needed

```python
# After preprocessing
preprocessed_image: np.ndarray
Shape: (240, 240)
dtype: float32
Values: Normalized to ~N(0, 1)
```

## 1.3 Data Augmentation (Training only)

Applied transformations:
- Horizontal flip (p=0.5)
- Small rotation (±10°, p=0.5)
- Small shift/scale (±5% translation, ±10% zoom, p=0.5)

```python
# After augmentation and ToTensorV2()
augmented_image: torch.Tensor
Shape: (3, 240, 240)  # Converted to 3-channel RGB
dtype: float32
```

**Note**: Single-channel grayscale is repeated 3 times for DINOv3 (expects RGB).

## 1.4 Batch Construction

```python
# DataLoader batch
batch_images: torch.Tensor
Shape: (64, 3, 240, 240)
       │   │   │    └─── Width
       │   │   └──────── Height  
       │   └──────────── Channels (R=G=B for grayscale)
       └──────────────── Batch size
```

---

# Part 2: Anchor Generation (One-time Setup)

**Strategy**: Eigenface (PCA + K-means clustering)

## 2.1 Load Training Images

```python
training_images: np.ndarray
Shape: (7500, 240, 240)
       │     │    └──── Width
       │     └───────── Height
       └─────────────── Number of training samples
```

## 2.2 Eigenface Decomposition

### Step 1: Mean Image
```python
# Flatten images
X: np.ndarray
Shape: (7500, 57600)  # 57600 = 240 × 240
       │     └───────── Flattened pixels

# Compute mean
mean_image: np.ndarray
Shape: (57600,)
```

### Step 2: Mean-Centering
```python
# Subtract mean from each image
X_centered: np.ndarray
Shape: (7500, 57600)
Φ_i = X_i - μ
```

### Step 3: PCA (Principal Component Analysis)
```python
# Fit PCA with n_components=50
pca = PCA(n_components=50)

# Eigenvectors (eigenfaces)
eigenvectors: np.ndarray
Shape: (50, 57600)
       │   └──────────── Flattened pixel space
       └──────────────── Number of principal components (M)

# Eigenvalues
eigenvalues: np.ndarray
Shape: (50,)

# Project images to eigenface space
coefficients: np.ndarray
Shape: (7500, 50)
       │     └──────── Eigenface coefficients
       └────────────── Training samples
```

**Explained**: Each image is now represented by 50 coefficients instead of 57,600 pixels.

### Step 4: K-means Clustering
```python
# Cluster in eigenface coefficient space
kmeans = KMeans(n_clusters=8)
cluster_labels: np.ndarray
Shape: (7500,)  # Cluster assignment for each image

# Cluster centroids
centroids: np.ndarray
Shape: (8, 50)
       │  └──── Eigenface coefficients
       └─────── Number of anchors (K)
```

### Step 5: Reconstruct Anchor Images
```python
# For each centroid k:
# X_anchor = μ + Σ(c_k[i] × eigenvector[i])

anchor_images: np.ndarray
Shape: (8, 240, 240)
       │   │    └──── Width
       │   └───────── Height
       └───────────── Number of anchors (K)
```

**These 8 anchor images** represent the 8 "prototypical" normal brain patterns.

---

# Part 3: Feature Extraction (DINOv3 Backbone)

## 3.1 Input to Backbone

```python
# Batch of images
x: torch.Tensor
Shape: (64, 3, 240, 240)
       │   │   │    └─── Width
       │   │   └──────── Height
       │   └──────────── RGB channels
       └──────────────── Batch size
Device: cuda
dtype: float32
```

## 3.2 Vision Transformer Processing

**Model**: `vit_small_patch16_dinov3.lvd1689m`

### Patch Embedding
```python
# Divide image into 16×16 patches
num_patches_h = 240 // 16 = 15
num_patches_w = 240 // 16 = 15
total_patches = 15 × 15 = 225

# Embed each patch
patch_embeddings: torch.Tensor
Shape: (64, 225, 384)
       │   │    └──── Embedding dimension (DINOv3 small)
       │   └───────── Number of patches
       └───────────── Batch size
```

### Add Special Tokens
DINOv3 adds:
- 1 **CLS token** (global representation)
- 4 **Register tokens** (learned memory)

```python
# After adding special tokens
features_with_tokens: torch.Tensor
Shape: (64, 230, 384)
       │   │    └──── Embedding dimension
       │   └───────── 1 CLS + 4 REG + 225 PATCH tokens
       └───────────── Batch size

Token order: [CLS, REG1, REG2, REG3, REG4, PATCH1, ..., PATCH225]
```

### Transformer Blocks
```python
# After 12 transformer layers (vit_small)
backbone_output: torch.Tensor
Shape: (64, 230, 384)
       │   │    └──── Still 384 dimensions
       │   └───────── Still 230 tokens
       └───────────── Batch size

# Features are FROZEN (backbone.requires_grad = False)
```

## 3.3 Extract CLS and Patch Tokens

```python
# CLS token (global representation)
cls_token: torch.Tensor
Shape: (64, 384)
       │   └──── Backbone embedding dim
       └──────── Batch size

# Patch tokens (skip 4 register tokens)
patch_tokens: torch.Tensor
Shape: (64, 225, 384)
       │   │    └──── Backbone embedding dim
       │   └───────── Number of patches (15×15)
       └───────────── Batch size
```

## 3.4 Reshape Patch Tokens to Spatial Grid

```python
# Reshape to 2D spatial layout
patch_grid: torch.Tensor
Shape: (64, 15, 15, 384)
       │   │   │   └──── Embedding dimension
       │   │   └──────── Width in patches
       │   └──────────── Height in patches
       └──────────────── Batch size
```

---

# Part 4: Trainable Projection Head

**Purpose**: Learn task-specific embeddings (frozen backbone + trainable head)

## 4.1 Projection Architecture

```python
projection_head = nn.Sequential(
    nn.Linear(384, 192),  # First layer: 384 → 192
    nn.ReLU(),
    nn.Linear(192, 128)   # Second layer: 192 → 128
)

# Total trainable parameters: 98,624
# (384×192 + 192) + (192×128 + 128) = 73,728 + 24,576 + 320 = 98,624
```

## 4.2 Project Global Features

```python
# Before projection
cls_token: torch.Tensor
Shape: (64, 384)

# After projection
global_feat_projected: torch.Tensor
Shape: (64, 128)
       │   └──── Projection dimension
       └──────── Batch size

# Apply L2 normalization
global_feat = F.normalize(global_feat_projected, dim=1)
Shape: (64, 128)
||global_feat[i]|| = 1.0  # Unit vectors
```

## 4.3 Project Dense Features

```python
# Before projection
patch_grid: torch.Tensor
Shape: (64, 15, 15, 384)

# Flatten patches
patches_flat: torch.Tensor
Shape: (64, 225, 384)
       │   │    └──── Backbone embedding dim
       │   └───────── 15 × 15 patches
       └───────────── Batch size

# Apply projection to each patch
dense_projected: torch.Tensor
Shape: (64, 225, 128)
       │   │    └──── Projection dimension
       │   └───────── Number of patches
       └───────────── Batch size

# Reshape back to spatial
dense_feat: torch.Tensor
Shape: (64, 15, 15, 128)
       │   │   │   └──── Projection dimension
       │   │   └──────── Width in patches
       │   └──────────── Height in patches
       └──────────────── Batch size
```

**Note**: Dense features are NOT L2-normalized here (normalized later for cosine distance).

---

# Part 5: Anchor Embeddings

## 5.1 Extract Anchor Features (Once at Initialization)

```python
# 8 anchor images
anchor_images: torch.Tensor
Shape: (8, 3, 240, 240)
       │  │   │    └─── Width
       │  │   └──────── Height
       │  └──────────── RGB channels
       └─────────────── Number of anchors (K)

# Pass through frozen DINOv3 backbone
anchor_backbone_features: torch.Tensor (CLS tokens)
Shape: (8, 384)
       │  └──── Backbone embedding dim
       └─────── Number of anchors
```

## 5.2 Store Original Anchor Embeddings

```python
# Stored in model (NOT trainable)
model.anchor_global_original: torch.Tensor
Shape: (8, 384)
       │  └──── Backbone embedding dim
       └─────── Number of anchors (K)
Buffer: True (not a parameter, but saved with model)
```

## 5.3 Project Anchors During Forward Pass

**Key insight**: Anchors are projected through the **same trainable head** as samples!

```python
# During training, project anchors
anchor_global_projected: torch.Tensor
Shape: (8, 128)
       │  └──── Projection dimension
       └─────── Number of anchors

# Apply L2 normalization (for cosine distance)
anchor_global = F.normalize(anchor_global_projected, dim=1)
Shape: (8, 128)
||anchor_global[k]|| = 1.0  # Unit vectors
```

---

# Part 6: Distance Computation

**Configuration**: `distance_metric: 'euclidean'` (L2 distance)

## 6.1 Global Distance Matrix

```python
# Sample embeddings
global_feat: torch.Tensor
Shape: (64, 128)
       │   └──── Projection dimension
       └──────── Batch size

# Anchor embeddings
anchor_global: torch.Tensor
Shape: (8, 128)
       │  └──── Projection dimension
       └─────── Number of anchors

# Compute pairwise L2 distances
# Distance = ||sample[i] - anchor[k]||_2
global_distances: torch.Tensor
Shape: (64, 8)
       │   └──── Distance to each anchor
       └──────── Batch size

# Example values (after normalization, L2 range [0, 2√128] ~ [0, 22.6])
global_distances[0] = [1.24, 0.87, 1.56, 0.92, 1.83, 1.45, 1.12, 1.67]
                        │     │                  │
                        │     └── Closest anchor (index 1, distance 0.87)
                        └── Distance to anchor 0
```

**For Cosine Distance** (if `distance_metric='cosine'`):
```python
# Cosine similarity
cosine_sim: torch.Tensor
Shape: (64, 8)
cosine_sim = global_feat @ anchor_global.T  # (64,128) × (128,8) → (64,8)

# Cosine distance = 1 - similarity
global_distances = 1.0 - cosine_sim
Range: [0, 2]  # 0=identical, 2=opposite
```

## 6.2 Dense Distance Computation (Patch-Level)

**Only computed if** `return_dense=True` and `use_dense=True` in loss.

```python
# Sample dense features
dense_feat: torch.Tensor
Shape: (64, 15, 15, 128)
       │   │   │   └──── Projection dimension
       │   │   └──────── Width in patches
       │   └──────────── Height in patches
       └──────────────── Batch size

# Anchor dense features (stored)
anchor_dense: torch.Tensor
Shape: (8, 15, 15, 128)
       │  │   │   └──── Projection dimension
       │  │   └──────── Width in patches
       │  └──────────── Height in patches
       └─────────────── Number of anchors

# Flatten spatial dimensions
dense_flat: torch.Tensor
Shape: (64, 225, 128)
       │   │    └──── Projection dimension
       │   └───────── 15 × 15 patches
       └───────────── Batch size

anchor_flat: torch.Tensor
Shape: (8, 225, 128)
       │  │    └──── Projection dimension
       │  └───────── 15 × 15 patches
       └─────────── Number of anchors

# Compute distance for each patch to corresponding anchor patch
# For each anchor k:
#   dist[batch, k, patch_idx] = ||dense[batch, patch_idx] - anchor[k, patch_idx]||_2

dense_distances: torch.Tensor
Shape: (64, 8, 225)
       │   │  └──── Distance for each patch location
       │   └─────── Distance to each anchor
       └─────────── Batch size

# Reshape to spatial
dense_distances: torch.Tensor
Shape: (64, 8, 15, 15)
       │   │  │   └──── Width in patches
       │   │  └──────── Height in patches
       │   └─────────── Distance to each anchor
       └─────────────── Batch size
```

---

# Part 7: Loss Computation

**Loss function**: Anchor-Margin Loss with attractor + repeller terms

**Config**: `alpha=1.0`, `beta=0.0` (NO REPELLER), `margin=1.0`

## 7.1 Attractor Loss (Pull samples to nearest anchor)

```python
# Find minimum distance to any anchor for each sample
min_distances: torch.Tensor
Shape: (64,)
min_distances, assigned_anchors = global_distances.min(dim=1)

# Example:
# min_distances = [0.87, 1.23, 0.65, 0.98, ...]
# assigned_anchors = [1, 3, 1, 0, ...]  # Which anchor is closest

# Attractor loss: L_A = (1/2) × mean(min_distances²)
loss_attract = 0.5 * (min_distances ** 2).mean()

# Example calculation:
# loss_attract = 0.5 × mean([0.87², 1.23², 0.65², ...])
#              = 0.5 × mean([0.7569, 1.5129, 0.4225, ...])
#              ≈ 0.5 × 0.95 = 0.475
```

**Interpretation**: Penalize distance from each sample to its nearest anchor.

## 7.2 Repeller Loss (Push anchors apart)

**Config**: `beta=0.0` → **REPELLER DISABLED**

```python
# Compute pairwise anchor distances
anchor_distances: torch.Tensor
Shape: (8, 8)  # Distance between each pair of anchors

# Repeller loss: L_R = (1/2) × Σ max(0, 2m - ||anchor_i - anchor_j||)²
# But beta=0.0, so:
loss_repel = 0.0

# Total loss
total_loss = alpha × loss_attract + beta × loss_repel
           = 1.0 × 0.475 + 0.0 × 0.0
           = 0.475
```

**Note**: With `beta=0.0`, anchors are NOT pushed apart during training!

## 7.3 Dense Loss (Patch-Level, Optional)

**Config**: `use_dense=False` → **DENSE LOSS DISABLED**

If enabled (`use_dense=True`):
```python
# Dense distances
dense_distances: torch.Tensor
Shape: (64, 8, 15, 15)

# Min distance to any anchor for each patch
dense_min, _ = dense_distances.min(dim=1)  # (64, 15, 15)

# Spatial reduction (mean or max)
if spatial_reduction == 'mean':
    loss_dense = 0.5 * (dense_min ** 2).mean()
else:  # max
    loss_dense = 0.5 * (dense_min ** 2).amax(dim=(1,2)).mean()

# Combined loss
total_loss = global_weight × loss_global + dense_weight × loss_dense
```

---

# Part 8: Training Step

## 8.1 Gradient Computation

```python
# Zero gradients
optimizer.zero_grad()

# Forward pass
outputs = model(batch_images, return_dense=False)
# outputs = {
#     'global_feat': (64, 128),
#     'global_distances': (64, 8),
#     'dense_feat': (64, 15, 15, 128)
# }

# Compute loss
loss_dict = criterion(
    embeddings=outputs['global_feat'],        # (64, 128)
    anchor_embeddings=model._get_projected_anchors()[0]  # (8, 128)
)
loss = loss_dict['total_loss']  # Scalar

# Backward pass
loss.backward()

# Update ONLY projection head (backbone is frozen!)
optimizer.step()
```

**Trainable parameters**: Only the 98,624 parameters in the projection head.

## 8.2 Training Progression

```
Epoch 1:
  Batch 1: loss=0.523, min_dist=0.89
  Batch 2: loss=0.498, min_dist=0.85
  ...
  Batch 117: loss=0.445, min_dist=0.78

Epoch 2:
  Batch 1: loss=0.412, min_dist=0.72
  ...
```

**Goal**: Minimize loss → samples get closer to their nearest anchor.

---

# Part 9: Inference (Anomaly Detection)

## 9.1 Test Image Processing

```python
# Test image
test_image: torch.Tensor
Shape: (1, 3, 240, 240)
       │  │   │    └─── Width
       │  │   └──────── Height
       │  └──────────── RGB channels
       └─────────────── Batch size (1 for inference)

# Extract features
outputs = model(test_image, return_dense=True)

# Global distances
global_distances: torch.Tensor
Shape: (1, 8)  # Distance to each anchor
Example: [1.85, 2.14, 1.92, 2.37, 2.08, 1.78, 2.21, 1.95]
                                            └── Closest anchor
```

## 9.2 Image-Level Anomaly Score

```python
# Minimum distance to any anchor
image_score: torch.Tensor
Shape: (1,)
image_score = global_distances.min(dim=1)[0]
Example: 1.78

# Assigned anchor
assigned_anchor: int
assigned_anchor = global_distances.argmin(dim=1)[0]
Example: 5  # Anchor #5 is closest
```

**Interpretation**:
- **Low score** (e.g., 0.5-1.0): Normal (close to training distribution)
- **High score** (e.g., 2.0-3.0): Anomalous (far from all anchors)

## 9.3 Pixel-Level Anomaly Map

```python
# Dense distances
dense_distances: torch.Tensor
Shape: (1, 8, 15, 15)
       │  │  │   └──── Width in patches
       │  │  └──────── Height in patches
       │  └─────────── Distance to each anchor
       └────────────── Batch size

# Min distance across anchors for each patch
pixel_scores: torch.Tensor
Shape: (1, 15, 15)
pixel_scores, _ = dense_distances.min(dim=1)

# Example (15×15 grid):
# [[0.82, 0.91, 0.88, ..., 0.95],
#  [0.87, 0.79, 0.93, ..., 1.02],
#  ...
#  [0.85, 0.92, 2.45, ..., 0.89]]  ← Anomalous patch (2.45)
```

## 9.4 Upsampling to Image Resolution

```python
# Upsample from 15×15 to 240×240
pixel_scores_upsampled: torch.Tensor
Shape: (1, 1, 240, 240)

pixel_scores_upsampled = F.interpolate(
    pixel_scores.unsqueeze(1),  # Add channel dim
    size=(240, 240),
    mode='bilinear',
    align_corners=False
)

# Final pixel-level anomaly map
pixel_map: torch.Tensor
Shape: (1, 240, 240)

# Can be visualized as heatmap
# Higher values = more anomalous regions
```

---

# Part 10: Evaluation Metrics

## 10.1 Image-Level AUROC

```python
# Test set
n_test_samples = 3715
n_normal = 3383 (91%)
n_anomalous = 332 (9%)

# Collect predictions
image_scores: np.ndarray
Shape: (3715,)  # Anomaly score for each image

labels: np.ndarray
Shape: (3715,)  # 0=normal, 1=anomaly

# Compute AUROC
from sklearn.metrics import roc_auc_score
image_auroc = roc_auc_score(labels, image_scores)
Example: 0.8235  # 82.35%
```

**Interpretation**: Model can distinguish normal vs anomalous images with 82.35% probability.

## 10.2 Pixel-Level AUROC

```python
# For anomalous images with masks
n_anomalous_with_masks = 332

# Collect pixel predictions
all_pixel_scores: np.ndarray
Shape: (332 × 240 × 240,) = (19,123,200,)  # All pixels from anomalous images

all_pixel_labels: np.ndarray
Shape: (19,123,200,)  # 0=normal pixel, 1=lesion pixel

# Compute AUROC
pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
Example: 0.8706  # 87.06%
```

**Interpretation**: Model can localize lesions with 87.06% pixel-level accuracy.

---

# Part 11: Complete Data Flow Summary

## 11.1 Training Flow

```
Raw Image (240×240)
    ↓
Preprocess: normalize, augment
    ↓
Batch Tensor (64, 3, 240, 240)
    ↓
DINOv3 Backbone (FROZEN)
    ├─→ CLS Token (64, 384)
    └─→ Patch Tokens (64, 225, 384)
        ↓
Projection Head (TRAINABLE)
    ├─→ Global Features (64, 128) [normalized]
    └─→ Dense Features (64, 15, 15, 128)
        ↓
Distance to Anchors
    ├─→ Global Distances (64, 8)
    └─→ Dense Distances (64, 8, 15, 15)
        ↓
Anchor-Margin Loss
    ├─→ Attractor: pull to nearest anchor
    └─→ Repeller: push anchors apart (disabled: beta=0)
        ↓
Backprop → Update Projection Head
```

## 11.2 Inference Flow

```
Test Image (1, 3, 240, 240)
    ↓
DINOv3 Backbone (FROZEN)
    ↓
Projection Head (TRAINED)
    ├─→ Global Features (1, 128)
    └─→ Dense Features (1, 15, 15, 128)
        ↓
Distance to Anchors
    ├─→ Global Distances (1, 8)
    └─→ Dense Distances (1, 8, 15, 15)
        ↓
Anomaly Scores
    ├─→ Image Score: min(global_distances) → scalar
    └─→ Pixel Map: min(dense_distances, dim=1) → (1, 15, 15)
        ↓ Upsample
    Pixel Map (1, 240, 240)
```

---

# Part 12: Embedding Spaces Explained

## 12.1 What Space Are We In?

### **Original Space**: Raw Pixels
- Dimension: 240 × 240 = **57,600 dimensions**
- Anchors computed here (eigenface reconstruction)

### **DINOv3 Embedding Space**: Backbone Features
- Dimension: **384 dimensions**
- Semantic features learned by self-supervised pretraining
- **Frozen** during training

### **Projected Space**: Task-Specific Embeddings
- Dimension: **128 dimensions**
- Learned by trainable projection head
- **This is the comparison space** where distances are computed
- Both samples AND anchors projected to this space

### **Anchor Transformation Pipeline**:
```
Anchor Images (8, 240, 240) [Pixel Space]
    ↓ DINOv3 Backbone (frozen)
Anchor Features (8, 384) [DINOv3 Embedding Space]
    ↓ Projection Head (trainable)
Anchor Embeddings (8, 128) [Projected Space]
    ↓ L2 Normalize
Final Anchors (8, 128) [Normalized Projected Space]
```

## 12.2 Why Multiple Spaces?

1. **Pixel Space** (57,600D):
   - Too high-dimensional
   - Sensitive to noise
   - Hard to cluster

2. **DINOv3 Space** (384D):
   - Rich semantic features
   - Pretrained on ImageNet
   - Good initialization

3. **Projected Space** (128D):
   - Compact representation
   - Task-specific (learned for anomaly detection)
   - Distance comparisons happen here

---

# Part 13: Distance Metrics Comparison

## 13.1 L2 (Euclidean) Distance

**Formula**: `distance = ||x - anchor||₂`

```python
global_distances = torch.cdist(
    global_feat,      # (64, 128)
    anchor_global,    # (8, 128)
    p=2
)
# Output: (64, 8)

# Range: [0, ∞)
# For normalized vectors: [0, 2√128] ≈ [0, 22.6]
# Typical values: 0.5-2.5
```

**Characteristics**:
- Measures absolute difference
- Sensitive to magnitude
- Geometric interpretation

## 13.2 Cosine Distance

**Formula**: `distance = 1 - cosine_similarity`

```python
cosine_sim = global_feat @ anchor_global.T  # (64, 128) × (128, 8)
global_distances = 1.0 - cosine_sim
# Output: (64, 8)

# Range: [0, 2]
# 0 = identical direction
# 1 = orthogonal
# 2 = opposite direction
# Typical values: 0.2-1.5
```

**Characteristics**:
- Measures angular difference
- Scale-invariant
- Common for normalized embeddings

---

# Part 14: Memory and Computation

## 14.1 Memory Footprint

### Model Parameters
```
DINOv3 Backbone: 22M parameters (frozen, not in optimizer)
Projection Head: 98,624 parameters (trainable)
Anchor Storage: 8 × 128 × 4 bytes = 4 KB
```

### Training Batch (64 samples)
```
Input: 64 × 3 × 240 × 240 × 4 bytes = 44.2 MB
Backbone Output: 64 × 230 × 384 × 4 bytes = 22.6 MB
Dense Features: 64 × 15 × 15 × 128 × 4 bytes = 7.4 MB
Global Features: 64 × 128 × 4 bytes = 32.8 KB

Total per batch: ~75 MB
```

### Training Epoch (7500 / 64 = 117 batches)
```
Memory per epoch: ~8.8 GB
```

## 14.2 Computation Time

### Per Image (Inference)
```
DINOv3 Forward: ~15 ms (GPU)
Projection Head: ~0.5 ms
Distance Computation: ~0.1 ms
Total: ~16 ms → 62 images/second
```

### Full Test Set (3715 images)
```
Batch size 64: 3715 / 64 = 58 batches
Time: 58 × 16 ms = 0.93 seconds
```

---

# Part 15: Key Takeaways

## 15.1 The Pipeline in 5 Steps

1. **Preprocess**: 240×240 grayscale → normalized, augmented
2. **Extract**: DINOv3 (frozen) → 384D embeddings
3. **Project**: Trainable head → 128D task-specific features
4. **Compare**: Compute L2/cosine distance to 8 anchors
5. **Score**: Min distance = anomaly score (low=normal, high=anomaly)

## 15.2 Critical Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| Backbone | DINOv3-small | Self-supervised, semantic features |
| Freeze backbone | Yes | Fast training, 98K params instead of 22M |
| Projection dim | 128 | Balance: compact but expressive |
| Num anchors | 8 | Enough diversity, not too many |
| Distance metric | L2 / Cosine | L2 for magnitude, cosine for direction |
| Repeller term | OFF (beta=0) | Anchors already diverse from clustering |
| Dense features | Optional | Patch-level localization |

## 15.3 What Makes This Work?

1. **DINOv3 Pretraining**: Rich semantic features from self-supervision
2. **Anchor Diversity**: 8 prototypes cover normal brain variations
3. **Projection Learning**: Task-specific 128D space optimized for separation
4. **Distance-Based Detection**: Simple, interpretable anomaly scoring
5. **Normalization**: Unit vectors ensure fair comparisons

---

# Appendix: Quick Reference

## Dimension Cheat Sheet

```
Raw Image:            (240, 240)
Preprocessed:         (240, 240)
Batch Input:          (64, 3, 240, 240)

DINOv3 Tokens:        (64, 230, 384)
  - CLS:              (64, 384)
  - Patches:          (64, 225, 384)

Projection:
  - Global:           (64, 128)
  - Dense:            (64, 15, 15, 128)

Anchors:
  - Original:         (8, 384)
  - Projected:        (8, 128)

Distances:
  - Global:           (64, 8)
  - Dense:            (64, 8, 15, 15)

Anomaly Scores:
  - Image:            (64,)
  - Pixel (dense):    (64, 15, 15)
  - Pixel (upsampled):(64, 240, 240)
```

## Config Parameters

```yaml
# Data
target_size: [240, 240]
batch_size: 64

# Anchors
n_anchors: 8
n_components: 50  # PCA
strategy: 'eigenface'

# Model
backbone: 'vit_small_patch16_dinov3.lvd1689m'
embed_dim: 384
projection_dim: 128
patch_size: 16

# Loss
margin: 1.0
alpha: 1.0  # Attractor
beta: 0.0   # Repeller (OFF)
distance_metric: 'euclidean'

# Training
epochs: 50
lr: 0.0001
```

---

**End of Walkthrough**
