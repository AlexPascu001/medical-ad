# Architecture Cheat Sheet - Dimensions Only

## 🎯 Complete Dimension Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT STAGE                                                     │
├─────────────────────────────────────────────────────────────────┤
│ Raw image:          (240, 240)          grayscale              │
│ Preprocessed:       (240, 240)          normalized             │
│ Batch:              (64, 3, 240, 240)   RGB tensor             │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ DINOV3 BACKBONE (FROZEN - 22M params)                          │
├─────────────────────────────────────────────────────────────────┤
│ Input:              (64, 3, 240, 240)                           │
│ Patches:            15×15 = 225 patches (patch_size=16)        │
│ All tokens:         (64, 230, 384)                             │
│   ├─ CLS:           (64, 384)           1 token                │
│   ├─ Register:      (64, 4, 384)        4 tokens               │
│   └─ Patches:       (64, 225, 384)      225 tokens             │
│ Reshaped patches:   (64, 15, 15, 384)   spatial grid           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ PROJECTION HEAD (TRAINABLE - 98,624 params)                    │
├─────────────────────────────────────────────────────────────────┤
│ Architecture:       384 → 192 → 128                            │
│                     Linear → ReLU → Linear                      │
│                                                                  │
│ Global input:       (64, 384)                                   │
│ Global output:      (64, 128)           [L2 normalized]        │
│                                                                  │
│ Dense input:        (64, 225, 384)      [flattened]            │
│ Dense output:       (64, 225, 128)      [projected patches]    │
│ Dense reshaped:     (64, 15, 15, 128)   [spatial]              │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ ANCHORS (K=8)                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Anchor images:      (8, 240, 240)       eigenface prototypes   │
│   ↓ DINOv3                                                      │
│ Original features:  (8, 384)             backbone space         │
│   ↓ Projection head                                             │
│ Projected:          (8, 128)             learned space          │
│   ↓ L2 normalize                                                │
│ Final anchors:      (8, 128)             ||·|| = 1             │
│                                                                  │
│ Dense anchors:      (8, 15, 15, 128)     patch embeddings      │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ DISTANCE COMPUTATION                                            │
├─────────────────────────────────────────────────────────────────┤
│ Global distances:   (64, 8)              batch × anchors        │
│   L2:               ||sample - anchor||₂                        │
│   Cosine:           1 - (sample · anchor)                       │
│                                                                  │
│ Dense distances:    (64, 8, 15, 15)     batch×anchors×spatial  │
│   Per-patch:        distance[b,k,h,w] = dist(sample[b,h,w],    │
│                                               anchor[k,h,w])     │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING: LOSS COMPUTATION                                      │
├─────────────────────────────────────────────────────────────────┤
│ Min distances:      (64,)                nearest anchor         │
│ Assigned anchors:   (64,)                index of nearest       │
│                                                                  │
│ Attractor loss:     scalar               0.5 × mean(min_dist²) │
│ Repeller loss:      0                    beta=0.0 (disabled)   │
│ Total loss:         scalar               alpha × L_A + beta × L_R │
│                                                                  │
│ Backprop updates:   98,624 params        projection head only  │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ INFERENCE: ANOMALY SCORING                                      │
├─────────────────────────────────────────────────────────────────┤
│ Image score:        (1,)                 min(global_distances)  │
│   Range:            0 to ∞               typically 0.5-3.0      │
│   Normal:           0.5-1.0              close to anchors       │
│   Anomaly:          2.0-3.0              far from anchors       │
│                                                                  │
│ Pixel scores:       (1, 15, 15)         min(dense_distances)   │
│   ↓ Bilinear upsample                                           │
│ Pixel map:          (1, 240, 240)       full resolution        │
│   Interpretation:   Higher = more anomalous                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Embedding Space Dimensions

| Space | Dimensions | Content |
|-------|-----------|---------|
| **Pixel** | 57,600 (240²) | Raw grayscale images |
| **DINOv3** | 384 | Pretrained semantic features |
| **Projected** | 128 | Learned task-specific embeddings |

**Key**: All distance comparisons happen in the **128D projected space**.

---

## 🔢 Batch Processing Shapes

### Forward Pass (batch_size=64)
```
Step                          Shape                    Notes
────────────────────────────────────────────────────────────────
Input batch                   (64, 3, 240, 240)       RGB images
↓ DINOv3
CLS tokens                    (64, 384)               Global features
Patch tokens                  (64, 225, 384)          15×15 patches
↓ Projection head
Global features               (64, 128)               Normalized
Dense features                (64, 15, 15, 128)       Spatial patches
↓ Distance to anchors
Global distances              (64, 8)                 To each anchor
Dense distances               (64, 8, 15, 15)         Per-patch distances
↓ Minimum distance
Image scores                  (64,)                   One per image
Pixel maps                    (64, 15, 15)            Spatial scores
```

---

## 🎯 Anchor Generation (One-time)

### Eigenface Strategy
```
Step                          Shape                    Notes
────────────────────────────────────────────────────────────────
Training images               (7500, 240, 240)        All normal
↓ Flatten
Flattened                     (7500, 57600)           Pixel vectors
↓ Compute mean
Mean image                    (57600,)                Average brain
↓ Subtract mean
Centered                      (7500, 57600)           Zero-centered
↓ PCA
Eigenfaces                    (50, 57600)             Top 50 components
Coefficients                  (7500, 50)              Projection coords
↓ K-means clustering
Cluster labels                (7500,)                 8 clusters
Centroids                     (8, 50)                 Cluster centers
↓ Reconstruct
Anchor images                 (8, 240, 240)           8 prototypes
↓ DINOv3
Anchor features               (8, 384)                Stored in model
↓ Project during training
Final anchors                 (8, 128)                Comparison space
```

---

## 💾 Memory Usage

### Model
```
Component                     Size
────────────────────────────────────────
DINOv3 backbone               88 MB      (22M × 4 bytes)
Projection head               0.4 MB     (98K × 4 bytes)
Anchors (global)              4 KB       (8 × 128 × 4 bytes)
Anchors (dense)               246 KB     (8 × 15 × 15 × 128 × 4)
Total model                   ~89 MB
```

### Training (batch_size=64)
```
Tensor                        Size
────────────────────────────────────────
Input batch                   44.2 MB    (64 × 3 × 240 × 240 × 4)
Backbone output               22.6 MB    (64 × 230 × 384 × 4)
Global features               32.8 KB    (64 × 128 × 4)
Dense features                7.4 MB     (64 × 15 × 15 × 128 × 4)
Gradients (projection)        0.4 MB     (98K × 4)
Per batch total               ~75 MB
Per epoch (117 batches)       ~8.8 GB
```

---

## ⚡ Performance

### Speed (GPU)
```
Operation                     Time           Throughput
──────────────────────────────────────────────────────────
Single image inference        ~16 ms         62 img/sec
Batch of 64                   ~20 ms         3200 img/sec
Full test set (3715 imgs)     ~0.93 sec
Training epoch (7500 imgs)    ~5 min
```

### Accuracy
```
Metric                        Value
──────────────────────────────────────
Image-level AUROC             82-83%
Pixel-level AUROC             87%
```

---

## 🔧 Hyperparameters

```yaml
# Architecture
backbone_dim: 384                 # DINOv3 small
projection_dim: 128               # Learned space
n_anchors: 8                      # Prototypes
patch_size: 16                    # ViT patch size
n_patches: 225                    # 15×15 grid

# Anchor generation
strategy: eigenface               # PCA + K-means
n_components: 50                  # PCA dimensions
max_images_for_pca: 5000         # Memory limit

# Training
batch_size: 64
epochs: 50
lr: 0.0001
weight_decay: 0.000001

# Loss
alpha: 1.0                        # Attractor weight
beta: 0.0                         # Repeller (OFF)
margin: 1.0                       # For repeller
distance_metric: euclidean        # or cosine
```

---

## 🎓 Key Concepts

### What Gets Trained?
- ✅ Projection head: 98,624 parameters
- ❌ DINOv3 backbone: 22M parameters (frozen)
- ❌ Anchors: Fixed after generation

### What Space Are Distances Computed In?
**128D projected space** - both samples and anchors are projected through the trainable head before distance computation.

### Why Normalize?
- Global features: L2 normalized for stable cosine similarity
- Anchors: Normalized after projection for fair comparison
- Dense features: Normalized only for cosine distance

### What's the Anomaly Score?
**Minimum distance to any anchor**
- Low score (0.5-1.0): Close to normal prototypes → normal
- High score (2.0-3.0): Far from all prototypes → anomaly

---

## 📈 Training Dynamics

```
Initialization:
  Sample to anchor distance: ~2.0 (random)
  Loss: ~2.0

Early training (epoch 1-10):
  Sample to anchor distance: 2.0 → 1.2
  Loss: 2.0 → 0.7
  Projection learns to pull samples toward anchors

Mid training (epoch 10-30):
  Sample to anchor distance: 1.2 → 0.8
  Loss: 0.7 → 0.3
  Fine-tuning cluster structure

Late training (epoch 30-50):
  Sample to anchor distance: 0.8 → 0.6
  Loss: 0.3 → 0.2
  Converged, samples tightly clustered
```

---

## 🔍 Example: Single Test Image

```python
# Input
image: (1, 3, 240, 240)

# Extract features
global_feat: (1, 128)
dense_feat: (1, 15, 15, 128)

# Compare to 8 anchors
anchors: (8, 128)
distances: (1, 8) = [1.85, 2.14, 1.92, 2.37, 2.08, 1.78, 2.21, 1.95]
                                                      ↑
                                               Closest anchor

# Anomaly score
image_score: 1.78    # Distance to nearest anchor
assigned: 5          # Anchor #5 is closest

# Pixel-level
dense_distances: (1, 8, 15, 15)
pixel_scores: (1, 15, 15) = min over 8 anchors
pixel_map: (1, 240, 240) = bilinear upsample

# Interpretation
if image_score < 1.0:
    "Normal - close to training distribution"
elif image_score < 2.0:
    "Borderline - review manually"
else:
    "Anomaly - far from all prototypes"
```

---

**That's it!** The entire architecture in dimension-focused notation.

For full explanations, see:
- `ARCHITECTURE_WALKTHROUGH.md` - Complete conceptual walkthrough
- `architecture_diagram.png` - Visual data flow
- `embedding_spaces.png` - Space transformation visualization
