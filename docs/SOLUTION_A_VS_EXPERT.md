# Solution A vs Expert's Decoupled Approach - Comparison

## Overview
Both approaches generate anchors in **384D DINOv3 semantic space** (not random pixel space), but differ fundamentally in how they handle the projection to 128D training space.

---

## Solution A: Re-Projection Each Forward

### Key Characteristics
- **Semantic anchors**: Generated via K-means in 384D DINOv3 space
- **Geometric targets**: NONE - anchors are re-projected each forward pass
- **Training dynamics**: Anchors "move" with the projection head as it trains
- **Pseudo-labels**: Computed ONCE in 384D space, then frozen

### Implementation
```python
# Anchor generation (ONCE at initialization)
dino_embeddings_384d = backbone.forward_features(images)[:, 0]
kmeans = KMeans(n_clusters=K)
anchor_embeddings_384d = kmeans.fit(dino_embeddings_384d)

# Pseudo-label computation (ONCE at training start)
distances_384d = cdist(dino_embeddings_384d, anchor_embeddings_384d)
fixed_labels = distances_384d.argmin(dim=1)  # Frozen throughout training

# Forward pass (EVERY iteration)
sample_projected = projection_head(dino_embeddings_384d)  # (B, 128)
anchor_projected = projection_head(anchor_embeddings_384d)  # (K, 128) - RE-PROJECTED!
loss = cam_loss(sample_projected, anchor_projected, fixed_assignments=fixed_labels)
```

### Pros
✅ **Semantic clustering preserved**: Anchors initialized via K-means in meaningful DINOv3 space  
✅ **Simpler implementation**: No need to manage separate semantic/geometric anchors  
✅ **Projection learns meaningful transformation**: Maps semantic space → anomaly detection space  
✅ **Anchors adapt**: Can adjust to better fit the projection head's learned representation  

### Cons
❌ **Moving targets**: Anchors change during training, making optimization harder  
❌ **Collapse risk**: Without strong regularization (diversity loss), all samples can map to single anchor  
❌ **Requires careful tuning**: Need diversity loss (delta=0.1), repeller loss (beta=0.5) to prevent collapse  
❌ **Less stable**: Gradient flow affects both projection weights AND anchor positions  

### Config Parameters
```yaml
anchor:
  use_embedding_space: true    # Generate in 384D DINOv3 space
  use_solution_a: true         # Enable re-projection each forward
  strategy: 'kmeans'
  n_anchors: 8

loss:
  delta: 0.1                   # CRITICAL: Diversity loss to prevent collapse
  diversity_temperature: 0.1   # Temperature for soft assignments
  alpha: 1.0                   # Attractor weight
  beta: 0.5                    # Repeller weight (push anchors apart)
```

---

## Expert's Decoupled Approach

### Key Characteristics
- **Semantic anchors (384D)**: Used ONLY for pseudo-label assignment (frozen)
- **Geometric targets (128D)**: FIXED training targets, NEVER change
- **Training dynamics**: Projection head learns to map samples → fixed geometric targets
- **Decoupling**: Semantic space (labeling) completely separate from geometric space (training)

### Implementation
```python
# Anchor generation (ONCE at initialization)
dino_embeddings_384d = backbone.forward_features(images)[:, 0]
kmeans = KMeans(n_clusters=K)
semantic_anchors_384d = kmeans.fit(dino_embeddings_384d)  # For labeling

# Geometric targets: random orthogonal vectors (NEVER change)
geometric_targets_128d = torch.randn(K, 128)
geometric_targets_128d = normalize(geometric_targets_128d)  # Fixed forever!

# Pseudo-label computation (ONCE at training start)
distances_384d = cdist(dino_embeddings_384d, semantic_anchors_384d)
fixed_labels = distances_384d.argmin(dim=1)  # Based on SEMANTIC anchors

# Forward pass (EVERY iteration)
sample_projected = projection_head(dino_embeddings_384d)  # (B, 128)
# Anchors are FIXED geometric targets (NOT re-projected!)
loss = cam_loss(sample_projected, geometric_targets_128d, fixed_assignments=fixed_labels)
```

### Pros
✅ **Fixed targets**: Geometric targets NEVER move, preventing "moving target" problem  
✅ **Stable optimization**: Gradient flow only affects projection weights  
✅ **No collapse risk**: Fixed targets prevent all samples from converging to single point  
✅ **Cleaner separation**: Semantic clustering (labeling) decoupled from geometric training (loss)  
✅ **More robust**: Less sensitive to hyperparameters, doesn't require strong diversity regularization  

### Cons
❌ **Random geometric targets**: 128D targets have no semantic meaning, just random orthogonal vectors  
❌ **Disconnect risk**: Projection must learn arbitrary mapping: Semantic_Cluster_K → Random_Vector_K  
❌ **More complex**: Need to manage two separate anchor sets  
❌ **Less adaptive**: Geometric targets can't adjust to projection head's learned representation  

### Config Parameters
```yaml
anchor:
  use_embedding_space: true       # Generate in 384D DINOv3 space
  use_solution_a: false           # Use decoupled approach (default if not set)
  strategy: 'kmeans'
  n_anchors: 8
  geometric_init: 'random_orthogonal'  # How to create 128D targets

loss:
  delta: 0.1                      # Diversity loss (optional, less critical than Solution A)
  alpha: 1.0                      # Attractor weight
  beta: 0.5                       # Repeller weight
```

---

## Side-by-Side Comparison

| Aspect | Solution A (Re-project) | Expert's Decoupled |
|--------|------------------------|-------------------|
| **Semantic anchors (384D)** | K-means in DINOv3 space | K-means in DINOv3 space |
| **Geometric targets (128D)** | ❌ None (re-project each forward) | ✅ Fixed random orthogonal vectors |
| **Anchor stability** | ❌ Move with projection head | ✅ FIXED, never change |
| **Collapse risk** | ⚠️ High (needs diversity loss) | ✅ Low (fixed targets prevent) |
| **Optimization difficulty** | ⚠️ Harder (moving targets) | ✅ Easier (stable targets) |
| **Semantic meaning in 128D** | ✅ Learned via projection | ❌ Random (no semantic link) |
| **Hyperparameter sensitivity** | ⚠️ High (delta, beta critical) | ✅ Low (more robust) |
| **Implementation complexity** | ✅ Simpler (single anchor set) | ⚠️ More complex (dual anchors) |

---

## Experiment Configurations

### Solution A Config
**File**: `solution_a_reproject.yaml`  
**Output**: `./experiments/solution_a_reproject`

```yaml
anchor:
  use_embedding_space: true
  use_solution_a: true        # CRITICAL: Enable re-projection
  n_anchors: 8

loss:
  delta: 0.1                  # Diversity loss (REQUIRED)
  alpha: 1.0
  beta: 0.5

training:
  epochs: 100
  early_stopping_patience: 1000  # Disabled
```

### Expert's Approach Config
**File**: `solution_a_384d.yaml`  
**Output**: `./experiments/solution_a_384d_embedding`

```yaml
anchor:
  use_embedding_space: true
  use_solution_a: false       # Use decoupled (default)
  geometric_init: 'random_orthogonal'
  n_anchors: 8

loss:
  delta: 0.1                  # Diversity loss (optional)
  alpha: 1.0
  beta: 0.5

training:
  epochs: 100
  early_stopping_patience: 1000  # Disabled
```

---

## Expected Results

### Solution A
- **Hypothesis**: Anchors adapt to projection space, potentially better fit
- **Risk**: May collapse without sufficient regularization (diversity + repeller)
- **Expected AUROC**: 0.70-0.85 (if no collapse), 0.50-0.60 (if collapse occurs)
- **Anchor distribution**: Requires strong diversity loss to maintain balance

### Expert's Decoupled
- **Hypothesis**: Fixed targets prevent collapse, more stable training
- **Risk**: Random geometric targets may be suboptimal for anomaly detection
- **Expected AUROC**: 0.75-0.90 (stable, robust)
- **Anchor distribution**: Naturally balanced due to fixed targets

---

## How to Run

### Run both experiments:
```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Run Solution A
python .\project\main.py --config .\project\configs\solution_a_reproject.yaml

# Run Expert's Approach
python .\project\main.py --config .\project\configs\solution_a_384d.yaml
```

### Or run all experiments (including ablations):
```bash
.\run_all_experiments.ps1
```
This will run **8 experiments total**:
1. Expert's approach (K=8, no early stopping)
2. Solution A (K=8, no early stopping)
3-8. Ablations (K=4,6,12 with/without early stopping)

---

## Key Research Questions

1. **Does re-projection help or hurt?**  
   Compare AUROC: Solution A vs Expert's Decoupled

2. **Does collapse occur in Solution A?**  
   Check anchor distribution: Should be ~250±100 per anchor, not 1900 to one

3. **Are random geometric targets sufficient?**  
   Expert's approach uses random 128D vectors - does this limit performance?

4. **Which approach is more stable?**  
   Compare training curves, loss stability, convergence speed

---

## Analysis After Running

### Metrics to Compare
- **Pixel AUROC**: Primary evaluation metric
- **Anchor distribution**: Samples per anchor (check for collapse)
- **Training stability**: Loss curves, convergence speed
- **Embedding visualization**: t-SNE plots of 128D projected space

### Red Flags (Solution A)
- Anchor distribution: 1800+ samples to one anchor, <50 to others → **COLLAPSE**
- AUROC < 0.65 → Likely collapse or poor convergence
- Loss oscillation → Moving target problem

### Success Indicators (Either Approach)
- Balanced anchor usage: ~250±150 samples per anchor (for K=8)
- AUROC > 0.80 → Strong anomaly detection performance
- Smooth loss curves → Stable training
