# Distance Metric Experiments Guide

## Overview

The codebase now supports **two distance metrics** with full consistency between training and inference:

1. **Cosine Distance**: `1 - cosine_similarity`
2. **L2 (Euclidean) Distance**: `||x - anchor||₂`

**Key improvement**: Distance metric is now used **consistently** in:
- ✅ Loss function (training)
- ✅ Model forward pass (training)
- ✅ Anomaly score computation (inference)

---

## ✅ Verification: Anchors ARE Prenormalized

The code analysis confirms:

### In Model (`model.py`)

**1. Projected anchors are normalized:**
```python
# Line 186 in _get_projected_anchors()
anchor_global_projected = F.normalize(anchor_global_projected, dim=1)
```

**2. Sample embeddings are normalized:**
```python
# Line 141 in DINOv3Backbone.forward()
global_feat = F.normalize(global_feat, dim=1)
```

**3. Dense features are normalized (for cosine):**
```python
# Line 234 in AnomalyDetector.forward()
dense_feat_norm = F.normalize(dense_feat, dim=-1)
anchor_dense_norm = F.normalize(anchor_dense, dim=-1)
```

✅ **Conclusion**: Both embeddings and anchors are prenormalized, making cosine distance appropriate!

---

## Distance Metrics Explained

### 1. Cosine Distance

**Formula:**
```
distance = 1 - cosine_similarity
         = 1 - (x · anchor) / (||x|| · ||anchor||)
```

Since embeddings are normalized (||x|| = ||anchor|| = 1):
```
distance = 1 - (x · anchor)
```

**Anomaly Score (Inference):**
```python
# Minimum cosine distance to any anchor
anomaly_score = min(1 - cosine_similarity(sample, anchors))
              = 1 - max(cosine_similarity(sample, anchors))
```

**Properties:**
- ✅ Angle-based: measures direction similarity
- ✅ Scale-invariant: only cares about direction
- ✅ Range: [0, 2] (or [0, 1] if normalized embeddings point same hemisphere)
- ✅ Works well with normalized embeddings (like DINOv3)

**Use case:** 
- When semantic similarity matters more than magnitude
- DINOv3 embeddings are already normalized
- Common in representation learning

---

### 2. L2 (Euclidean) Distance

**Formula:**
```
distance = ||x - anchor||₂
         = sqrt(Σ(x_i - anchor_i)²)
```

**Anomaly Score (Inference):**
```python
# Minimum L2 distance to any anchor
anomaly_score = min(||sample - anchor||₂)
```

**Properties:**
- ✅ Magnitude-based: measures absolute difference
- ✅ Sensitive to scale
- ✅ Range: [0, ∞) (but typically [0, 2√D] for normalized embeddings)
- ✅ Geometric interpretation: straight-line distance

**Use case:**
- When absolute feature differences matter
- Traditional metric learning
- Explicit geometric distance

---

## Configuration

### Cosine Distance Config (`configs/cosine_distance.yaml`)

```yaml
loss:
  margin: 0.5               # Smaller margin for cosine (range [0,2])
  alpha: 1.0
  beta: 0.0
  distance_metric: 'cosine' # Use cosine distance
```

### L2 Distance Config (`configs/l2_distance.yaml`)

```yaml
loss:
  margin: 1.0               # Larger margin for L2
  alpha: 1.0
  beta: 0.0
  distance_metric: 'euclidean'  # Use L2 distance
```

---

## Running Experiments

### Single Experiment

```bash
# Cosine distance experiment → experiments/bmad_eigenface_k8_cos/
python main.py --config configs/cosine_distance.yaml --auto-name

# L2 distance experiment → experiments/bmad_eigenface_k8_l2/
python main.py --config configs/l2_distance.yaml --auto-name
```

### Batch Comparison (All Combinations)

```bash
# Test both distance metrics with multiple strategies
python run_anchor_experiments.py \
    --strategies eigenface kmeans random \
    --n-anchors 8 16 \
    --distance-metrics cosine euclidean

# Results in:
# - experiments/bmad_eigenface_k8_cos/
# - experiments/bmad_eigenface_k8_l2/
# - experiments/bmad_kmeans_k8_cos/
# - experiments/bmad_kmeans_k8_l2/
# ... etc.
```

---

## Expected Results

### Hypothesis

**Cosine distance may perform better** because:
1. DINOv3 embeddings are already normalized
2. Semantic similarity (angle) more important than magnitude
3. Widely used in vision transformers

| Distance Metric | Expected Val AUROC | Expected Test AUROC |
|----------------|-------------------|---------------------|
| **Cosine**     | 0.83-0.87         | 0.80-0.83          |
| **L2**         | 0.80-0.85         | 0.78-0.81          |

### Comparison Matrix

| Metric   | Sensitivity | Scale-Invariant | Range     | Best For                |
|----------|-------------|-----------------|-----------|-------------------------|
| Cosine   | Angle       | ✅ Yes          | [0, 2]    | Semantic similarity     |
| L2       | Magnitude   | ❌ No           | [0, ∞)    | Geometric distance      |

---

## Implementation Details

### Model Forward Pass (`model.py`)

```python
# Cosine distance
if self.distance_metric == 'cosine':
    cosine_sim = torch.mm(global_feat, anchor_global.t())  # (B, K)
    global_distances = 1.0 - cosine_sim

# L2 distance  
else:
    global_distances = torch.cdist(global_feat, anchor_global, p=2)
```

### Loss Function (`loss.py`)

```python
# Loss supports both metrics
loss = AnchorMarginLoss(
    margin=margin,
    alpha=1.0,
    beta=0.0,
    distance_metric='cosine'  # or 'euclidean'
)
```

### Anomaly Score (Inference)

```python
# Same for both: minimum distance to any anchor
anomaly_score = global_distances.min(dim=1)[0]  # (B,)
```

**For cosine:**
- `anomaly_score = 1 - max(cos_sim)` 
- Range: [0, 2], lower = more normal

**For L2:**
- `anomaly_score = min(L2_dist)`
- Range: [0, ∞), lower = more normal

---

## Experiment Matrix

Full factorial design:

| Strategy   | K  | Distance | Exp Name                  |
|------------|----|---------|-----------------------------|
| Random     | 8  | Cosine  | `bmad_random_k8_cos`       |
| Random     | 8  | L2      | `bmad_random_k8_l2`        |
| K-means    | 8  | Cosine  | `bmad_kmeans_k8_cos`       |
| K-means    | 8  | L2      | `bmad_kmeans_k8_l2`        |
| Eigenface  | 8  | Cosine  | `bmad_eigenface_k8_cos`    |
| Eigenface  | 8  | L2      | `bmad_eigenface_k8_l2`     |

**Total combinations**: 3 strategies × 3 K values × 2 distances = **18 experiments**

---

## Analysis Example

```python
import json
from pathlib import Path
import pandas as pd

# Collect results
results = []
for strategy in ['random', 'kmeans', 'eigenface']:
    for k in [4, 8, 16]:
        for dist in ['cos', 'l2']:
            exp_dir = Path(f'experiments/bmad_{strategy}_k{k}_{dist}')
            metrics_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
            
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                
                results.append({
                    'strategy': strategy,
                    'k': k,
                    'distance': dist,
                    'image_auroc': metrics['image_auroc'],
                    'pixel_auroc': metrics.get('pixel_auroc', None)
                })

# Create dataframe
df = pd.DataFrame(results)

# Compare distance metrics
print("Distance Metric Comparison (averaged over strategies):")
print(df.groupby('distance')['image_auroc'].mean())

# Best configuration
best = df.loc[df['image_auroc'].idxmax()]
print(f"\nBest: {best['strategy']} K={best['k']} {best['distance']}: AUROC={best['image_auroc']:.4f}")
```

---

## Key Takeaways

### ✅ Consistency Achieved

- **Before**: Loss used L2, but inference used cosine → mismatch!
- **Now**: Both training and inference use same metric → consistent!

### 📊 What to Test

1. **Distance metric effect**: Does cosine outperform L2?
2. **Interaction with anchors**: Does distance choice matter more for random vs eigenface anchors?
3. **Optimal configuration**: Best (strategy, K, distance) combination?

### 🎯 Recommended Experiments

**Priority 1 (Quick baseline):**
```bash
python main.py --config configs/cosine_distance.yaml --auto-name
python main.py --config configs/l2_distance.yaml --auto-name
```

**Priority 2 (Full comparison):**
```bash
python run_anchor_experiments.py --strategies eigenface --n-anchors 8 --distance-metrics cosine euclidean
```

**Priority 3 (Complete grid):**
```bash
python run_anchor_experiments.py  # All combinations
```

---

## Summary

✅ **Anchors are prenormalized** (verified in code)  
✅ **Two distance metrics**: cosine and L2  
✅ **Full consistency**: same metric in training and inference  
✅ **Auto-naming**: experiments named by strategy_k<N>_<metric>  
✅ **Batch runner**: test all combinations easily  

**Next steps:**
1. Run cosine vs L2 experiments
2. Compare AUROC across distance metrics
3. Determine if cosine is better for normalized embeddings
