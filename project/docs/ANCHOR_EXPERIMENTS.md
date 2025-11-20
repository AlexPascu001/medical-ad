# Anchor Strategy Experiments Guide

## Overview

The codebase now supports **three anchor generation strategies** with configurable number of anchors as a hyperparameter:

1. **Random** (Baseline) - Randomly select training images as anchors
2. **K-Means** - Cluster images in pixel space, use centroids as anchors  
3. **Eigenface** (Original) - PCA + clustering in eigenface space

Experiment names are **auto-generated** based on strategy and number of anchors:
- Format: `bmad_<strategy>_k<n_anchors>`
- Examples: `bmad_random_k8`, `bmad_kmeans_k16`, `bmad_eigenface_k4`

---

## Quick Start

### 1. Single Experiment

Run with a specific strategy and auto-generated experiment name:

```bash
# Random baseline with 8 anchors → experiments/bmad_random_k8/
python main.py --config configs/random_baseline.yaml --auto-name

# K-means with 8 anchors → experiments/bmad_kmeans_k8/
python main.py --config configs/kmeans.yaml --auto-name

# Eigenface with 8 anchors → experiments/bmad_eigenface_k8/
python main.py --config configs/default.yaml --auto-name
```

### 2. Batch Experiments (Multiple Strategies & Numbers)

Run comprehensive comparison experiments:

```bash
# Test all strategies with K=[4, 8, 16] anchors
python run_anchor_experiments.py

# Custom: only random and kmeans with K=[8, 16, 32]
python run_anchor_experiments.py --strategies random kmeans --n-anchors 8 16 32

# Skip anchor generation if already exists (faster re-runs)
python run_anchor_experiments.py --skip-anchors
```

---

## Configuration Files

### Random Baseline (`configs/random_baseline.yaml`)
```yaml
anchor:
  strategy: 'random'
  n_anchors: 8  # Change this to 4, 16, 32, etc.
```

### K-Means (`configs/kmeans.yaml`)
```yaml
anchor:
  strategy: 'kmeans'
  n_anchors: 8
  max_images_for_pca: 5000  # Max images for clustering
```

### Eigenface (`configs/default.yaml`)
```yaml
anchor:
  strategy: 'eigenface'
  n_components: 50  # PCA components
  n_anchors: 8
```

---

## Anchor Strategies Explained

### 1. Random (Baseline)

**Method:**
- Randomly select K training images
- Use them directly as anchors
- No learning or optimization

**Pros:**
- Fastest generation
- No hyperparameters (except K)
- Good baseline for comparison

**Cons:**
- May select outliers or redundant samples
- No guarantee of diversity

**Use case:** Baseline to measure how much anchor quality matters

---

### 2. K-Means

**Method:**
- Flatten images to vectors
- Run k-means clustering in image space
- Use cluster centroids as anchors

**Pros:**
- Simple and interpretable
- Guaranteed diversity (centroids are separated)
- Directly optimizes pixel-space similarity

**Cons:**
- Sensitive to pixel-level noise
- May not capture semantic features
- Slower than random

**Use case:** Middle ground between random and eigenface

---

### 3. Eigenface (Original)

**Method:**
- Compute mean image, center data
- PCA to get eigenfaces (principal components)
- Cluster in eigenface coefficient space
- Reconstruct anchors from centroids

**Pros:**
- Captures main modes of variation
- More robust to noise (PCA denoising)
- Semantic grouping via eigenfaces

**Cons:**
- Most complex
- Requires tuning n_components
- Slowest generation

**Use case:** Best anchor quality, captures data distribution

---

## Number of Anchors (K)

The number of anchors K is a **key hyperparameter** affecting:
- **Coverage**: More anchors = better coverage of normal variation
- **Specificity**: Fewer anchors = tighter normal class boundary
- **Training speed**: More anchors = slower training

**Recommended values to test:**
- K=4: Minimal coverage, very specific
- K=8: Default, good balance
- K=16: Better coverage, may overfit
- K=32: High coverage, slower training

---

## Expected Results

### Hypothesis

**Random < K-means < Eigenface** in terms of AUROC

| Strategy   | K | Expected Val AUROC | Expected Test AUROC |
|------------|---|-------------------|---------------------|
| Random     | 8 | 0.70-0.75         | 0.68-0.73           |
| K-means    | 8 | 0.78-0.83         | 0.76-0.80           |
| Eigenface  | 8 | 0.83-0.87         | 0.80-0.83           |

### Effect of K

| K  | Coverage | Specificity | Expected AUROC |
|----|----------|-------------|----------------|
| 4  | Low      | High        | 0.75-0.80      |
| 8  | Medium   | Medium      | 0.80-0.85      |
| 16 | High     | Low         | 0.82-0.87      |
| 32 | Very High| Very Low    | 0.80-0.85 (may overfit) |

---

## Experiment Workflow

### Full Comparison Study

```bash
# 1. Run all experiments (9 total: 3 strategies × 3 K values)
python run_anchor_experiments.py --strategies random kmeans eigenface --n-anchors 4 8 16

# Results will be in:
# experiments/bmad_random_k4/
# experiments/bmad_random_k8/
# experiments/bmad_random_k16/
# experiments/bmad_kmeans_k4/
# experiments/bmad_kmeans_k8/
# experiments/bmad_kmeans_k16/
# experiments/bmad_eigenface_k4/
# experiments/bmad_eigenface_k8/
# experiments/bmad_eigenface_k16/
```

### Compare Results

```python
import json
from pathlib import Path

strategies = ['random', 'kmeans', 'eigenface']
k_values = [4, 8, 16]

for strategy in strategies:
    for k in k_values:
        exp_dir = Path(f'experiments/bmad_{strategy}_k{k}')
        metrics_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
        
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"{strategy:10s} K={k:2d}: AUROC={metrics['image_auroc']:.4f}")
```

---

## Tips & Best Practices

### Memory Considerations

For large datasets, limit images used for anchor generation:
```yaml
anchor:
  max_images_for_pca: 5000  # Use subset for faster generation
```

### Reproducibility

Always set seed in config:
```yaml
seed: 42  # For reproducible results
```

### Quick Testing

Use `--skip-anchors` to reuse anchors when testing other hyperparameters:
```bash
python main.py --config configs/default.yaml --auto-name --skip-anchors
```

---

## File Structure

```
configs/
  ├── random_baseline.yaml  # Random strategy config
  ├── kmeans.yaml           # K-means strategy config
  └── default.yaml          # Eigenface strategy config

experiments/
  ├── bmad_random_k8/       # Auto-generated experiment dirs
  ├── bmad_kmeans_k8/
  └── bmad_eigenface_k8/

project/
  ├── anchors.py            # All anchor strategies
  ├── main.py               # Training with auto-naming
  └── run_anchor_experiments.py  # Batch experiment runner
```

---

## Testing

Verify all strategies work:
```bash
python test_anchor_strategies.py
# Generates: test_anchor_strategies.png (visualizes all 3 strategies)
```

---

## Summary

✅ **3 anchor strategies implemented**: random, k-means, eigenface  
✅ **Configurable K**: test 4, 8, 16, 32+ anchors  
✅ **Auto-naming**: experiments named by strategy and K  
✅ **Batch runner**: test all combinations easily  
✅ **Reproducible**: seed control for all strategies  

**Next steps:**
1. Run baseline experiments with all 3 strategies
2. Compare AUROC across strategies
3. Find optimal K for each strategy
4. Analyze which strategy works best for your data
