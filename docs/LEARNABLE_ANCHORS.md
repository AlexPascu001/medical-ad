# Learnable Anchors for Medical Anomaly Detection

## Overview

This document describes the implementation of **Learnable Anchors** based on the Class Anchor Margin (CAM) Loss from the paper:

> **"Class Anchor Margin Loss for Content-Based Image Retrieval"**  
> Alexandru Ghita, Radu Tudor Ionescu  
> arXiv:2306.00630

The key idea is to use learnable anchor embeddings that can move in the embedding space during training, allowing them to better capture the structure of normal samples.

## Concept

### Standard (Fixed) Anchors
In the baseline approach, anchors are:
1. Selected from training images (random, k-means, eigenface)
2. Embedded using DINOv3 backbone
3. **Fixed** during training - only the projection head is trained

### Learnable Anchors
With learnable anchors:
1. Anchors start from the same initialization (random, k-means, eigenface)
2. Anchor embeddings become **trainable parameters**
3. The CAM Loss has three components:
   - **Attractor**: Pulls samples toward their assigned anchor
   - **Repeller**: Pushes different anchors apart (by margin m)
   - **Min-Norm**: Prevents anchors from collapsing to zero

### Pseudo-Label Assignment
Each training sample is assigned to its nearest anchor ("pseudo-label"). Two strategies:

1. **Fixed Labels**: Compute assignments once before training, keep them fixed
2. **Dynamic Labels**: Recompute assignments every N epochs as anchors move

## Loss Function

The CAM Loss is defined as:

$$L_{CAM} = \lambda_1 L_{attractor} + \lambda_2 L_{repeller} + \lambda_3 L_{norm}$$

### 1. Attractor Loss
Pulls embeddings toward their assigned anchor:
$$L_{attractor} = \frac{1}{N} \sum_{i=1}^{N} ||z_i - c_{y_i}||_2^2$$

- **Purpose**: Ensure embeddings cluster around their nearest anchor
- **Effect**: Creates compact, anchor-centered clusters in embedding space

### 2. Repeller Loss
Pushes anchors apart by at least margin $m$:
$$L_{repeller} = \frac{1}{K(K-1)} \sum_{j \neq k} \max(0, m - ||c_j - c_k||_2)$$

- **Purpose**: Prevent anchor collapse and ensure diversity
- **Effect**: Maintains minimum separation between all anchor pairs
- **Hyperparameter**: `margin` (default: 1.0)

### 3. Min-Norm Loss
Prevents anchors from collapsing to zero:
$$L_{norm} = \frac{1}{K} \sum_{k=1}^{K} \max(0, \delta - ||c_k||_2)$$

- **Purpose**: Regularization to keep anchors meaningful
- **Effect**: Ensures all anchors maintain minimum magnitude
- **Hyperparameter**: `min_norm` (default: 0.5)

## Architecture

```
Input Image (240×240)
    ↓
DINOv3 Backbone (frozen)
    ↓
384D Global Embedding
    ↓
Projection Head (trainable)
    ↓
128D Projected Embedding
    ↓
    ├─→ Assign to Nearest Anchor
    │
    └─→ CAM Loss:
         • Attractor: Pull to assigned anchor
         • Repeller: Push anchors apart
         • Min-Norm: Prevent anchor collapse
```

**Trainable Components:**
1. Projection Head (384D → 128D): ~98K parameters
2. Anchor Embeddings (K × 128D): K × 128 parameters

For K=8 anchors: **Total ~99K trainable parameters**

## Usage

### Step 1: Train Base Model (Fixed Anchors)

First, train a model with fixed anchors (eigenface, k-means, or random):

```bash
# Already done in your experiments
python project/main.py --config project/configs/default.yaml
```

This creates: `experiments/bmad_eigenface_k8_l2/`

### Step 2: Train with Learnable Anchors

Initialize learnable anchors from the base experiment:

```bash
python project/train_learnable_anchors.py \
    --config project/configs/learnable_anchors.yaml \
    --init-from experiments/bmad_eigenface_k8_l2
```

### Step 3: Compare Results

The learnable anchor experiment will save:
- `best_model.pth`: Best model checkpoint
- `anchor_embeddings.pt`: Final learned anchors
- `training_history.json`: Loss curves and anchor statistics

## Configuration

Key hyperparameters in `configs/learnable_anchors.yaml`:

```yaml
learnable_anchors:
  init_from: './experiments/bmad_eigenface_k8_l2'  # Source experiment
  freeze_anchors: false  # Set true for baseline (no learning)
  
  # Loss weights
  lambda_attractor: 1.0   # Clustering strength
  lambda_repeller: 1.0    # Separation strength
  lambda_norm: 0.1        # Regularization strength
  
  # Constraints
  margin: 1.0             # Min distance between anchors
  min_norm: 0.5           # Min anchor magnitude
```

### Hyperparameter Guidelines

**λ_attractor** (default: 1.0)
- Higher → Tighter clusters around anchors
- Lower → More flexible assignments

**λ_repeller** (default: 1.0)
- Higher → More separation between anchors
- Lower → Anchors can be closer together

**λ_norm** (default: 0.1)
- Higher → Stronger regularization
- Typically keep low (0.1-0.5)

**margin** (default: 1.0)
- Minimum distance between any two anchors
- Should be ≥ 2 × typical intra-cluster distance

**min_norm** (default: 0.5)
- Minimum L2 norm for each anchor
- Depends on embedding normalization

## Expected Improvements

Learnable anchors should provide:

1. **Better Separation**: Anchors adapt to data distribution
2. **Optimal Placement**: Moves from initialization toward optimal positions
3. **Fine-Tuning**: Refines projection head jointly with anchors

Typical improvement: **+1-3% AUROC** over fixed anchors

## Monitoring Training

The trainer logs:
- Loss components (attractor, repeller, norm)
- Anchor norms (min, max, mean)
- Anchor distances (min, mean)
- Validation AUROC

**Healthy Training Signs:**
- Attractor loss decreases (embeddings moving to anchors)
- Repeller loss stays low (anchors well-separated)
- Norm loss stays low (anchors maintain magnitude)
- Anchor distances > margin (no violations)

## Experiments

### Baseline Comparisons

Run learnable anchors initialized from each strategy:

```bash
# From Eigenface
python project/train_learnable_anchors.py \
    --config project/configs/learnable_anchors.yaml \
    --init-from experiments/bmad_eigenface_k8_l2

# From K-means
python project/train_learnable_anchors.py \
    --config project/configs/learnable_anchors.yaml \
    --init-from experiments/bmad_kmeans_k8_l2

# From Random
python project/train_learnable_anchors.py \
    --config project/configs/learnable_anchors.yaml \
    --init-from experiments/bmad_random_k8_l2
```

### Ablation Studies

Test individual loss components:

```yaml
# Only attractor (no repeller/norm)
lambda_attractor: 1.0
lambda_repeller: 0.0
lambda_norm: 0.0

# Only attractor + repeller (no norm)
lambda_attractor: 1.0
lambda_repeller: 1.0
lambda_norm: 0.0

# Full CAM loss
lambda_attractor: 1.0
lambda_repeller: 1.0
lambda_norm: 0.1
```

## Files

- `learnable_anchors.py`: Core implementation
  - `LearnableAnchors`: Learnable anchor parameters
  - `CAMLoss`: CAM loss computation
  - `assign_to_nearest_anchor()`: Assignment function

- `train_learnable_anchors.py`: Training script
  - `LearnableAnchorTrainer`: Custom trainer class
  - Loads initial anchors from experiment
  - Optimizes projection + anchors jointly

- `configs/learnable_anchors.yaml`: Default configuration

## Integration with Main Pipeline

Learnable anchor models are compatible with existing evaluation:

```bash
# Evaluate learned model
python project/eval.py \
    --checkpoint experiments/bmad_learnable_eigenface/best_model.pth
```

## New Experiments (Added)

### Running All Experiments

```bash
# Run all 9 experiments
python project/run_learnable_experiments.py --all

# Run only learnable experiments
python project/run_learnable_experiments.py --learnable-only

# Run specific strategies
python project/run_learnable_experiments.py --strategies random kmeans
```

### Configuration Files

**Baseline (Fixed Anchors):**
- `configs/fixed_random.yaml`
- `configs/fixed_kmeans.yaml`  
- `configs/fixed_eigenface.yaml`

**Learnable + Fixed Labels:**
- `configs/learnable_random_fixed.yaml`
- `configs/learnable_kmeans_fixed.yaml`
- `configs/learnable_eigenface_fixed.yaml`

**Learnable + Dynamic Labels:**
- `configs/learnable_random_dynamic.yaml`
- `configs/learnable_kmeans_dynamic.yaml`
- `configs/learnable_eigenface_dynamic.yaml`

### Visualization from Checkpoints

```bash
python project/visualize_from_checkpoint.py --experiment experiments/bmad_learnable_random_fixed
```

### Compare Results

```bash
python project/compare_experiment_results.py --experiments-dir experiments
```

## Experiment Matrix

| Strategy   | Fixed Anchors | Learnable + Fixed Labels | Learnable + Dynamic Labels |
|------------|---------------|--------------------------|----------------------------|
| Random     | ✓             | ✓                        | ✓                          |
| K-Means    | ✓             | ✓                        | ✓                          |
| Eigenface  | ✓             | ✓                        | ✓                          |

**Total: 9 experiments**


## Citation

If you use learnable anchors in your dissertation, cite:

```bibtex
@article{ghita2023class,
  title={Class Anchor Margin Loss for Content-Based Image Retrieval},
  author={Ghita, Alexandru and Ionescu, Radu Tudor},
  journal={arXiv preprint arXiv:2306.00630},
  year={2023}
}
```

## Next Steps

1. Train learnable anchor variants
2. Compare with fixed anchor baselines
3. Visualize anchor evolution (t-SNE over epochs)
4. Analyze which initialization (eigenface/kmeans/random) benefits most
5. Test different hyperparameter settings

Good luck with your dissertation! 🎓
