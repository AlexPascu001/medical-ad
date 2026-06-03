# Implementation Summary: Projection Head Pre-Training

**Date:** January 11, 2026  
**Objective:** Fix temporal misalignment between anchors and samples during training

## Problem Statement

The root cause of poor AUROC (0.57 vs expected 0.80) was identified as:

**Temporal Misalignment**: Anchors are projected through a **randomly initialized** projection head at t=0, then stored as fixed targets. During training, the projection head learns and evolves, but anchors remain at their random t=0 positions. This creates a misalignment between the anchor space and the sample space.

## Solution Implemented

**Combination of Solution 1 + 3:**
1. **Solution 3**: Orthogonal initialization of projection head (gain=0.1)
2. **Solution 1**: Warm-start pre-training with temporary anchors before projecting real anchors

## Files Modified

### 1. `project/model.py`
- **Added**: `_init_projection_head()` method to DINOv3Backbone class
- **Purpose**: Initialize projection head with orthogonal weights (gain=0.1) and zero biases
- **Benefits**: Preserves DINOv3 semantic structure, better gradient flow from start

### 2. `project/pretrain.py` (NEW)
- **Created**: Complete pre-training module with caching support
- **Key Features**:
  - Generates temporary anchors (kmeans/random) from training samples
  - Projects temporary anchors through projection head
  - Trains ONLY projection head (DINOv3 frozen) using CAM loss for 5-10 epochs
  - Discards temporary anchors after pre-training
  - Caches pre-trained weights by (backbone, projection_dim, n_anchors, strategy) hash
  - Reuses cached weights across experiments for efficiency

### 3. `project/main.py`
- **Added**: Pre-training stage (Stage 2.5) after backbone creation, before anchor generation
- **Integration**: Calls `pretrain_projection_head()` with config-based control
- **Cache**: Uses `./cache/pretrained_projections/` directory

### 4. `project/configs/default.yaml`
- **Added**: `pretraining` configuration section with all parameters:
  - `enabled`: Toggle pre-training (default: false)
  - `epochs`: Number of pre-training epochs (5-10 recommended)
  - `lr`: Learning rate for pre-training (1e-3, higher than main training)
  - `batch_size`: Batch size (64)
  - `temp_anchors`: Number of temporary anchors (8, same as n_anchors)
  - `temp_strategy`: 'kmeans' or 'random' for temporary anchors
  - `loss_alpha`, `loss_beta`: CAM loss weights (1.0, 0.0 = no repeller)
  - `distance_metric`: 'euclidean' or 'cosine'

### 5. `project/configs/pretrain_test.yaml` (NEW)
- **Created**: Test configuration with pre-training enabled
- **Purpose**: Easy testing of new implementation

## Implementation Details

### Pre-Training Workflow

```
1. Generate temporary anchors (kmeans/random)
   ↓
2. Project temporary anchors through projection head
   ↓
3. Train projection head with CAM loss (DINOv3 frozen)
   - Attractor: Pull samples toward nearest temporary anchor
   - Repeller: Push temporary anchors apart (optional)
   ↓
4. Save pre-trained weights to cache
   ↓
5. Discard temporary anchors
   ↓
6. Project REAL anchors through pre-trained head
   ↓
7. Main training with properly aligned anchors
```

### Caching Strategy

- **Cache Key**: MD5 hash of (backbone_name, projection_dim, n_anchors, strategy)
- **Cache Location**: `./cache/pretrained_projections/pretrained_projection_{cache_key}.pt`
- **Cache Info**: JSON metadata with timestamp, config, final loss
- **Reuse**: Automatic reuse across experiments with same configuration
- **Force Retrain**: `force_retrain=True` parameter to override cache

### Loss Configuration

Pre-training uses the same CAM loss as main training:
- **Attractor** (α=1.0): Pulls samples toward nearest temporary anchor
- **Repeller** (β=0.0): Disabled (no repeller term)
- **Min-Norm** (γ=0.0): Disabled (temporary anchors are discarded)

## Expected Benefits

1. **Orthogonal Initialization**: +1-3% AUROC improvement
   - Better gradient flow from start
   - Preserves DINOv3 semantic structure

2. **Pre-Training**: +3-8% AUROC improvement
   - Projection head learns meaningful space before anchors are fixed
   - Eliminates temporal misalignment

3. **Combined**: +5-10% AUROC improvement
   - Target: 0.57 → 0.67-0.72 AUROC (ideally approaching 0.80)

## Usage

### Quick Test (Pre-training Enabled)

```bash
python project/main.py --config project/configs/pretrain_test.yaml --exp-name pretrain_test
```

### Custom Configuration

```yaml
pretraining:
  enabled: true           # Enable pre-training
  epochs: 5               # 5-10 epochs recommended
  lr: 0.001               # Higher than main training
  temp_strategy: 'kmeans' # or 'random'
```

### Ablation Studies

Compare performance across configurations:

1. **Baseline**: `pretraining.enabled: false` (current system)
2. **Orthogonal Init Only**: Pre-training disabled, but init code remains active
3. **Pre-training (5 epochs)**: `pretraining.epochs: 5`
4. **Pre-training (10 epochs)**: `pretraining.epochs: 10`
5. **Combined**: Both orthogonal init + pre-training (default)

## Next Steps

1. **Test Implementation**: Run quick test with pretrain_test.yaml
2. **Verify Caching**: Check that second run reuses cached weights
3. **Run Ablation Studies**: Compare baseline vs orthogonal vs pre-training vs both
4. **Monitor Metrics**: Track image/pixel AUROC improvements
5. **Optimize Hyperparameters**: Tune pre-training epochs, LR if needed

## Technical Notes

- Pre-training uses **same data** as main training (no separate pre-training dataset)
- Temporary anchors are **discarded** after pre-training (not used in main training)
- DINOv3 backbone remains **frozen** during pre-training (only projection head trains)
- Cache is **shared** across experiments with same (backbone, projection_dim, n_anchors, strategy)
- Pre-training is **idempotent**: re-running with same config reuses cache

## Validation

- [x] `pretrain.py` imports successfully
- [x] Orthogonal initialization added to model.py
- [x] Pre-training integrated into main.py
- [x] Config section added to default.yaml
- [x] Test config created (pretrain_test.yaml)
- [ ] Test run to verify end-to-end functionality
- [ ] Cache reuse verification
- [ ] AUROC improvement measurement

---

**Status**: Implementation complete, ready for testing
