# Quick Start Guide: Testing Pre-Training

## 1. Quick Test (5 minutes)

Test that everything works with a minimal run:

```bash
# Test with pre-training enabled (5 epochs pre-training + 5 main epochs)
python project/main.py \
  --config project/configs/pretrain_test.yaml \
  --exp-name quick_test \
  training.epochs=5
```

**Expected behavior:**
1. Shows "PRE-TRAINING PROJECTION HEAD" banner
2. Generates 8 temporary kmeans anchors
3. Trains projection head for 5 epochs
4. Saves cache to `./cache/pretrained_projections/`
5. Discards temporary anchors
6. Generates real random anchors (now projected through pre-trained head)
7. Runs main training for 5 epochs
8. Reports AUROC

## 2. Full Pre-Training Test (30-60 minutes)

Full test with pre-training:

```bash
python project/main.py \
  --config project/configs/pretrain_test.yaml \
  --exp-name pretrain_full
```

**Configuration:**
- Pre-training: 5 epochs (kmeans, 8 temporary anchors)
- Main training: 50 epochs (random, 8 real anchors)
- Expected improvement: +5-10% AUROC over baseline

## 3. Verify Cache Reuse

Run the same config twice to verify caching:

```bash
# First run: pre-trains projection head
python project/main.py \
  --config project/configs/pretrain_test.yaml \
  --exp-name cache_test_1

# Second run: should load from cache (instant)
python project/main.py \
  --config project/configs/pretrain_test.yaml \
  --exp-name cache_test_2
```

**Expected behavior:**
- First run: Shows "PRE-TRAINING PROJECTION HEAD" and trains for 5 epochs
- Second run: Shows "LOADING CACHED PRE-TRAINED PROJECTION HEAD" (instant)

## 4. Ablation Study: Compare All Configurations

### Baseline (No Pre-training)

```bash
# Disable pre-training in config
python project/main.py \
  --config project/configs/default.yaml \
  --exp-name baseline_no_pretrain \
  anchor.strategy=random \
  pretraining.enabled=false
```

### Orthogonal Init Only

```bash
# Pre-training disabled, but orthogonal init still active
python project/main.py \
  --config project/configs/default.yaml \
  --exp-name orthogonal_init_only \
  anchor.strategy=random \
  pretraining.enabled=false
```

**Note**: Orthogonal init is always active (hard-coded in model.py), so this tests orthogonal init alone.

### Pre-training Only (5 epochs)

```bash
python project/main.py \
  --config project/configs/pretrain_test.yaml \
  --exp-name pretrain_5ep \
  pretraining.epochs=5
```

### Pre-training (10 epochs)

```bash
python project/main.py \
  --config project/configs/pretrain_test.yaml \
  --exp-name pretrain_10ep \
  pretraining.epochs=10
```

### Combined (Default)

```bash
# Both orthogonal init + pre-training (5 epochs)
python project/main.py \
  --config project/configs/pretrain_test.yaml \
  --exp-name pretrain_combined
```

## 5. Compare Results

After running all ablation experiments:

```bash
# Compare AUROC across experiments
python project/compare_experiment_results.py \
  --experiments-dir experiments \
  --filter "baseline_no_pretrain|orthogonal_init_only|pretrain_*"
```

## 6. Advanced: Custom Pre-training Configuration

Create a custom config for specific experiments:

```yaml
# my_pretrain_config.yaml
pretraining:
  enabled: true
  epochs: 10              # More pre-training
  lr: 0.0005              # Lower LR for stability
  temp_strategy: 'random' # Try random instead of kmeans
  loss_beta: 0.5          # Add repeller term
```

Run with custom config:

```bash
python project/main.py \
  --config my_pretrain_config.yaml \
  --exp-name custom_pretrain
```

## 7. Monitor Pre-Training Progress

Pre-training logs show:

```
================================================================================
PRE-TRAINING PROJECTION HEAD
================================================================================
Strategy: Fix temporal misalignment by pre-training projection head
  before projecting real anchors

Pre-training Configuration:
  Epochs: 5
  Learning rate: 0.001
  Batch size: 64
  Temporary anchors: 8 (kmeans)
  Loss weights: α=1.0, β=0.0
  Distance metric: euclidean
  Cache key: a1b2c3d4e5f6g7h8

Step 1: Generating temporary anchors...
✓ Generated 8 temporary anchors using kmeans

Step 2: Projecting temporary anchors through projection head...
✓ Temporary anchor embeddings: torch.Size([8, 128])

Step 3: Setting up pre-training...
Trainable parameters: 98,432 (projection head only)

Step 4: Pre-training projection head...
================================================================================
Pre-train Epoch 1/5: 100%|█████████| 150/150 [00:30<00:00, loss=0.4523, attract=0.4523]

Epoch 1/5:
  Loss: 0.4523 (Attract: 0.4523, Repel: 0.0000)

[... 4 more epochs ...]

Step 5: Saving pre-trained projection head to cache...
✓ Saved to ./cache/pretrained_projections/pretrained_projection_a1b2c3d4e5f6g7h8.pt

Step 6: Discarding temporary anchors...
✓ Temporary anchors discarded (will generate real anchors next)

================================================================================
PRE-TRAINING COMPLETE
================================================================================
✓ Projection head is now pre-trained and ready for real anchor projection
✓ This fixes the temporal misalignment issue
✓ Real anchors will be projected through this pre-trained head
```

## 8. Troubleshooting

### Error: "Python was not found"

Configure Python environment:

```bash
# Activate virtual environment first
D:/Documents/FMI/Disertatie/medical-ad/venv/Scripts/activate
python project/main.py --config project/configs/pretrain_test.yaml
```

### Error: "Cannot find module 'pretrain'"

Make sure you're in the correct directory:

```bash
cd D:/Documents/FMI/Disertatie/medical-ad
python project/main.py --config project/configs/pretrain_test.yaml
```

### Cache Issues

Clear cache to force re-training:

```bash
# Remove all cached pre-trained weights
Remove-Item -Path "./cache/pretrained_projections/*" -Force
```

Or use `force_retrain=True` in code (modify main.py temporarily):

```python
pretrain_projection_head(
    ...
    force_retrain=True  # Force re-training even if cache exists
)
```

## 9. Expected Timeline

- **Quick test** (training.epochs=5): ~5 minutes
  - Pre-training: ~1-2 minutes
  - Main training: ~2-3 minutes
  
- **Full test** (training.epochs=50): ~30-60 minutes
  - Pre-training: ~1-2 minutes (then cached)
  - Main training: ~25-55 minutes
  
- **Ablation study** (5 experiments × 50 epochs): ~2-4 hours
  - First run: ~30-60 min (pre-trains + caches)
  - Subsequent runs: instant pre-training (loads cache) + ~25-55 min main training

## 10. Success Criteria

✅ **Pre-training works** if you see:
- "PRE-TRAINING PROJECTION HEAD" banner
- 5 epochs of pre-training complete
- Cache file created in `./cache/pretrained_projections/`
- "PRE-TRAINING COMPLETE" banner

✅ **Cache works** if second run shows:
- "LOADING CACHED PRE-TRAINED PROJECTION HEAD"
- No pre-training epochs (instant)

✅ **Performance improves** if:
- AUROC improves by +5-10% over baseline
- Target: 0.57 → 0.67-0.72 (ideally 0.80)

---

**Ready to test!** Start with the quick test, then run full ablation study.
