# Quick Start Guide - Using Enhanced Logging & Analysis

## What's New?

I've enhanced your medical anomaly detection project with:

1. **Comprehensive Training Logs** - Track all loss components and metrics per epoch
2. **Automatic Visualization** - Beautiful training curves saved automatically
3. **Analysis Tools** - Identify data issues, distribution shifts, and model problems
4. **Bug Fixes** - Fixed critical issues in loss function and logging

---

## Step 1: Retrain with Enhanced Logging

The training script now automatically tracks detailed metrics!

### Option A: Use Recommended Config (Fixed Issues)

```bash
cd project
python main.py --config configs/recommended.yaml
```

**What's different:**
- `beta: 0.0` - Disabled harmful repeller loss
- `use_dense: true` - Enabled pixel-level training
- Enhanced logging automatically enabled

### Option B: Use Experimental Simple Config

```bash
python main.py --config configs/experimental_simple.yaml
```

**What's different:**
- K-means anchors instead of PCA
- No projection head (simpler)
- More anchors (16 instead of 8)

---

## Step 2: Review Training Curves

After training completes, check:

**File:** `experiments/bmad_fixed/training_curves.png`

This shows 9 subplots:
1. **Total Loss** - Should DECREASE for both train and val
2. **Attractor Loss** - Should DECREASE (pulling samples to anchors)
3. **Repeller Loss** - Should be ~0 if beta=0
4. **Dense Loss** - Should DECREASE if enabled
5. **Image AUROC** - Should be stable/increasing on validation
6. **Pixel AUROC** - Should improve if dense loss is used
7. **Learning Rate** - Shows schedule
8. **Loss Components** - Breakdown of final epoch
9. **Train vs Val** - Overfitting check

### What to Look For:

✅ **Good Signs:**
- Train loss decreasing
- Val loss close to train loss
- AUROC stable or improving
- No huge gap between train and val loss

❌ **Bad Signs:**
- Train loss increasing (CRITICAL - means loss function is broken)
- Large train/val gap (overfitting)
- AUROC declining (model degrading)
- AUROC < 0.5 (inverted predictions!)

---

## Step 3: Run Analysis Script

Diagnose deeper issues with:

```bash
python analyze_issues.py --config configs/recommended.yaml
```

This creates `experiments/bmad_fixed/analysis/` with:

### 1. `data_distribution_analysis.png`
Shows if train/val/test have different distributions:
- Mean intensity distributions
- Std dev distributions  
- Min/max value distributions

**Look for:** Overlapping distributions across splits. If test distribution is very different, you have a distribution shift problem!

### 2. `embedding_analysis.png`
Shows embedding quality:
- L2 norms of embeddings
- Distance to nearest anchor

**Look for:** 
- Normal samples should be CLOSE to anchors
- Anomaly samples should be FAR from anchors
- If reversed, you have inverted scoring!

### 3. `anchor_pairwise_distances.png`
Heatmap of anchor separation

**Look for:**
- Diagonal should be 0
- Off-diagonal should be > 0.5 (well-separated)
- If anchors too close (<0.1), they're redundant

---

## Step 4: Compare Results

### Check Validation vs Test AUROC

**File:** `experiments/bmad_fixed/evaluation/evaluation_metrics.json`

```json
{
  "image_auroc": 0.85,  // ← Should be close to validation AUROC
  "pixel_auroc": 0.78,  // ← Should be reasonable if dense loss used
  ...
}
```

If test AUROC is much lower than validation:
1. Run `analyze_issues.py` to check distribution shift
2. Check if val/test have overlap (data leakage)
3. Verify data preprocessing is consistent

---

## Understanding the Metrics

### Training Metrics (per epoch):

| Metric | Description | Desired Trend |
|--------|-------------|---------------|
| `train_loss` | Total loss | ↓ Decreasing |
| `train_loss_attract` | Pull samples to anchors | ↓ Decreasing |
| `train_loss_repel` | Push anchors apart | Should be 0 if beta=0 |
| `train_loss_dense` | Pixel-level loss | ↓ Decreasing |
| `val_loss` | Validation total loss | ↓ Decreasing, close to train |
| `val_image_auroc` | Image-level anomaly detection | ↑ Increasing/stable |
| `val_pixel_auroc` | Pixel-level localization | ↑ Increasing |

### Loss Components:

**Attractor Loss:** 
```
L_attract = (1/2) * ||embedding - nearest_anchor||²
```
- Pulls normal samples toward their assigned anchor
- Should DECREASE during training

**Repeller Loss:** 
```
L_repel = (1/2) * Σ max(0, 2m - ||anchor_i - anchor_j||)²
```
- Pushes different anchors apart
- **SHOULD BE DISABLED (beta=0) for single-class anomaly detection!**

**Dense Loss:**
- Same as attractor but per-patch
- Helps with pixel-level localization

---

## Troubleshooting

### Problem: Train loss is increasing

**Cause:** Repeller loss is harmful for single-class data
**Fix:** Set `beta: 0.0` in config

### Problem: Val AUROC good but Test AUROC bad

**Causes:**
1. Distribution shift between val and test
2. Data leakage (overlapping samples)
3. Overfitting to validation set

**Fix:**
1. Run `analyze_issues.py` to check distributions
2. Verify data splits are clean
3. Use cross-validation

### Problem: AUROC < 0.5

**Cause:** Model is inverting predictions (normal=high score, anomaly=low score)
**Fix:** 
1. Check if train loss is increasing
2. Disable repeller loss
3. Verify distance metric is correct

### Problem: Model not training (no trainable parameters)

**Cause:** `projection_dim: null` and `freeze_backbone: true`
**Expected:** Model will skip training and just evaluate with DINOv3 features
**Fix:** Set `projection_dim: 128` to add trainable head

---

## Advanced: Grid Search

Want to find best hyperparameters? Try:

```bash
# Edit configs/grid_search.yaml with parameter ranges
python run_experiments.sh
```

This will test combinations of:
- Number of anchors: [4, 8, 16]
- Projection dimensions: [64, 128, 256]
- Dense weights: [0.3, 0.5, 1.0]
- Anchor strategies: ['eigenface', 'kmeans']

---

## Files Generated

After training with enhanced logging:

```
experiments/bmad_fixed/
├── config.yaml                    # Copy of config used
├── anchor_embeddings.pt           # Saved anchor embeddings
├── training_history.json          # All metrics per epoch
├── training_curves.png            # 📊 9-plot visualization
├── best_model.pth                 # Best checkpoint
├── final_model.pth                # Final checkpoint
└── evaluation/
    ├── evaluation_metrics.json    # Test set metrics
    ├── roc_curve.png              # ROC curve
    ├── score_distributions.png    # Score histograms
    ├── normal_samples.png         # Visualized predictions
    ├── anomaly_samples.png        # Visualized predictions
    └── anchor_analysis.png        # Anchor assignments
```

After running analysis:

```
experiments/bmad_fixed/analysis/
├── data_distribution_analysis.png      # Distribution shift check
├── embedding_analysis.png              # Embedding quality
└── anchor_pairwise_distances.png       # Anchor separation
```

---

## Expected Results (After Fixes)

With `beta: 0.0` and enhanced training:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Train Loss Trend | ↑ Increasing | ↓ Decreasing |
| Val Image AUROC | 0.87 | 0.85-0.90 |
| Test Image AUROC | 0.37 | 0.80-0.88 |
| Pixel AUROC | 0.86 | 0.75-0.85 |

The test AUROC should now match validation!

---

## Questions?

Check `ANALYSIS_REPORT.md` for detailed explanations of:
- Why repeller loss is harmful
- How the loss function works
- What each metric means
- Recommended next steps

---

## Quick Commands Reference

```bash
# Train with recommended config
python main.py --config configs/recommended.yaml

# Train with experimental config
python main.py --config configs/experimental_simple.yaml

# Analyze trained model
python analyze_issues.py --config configs/recommended.yaml

# Evaluate only (skip training)
python main.py --config configs/recommended.yaml --eval-only

# Use existing anchors (skip regeneration)
python main.py --config configs/recommended.yaml --skip-anchors
```

---

Good luck! 🚀
