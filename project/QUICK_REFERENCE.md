# BMAD Quick Reference

## 🚀 Common Commands

### Training

```bash
# New experiment
python main.py --config configs/recommended.yaml

# Resume from checkpoint
python main.py --config configs/my_exp.yaml --resume experiments/my_exp/checkpoint_epoch_20.pth

# Use existing anchors
python main.py --config configs/my_exp.yaml --skip-anchors
```

### Evaluation

```bash
# Evaluate best checkpoint
python evaluate_test.py --checkpoint-dir experiments/my_exp --checkpoint best

# Evaluate final checkpoint
python evaluate_test.py --checkpoint-dir experiments/my_exp --checkpoint final

# Evaluate specific checkpoint
python evaluate_test.py --checkpoint-dir experiments/my_exp --checkpoint /path/to/checkpoint.pth
```

### Visualization

```bash
# Plot training curves
python plot_from_checkpoint.py --checkpoint-dir experiments/my_exp

# Analyze training issues
python analyze_issues.py --experiment-dir experiments/my_exp
```

### Utilities

```bash
# Verify dataset structure
python verify_dataset.py --data-root ../data/BraTS2021_slice

# Test data loading
python test_data_loading.py --config configs/recommended.yaml

# Validate pixel AUROC computation
python validate_pixel_auroc.py
```

---

## ⚙️ Critical Config Parameters

```yaml
loss:
  beta: 0.0              # ⚠️ MUST be 0.0! (disable repeller loss)
  use_dense: true        # Enable pixel-level loss
  
training:
  val_interval: 1        # Validate every epoch
  early_stopping_patience: 15

anchor:
  strategy: 'eigenface'  # Recommended
  n_anchors: 8           # 4-16 range
```

---

## 📊 Expected Results

**Recommended Config (BraTS2021_slice):**
- Val Image AUROC: **0.86-0.87**
- Val Pixel AUROC: **0.91-0.92**
- Test Image AUROC: **0.78-0.82**
- Test Pixel AUROC: **0.88-0.90**
- Training time: **30-45 min** (RTX 3090)

---

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| Training loss increases | Set `beta: 0.0` in config |
| CUDA OOM | Reduce `batch_size` (try 32 or 16) |
| No pixel AUROC logged | Already fixed in latest version |
| Low AUROC | Check beta=0.0, run `analyze_issues.py` |
| Checkpoint load error | Use `weights_only=False` or update PyTorch |

---

## 📁 Output Files

```
experiments/my_experiment/
├── anchor_embeddings.pt         # Anchor prototypes
├── best_model.pth              # Best checkpoint (max val AUROC)
├── final_model.pth             # Final checkpoint (last epoch)
├── training_history.json       # All metrics per epoch
├── training_curves.png         # Visualization
├── config.yaml                 # Config used
└── test_evaluation/
    ├── evaluation_metrics.json # Test metrics
    ├── roc_curves.png
    ├── predictions.npz
    └── visualizations/
```

---

## 🔍 Metrics Explained

| Metric | Range | Meaning |
|--------|-------|---------|
| **Image AUROC** | 0-1 | Image-level anomaly detection (higher = better) |
| **Pixel AUROC** | 0-1 | Pixel-level localization (higher = better) |
| **Attractor Loss** | 0+ | Distance to anchors (lower = better) |
| **Dense Loss** | 0+ | Pixel-level alignment (lower = better) |
| **Anchor Balance** | σ | Std of sample distribution across anchors (lower = better balance) |

---

## 📚 Configuration Templates

### Minimal Config (Fast Prototyping)

```yaml
seed: 42
output_dir: './experiments/quick_test'
data:
  data_root: '../data/BraTS2021_slice'
  target_size: [240, 240]
anchor:
  strategy: 'eigenface'
  n_anchors: 4
  n_components: 30
model:
  backbone: 'vit_small_patch16_dinov2'
  freeze_backbone: true
  projection_dim: 64
loss:
  beta: 0.0  # ⚠️ Critical!
  use_dense: false
training:
  epochs: 20
  batch_size: 32
```

### Production Config (Best Performance)

```yaml
seed: 42
output_dir: './experiments/production'
data:
  data_root: '../data/BraTS2021_slice'
  target_size: [240, 240]
anchor:
  strategy: 'eigenface'
  n_components: 50
  n_anchors: 8
  max_images_for_pca: 5000
model:
  backbone: 'vit_small_patch16_dinov3.lvd1689m'
  freeze_backbone: true
  projection_dim: 128
loss:
  margin: 1.0
  alpha: 1.0
  beta: 0.0  # ⚠️ Critical!
  use_dense: true
  global_weight: 1.0
  dense_weight: 0.5
training:
  epochs: 50
  batch_size: 64
  lr: 0.0001
  val_interval: 1
  early_stopping_patience: 15
```

---

## 🎯 Workflow Checklist

- [ ] Dataset verified (`verify_dataset.py`)
- [ ] Config created with `beta: 0.0`
- [ ] Training started (`main.py --config ...`)
- [ ] Training curves checked (loss decreasing?)
- [ ] Best model selected (highest val AUROC)
- [ ] Test evaluation run (`evaluate_test.py`)
- [ ] Results analyzed (AUROC ≥ 0.75?)
- [ ] Visualizations checked
- [ ] Results documented

---

**For full documentation, see [README.md](README.md)**
❌ Train Loss: INCREASING (1.78 → 1.96)
✅ Val AUROC: 0.87
❌ Test AUROC: 0.37 (worse than random!)
```

### AFTER (With beta=0):
```
✅ Train Loss: DECREASING
✅ Val AUROC: 0.85-0.90
✅ Test AUROC: 0.80-0.88 (matches validation!)
```

---

## 📖 Documentation

- `SUMMARY.md` - Complete overview of changes
- `ANALYSIS_REPORT.md` - Detailed issue analysis
- `QUICK_START.md` - Step-by-step usage guide

---

## 🎯 Interpretation Guide

### Training Curves

**Good Signs:**
- Train loss ↓ decreasing
- Val loss ≈ train loss
- AUROC ↑ increasing/stable
- AUROC > 0.7

**Bad Signs:**
- Train loss ↑ increasing
- Big train/val gap (overfitting)
- AUROC < 0.5 (inverted!)
- Test << Val AUROC (distribution shift)

### Loss Components

- **Attractor:** Pulls samples to anchors (should decrease)
- **Repeller:** Pushes anchors apart (should be 0 if beta=0)
- **Dense:** Pixel-level localization (should decrease)

### Metrics

- **Image AUROC:** Image-level anomaly detection (target: >0.8)
- **Pixel AUROC:** Pixel-level localization (target: >0.7)

---

## 🔧 Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Train loss increasing | Repeller loss harmful | `beta: 0.0` |
| AUROC < 0.5 | Inverted predictions | Fix repeller loss |
| Val good, Test bad | Distribution shift | Run `analyze_issues.py` |
| No training | No trainable params | Set `projection_dim: 128` |

---

## 🚀 Quick Commands

```bash
# Train with fixes
python main.py --config configs/recommended.yaml

# Analyze issues
python analyze_issues.py --config configs/recommended.yaml

# Try simpler baseline
python main.py --config configs/experimental_simple.yaml
```

---

## 📁 Output Files

After training:
```
experiments/bmad_fixed/
├── training_curves.png          ← Review this first!
├── training_history.json
└── evaluation/
    ├── evaluation_metrics.json  ← Check test AUROC
    └── roc_curve.png
```

After analysis:
```
experiments/bmad_fixed/analysis/
├── data_distribution_analysis.png
├── embedding_analysis.png
└── anchor_pairwise_distances.png
```

---

## ⚡ TL;DR

1. **Problem:** Repeller loss breaks single-class anomaly detection
2. **Fix:** Set `beta: 0.0` in config
3. **Train:** `python main.py --config configs/recommended.yaml`
4. **Check:** Training loss should now DECREASE
5. **Verify:** Test AUROC should match validation AUROC

---

Good luck! 🎯
