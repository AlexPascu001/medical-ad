# Contrastive Loss Functions for Learnable Anchors

## Overview

Three new loss functions have been added that are **better suited for learnable anchors** than the original CAM loss:

1. **Center Loss** - Simple and effective
2. **InfoNCE Loss** - Soft contrastive learning
3. **Hybrid Loss** - Best of both (RECOMMENDED)

## Why Use These Instead of CAM Loss?

The original CAM loss had a critical bug (`.detach()`) that prevented anchor learning. Even with the bug fixed, CAM loss has issues:

- ❌ Attractor only pulls samples → anchors (one-way)
- ❌ Repeller/min-norm losses are weak compared to attractor
- ❌ Mode collapse still occurs

The new contrastive losses:

- ✅ Pull both samples AND anchors toward each other (two-way learning)
- ✅ Use proven contrastive learning techniques (Center Loss, InfoNCE)
- ✅ Better balance between attraction and separation
- ✅ Gradients flow correctly to learnable anchors

## Loss Functions

### 1. Center Loss (Simple, Effective)

**What it does:**
- Pulls samples toward their nearest anchor
- Pulls anchors toward their assigned samples
- Pushes anchors apart

**Use when:** You want a simple, interpretable loss

**Config:**
```yaml
loss:
  type: 'center'
  lambda_center: 1.0    # Pull samples + anchors together
  lambda_repel: 0.1     # Push anchors apart
  margin: 1.0
```

**Papers:** 
- [A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 2016)](https://ydwen.github.io/papers/WenECCV16.pdf)

---

### 2. InfoNCE Loss (Soft Contrastive)

**What it does:**
- Uses temperature-scaled softmax for soft assignments
- Cross-entropy loss: pull to assigned anchor, push from others
- More flexible than hard assignments

**Use when:** You want soft assignments and temperature control

**Config:**
```yaml
loss:
  type: 'infonce'
  temperature: 0.07     # Lower = harder assignments
  lambda_repel: 0.1     # Push anchors apart
  margin: 1.0
```

**Papers:**
- [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709)

---

### 3. Hybrid Loss (RECOMMENDED)

**What it does:**
- Combines Center Loss (hard L2 pull) + InfoNCE (soft contrastive)
- Gets benefits of both approaches
- Most robust and flexible

**Use when:** You want the best of both worlds (RECOMMENDED!)

**Config:**
```yaml
loss:
  type: 'hybrid'
  lambda_center: 1.0    # Hard pull (L2)
  lambda_infonce: 0.5   # Soft contrastive
  lambda_repel: 0.1     # Anchor separation
  temperature: 0.07     # InfoNCE temperature
  margin: 1.0
```

---

## Quick Start

### 1. Run with Center Loss

```bash
venv/Scripts/python.exe project/main.py --config project/configs/center_loss.yaml
```

### 2. Run with InfoNCE Loss

```bash
venv/Scripts/python.exe project/main.py --config project/configs/infonce_loss.yaml
```

### 3. Run with Hybrid Loss (Recommended)

```bash
venv/Scripts/python.exe project/main.py --config project/configs/hybrid_loss.yaml
```

---

## Hyperparameters

### Center Loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_center` | 1.0 | Weight for center loss (pull samples + anchors) |
| `lambda_repel` | 0.1 | Weight for anchor separation |
| `margin` | 1.0 | Minimum distance between anchors |

### InfoNCE Loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.07 | Temperature for softmax (0.01-0.1) |
| `lambda_repel` | 0.1 | Weight for anchor separation |
| `margin` | 1.0 | Minimum distance between anchors |

### Hybrid Loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_center` | 1.0 | Weight for hard pull (L2) |
| `lambda_infonce` | 0.5 | Weight for soft contrastive |
| `lambda_repel` | 0.1 | Weight for anchor separation |
| `temperature` | 0.07 | Temperature for InfoNCE |
| `margin` | 1.0 | Minimum distance between anchors |

---

## Expected Behavior

With learnable anchors and contrastive losses, you should see:

### During Training
- Anchors move toward their assigned sample clusters
- Samples pull their anchors closer
- Anchor separation maintained by repeller term
- No mode collapse (anchors stay separated)

### In t-SNE Visualizations
- Anchors should be **inside** or **near** sample clusters
- Each anchor surrounded by its assigned samples
- Clear separation between different anchor regions
- Progressive convergence over epochs

### Metrics
- Better anchor utilization (more even distribution)
- Higher coverage ratio (samples closer to anchors)
- Improved AUROC (better anomaly detection)

---

## Comparison: CAM vs Contrastive

| Aspect | CAM Loss | Contrastive Losses |
|--------|----------|-------------------|
| Anchor Learning | One-way (samples→anchors) | Two-way (samples↔anchors) |
| Assignment | Hard (nearest) | Hard (Center) or Soft (InfoNCE) |
| Gradient Flow | Weak to anchors | Strong to anchors |
| Mode Collapse | High risk | Lower risk |
| Tuning | 3 hyperparameters (α,β,γ) | 2-4 hyperparameters |
| Interpretability | Medium | High |
| Recommended For | Fixed anchors | **Learnable anchors** |

---

## Troubleshooting

### Anchors still collapsing?
- Increase `lambda_repel` (try 0.5 or 1.0)
- Increase `margin` (try 2.0 or 3.0)
- Decrease learning rate

### Anchors not moving?
- Check gradients with `test_anchor_gradients.py`
- Ensure `learnable: true` in config
- Increase `lambda_center` (Center/Hybrid loss)

### Poor performance?
- Try different loss types (Hybrid usually best)
- Adjust temperature (InfoNCE: 0.05-0.1)
- Check anchor initialization (random usually best)

---

## Testing

Test all losses work correctly:

```bash
venv/Scripts/python.exe project/test_contrastive_losses.py
```

Test anchor gradients:

```bash
venv/Scripts/python.exe project/test_anchor_gradients.py
```

Both should show ✅ PASS for all tests.

---

## Next Steps

1. **Run experiments** with all 3 loss types
2. **Compare t-SNE visualizations** across epochs
3. **Analyze anchor coverage** with `analyze_anchor_coverage.py`
4. **Choose best loss** based on AUROC + visualization quality
5. **Tune hyperparameters** for chosen loss

Good luck! 🚀
