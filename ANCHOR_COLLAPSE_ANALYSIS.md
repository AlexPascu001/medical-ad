# Analysis: Anchor Collapse & Performance Degradation

## Problem Summary

**Observed Issues:**
1. **Extreme anchor imbalance**: 1746/2000 samples (87%) assigned to Anchor 1
2. **Performance degradation**: Best AUROC 0.8989 at epoch 0 → drops to 0.73-0.80 by epoch 16
3. **Loss decreases but AUROC worsens**: Training loss goes down, but discriminability decreases

## Root Cause Analysis

### Issue 1: Why One Anchor Dominates

**The Problem Chain:**
```
Pre-training (epochs 1-5):
  ├─ Generate 8 temporary kmeans anchors from subset of data
  ├─ Project through projection head (orthogonal init)
  ├─ Train projection head to pull samples toward temp anchors
  ├─ Loss: α=1.0 (attractor), β=0.0 (NO repeller)
  └─ Result: Projection head learns, but NO constraint keeping anchors apart

Anchor Generation:
  ├─ Generate 8 DIFFERENT real kmeans anchors (same data, same seed, but different run)
  ├─ Project real anchors through pre-trained head
  └─ Problem: Real anchors land in positions determined by pre-trained head
                but head was trained on DIFFERENT temporary anchors!

Main Training (epochs 1-50):
  ├─ Projection head continues training with attractor-only (β=0)
  ├─ Pulls samples toward nearest anchor
  ├─ NO repeller term → anchors can drift together in projection space
  ├─ One anchor becomes dominant (closest to most samples)
  └─ Result: Projection space COLLAPSES around dominant anchor
```

**Key Insight:** Without repeller loss (β=0), there's NO force keeping anchors separated. The projection head learns to minimize attractor loss by collapsing samples toward one anchor.

### Issue 2: Why Performance Peaks at Epoch 0

**Epoch 0 = Right After Pre-training:**
- Pre-trained projection head creates reasonable embedding space
- Real anchors just projected through it
- No overfitting yet
- **AUROC = 0.8989** ← Best performance!

**Epochs 1-16:**
- Projection head trains with fixed anchors
- Attractor loss pulls samples toward anchors
- **Without repeller, projection space collapses:**
  - All normal samples → cluster tightly near dominant anchor
  - Anomalies also get pulled toward same anchor
  - **Reduced separation between normal/anomaly**
  - Loss ↓ (samples closer to anchors)
  - AUROC ↓ (less discriminability)

**Visualization Evidence:**
- t-SNE (epoch 16): Most samples clustered in center, all lines pointing to 1-2 anchors
- PCA (epoch 16): Tight cluster around dominant anchor (Anchor 1, 4, 7)
- Test samples: Normal and anomaly OVERLAPPING in projection space
- Score distributions: Heavy overlap between normal (blue) and anomaly (red)

## Solutions (Ranked by Expected Impact)

### ✅ Solution 1: Enable Repeller Loss (HIGHEST PRIORITY)

**Problem:** β=0 allows anchors to collapse together in projection space

**Fix:** Set β=0.5 during main training to push anchors apart

```yaml
loss:
  beta: 0.5  # Add repeller term (was 0.0)
```

**Why This Works:**
- Repeller term: `L_R = 0.5 * Σ max(0, 2m - ||c_i - c_j||)²`
- Penalizes anchors that are closer than 2*margin (2.0 for margin=1.0)
- Forces anchors to maintain separation in projection space
- Prevents collapse to dominant anchor

**Expected Improvement:**
- More balanced anchor coverage (200-300 samples per anchor instead of 1746)
- Better use of embedding space
- Improved AUROC by maintaining discriminability
- Target: 0.85-0.90 sustained AUROC

---

### ✅ Solution 2: Use Same Anchors for Pre-training and Main Training

**Problem:** Pre-training uses temporary anchors, main training uses different real anchors → mismatch

**Fix:** Cache temporary anchors and reuse them as real anchors

**Implementation:**
```python
# In pretrain.py, after generating temporary anchors:
# Save temporary anchors to experiment directory
torch.save({
    'temp_anchor_images': temp_anchor_images,
    'temp_anchor_global': temp_anchor_global
}, save_dir / 'pretrain_anchors.pt')

# In main.py, when generating anchors:
if (save_dir / 'pretrain_anchors.pt').exists():
    # Reuse pre-training anchors instead of generating new ones
    pretrain_anchors = torch.load(save_dir / 'pretrain_anchors.pt')
    anchor_images = pretrain_anchors['temp_anchor_images']
    anchor_global = pretrain_anchors['temp_anchor_global']
```

**Why This Works:**
- Projection head trained on specific anchor positions
- Using SAME anchors ensures consistency
- No position mismatch

**Expected Improvement:**
- Better initial anchor positions
- More stable training
- Might prevent early collapse

---

### ✅ Solution 3: Increase Margin for Better Anchor Separation

**Problem:** margin=1.0 might be too small for 128D projection space

**Fix:** Increase margin to 2.0 or 3.0

```yaml
loss:
  margin: 2.0  # Increase from 1.0
  beta: 0.5    # Enable repeller
```

**Why This Works:**
- Larger margin → repeller enforces larger minimum distance
- Repeller: `max(0, 2*margin - distance)` → penalizes if distance < 4.0 (instead of < 2.0)
- Anchors forced to spread out more

**Expected Improvement:**
- Even better anchor separation
- More diverse anchor coverage

---

### ✅ Solution 4: Dynamic Anchor Reassignment

**Problem:** Fixed pseudo-labels mean samples stuck with initial (possibly bad) anchor assignment

**Fix:** Enable periodic reassignment

```yaml
training:
  fixed_pseudo_labels: false  # Allow reassignment
  dynamic_reassignment: true
  reassignment_interval: 5    # Reassign every 5 epochs
```

**Why This Works:**
- As projection head learns, optimal anchor assignments change
- Reassignment balances anchor coverage
- Prevents one anchor from accumulating all samples

**Caution:** This conflicts with the paper's fixed pseudo-label approach. May need testing.

---

### ✅ Solution 5: Add Entropy Regularization for Balanced Coverage

**Problem:** No explicit constraint on anchor usage distribution

**Fix:** Add entropy term to encourage uniform anchor usage

```python
# In loss.py
def compute_anchor_entropy_loss(assigned_anchors, n_anchors):
    """Encourage balanced anchor usage via entropy maximization"""
    # Count assignments
    counts = torch.bincount(assigned_anchors, minlength=n_anchors).float()
    probs = counts / counts.sum()
    
    # Maximum entropy when uniform
    max_entropy = np.log(n_anchors)
    current_entropy = -(probs * torch.log(probs + 1e-8)).sum()
    
    # Loss = difference from max entropy
    return max_entropy - current_entropy

# In AnchorMarginLoss.forward():
entropy_loss = compute_anchor_entropy_loss(assigned_anchors, K)
total_loss += 0.1 * entropy_loss  # Small weight
```

**Expected Improvement:**
- Explicit pressure for balanced coverage
- Prevents pathological cases like 1746/2000

---

## Recommended Experiment Plan

### Experiment 1: Baseline with Repeller (QUICK TEST)

**Config:** `pretrain_repeller.yaml`
```yaml
loss:
  beta: 0.5  # Enable repeller
pretraining:
  loss_beta: 0.5  # Also in pre-training
```

**Expected Result:**
- More balanced anchor coverage
- AUROC improvement from 0.73 → 0.85+
- Sustained performance (not peaking at epoch 0)

**Run Time:** ~1 hour

---

### Experiment 2: Reuse Pre-training Anchors + Repeller

**Changes:**
1. Modify `pretrain.py` to save temporary anchors
2. Modify `main.py` to reuse saved anchors
3. Enable repeller (β=0.5)

**Expected Result:**
- Perfect alignment between pre-training and main training
- Even better performance: 0.88-0.92 AUROC
- Most balanced anchor coverage

**Run Time:** ~1 hour (after implementation)

---

### Experiment 3: Larger Margin

**Config:**
```yaml
loss:
  margin: 2.0
  beta: 0.5
```

**Expected Result:**
- Best anchor separation
- Might sacrifice some attractor tightness
- AUROC: 0.85-0.90

**Run Time:** ~1 hour

---

## Quick Diagnostic: Check Anchor Positions

Before running experiments, let's check if anchors are actually collapsing in projection space:

```python
# Add to evaluate_test.py or create check_anchor_separation.py
import torch

# Load model checkpoint
checkpoint = torch.load('experiments/bmad_pretrain_random_test/pretrain_test/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Get projected anchors
anchor_global = model._get_projected_anchors()[0]  # (8, 128)

# Check pairwise distances
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(anchor_global.cpu().numpy(), metric='euclidean'))

print("Anchor pairwise distances:")
print(distances)
print(f"\nMin distance: {distances[distances > 0].min():.4f}")
print(f"Max distance: {distances.max():.4f}")
print(f"Mean distance: {distances[distances > 0].mean():.4f}")
print(f"\nMargin: 1.0, so repeller wants distances > 2.0")
print(f"Violations (distance < 2.0): {(distances[distances > 0] < 2.0).sum() // 2}")
```

If min distance < 2.0 → anchors ARE collapsing → repeller will help!

---

## Summary

**Root Cause:** No repeller loss (β=0) + projection head training = anchor collapse

**Primary Fix:** Enable repeller loss (β=0.5) during both pre-training and main training

**Expected Impact:**
- Anchor coverage: 1746 → ~250 per anchor (8 anchors)
- AUROC: 0.73 → 0.85-0.90
- Sustained performance (not peaking at epoch 0)

**Next Steps:**
1. Run `pretrain_repeller.yaml` experiment (1 hour)
2. Check anchor coverage in visualizations
3. If still imbalanced, try Solution 2 (reuse pre-training anchors)
4. If still issues, add entropy regularization (Solution 5)
