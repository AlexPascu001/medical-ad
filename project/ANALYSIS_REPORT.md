# Medical Anomaly Detection - Issues Analysis & Recommendations

## Summary of Findings

After analyzing your BMAD medical anomaly detection project, I've identified several **critical issues** that explain the poor test performance (AUROC 0.37) despite good validation performance (AUROC 0.87).

---

## 🔴 Critical Issues Identified

### 1. **Training Loss is INCREASING (Major Red Flag!)**

**Observation:**
- Training loss increases from 1.78 → 1.96 over 35 epochs
- This is **opposite** of what should happen

**Root Cause:**
The **repeller loss** is pushing anchors apart over time, which also pushes normal training samples further from their assigned anchors. This creates a conflict:
- **Attractor** tries to pull samples close to anchors
- **Repeller** pushes anchors apart, indirectly pushing samples away
- As training progresses, the repeller dominates

**Why This Happens:**
In the paper (https://arxiv.org/abs/2306.00630), the repeller loss is designed for **multi-class** classification where you have DIFFERENT class anchors that should be separated. In your case:
- You only have ONE class (normal brain MRI)
- All anchors represent variations of the SAME normal class
- The repeller loss is incorrectly pushing apart anchors that should represent the same distribution

**Impact:**
This makes the model learn to produce HIGHER anomaly scores over time, even for normal samples, breaking the entire anomaly detection logic.

---

### 2. **Severe Class Imbalance in Test Set**

**Observation:**
```
Val:  640 normal, 2135 anomaly  (ratio ~1:3.3)
Test: 640 normal, 3075 anomaly  (ratio ~1:4.8)
```

**Impact:**
- AUROC is sensitive to class imbalance
- The test set has more severe imbalance than validation
- This partially explains the AUROC discrepancy

---

### 3. **ROC Curve Mismatch (0.63 terminal vs 0.37 plot)**

**Root Cause:**
Looking at `evaluation_metrics.json`:
- `image_auroc: 0.3739` (what the plot shows)
- This is BELOW 0.5 (worse than random!)

But training logs show validation AUROC ~0.87. **These are evaluating different datasets!**

**Explanation:**
- During training: model evaluates on **validation set** → AUROC ~0.87
- After training: `evaluate_comprehensive()` evaluates on **test set** → AUROC 0.37

The 0.63 you saw in terminal might have been from a different run or intermediate checkpoint.

---

### 4. **Inverted Anomaly Scoring**

**Critical Bug:**
When AUROC < 0.5, it means your model is **inverting** the prediction:
- Normal samples get HIGH scores (far from anchors)
- Anomalous samples get LOW scores (close to anchors)

This is the OPPOSITE of what should happen! This is caused by issue #1 (increasing loss).

---

### 5. **Dense Loss Not Being Used**

**Observation:**
- Config has `use_dense: false`
- Pixel AUROC is still computed but not optimized during training
- Missing opportunity for localization

---

### 6. **No Validation Loss Tracking** (Fixed in my updates)

Original code didn't compute validation loss, only AUROC. This made it impossible to detect overfitting in the loss function.

---

## ✅ Changes I've Implemented

### 1. **Enhanced Logging System**

I modified `train.py` to track comprehensive metrics per epoch:

**Training Metrics:**
- Total loss, Attractor loss, Repeller loss, Dense loss
- Anchor assignment balance

**Validation Metrics:**
- Validation loss (all components)
- Image AUROC, Pixel AUROC
- Loss breakdown (attractor, repeller, dense)

**Per-Epoch Statistics:**
- Learning rate tracking
- Epoch indices

### 2. **Comprehensive Visualization**

Added `plot_training_curves()` function that creates a 3x3 grid showing:
1. Total Loss (train + val)
2. Attractor Loss evolution
3. Repeller Loss evolution  
4. Dense Loss (if enabled)
5. Image AUROC over epochs
6. Pixel AUROC over epochs
7. Learning Rate schedule
8. Final epoch loss components breakdown
9. Train vs Val loss comparison (overfitting check)

This will be automatically saved as `training_curves.png` after training.

### 3. **Analysis Script**

Created `analyze_issues.py` that performs:
- **Data distribution analysis**: Check for distribution shift between train/val/test
- **Embedding analysis**: Visualize embedding norms and distances
- **Val/Test overlap check**: Ensure no data leakage
- **Anchor quality analysis**: Check if anchors are well-separated

---

## 🔧 Recommended Fixes

### **Priority 1: Fix the Repeller Loss Issue**

**Option A: Remove Repeller Loss for Single-Class Anomaly Detection**

Since you only have one class (normal), the repeller loss is harmful. Modify your config:

```yaml
loss:
  margin: 1.0
  alpha: 1.0
  beta: 0.0  # ← SET TO ZERO (disable repeller)
  distance_metric: 'euclidean'
```

**Option B: Use Only Attractor Loss**

The paper's loss was designed for multi-class retrieval. For anomaly detection:
- Train ONLY with attractor loss (pull normal samples to anchors)
- At test time, anomalies will naturally be far from anchors

### **Priority 2: Enable Dense Loss for Better Localization**

```yaml
loss:
  use_dense: true
  dense_weight: 0.5
```

This will help the model learn pixel-level representations.

### **Priority 3: Address Class Imbalance**

Add weighted sampling or adjust evaluation:

```python
# In data.py, add balanced sampling for validation
from torch.utils.data import WeightedRandomSampler

# Or use stratified evaluation
```

### **Priority 4: Check for Distribution Shift**

Run my analysis script:
```bash
python analyze_issues.py --config configs/default.yaml
```

This will generate plots showing if train/val/test have different distributions.

### **Priority 5: Experiment with Anchor Strategies**

Try both:
- `strategy: 'kmeans'` - Direct clustering in image space
- `strategy: 'eigenface'` - PCA + clustering

And vary the number of anchors:
```yaml
anchor:
  n_anchors: [4, 8, 16, 32]  # Try different values
```

---

## 📊 How to Use the New Features

### 1. **Retrain with Enhanced Logging**

Simply run your training script as before:
```bash
python main.py --config configs/default.yaml
```

After training completes, you'll see:
- `training_history.json` - Contains all metrics
- `training_curves.png` - Comprehensive visualization

### 2. **Analyze the Issues**

```bash
python analyze_issues.py --config configs/default.yaml
```

This will create `experiments/bmad_baseline/analysis/` with:
- `data_distribution_analysis.png`
- `embedding_analysis.png`
- `anchor_pairwise_distances.png`

### 3. **Interpret the Results**

**Good Signs:**
- Train loss should DECREASE
- Val loss should be close to train loss (no overfitting)
- Val and Test AUROC should be similar (no distribution shift)
- Anchors should be well-separated (pairwise distances > 0.5)

**Bad Signs:**
- Train loss increasing → Problem with loss function
- Large train/val loss gap → Overfitting
- Large val/test AUROC gap → Distribution shift
- Anchors too close → Poor anchor generation

---

## 🎯 Expected Improvements

After applying the fixes:

1. **Training Loss**: Should DECREASE monotonically
2. **Validation AUROC**: Should remain stable ~0.85-0.90
3. **Test AUROC**: Should match validation ±0.05
4. **Pixel AUROC**: Should improve with dense loss

---

## 🔬 Additional Investigations

### Check Paper Implementation Details

From the BMAD paper (https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.pdf):

1. **Verify preprocessing**: Are you using the same normalization?
2. **Check data splits**: Are you using patient-wise or image-wise splits?
3. **Anchor generation**: The paper might have specific anchor strategies

### Check DINOv3 Paper Implementation

From the anomaly detection paper (https://arxiv.org/pdf/2508.10104):

1. **Feature extraction**: Which layer's features are used?
2. **Normalization**: Are features L2-normalized before distance computation?
3. **Anomaly scoring**: Is it min distance, max distance, or mean distance?

### Verify Loss Implementation

From CAMaL paper (https://arxiv.org/pdf/2306.00630):

1. **Multi-class vs single-class**: The paper uses multiple class anchors
2. **Distance metric**: Euclidean vs cosine - does it matter?
3. **Margin value**: Is margin=1.0 appropriate for your embedding space?

---

## 📝 Summary

**Main Issue**: The repeller loss is pushing apart anchors that represent the SAME class (normal brain MRI), causing training loss to increase and model performance to degrade.

**Quick Fix**: Set `beta: 0.0` in your config to disable repeller loss.

**Long-term**: The current implementation assumes multi-class classification, but you need single-class anomaly detection. Consider:
1. Using only attractor loss
2. Or implementing anchor diversity constraints differently
3. Or using a simple distance-based approach without repeller

**Next Steps**:
1. Run analysis script to understand data distribution
2. Retrain with `beta: 0.0` (no repeller)
3. Compare results and examine training curves
4. If still poor, investigate data splits and preprocessing

---

Would you like me to implement any specific fix or create additional analysis tools?
