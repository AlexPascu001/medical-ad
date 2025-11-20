# 📚 Complete Architecture Documentation Index

This folder contains comprehensive documentation of the BMAD (Brain MRI Anomaly Detection) architecture with **actual dimensions** at every step.

---

## 📖 Documentation Files

### 1. **ARCHITECTURE_WALKTHROUGH.md** (⭐ MAIN DOCUMENT - 1040 lines)
**Complete end-to-end walkthrough with actual dimensions**

**Contents**:
- Part 1: Data Preprocessing (240×240 → normalized)
- Part 2: Anchor Generation (eigenface with actual shapes)
- Part 3: Feature Extraction (DINOv3 backbone: 384D)
- Part 4: Trainable Projection Head (384 → 192 → 128)
- Part 5: Anchor Embeddings (8 prototypes in 128D)
- Part 6: Distance Computation (L2 vs cosine)
- Part 7: Loss Computation (attractor + repeller)
- Part 8: Training Step (gradient flow)
- Part 9: Inference (anomaly scoring)
- Part 10: Evaluation Metrics (AUROC)
- Part 11: Complete Data Flow Summary
- Part 12: Embedding Spaces Explained
- Part 13: Distance Metrics Comparison
- Part 14: Memory and Computation
- Part 15: Key Takeaways
- Appendix: Quick Reference Tables

**When to use**: Deep dive into how everything works conceptually and mathematically.

---

### 2. **DIMENSIONS_CHEATSHEET.md** (🎯 QUICK REFERENCE - 500 lines)
**Visual ASCII flowchart with dimensions at every step**

**Contents**:
- Complete dimension flow (ASCII diagram)
- Embedding space dimensions table
- Batch processing shapes
- Anchor generation dimensions
- Memory usage breakdown
- Performance metrics
- Hyperparameter reference
- Key concepts summary
- Training dynamics
- Example: single test image

**When to use**: Quick lookup of tensor shapes, need ASCII visual reference.

---

### 3. **DISTANCE_METRICS.md** (📐 COMPARISON - 300 lines)
**Distance metric experiments: cosine vs L2**

**Contents**:
- Verification of normalization
- Cosine distance explained
- L2 distance explained
- Configuration examples
- Expected results
- Implementation details
- Experiment matrix
- Analysis examples

**When to use**: Understanding distance metrics, planning experiments.

---

### 4. **architecture_diagram.png** (🖼️ VISUAL FLOW)
**Complete data flow diagram with color-coded stages**

**Visual elements**:
- Input data (blue): 240×240 → (64, 3, 240, 240)
- DINOv3 backbone (orange): frozen, 384D embeddings
- Projection head (green): trainable, 384 → 128
- Anchors (red): 8 prototypes, side branch
- Loss computation (purple): attractor + repeller
- Inference (yellow): anomaly scoring

**When to use**: Presenting to others, understanding overall flow.

---

### 5. **embedding_spaces.png** (🧠 SPACE TRANSFORMATION)
**Three-panel visualization of embedding spaces**

**Panels**:
1. **Pixel Space** (57,600D): Raw images, high-dimensional
2. **DINOv3 Space** (384D): Semantic features, pretrained
3. **Projected Space** (128D): Task-specific, learned

**When to use**: Understanding what "space" means, explaining projection.

---

## 🎓 Learning Path

### For First-Time Readers:
1. Start with **DIMENSIONS_CHEATSHEET.md** (10 min)
   - Get overall picture with ASCII diagram
   - Understand tensor shapes
   
2. Look at **architecture_diagram.png** (5 min)
   - Visual understanding of data flow
   - See color-coded stages

3. Read **ARCHITECTURE_WALKTHROUGH.md** Part 1-5 (30 min)
   - Deep dive into preprocessing
   - Understand feature extraction
   - Learn about projection

4. Look at **embedding_spaces.png** (5 min)
   - Understand space transformation
   - See clustering effect

5. Continue **ARCHITECTURE_WALKTHROUGH.md** Part 6-15 (60 min)
   - Distance computation
   - Loss functions
   - Training and inference

### For Quick Reference:
- **Tensor shape**: DIMENSIONS_CHEATSHEET.md
- **Distance metrics**: DISTANCE_METRICS.md
- **Visual overview**: architecture_diagram.png
- **Memory usage**: DIMENSIONS_CHEATSHEET.md → Memory Usage section
- **Hyperparameters**: DIMENSIONS_CHEATSHEET.md → Hyperparameters section

### For Presentations:
1. **architecture_diagram.png** - Main slide
2. **embedding_spaces.png** - Explain spaces
3. Key numbers from DIMENSIONS_CHEATSHEET.md
4. Results from ARCHITECTURE_WALKTHROUGH.md Part 10

---

## 🔑 Key Numbers (Quick Facts)

```
Input:        240×240 grayscale MRI
Backbone:     DINOv3-small (384D, 22M params, frozen)
Projection:   384 → 192 → 128 (98K params, trainable)
Anchors:      8 prototypes in 128D space
Patches:      15×15 = 225 patches (patch size 16×16)
Batch size:   64 (training)
Distance:     L2 or cosine
Loss:         Attractor (α=1.0), Repeller off (β=0.0)
Training:     ~5 min/epoch, 50 epochs
Accuracy:     82-83% image AUROC, 87% pixel AUROC
Memory:       ~89 MB model, ~8.8 GB/epoch
Speed:        ~16 ms/image (GPU), 62 img/sec
```

---

## 📐 Dimension Flow (Ultra-Quick)

```
(240, 240)                      Raw image
  ↓
(64, 3, 240, 240)               Batch
  ↓
(64, 384)                       DINOv3 CLS
  ↓
(64, 128)                       Projection (normalized)
  ↓ Compare to anchors (8, 128)
(64, 8)                         Distances
  ↓ Min
(64,)                           Anomaly scores
```

---

## 🎯 What Space Are We In?

| Space | Dim | Purpose |
|-------|-----|---------|
| Pixel | 57,600 | Raw images (anchor generation) |
| DINOv3 | 384 | Pretrained semantic features |
| **Projected** | **128** | **Comparison space** (distances) |

**Critical**: Both samples and anchors are projected to the **same 128D space** using the trainable projection head before distance computation!

---

## 🧪 Test Scripts

### **test_dimensions.py**
Quick test to verify actual dimensions with your config:
```bash
python test_dimensions.py
```

Output:
```
Input shape: (8, 3, 240, 240)
Backbone: 384D, patch size 16x16
Projection: 128D
Global features: (8, 128)
Dense features: (8, 15, 15, 128)
```

### **generate_architecture_diagrams.py**
Generate visual diagrams:
```bash
python generate_architecture_diagrams.py
```

Output:
- `architecture_diagram.png` (data flow)
- `embedding_spaces.png` (space transformation)

---

## 💡 Core Concept (ELI5)

**You have 8 "normal brain" prototypes in a 128D learned space.**

Every test image:
1. Gets converted to a point in that 128D space
2. Is compared to all 8 prototypes
3. Gets a score = distance to nearest prototype

**Close to a prototype** = normal brain  
**Far from all prototypes** = anomaly (tumor/lesion)

The 128D space is learned by a small neural network (98K parameters) that optimizes the space so normal brains cluster tightly around the prototypes.

---

## 📊 Example: One Image

```python
# Test image
shape: (1, 3, 240, 240)

# After DINOv3 + projection
embedding: (1, 128)  # One point in 128D space

# Compare to 8 anchors
anchors: (8, 128)
distances: [1.85, 2.14, 1.92, 2.37, 2.08, 1.78, 2.21, 1.95]
                                            ↑
                                    Nearest anchor

# Anomaly score
score: 1.78  # Borderline

# Interpretation
< 1.0:  Normal (close to prototype)
1.0-2.0: Borderline (review)
> 2.0:  Anomaly (far from all prototypes)
```

---

## 🚀 Usage

### View Documentation
```bash
# In VS Code
code ARCHITECTURE_WALKTHROUGH.md
code DIMENSIONS_CHEATSHEET.md
code DISTANCE_METRICS.md

# View images
start architecture_diagram.png
start embedding_spaces.png
```

### Run Tests
```bash
# Verify dimensions
python test_dimensions.py

# Generate diagrams
python generate_architecture_diagrams.py
```

---

## 📝 Additional Resources

- **README.md** - Project overview and setup
- **ANCHOR_EXPERIMENTS.md** - Anchor strategy comparison
- **configs/*.yaml** - Configuration files with actual hyperparameters
- **Code files**:
  - `model.py` - DINOv3Backbone and AnomalyDetector
  - `anchors.py` - Anchor generation strategies
  - `loss.py` - Anchor-margin loss
  - `train.py` - Training loop
  - `eval.py` - Evaluation with AUROC

---

## 🎓 FAQ

### Q: What dimensions matter most?
**A**: Focus on:
- Input: (64, 3, 240, 240)
- DINOv3: (64, 384)
- Projected: (64, 128) ← **This is where comparison happens**
- Anchors: (8, 128)
- Distances: (64, 8)

### Q: What space are distances computed in?
**A**: The **128D projected space**. Both samples and anchors are projected through the trainable head before distance computation.

### Q: Why 128 dimensions?
**A**: Balance between:
- **Too few** (e.g., 32): Not enough expressiveness
- **Too many** (e.g., 512): Overfitting, slow
- **128**: Sweet spot for brain MRI

### Q: What gets trained?
**A**: Only the projection head (98K params). DINOv3 is frozen (22M params).

### Q: How are anchors generated?
**A**: 
1. PCA on 7500 training images → 50 components
2. K-means clustering in eigenface space → 8 clusters
3. Reconstruct 8 anchor images from centroids
4. Extract DINOv3 features (384D)
5. Store for later projection

### Q: Why L2 normalize?
**A**: For cosine distance to work properly. Also stabilizes training.

### Q: What's the anomaly threshold?
**A**: Not fixed—depends on desired sensitivity/specificity. Typical:
- score < 1.0: Normal
- score > 2.0: Anomaly
- Use AUROC to evaluate overall separation

---

## 📧 Summary

**Three documents, two images, one test script:**

1. **ARCHITECTURE_WALKTHROUGH.md** - Complete conceptual explanation
2. **DIMENSIONS_CHEATSHEET.md** - Quick dimension reference
3. **DISTANCE_METRICS.md** - Distance metric guide
4. **architecture_diagram.png** - Visual data flow
5. **embedding_spaces.png** - Space transformation
6. **test_dimensions.py** - Verify actual dimensions

**Start here**: DIMENSIONS_CHEATSHEET.md + architecture_diagram.png (15 min)  
**Deep dive**: ARCHITECTURE_WALKTHROUGH.md (2 hours)  
**Experiments**: DISTANCE_METRICS.md + run experiments

---

**You now have complete documentation of every dimension in the pipeline! 🎉**
