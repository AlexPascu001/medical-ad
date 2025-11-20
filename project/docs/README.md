# BMAD: Brain MRI Anomaly Detection

A medical anomaly detection system using DINOv3 backbone with anchor-based learning for detecting brain tumor anomalies in MRI scans.

## 🎯 Overview

This project implements an anchor-based anomaly detection approach for brain MRI using:
- **DINOv3** (Vision Transformer) as the feature extraction backbone
- **Anchor-based margin loss** for learning normal brain representations
- **Dense loss** for pixel-level anomaly localization
- **BraTS2021** dataset for training and evaluation

## 📋 Table of Contents

1. [Installation](#installation)
2. [Dataset Structure](#dataset-structure)
3. [Quick Start](#quick-start)
4. [Complete Tutorial](#complete-tutorial)
5. [Configuration](#configuration)
6. [Utilities & Analysis](#utilities--analysis)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone <repository-url>
cd medical-ad/project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
Key dependencies:
- PyTorch 2.0+
- torchvision
- timm (for DINOv3 models)
- scikit-learn
- albumentations
- matplotlib
- pyyaml
- tqdm

---

## 📁 Dataset Structure

The project expects the BraTS2021_slice dataset in the following structure:

```
data/BraTS2021_slice/
├── train/
│   └── good/                  # Normal training images (PNG files)
│       ├── 00001_60.png
│       ├── 00001_61.png
│       └── ...
├── valid/
│   ├── good/                  # Normal validation images
│   │   ├── img/
│   │   │   └── *.png
│   │   └── label/             # Masks (all zeros for normal)
│   │       └── *.png
│   └── Ungood/                # Anomalous validation images
│       ├── img/
│       │   └── *.png
│       └── label/             # Pixel-level anomaly masks
│           └── *.png
└── test/
    ├── good/                  # Normal test images
    │   ├── img/
    │   │   └── *.png
    │   └── label/
    │       └── *.png
    └── Ungood/                # Anomalous test images
        ├── img/
        │   └── *.png
        └── label/
            └── *.png
```

**Key points:**
- Training set: **Only normal (healthy) brain scans**
- Validation/Test: **Both normal and anomalous (with tumors)**
- Labels: Pixel-level binary masks (0=normal, 255=anomaly)

---

## 🚀 Quick Start

### 1. Train a Model (End-to-End)

```bash
# Train from scratch with recommended config
python main.py --config configs/recommended.yaml

# This will:
# - Generate anchors from training data
# - Train the model
# - Validate after each epoch
# - Save best model and training history
```

### 2. Evaluate on Test Set

```bash
# Evaluate the best checkpoint
python evaluate_test.py --checkpoint-dir experiments/bmad_fixed --checkpoint best

# Results saved to: experiments/bmad_fixed/test_evaluation/
```

### 3. Visualize Training Progress

```bash
# Generate training curves from saved history
python plot_from_checkpoint.py --checkpoint-dir experiments/bmad_fixed
```

---

## 📚 Complete Tutorial

### Step 1: Prepare Configuration

Create or modify a config file (e.g., `configs/my_experiment.yaml`):

```yaml
seed: 42
output_dir: './experiments/my_experiment'

data:
  data_root: '../data/BraTS2021_slice'
  target_size: [240, 240]  # Resize images to 240x240

anchor:
  strategy: 'eigenface'     # Use PCA + K-means for anchors
  n_components: 50          # PCA components
  n_anchors: 8              # Number of anchor prototypes
  max_images_for_pca: 5000  # Max images for PCA computation

model:
  backbone: 'vit_small_patch16_dinov3.lvd1689m'
  freeze_backbone: true     # Freeze DINOv3, only train projection
  projection_dim: 128       # Projection head output dimension

loss:
  margin: 1.0
  alpha: 1.0
  beta: 0.0                 # ⚠️ IMPORTANT: Set to 0.0 (disable repeller)
  use_dense: true           # Enable pixel-level loss
  global_weight: 1.0
  dense_weight: 0.5

training:
  epochs: 50
  batch_size: 64
  lr: 0.0001
  val_interval: 1           # Validate every epoch
  early_stopping_patience: 15
```

**Critical Configuration Notes:**
- `beta: 0.0` - **MUST be 0.0!** Repeller loss is harmful for single-class anomaly detection
- `use_dense: true` - Enables pixel-level localization
- `freeze_backbone: true` - Only trains the projection head (faster, less prone to overfitting)

---

### Step 2: Generate Anchors (Optional - Standalone)

Anchors are automatically generated during training, but you can pre-generate them:

```bash
python main.py --config configs/my_experiment.yaml --eval-only
```

This creates: `experiments/my_experiment/anchor_embeddings.pt`

**What are anchors?**
- Anchors are prototype representations of "normal" brain scans
- Generated using PCA + K-means clustering on training features
- Used as reference points to compute anomaly scores

---

### Step 3: Train the Model

#### Option A: Fresh Training

```bash
python main.py --config configs/my_experiment.yaml
```

**Training Process:**
1. Loads DINOv3 backbone (frozen)
2. Generates anchors from training data
3. Trains projection head for 50 epochs
4. Validates every epoch, tracks:
   - Image-level AUROC
   - Pixel-level AUROC (if anomalous samples exist)
   - Loss components (attractor, repeller, dense)
5. Saves best model based on validation Image AUROC
6. Early stops if no improvement for 15 epochs

**Outputs:**
```
experiments/my_experiment/
├── anchor_embeddings.pt      # Anchor prototypes
├── best_model.pth            # Best checkpoint (highest val AUROC)
├── final_model.pth           # Final checkpoint (last epoch)
├── training_history.json     # All training metrics
├── training_curves.png       # Loss/AUROC plots
├── config.yaml               # Copy of config used
└── checkpoints/
    ├── checkpoint_epoch_5.pth
    ├── checkpoint_epoch_10.pth
    └── ...
```

#### Option B: Resume Training

```bash
python main.py --config configs/my_experiment.yaml --resume experiments/my_experiment/checkpoint_epoch_10.pth
```

#### Option C: Use Pre-generated Anchors

```bash
# If anchors already exist, skip regeneration
python main.py --config configs/my_experiment.yaml --skip-anchors
```

---

### Step 4: Monitor Training

During training, you'll see:

```
Epoch 24 Summary:
  Train Loss: 0.0004
    Attractor: 0.0004
    Repeller: 0.0000 (disabled, beta=0)
    Dense: 0.0002 (Attr: 0.0002)
  Anchor Balance: min=0.112, max=0.145, std=0.012
  Time: 45.3s

  Validation Results:
    Val Loss: 0.0005
      Attractor: 0.0004
      Repeller: 0.0000 (disabled, beta=0)
      Dense: 0.0003
    Image AUROC: 0.8601
    Pixel AUROC: 0.9170
  ✓ New best model! AUROC: 0.8601
```

**Key Metrics:**
- **Attractor Loss**: Pulls normal samples toward anchors (should decrease)
- **Repeller Loss**: Should be 0.0 (disabled via beta=0.0)
- **Dense Loss**: Pixel-level alignment (should decrease)
- **Image AUROC**: Image-level anomaly detection (0-1, higher better)
- **Pixel AUROC**: Pixel-level localization (0-1, higher better)
- **Anchor Balance**: Distribution of samples across anchors (balanced = better)

---

### Step 5: Generate Training Visualizations

After training completes:

```bash
python plot_from_checkpoint.py --checkpoint-dir experiments/my_experiment
```

Generates `training_curves.png` with:
- Total Loss (train + val)
- Attractor Loss
- Image AUROC
- Pixel AUROC

**Example interpretation:**
- Loss should **decrease** over time
- AUROC should **increase** or stabilize
- Large gap between train/val loss = overfitting
- Pixel AUROC > Image AUROC = good localization

---

### Step 6: Evaluate on Test Set

```bash
python evaluate_test.py \
    --checkpoint-dir experiments/my_experiment \
    --checkpoint best \
    --output-dir experiments/my_experiment/test_evaluation
```

**Arguments:**
- `--checkpoint-dir`: Experiment directory
- `--checkpoint`: Which checkpoint to use:
  - `best`: Best validation AUROC
  - `final`: Last epoch
  - `<path>`: Custom checkpoint path
- `--output-dir`: Where to save results (optional)

**Outputs:**
```
experiments/my_experiment/test_evaluation/
├── evaluation_metrics.json   # Detailed metrics
├── predictions.npz           # Anomaly scores & predictions
├── roc_curves.png            # ROC curves
├── score_distributions.png   # Score histograms
├── pr_curve.png              # Precision-Recall curve
└── visualizations/
    ├── normal_samples/       # Normal predictions
    │   └── *.png
    └── anomaly_samples/      # Anomaly predictions with masks
        └── *.png
```

**Metrics in `evaluation_metrics.json`:**
```json
{
  "image_auroc": 0.8035,      // Image-level detection
  "image_aupr": 0.9412,       // Average Precision
  "pixel_auroc": 0.8907,      // Pixel-level localization
  "pixel_aupr": 0.2867,       // Pixel-level AP
  "num_normal": 640,
  "num_anomaly": 3075,
  "operating_points": {       // Different FPR thresholds
    "fpr_0.01": {...},
    "fpr_0.05": {...},
    "fpr_0.10": {...}
  },
  "confidence_intervals": {   // Bootstrap CI for AUROC
    "auroc_mean": 0.8030,
    "auroc_std": 0.0100,
    "auroc_lower": 0.7826,
    "auroc_upper": 0.8226
  }
}
```

---

### Step 7: Analyze Results

#### Visualize Predictions

Check `test_evaluation/visualizations/`:
- Green boxes = correctly classified
- Red boxes = misclassified
- Heatmaps show anomaly score intensity

#### Check Training Curves

```bash
python plot_from_checkpoint.py --checkpoint-dir experiments/my_experiment
```

#### Analyze Anchor Quality

```bash
python analyze_issues.py --experiment-dir experiments/my_experiment
```

**This checks:**
1. **Distribution Shift**: KS test between train/val embeddings
2. **Anchor Quality**: Silhouette score, separation
3. **Embedding Analysis**: t-SNE visualization
4. **Recommendations**: Actionable improvement suggestions

---

## ⚙️ Configuration

### Configuration Files

Located in `configs/`:

| File | Description |
|------|-------------|
| `recommended.yaml` | **Recommended config** - fixes critical issues |
| `default.yaml` | Basic config for quick experiments |
| `grid_search.yaml` | Multiple configs for hyperparameter search |

### Key Parameters

#### Anchor Configuration
```yaml
anchor:
  strategy: 'eigenface'      # 'eigenface' or 'kmeans'
  n_anchors: 8               # Number of prototypes (4-16)
  n_components: 50           # PCA components (eigenface only)
  max_images_for_pca: 5000   # Limit for PCA computation
```

#### Loss Configuration
```yaml
loss:
  margin: 1.0                # Margin for contrastive loss
  alpha: 1.0                 # Attractor weight
  beta: 0.0                  # ⚠️ Repeller weight (MUST be 0.0)
  use_dense: true            # Enable pixel-level loss
  global_weight: 1.0         # Image-level loss weight
  dense_weight: 0.5          # Pixel-level loss weight
```

**⚠️ CRITICAL:**  
`beta: 0.0` is **required**! Repeller loss pushes anchors apart, which causes training loss to **increase** instead of decrease for single-class anomaly detection.

#### Training Configuration
```yaml
training:
  epochs: 50
  batch_size: 64             # Reduce if OOM
  lr: 0.0001                 # Learning rate
  val_interval: 1            # Validate every N epochs
  early_stopping_patience: 15
```

---

## 🛠️ Utilities & Analysis

### 1. Verify Dataset

Check dataset structure and statistics:

```bash
python verify_dataset.py --data-root ../data/BraTS2021_slice
```

### 2. Test Data Loading

Verify dataloaders work correctly:

```bash
python test_data_loading.py --config configs/recommended.yaml
```

### 3. Validate Pixel AUROC Computation

Test that pixel AUROC is computed correctly:

```bash
python validate_pixel_auroc.py
```

### 4. Plot from Checkpoint

Generate plots without retraining:

```bash
python plot_from_checkpoint.py --checkpoint-dir experiments/my_experiment
```

### 5. Analyze Training Issues

Diagnose training problems:

```bash
python analyze_issues.py --experiment-dir experiments/my_experiment
```

**Outputs:**
- Distribution shift detection (KS test)
- Anchor quality metrics (silhouette score)
- t-SNE embedding visualization
- Actionable recommendations

---

## 🐛 Troubleshooting

### Training Loss Increases

**Problem:** Training loss goes up instead of down  
**Cause:** Repeller loss (beta > 0) pushes same-class anchors apart  
**Fix:** Set `beta: 0.0` in config

### No Pixel AUROC During Validation

**Problem:** `val_pixel_auroc` is empty in `training_history.json`  
**Cause:** Fixed in latest version  
**Solution:** Already fixed - `eval.py` now properly computes pixel AUROC

### CUDA Out of Memory

**Problem:** GPU memory error during training  
**Fix:** Reduce batch size in config:
```yaml
training:
  batch_size: 32  # or 16
```

### Low Validation AUROC

**Possible causes:**
1. **Too few training samples** - Need 1000+ normal samples
2. **Poor anchor quality** - Run `analyze_issues.py` to check
3. **Distribution shift** - Val set too different from train
4. **Incorrect beta** - Must be 0.0!

**Debug steps:**
```bash
# Check anchor quality
python analyze_issues.py --experiment-dir experiments/my_experiment

# Verify dataset
python verify_dataset.py --data-root ../data/BraTS2021_slice

# Check config
cat experiments/my_experiment/config.yaml | grep beta  # Should be 0.0
```

### Poor Test Performance

**If Val AUROC good but Test AUROC poor:**
1. **Overfitting** - Reduce projection_dim, add weight_decay
2. **Distribution shift** - Test set very different
3. **Wrong checkpoint** - Try `--checkpoint final` instead of `best`

**If both Val and Test AUROC poor:**
1. **Wrong beta** - Check it's 0.0
2. **Bad anchors** - Try increasing `n_anchors` or `n_components`
3. **Insufficient training** - Increase epochs

### Pixel AUROC Not Computed

**Symptoms:**
- Training completes but `pixel_auroc` is empty/NaN in history
- Error: "inconsistent numbers of samples" during validation

**Root cause:**
- Size mismatch between model outputs and mask dimensions
- Model outputs 256×256 maps but masks are 240×240 (or vice versa)

**Fix applied in eval.py:**
- Automatically resizes pixel scores to match mask dimensions using bilinear interpolation
- Handles both 256→240 and 240→240 cases correctly

**Verify fix works:**
```bash
python validate_pixel_auroc.py --checkpoint experiments/bmad_baseline/best_model.pth
# Should show: ✓ SUCCESS: Pixel AUROC computed correctly!
```

**Manual fix (if needed):**
1. Ensure `target_size` in config matches your mask size (usually [240, 240])
2. Pass correct `target_size` to `evaluate_model()` function
3. Check masks are properly loaded (use `verify_dataset.py`)

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Checkpoint Loading Errors

**Problem:** `weights_only` error when loading checkpoint  
**Fix:** Update PyTorch to 2.0+ or use `weights_only=False`

---

## 📊 Expected Performance

### BraTS2021_slice Dataset

**With recommended config:**
- **Image AUROC**: 0.80-0.87 (validation), 0.78-0.82 (test)
- **Pixel AUROC**: 0.89-0.92 (validation), 0.88-0.90 (test)
- **Training time**: ~40-50 epochs to convergence (~30-45 min on RTX 3090)

**Performance factors:**
- Frozen DINOv3 backbone → faster, less overfitting
- Trainable projection head → adapts to medical domain
- Dense loss → improves localization
- Beta=0.0 → stable training

---

## 📁 Project Structure

```
project/
├── README.md                    # This file
├── main.py                      # Main training script
├── train.py                     # Trainer class
├── eval.py                      # Evaluation functions
├── evaluate_test.py             # Test set evaluation
├── model.py                     # DINOv3 + Detector
├── loss.py                      # Anchor margin losses
├── data.py                      # Dataset & dataloaders
├── anchors.py                   # Anchor generation
├── utils.py                     # Utilities
├── plot_from_checkpoint.py      # Plotting script
├── analyze_issues.py            # Diagnostic tool
├── validate_pixel_auroc.py      # Pixel AUROC test
├── verify_dataset.py            # Dataset checker
├── requirements.txt             # Dependencies
├── configs/                     # Configuration files
│   ├── recommended.yaml         # ⭐ Recommended config
│   ├── default.yaml
│   └── grid_search.yaml
└── experiments/                 # Output directory
    └── <experiment_name>/
        ├── anchor_embeddings.pt
        ├── best_model.pth
        ├── training_history.json
        ├── training_curves.png
        └── test_evaluation/
```

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{bmad2024,
  title={BMAD: Brain MRI Anomaly Detection using DINOv3 and Anchor-based Learning},
  author={Your Name},
  year={2024}
}
```

---

## 📝 License

[Your License Here]

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

## 🔗 References

- **DINOv3**: [Paper](https://arxiv.org/abs/2304.07193) | [Code](https://github.com/facebookresearch/dinov2)
- **BraTS**: [Website](http://braintumorsegmentation.org/)
- **CAMaL Loss**: Class Anchor Margin Loss for anomaly detection

---

**Last Updated:** October 29, 2025
