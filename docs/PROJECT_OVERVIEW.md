# Medical-AD Project Overview

## 1) Purpose and Scope

This repository implements **brain MRI anomaly detection** on BraTS slice data using:

- A **DINOv3-based encoder** for semantic representation learning.
- **Anchor-based anomaly scoring** (Class-Anchor style training).
- Optional **stage-2 reconstruction branch** for complementary anomaly cues.
- Comprehensive **image-level and pixel-level evaluation**.
- Optional **score-level ensembles** across experiments, with explicit score normalization.

The project is experimentation-heavy and supports multiple anchor strategies, loss variants, and large anchor-count studies.

---

## 2) High-Level System Architecture

### Core pipeline

1. **Data loading + preprocessing** (BraTS slice folders; normal-only train split, mixed val/test).
2. **Anchor preparation** in DINO embedding space.
3. **Model creation** (`DINOv3Backbone` + `AnomalyDetector`).
4. **Stage-1 training** (anchor-distance objective).
5. Optional **Stage-2 training** (reconstruction + consistency).
6. **Evaluation** with AUROC/AUPR + visualizations + score exports.
7. Optional **ensemble post-processing** over per-sample score CSVs.

### Key implementation files

- `project/main.py`: end-to-end orchestration (train/eval, stage-2, config defaults).
- `project/model.py`: backbone wrapper, anchor logic, stage-2 branch, inference scoring.
- `project/train.py`: stage-1 + stage-2 training loops and checkpoint/history handling.
- `project/eval.py`: metrics, confidence intervals, plots, and per-sample score export.
- `project/anchors.py`: anchor generation and embedding utilities.
- `project/loss.py`: canonical anchor-margin loss implementation used in main pipeline.
- `project/evaluate_reproject_top3_ensemble.py`: normalized score-level ensemble evaluation.

---

## 3) Data and Splits

Expected dataset structure (BraTS slices):

- `train/good/*.png` (normal only)
- `valid/good/img`, `valid/good/label`
- `valid/Ungood/img`, `valid/Ungood/label`
- `test/good/img`, `test/good/label`
- `test/Ungood/img`, `test/Ungood/label`

Important behavior:

- Training uses only normal images.
- Validation/test labels are binary (`0=normal`, `1=anomaly`).
- Pixel masks are loaded when present and used for pixel-level metrics.
- Grayscale slices are converted to 3-channel tensors for DINOv3 compatibility.

---

## 4) Backbone and Representation Flow

`project/model.py` contains the feature stack:

- **DINOv3 ViT backbone** (default `vit_small_patch16_dinov3.lvd1689m`).
- Optional **trainable projection head** (`384 -> 192 -> 128`).
- Optional **pixel decoder** path (multi-scale feature pyramid decoder).

Model outputs include:

- Global embedding and distances to anchors.
- Optional dense/pixel outputs.
- Optional stage-2 reconstruction outputs.

---

## 5) Anchor Concepts and Modes

### A) Reprojecting each pass (reproject anchors each forward)

- Anchors are generated in **384D semantic DINO space**.
- During forward pass, anchors are passed through the current projection head.
- This can adapt anchors with representation learning, but introduces moving-target dynamics.

### B) Decoupled/fixed-target style

- Conceptually separates semantic assignment from geometric training targets.
- Generally more stable in some settings, especially at scale.

### C) Learnable anchors (separate path)

- Implemented in dedicated learnable-anchor flow (`learnable_anchors.py`, `train_learnable_anchors.py`).
- Not the same training path as default `main.py` + `loss.py` pipeline.

Potentially missed detail:

- There are **two CAM-related implementations** in the repo (`loss.py` vs `learnable_anchors.py`) with differences in formulations and usage context. Main experiments use the canonical path in `loss.py`.

---

## 6) Stage-1 Training (Anchor-Based Detection)

Stage-1 optimizes anchor-structured embedding behavior:

- Pull sample embedding toward assigned anchor (attractor term).
- Push anchors apart (repeller/diversity behavior, depending on config/loss variant).
- Optional dense branch behavior for localization path compatibility.

Pseudo-label behavior:

- Commonly uses **fixed pseudo-label assignment** computed once (semantic anchor assignment), then reused through training.

Training outputs:

- `best_model.pth`, `final_model.pth`.
- `training_history.json`, `training_summary.json` (if enabled in flow), training curves, visualizations.

---

## 7) Stage-2 Reconstruction Branch

Enabled via `stage2.enabled: true` in config.

Stage-2 in `AnomalyDetector`:

- Adds copied stage-2 projection branch + fusion block + reconstruction decoder.
- Uses anchor assignment from stage-1 distance path.
- Can freeze encoder and/or anchors during stage-2.

Stage-2 losses:

1. **Reconstruction loss** (`mse` or `l1`) between reconstructed and input normalized image.
2. **Consistency loss** (`cosine` or `l2`) between stage-2 feature and assigned anchor embedding.

Stage-2 pixel map options:

- `reconstruction_l2`: channel-mean squared residual map.
- `reconstruction_l1`: channel-mean absolute residual map.

Potentially missed detail:

- Stage-2 training is self-supervised (no mask supervision in loss); masks are for evaluation metrics.

---

## 8) Inference Scoring (Image and Pixel)

### Image-level scores

From `compute_anomaly_scores` in `model.py`:

- **Anchor image score**: min distance to anchors.
- **Reconstruction image score** (if stage-2 enabled): mean reconstruction error over image.
- **Optional raw combined score** can be emitted in model output; authoritative combined metric is computed in eval.

### Pixel-level scores

Priority order at inference:

1. **Stage-2 reconstruction pixel map** (if enabled) is set as primary `pixel_scores`.
2. Else anchor-pixel map from pixel decoder/dense fallback (`min` over anchors per pixel/patch).

Evaluation reports may include:

- `pixel_auroc/aupr` (primary map used).
- `reconstruction_pixel_auroc/aupr` (if reconstruction map available).
- `anchor_pixel_auroc/aupr` (if anchor pixel map available).

---

## 9) Evaluation and Artifacts

`evaluate_comprehensive` and related routines in `project/eval.py` produce:

- Image metrics: AUROC, AUPR, operating points, confidence intervals.
- Pixel metrics (if masks + maps available).
- Plots: ROC curves, score distributions, qualitative normal/anomaly samples.
- JSON metrics file: `evaluation/evaluation_metrics.json`.
- Per-sample score export: `evaluation/evaluation_image_scores.csv`.

Per-sample CSV is critical for calibrated ensemble fusion.

---

## 10) Ensemble Aggregation and Score Calibration

Problem observed in practice:

- Different experiments produce scores on different ranges (e.g., narrow low ranges vs broader ranges).
- Raw averaging can unfairly bias higher-scale models.

Current solution:

- `project/evaluate_reproject_top3_ensemble.py` loads per-sample score CSVs.
- Applies per-model normalization before weighted aggregation.
- Supported methods: `none`, `minmax`, `zscore`, `robust`, `rank`.
- Writes fused results:
  - `ensemble_scores.csv`
  - `ensemble_metrics.json`

Potentially missed detail:

- Ensemble merging is done by `(path, label)` intersection. If data/sample identity differs across runs, sample count drops.

---

## 11) Configuration System and Important Flags

Default config lives in `project/configs/default.yaml`.

Notable knobs:

- `anchor.*`: strategy, count, embedding-space behavior, reproject behavior.
- `model.*`: backbone and projection head.
- `loss.*`: margin and component weights.
- `training.*`: optimizer, epochs, early stopping, pseudo-label mode.
- `stage2.*`:
  - reconstruction and consistency losses,
  - pixel map generation (`pixel_map.enabled/type`),
  - pixel metric toggle (`pixel_metrics.enabled`),
  - score combination (`score_combination.enabled/alpha/normalization`).
- `eval.*`: pixel metric computation and bootstrap settings.

---

## 12) Experiment Management

The repository contains many experiment directories under `experiments/` and several runner scripts.

Useful scripts:

- `run_two_stage_reproject_k1_k512.ps1`: sequential stage-2 runs for k=1 and k=512.
- `run_large_anchor_experiments.ps1`: broad large-anchor study runs.

Evaluation-only mode:

- `main.py --eval-only` reuses existing experiment directory (important for regenerating metrics/score CSVs in-place).

---

## 13) Common Pitfalls (Easy to Miss)

1. **Config path typos / cwd mismatch**
   - Running from `project/` vs repo root changes relative path resolution.
   - Prefer root-level invocation with explicit config path.

2. **Output directory uniqueness behavior**
   - Training mode can auto-append suffixes (`_1`, `_2`, ...), creating many sibling experiment folders.
   - Eval-only now expects existing target folder and reuses it.

3. **Score scale mismatch in ensembles**
   - Never trust raw average across heterogeneous experiments without normalization.

4. **Warnings vs fatal errors in PowerShell runners**
   - Some data augmentation warnings print on stderr but are non-fatal.

5. **Pixel metric availability**
   - Pixel metrics require masks and available pixel map outputs.
   - Some runs only produce image-level outputs.

6. **Large anchor count complexity**
   - High-K setups increase compute/memory and can clutter diagnostics; adaptive visualization is used for readability.

7. **Multiple CAM code paths**
   - Ensure you know whether experiment uses canonical `loss.py` pipeline or learnable-anchor-specific implementation.

---

## 14) Recommended Operational Workflow

1. Choose config and run training/evaluation.
2. Confirm `evaluation_metrics.json` and `evaluation_image_scores.csv` are generated.
3. For multi-model fusion, run normalized ensemble script (`rank` or `robust` are good starting points for scale mismatch).
4. Compare methods using AUROC/AUPR and inspect score distributions qualitatively.
5. Keep experiment naming and config snapshots consistent to avoid merge/identity confusion.

---

## 15) Quick Command Examples

Train/evaluate:

```powershell
D:/Documents/FMI/Disertatie/medical-ad/venv/Scripts/python.exe project/main.py --config project/configs/two_stage_k1.yaml
```

Eval-only refresh (same experiment folder):

```powershell
D:/Documents/FMI/Disertatie/medical-ad/venv/Scripts/python.exe project/main.py --config project/configs/reproject_k1024_early.yaml --eval-only
```

Normalized ensemble:

```powershell
D:/Documents/FMI/Disertatie/medical-ad/venv/Scripts/python.exe project/evaluate_reproject_top3_ensemble.py \
  --experiments solution_a_reproject_k1_1 reproject_k256_early reproject_k1024_early \
  --normalization rank \
  --score-column image_score \
  --weights 1 1 1 \
  --output experiments/reproject_top3_ensemble_rank
```

---

## 16) Final Notes

This project has evolved into a robust experimental platform rather than a single fixed training recipe. The most important implementation-level takeaways are:

- Anchor generation and assignment semantics matter as much as architecture.
- Stage-2 gives complementary signals, especially for pixel-level localization via reconstruction residuals.
- Calibration/normalization is mandatory when aggregating scores across heterogeneous runs.
- Reproducibility depends heavily on consistent config paths, output directory targeting, and split/sample identity alignment.
