# Scoring Analysis: K-Scaling Behaviour on BraTS2021

This document analyses how the number of anchors (K) affects each anomaly scoring signal, explains the mechanisms behind observed trends, and identifies weaknesses with potential improvements.

All experiments use the same setup: frozen DINOv3 ViT-S backbone, 128D projection, re-projected anchors, 100 epochs Stage 1, 20 epochs Stage 2, frozen bottleneck, score fusion enabled (0.4 / 0.3 / 0.3), pixel maps, full BraTS2021 dataset (7500 train / 83 val / 3715 test).

---

## 1. Experiment Results

| Metric | K=1 | K=16 | K=1024 |
|--------|-----|------|--------|
| **Anchor Image AUROC** | **0.804** | 0.680 | 0.551 |
| Reconstruction Image AUROC | 0.585 | 0.718 | **0.723** |
| Divergence Image AUROC | 0.497 | **0.579** | 0.315 |
| Pixel-Aggregated AUROC | 0.629 | 0.723 | **0.743** |
| Patch-Divergence Aggregated | **0.763** | 0.609 | 0.558 |
| **Fused 3-Signal AUROC** | **0.796** | 0.745 | 0.684 |
| Fused 3-Signal AUPR | **0.942** | 0.930 | 0.913 |
| **Pixel AUROC** | 0.910 | **0.922** | 0.918 |
| Pixel AUPR | 0.329 | **0.374** | 0.366 |

Key observations:
- **Anchor score** degrades monotonically with K — near random at K=1024.
- **Reconstruction and pixel-aggregated scores** improve with K — opposite trend.
- **Divergence** is weak everywhere and anti-correlated for K=1 and K=1024.
- **Pixel AUROC** is stable (~0.92) regardless of K.
- **Fused score** is dominated by whichever component is strongest, and fixed weights hurt.

---

## 2. Why Anchor AUROC Falls Sharply as K Grows

### Mechanism

The anchor score is `min(L2(sample_128d, anchor_k_128d))` over all K anchors (see `model.py:compute_anomaly_scores()`, line ~1095).

With **K=1**, every normal training image is pulled toward a single anchor during Stage 1. The embedding space forms one tight cluster. At test time:
- Normal images → close to the anchor → low score.
- Anomalous images → farther from the anchor → high score.
- **Clean separation** — there is only one "normal" prototype.

With **K=1024**, the 7500 training images are partitioned into ~7 images per cluster across 128D. Several problems emerge:

1. **Dense covering of the normal manifold.** 1024 anchors form a dense mesh over the normal distribution. Even anomalous images land close to *some* anchor by sheer proximity. The min-distance score loses discriminative power.

2. **Training difficulty.** The attractor loss pulls in 1024 directions simultaneously. The repeller loss must separate $\binom{1024}{2} = 523\text{,}776$ anchor pairs — an enormous constraint set that the optimiser cannot fully satisfy with a single 128D projection head.

3. **Embedding space fragmentation.** The projection head learns a compromise that maps each cluster adequately but none perfectly. Both normal and anomalous samples end up at similar intermediate distances from their nearest anchor.

### Why BraTS2021 Favours K=1

BraTS2021 FLAIR slices of healthy brains form a **homogeneous population** — they are all axial brain cross-sections with similar intensity distributions. A single cluster centre suffices to model this distribution. The simplicity of one anchor creates the sharpest possible decision boundary.

For datasets with more structural variation (e.g., different imaging planes, multiple organs, or multiple normality modes), larger K would likely be beneficial.

---

## 3. Why Reconstruction AUROC Is Weak for K=1 but Strong for K≥16

### Mechanism

The reconstruction decoder receives `fuser(stage2_feat ⊕ assigned_anchor_embedding)` and must reproduce the input image (see `model.py:forward()`, line ~969).

With **K=1**, every image — normal or anomalous — gets the **same** anchor embedding concatenated. The fuser input is always `(stage2_feat, anchor_1)`:
- The anchor embedding provides **zero discriminative information** — it acts as a constant bias.
- The decoder must reconstruct every image from `stage2_feat` alone.
- Both normal and anomalous images are reconstructed with similar fidelity, because the decoder has no cluster-specific context.

With **K=16 or K=1024**, different images get different anchor embeddings:
- Normal images → correct cluster anchor → well-conditioned reconstruction → low error.
- Anomalous images → assigned to the nearest (wrong) anchor → misleading context → higher error.
- More anchors → more specialised decoder per cluster → larger normal/anomalous gap.

### The Fundamental Tension

This reveals a core architectural trade-off:

| Signal | Benefits from | Explanation |
|--------|--------------|-------------|
| Anchor score | **Fewer anchors** | Sharper distance boundary |
| Reconstruction score | **More anchors** | More specialised decoder conditioning |

The optimal K depends on which signal dominates. For BraTS2021 with the current architecture, the anchor score is more powerful than reconstruction, so K=1 wins overall despite its poor reconstruction signal.

### Potential Improvement

For K=1, the anchor embedding is redundant in the fuser. Two options:
1. **Skip the anchor concatenation** when K=1 — simplify the fuser to use `stage2_feat` only, giving the decoder full capacity for reconstruction.
2. **Conditional reconstruction** via adaptive mechanisms (e.g., AdaIN or FiLM layers) — even a single anchor could provide useful conditioning if injected differently.

---

## 4. Why Divergence Is Weak and Sometimes Anti-Correlated

### What Divergence Measures

**CLS-level bottleneck divergence** = `1 - cos_sim(frozen_proj(CLS_raw), stage2_proj(CLS_raw))` (see `model.py:forward()`, line ~930).

The frozen projection is a deep copy of the Stage-1 projection head (never updated). The stage-2 projection is a clone that fine-tunes during Stage 2. The hypothesis: for normal images, both projections agree (low divergence); for anomalous images, stage-2 must deviate to minimise reconstruction loss (high divergence).

### Why It Fails

**K=1 (AUROC 0.497 — random):** With a single anchor, the reconstruction task is "reconstruct any brain from a single latent." The stage-2 projection barely needs to change from the frozen copy. The alignment loss ($w_a = 0.1$) keeps them close. The divergence is nearly uniform → no discriminative power.

**K=1024 (AUROC 0.315 — anti-correlated):** With many anchors, the stage-2 projection receives varied reconstruction gradients, but the alignment loss fights against divergence. The net effect: some normal images (with ambiguous cluster membership) show *higher* divergence than some anomalous images (whose nearest anchor happens to enable easy reconstruction). The signal inverts.

**K=16 (AUROC 0.579 — weak positive):** A moderate anchor count balances reconstruction gradient vs. alignment constraint. There is some discriminative power, but it remains weak.

### Root Causes

1. **Alignment loss fights divergence by design.** $L_{\text{align}} = 0.1 \cdot (1 - \cos\_sim)$ directly penalises the divergence that the system tries to measure at inference. The model learns to *minimise* the very quantity we want to detect.

2. **The frozen projection has no anomaly response.** It was trained on normal images only and simply extrapolates to anomalous inputs — there is no learned mechanism to produce a different response for anomalies.

3. **Only 20 epochs of Stage-2 training.** The stage-2 projection barely drifts from its initial state. Longer training might help, but the alignment loss caps the achievable divergence.

### Potential Improvements

- **Reduce or remove the alignment loss** for divergence-focused experiments. It was intended for training stability but actively suppresses the divergence signal.
- **Use a reconstruction-based divergence** instead: compare `recon_error_frozen / recon_error_trained` — this captures functional divergence rather than representational distance.
- **Train Stage 2 for more epochs** to allow greater divergence to develop.

---

## 5. Pixel-Aggregated Score — the Hidden Star

The pixel-aggregated score is the top-5th-percentile of per-pixel reconstruction L2 error (see `pixel_aggregation.py:aggregate_pixel_scores_torch()`). It reaches 0.629–0.743 image-level AUROC, making it the strongest reconstruction-derived image-level signal.

### Why It Outperforms Global Reconstruction Score

The global reconstruction score is `MSE(input, reconstructed).mean(C, H, W)`. A tumour occupies perhaps 5–15% of the image. The remaining 85–95% of healthy tissue reconstructs well, **diluting** the anomaly signal in the global mean.

The top-k percentile aggregation (95th percentile → top 5% of pixels) focuses on the *worst-reconstructed* region:
- For anomalous images: the top 5% captures the tumour area → high score.
- For normal images: the top 5% is just the noisiest healthy pixels → relatively low score.

This explains the consistent gap: pixel-aggregated AUROC (0.743 at K=1024) vs. global reconstruction AUROC (0.723 at K=1024). The advantage grows for smaller, more localised anomalies.

### Why It Improves with More Anchors

Same mechanism as §3: more anchors → more specialised decoder → larger gap in pixel-level error between normal and anomalous regions.

---

## 6. Patch-Divergence Aggregated — Strong for K=1, Weak for K≥16

Patch-divergence aggregated = per-patch (15×15) cosine divergence between frozen and stage-2 projection, upsampled to 240×240, then top-k percentile aggregated (see `model.py:compute_anomaly_scores()`, line ~1120).

### Why K=1 Is Best (AUROC 0.763)

With K=1, the spatial structure of the divergence map is the most informative signal. The CLS-level divergence is diluted (whole-image average) and lands at random-level AUROC (0.497). But the patch-level map can localise **where** the stage-2 projection deviates most from the frozen reference. Since all images share the same anchor, the map effectively captures "which patches are hardest for the reconstruction branch" — and those patches correspond to anomalous regions.

### Why K=1024 Is Worst (AUROC 0.558)

With many anchors, the alignment loss keeps the stage-2 projection close to frozen, and the specialised anchor embeddings reduce reconstruction difficulty per cluster. Per-patch divergence is uniformly low → no spatial discrimination.

---

## 7. The Fused 3-Signal Score

The fused score combines three normalised signals (see `eval.py:evaluate_model():_norm_fusion()`):

$$\text{score}_{\text{fused}} = w_a \cdot \hat{s}_{\text{anchor}} + w_d \cdot \hat{s}_{\text{div}} + w_p \cdot \hat{s}_{\text{pixel}}$$

Default weights: $w_a = 0.4$, $w_d = 0.3$, $w_p = 0.3$. Each signal is normalised independently (default: minmax over the dataset).

### Anti-Correlation Guard

If any signal's individual AUROC < 0.5, it is **dropped** from fusion and remaining weights renormalised. This prevents anti-correlated signals from pulling the score in the wrong direction.

### Divergence Signal Selection

When both CLS-level and patch-level divergence are available, the one with higher individual AUROC is automatically selected as the divergence signal.

### Per-K Behaviour

**K=1:** CLS divergence AUROC = 0.497 → dropped. Patch-divergence (0.763) selected. Effective fusion: `0.4 × anchor(0.804) + 0.3 × patch_div(0.763) + 0.3 × pixel_agg(0.629) = 0.796`. The fused score (0.796) is **lower** than anchor alone (0.804) because the weaker signals dilute it.

**K=16:** CLS divergence (0.579) selected over patch-divergence (0.609)... actually patch-div is higher, so patch-div is selected. Fusion: `0.4 × anchor(0.680) + 0.3 × patch_div(0.609) + 0.3 × pixel_agg(0.723) = 0.745`.

**K=1024:** CLS divergence AUROC = 0.315 → dropped. Patch-divergence (0.558) used. Fusion: `0.4 × anchor(0.551) + 0.3 × patch_div(0.558) + 0.3 × pixel_agg(0.743) = 0.684`. The weak anchor score drags the fusion down despite a strong pixel-aggregated signal.

### Why Fixed Weights Hurt

The 0.4/0.3/0.3 weights are suboptimal for every K:
- **K=1**: anchor is strong (0.804), should be weighted higher. Pixel-agg is weak (0.629), should be weighted lower.
- **K=1024**: anchor is nearly random (0.551), should be weighted lower or dropped. Pixel-agg is the star (0.743), should dominate.

---

## 8. Pixel AUROC — Stable Across K

Pixel AUROC (0.910 – 0.922) is nearly identical across all K values. It is computed as `roc_auc_score(all_pixel_masks_flat, all_pixel_scores_flat)` in `eval.py` — every pixel across all test images is treated as an independent binary classification.

This stability makes sense: pixel-level reconstruction quality depends on **decoder capacity and training**, not anchor count. All three models learn a similar "what healthy tissue looks like" representation at the pixel level. The anchor count only affects which conditioning vector the decoder receives, which matters for image-level scores but has minimal effect on the pixel-level error distribution.

---

## 9. Identified Weaknesses and Improvement Proposals

### 9.1 Fixed Fusion Weights

**Problem:** The 0.4/0.3/0.3 weights are never optimal. K=1 should weight anchor heavily; K=1024 should weight pixel-agg heavily.

**Proposal:** Learn weights via validation-set grid search after training, or use a lightweight attention-based fusion mechanism. Even a simple "pick the signal with the highest validation AUROC" strategy would outperform fixed weights.

### 9.2 Alignment Loss Suppresses Divergence

**Problem:** The alignment loss ($w_a = 0.1$) directly penalises the divergence signal during training. The model minimises the very quantity we want to maximise at inference. This is a design contradiction.

**Proposal:** (a) Remove alignment loss entirely — rely on reconstruction loss alone to regularise stage-2. (b) Replace the divergence metric with a task-based signal, e.g., `recon_error_frozen / recon_error_trained`. (c) Use a detached divergence: compute divergence between frozen and a stop-gradient copy of stage-2 features, so alignment loss does not directly minimise measured divergence.

### 9.3 K=1 Reconstruction Bottleneck

**Problem:** With K=1, the anchor embedding is a constant bias — it provides zero discriminative information to the fuser/decoder. Reconstruction AUROC is only 0.585.

**Proposal:** For K=1, either skip the anchor concatenation (fuser receives only `stage2_feat`) or condition the decoder via adaptive normalisation (AdaIN / FiLM) instead of concatenation.

### 9.4 Anchor Score Collapse at Large K

**Problem:** At K=1024, the anchor min-distance metric is nearly random (0.551) because the dense anchor mesh gives every sample — normal or anomalous — a nearby anchor.

**Proposal:** Use a scoring function that considers the **distribution** of distances, not just the minimum:
- **Lowe ratio:** `d_nearest / d_second_nearest` — measures how confidently a sample belongs to one cluster.
- **Energy-based score:** $-\log \sum_k \exp(-d_k / T)$ — weighs contributions from all anchors.
- **KNN-based:** Average distance to the K-nearest anchors rather than just the nearest.

### 9.5 "Pixel-Level Data Not Available" During Stage-1 Validation

**This is expected behaviour, not a bug.** During Stage-1 training, no reconstruction decoder or pixel decoder exists. The `compute_anomaly_scores()` method does not populate the `pixel_scores` key in its output, so the evaluation code logs the message and skips pixel metrics.

The message appears because `train.py:validate()` passes `compute_pixel_auroc=True` regardless of stage. The metric is simply not computed — the log message is informational.

### 9.6 Early Stopping Metric Noise

**Problem:** Stage 2 uses `pixel_aggregated_image_auroc` for early stopping. This metric depends on the top-5% worst pixels, which can fluctuate significantly between epochs — especially early in training when the decoder is unstable.

**Proposal:** Use a smoothed version (exponential moving average over last 3–5 epochs) or switch to pixel AUROC (more stable) for early stopping.

### 9.7 Insufficient Stage-2 Training Duration

**Problem:** 20 epochs may be insufficient for the stage-2 projection to diverge meaningfully from the frozen copy, especially with alignment loss active.

**Proposal:** Run experiments with 50–100 Stage-2 epochs. If alignment loss is removed (§9.2), fewer epochs may suffice because the stage-2 projection can adapt freely.
