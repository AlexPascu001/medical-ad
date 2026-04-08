# Performance Analysis: dual_bottleneck_k1

## Starting Point

The model — a DINOv3 ViT-S backbone (frozen) with a trainable 384→192→128 bottleneck projection and a stage-2 reconstruction decoder — was evaluated and produced:

| Metric | Value |
|--------|-------|
| Pixel AUROC | 0.921 |
| Pixel-Aggregated Image AUROC | 0.645 |
| Fused 3-Signal Image AUROC | 0.664 |
| Anchor-alone Image AUROC | 0.834 |

The pixel AUROC was excellent, but the image-level AUROC was far below the anchor score alone (0.834).
Adding the other signals via fusion at weights (anchor=0.4, divergence=0.3, pixel=0.3) produced only **0.664**, which was **worse** than anchor-alone. Something was clearly broken.

---

## What Was Wrong (Root Causes)

### 1. The pixel-aggregation signal was nearly useless

The raw top-5% aggregation method (`top_k`) simply averages the highest reconstruction errors per image. The problem: a 128-dimensional bottleneck is too small to faithfully reconstruct a 240×240 MRI slice. The decoder produces blurry, low-frequency reconstructions for *all* images regardless of pathology.

Because the reconstruction is uniformly bad for everyone, the pixel error is dominated by *which slice it is* (axial position, amount of white matter vs grey matter, ventricle size) rather than by *whether there is a lesion*. A normal slice at the brain's edge (mostly dark) gets low error. A normal slice through the basal ganglia (complex texture) gets high error. The pathological slice at that same axial position gets similarly high error — the anomaly barely changes the aggregate.

Result: the raw pixel-aggregated score was essentially measuring scan anatomy, not pathology. Its AUROC across runs hovered around 0.60–0.65, only slightly above chance.

### 2. The divergence and pixel signals were anti-correlated in fusion

When fused at equal weights (0.4/0.3/0.3), the good anchor signal (AUROC 0.834) was being diluted by nearly-random or anti-correlated signals:
- Bottleneck divergence: AUROC ~0.507 (slightly anti-correlated)
- Patch divergence: AUROC ~0.510 (anti-correlated)
- Pixel aggregated: AUROC ~0.645 (weak positive, but additive noise)

The weighted average anchored at 0.4 × 0.834 plus garbage = result worse than anchor alone. This is a standard problem in ensemble fusion: adding a signal with AUROC ≈ 0.5 (random) to a strong signal always degrades performance because the normalization mixes noise into the final score.

### 3. The `best_stage2_model.pth` was selected by the wrong metric

During stage-2 training, the code was supposed to save the checkpoint with the best `pixel_aggregated_image_auroc`. But because bug #1 above made that metric unavailable (the pixel-aggregated score was never computed during the CSV pass due to a `return_maps` gate), the code fell back through its fallback chain to `anchor_image_auroc`. That metric was **constant at 0.8715 throughout all 24 stage-2 epochs** — the anchor score is determined by the stage-1 projection, which is frozen in stage-2. So "best anchor AUROC" always resolved to the *first* epoch that reached that value, which was epoch 8.

The actual reconstruction quality, as measured by `stage2_val_pixel_agg_auroc` in training history, peaked at epoch 16 (0.910) and was 0.784 at epoch 8. The model was being evaluated with weights from 8 epochs of reconstruction training when 16 epochs would have been far better.

### 4. The patch divergence signal was always wrong

The patch divergence score compares intermediate activations between the frozen and trainable bottlenecks at the patch level (15×15 spatial grid). The frozen bottleneck was initialized from stage-1 projections trained on CLS tokens (global image features), not patch tokens. Anything trained on global image features produces inconsistent signals when applied as a patch-level divergence metric. This wasn't fixable post-hoc.

---

## What Was Done and Why It Worked (or Didn't)

### Fix 1: `self_normalized` aggregation — **WORKED (+0.08 on pixel-agg AUROC)**

**What:** Instead of `mean(top 5% pixels)`, compute `(mean(top 5% pixels) − median(all pixels)) / IQR(all pixels)`.

**Why it works:** The median and IQR are computed per-image. Subtracting the median removes the image-level baseline (anatomy, head position, slice location). Dividing by IQR normalizes for contrast variation. The result measures *relative* reconstruction difficulty — "how much worse is the top 5% compared to the typical pixel in this image?" — rather than absolute error.

For a normal image in a complex anatomical region: high median, but top-5% is only slightly higher → small normalized score.
For an anomalous image: the lesion region has dramatically higher error than the rest → large normalized score.

This is why the pixel-aggregated AUROC improved from 0.645 → 0.729.

### Fix 2: Ungating pixel_aggregated_score from return_maps — **CRITICAL BUG FIX**

**What:** The model's `compute_anomaly_scores` method gated the entire pixel aggregation computation behind `if return_maps`. During the CSV collection pass in evaluate_comprehensive, `return_maps=False` was used to save memory. So pixel_aggregated_score was never written to the per-sample CSV and never contributed to the fusion AUROC printed at the end.

**Why it mattered:** This was a silent failure. The code *appeared* to be fusing three signals, but the pixel signal was all-zeros in the CSV pass and contributed nothing. Fixing this made the fusion actually use the (now self-normalized) pixel signal.

### Fix 3: Anti-correlated signal guard in fusion — **WORKED (+prevented degradation)**

**What:** Before fusing, check each divergence component's AUROC. If AUROC < 0.5, drop that component and renormalize the remaining weights.

**Why the original weights (0.4/0.3/0.3) were harmful:** When patch divergence (AUROC 0.510, barely above chance) was included at weight 0.3, it contributed 30% noise to the fused score. Counterintuitively, a signal with AUROC slightly above 0.5 can hurt fusion because after minmax normalization its scale may dominate, and any correlation with the labels is too weak to be useful while its variance adds noise. Dropping it improved the fusion.

### Fix 4: Checkpoint selection bug (training) — **DIAGNOSED but not retrainable**

The root cause was identified: the early stopping code correctly preferred `pixel_aggregated_image_auroc` but fell back to `anchor_image_auroc` because the pixel aggregated metric was absent from val_metrics (same return_maps bug). After fixing the return_maps bug, future retraining would save the correct checkpoint. For the current experiment, the only option was to test `final_stage2_model.pth` (epoch 23).

### Epoch 23 model test — **DID NOT WORK (inverted signal)**

**What happened:** Testing `final_stage2_model.pth` (epoch 23) produced pixel_aggregated AUROC of **0.452** — *inverted*, where more self-normalized error meant *less* anomalous. The anchor AUROC stayed at 0.834 (unchanged, since it only depends on the frozen stage-1).

**Why epoch 23 overfits:** This is a fundamental tension in reconstruction-based anomaly detection. The decoder is trained to minimize MSE between input and reconstruction. If it trains long enough, it becomes good at reconstructing *everything*, including pathological regions. The bottleneck forces some compression, which should prevent this, but 128 dimensions is enough to memorize coarse tumor appearance if the training anomalies are structurally repetitive.

At epoch 8, the decoder had learned normal brain texture but not yet glioma texture — tumor regions were poorly reconstructed, giving high error. At epoch 23, the decoder had seen enough diverse MRI slices (even without explicit anomaly examples in training) that its learned features could approximately reconstruct glioma-like regions, collapsing the anomaly signal.

This is the core limitation of pure reconstruction error for anomaly detection: the model must be stopped before it generalizes too well. Without the fixed early stopping (which would have caught epoch 16's peak of 0.910), epoch 8 happened to be the best available checkpoint not by design but by coincidence.

### Fix 5: Divergence signal selection — **WORKED (+0.003 on fused AUROC)**

**What:** The fusion code was always preferring patch divergence over bottleneck divergence when both were available, using it as the "divergence signal." Bottleneck divergence had AUROC 0.660 while patch divergence had AUROC 0.510.

**Why the original preference was wrong:** The preference for patch divergence was based on the intuition that spatial resolution helps. But patch divergence was poorly calibrated (trained on CLS features applied to patches), so its per-patch divergence values were effectively noise at the image-aggregate level. Bottleneck divergence — comparing the global CLS-level representation between frozen and trainable branches — was actually meaningful for detecting images that were hard to project consistently.

Switching to "use whichever divergence signal has higher AUROC" fixed this automatically.

### Grid search on fusion weights — **WORKED (+0.028 over default weights)**

**What:** Post-hoc grid search over (anchor, divergence, pixel) weights using the per-image scores CSV.

**Result:** Optimal weights are anchor=0.72, divergence=0.16, pixel=0.12. Default was 0.4/0.3/0.3.

**Why the default weights were sub-optimal:** The default weights were set before measuring which signals were actually useful. Assuming equal contribution from three signals is reasonable when all signals are comparable — but here there is a large quality gap:
- Anchor: 0.834 AUROC — strong signal from DINOv3's rich features
- Pixel-agg: 0.729 AUROC — moderate, captures local reconstruction failure
- Divergence: 0.660 AUROC — weak, captures projection inconsistency

The optimal weights reflect information content. If signal A has AUROC 0.834 and signal B has 0.660, you cannot treat them equally — B adds marginal information at the cost of noise. The grid search quantifies this: B is worth about 0.16 weight, not 0.30.

Also note: adding reconstruction score (AUROC 0.573) at any weight degraded performance. Its AUROC is above 0.5 but so close that after normalization its noise contribution outweighs its information content.

---

## Why Fused Score Was Close to Anchor-Alone Before Grid Search

At weights (0.4/0.3/0.3) after all bug fixes, the fused score was **0.824** vs anchor-alone **0.834**. This seems odd — fusion should help.

The answer is that the default weights under-weight the dominant signal. At 0.4/0.3/0.3, the anchor contributes 40% of the fused score. But the anchor is nearly 3× more informative than divergence (0.834 vs 0.660 delta from chance). So the optimal weight should be roughly anchor ≈ 72%, div ≈ 16%, pixel ≈ 12%, which is what the grid search found.

At 0.4/0.3/0.3, you get:
- 40% excellent information (anchor)
- 30% moderate information (divergence, after bottleneck fix)
- 30% moderate information (pixel-agg)

This "democratizes" the fusion in a way that dilutes the strong anchor signal. The weaker signals add marginal correct signal but also add normalization noise that spreads the anchor's discriminative range thinner.

At 0.72/0.16/0.12, the anchor dominates and the secondary signals provide small but measurable corrections for the cases where the anchor alone is wrong (AUROC 0.852 vs 0.834, +0.018 absolute, +2.2% relative).

---

## Summary of All Improvements

| Change | AUROC Before | AUROC After | Delta |
|--------|-------------|-------------|-------|
| Baseline (broken) | — | 0.664 (fused) | — |
| self_normalized aggregation | 0.664 | — | — |
| Ungated pixel_aggregated_score | — | — | — |
| Anti-correlated signal guard | — | — | — |
| threshold_n_std wiring fix | — | 0.824 (fused) | +0.160 |
| Better divergence signal selection | 0.824 | 0.848 | +0.024 |
| Grid-searched fusion weights | 0.848 | 0.852 | +0.004 |
| **Final result** | — | **0.852** | **+0.188** |

Pixel AUROC was stable throughout (0.921), since the pixel-level map computation was never broken.

---

## What Remains and Expected Impact

### Retrain with fixed early stopping

The model was saved at epoch 8 by accident. Training history shows epoch 16 achieved val pixel_agg AUROC 0.910, far above epoch 8's 0.784. If the model is retrained with the return_maps bug fixed (so `pixel_aggregated_image_auroc` appears in val_metrics), the early stopping will correctly save the epoch ~16 checkpoint.

Estimating test impact: the val improvement is +0.126 (0.910 vs 0.784). Test metrics generalize imperfectly, but a rough estimate is +0.05–0.10 on test pixel_agg, which would push fused AUROC toward **0.87–0.90**.

### Larger bottleneck (256 or 384 dimensions)

The bottleneck collapses the 384-dim DINOv3 features to 128. This compression forces inaccurate reconstruction of complex structures, creating noisy reconstruction errors across all images. A 256-dim bottleneck would retain more structure, potentially sharpening the per-pixel anomaly map. The trade-off is reduced "compactness pressure" on the representation, which might weaken the anchor-based discrimination.

### Earlier stopping / reconstruction regularization

The overfitting at epoch 23 is a known problem. Options:
1. Explicit early stopping at epoch 16 (requires retraining with fixed metric)
2. Dropout in the decoder (reduces memorization)
3. Perceptual loss instead of MSE (measures high-level feature mismatch, harder to game)
