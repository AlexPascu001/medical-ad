# Thesis Research Story

This is the thesis-facing story of the project: what the hypothesis was, what was
tried, what broke, what improved, what the final results mean, and how to present
the work honestly.

## One-Sentence Version

This dissertation investigated whether a supervised class-anchor metric-learning
objective can be adapted to one-class brain MRI anomaly detection with DINOv3
features; the adapted method achieved decent but sub-baseline results, and the
main contribution is the empirical explanation of why the transfer only partly
works.

## Research Hypothesis

The starting hypothesis was reasonable:

1. DINOv3 provides strong visual features for dense and global image reasoning.
2. Normal brain MRI slices should occupy a structured region of feature space.
3. K-means centroids or representative normal samples can serve as anchors for
   that normal region.
4. A CAM-style attractor loss can pull normal samples toward those anchors.
5. At test time, anomalous slices should be farther from all normal anchors, so
   nearest-anchor distance can be used as an anomaly score.

In short: replace supervised CAM class anchors with unsupervised normal-data
anchors, then use distance to the normal anchor set for anomaly detection.

## What Made The Hypothesis Non-Trivial

The project was not just "apply CAM to BMAD." Three mismatches made the problem
scientifically interesting:

1. CAM is supervised and multi-class; BMAD brain MRI anomaly detection is
   one-class.
2. CAM's repeller term assumes anchors represent real different classes; here
   anchors are artificial partitions of normal anatomy.
3. DINOv3 is valuable partly because frozen dense features are strong; training
   objectives that reshape those features can damage or miscalibrate the signal.

That is why the thesis should frame the work as an empirical transfer study, not
as a straightforward architecture proposal.

## The Research Path

### 1. End-To-End Anchor Pipeline

The first complete system trained normal slices against fixed anchors and used
nearest-anchor distance as the anomaly score. Early recovered docs show that this
phase had real engineering problems:

- image and pixel AUROC computation had bugs or mismatches;
- anomaly score direction had to be checked;
- dense/pixel scoring was not clean at first;
- validation tracking and class imbalance needed attention;
- anchor strategy and distance metric were open questions.

Thesis role: this phase explains why the first contribution was building a
measurable pipeline, not just running final experiments.

### 2. Anchor Strategy And Collapse

The first large research question was how anchors should be chosen:

- random normal samples;
- k-means centroids or closest samples;
- eigenface/PCA-style anchors;
- L2 versus cosine distance;
- different K values.

The early interpretation focused on anchor collapse: without a repeller, samples
could drift toward a single dominant anchor, and AUROC could peak before training
really improved the representation. This led to repeller loss, pretraining, and
more careful anchor initialization.

Thesis nuance: the early engineering conclusion was "repeller may prevent
collapse." The later scientific conclusion is subtler: the repeller can prevent
one type of collapse, but in a one-class setting it can also fragment normality
into artificial separated regions.

### 3. Learnable Anchors And Repeated Trials

Learnable anchors were explored because fixed anchors might become stale when the
representation changes. This branch tested whether anchors should move with the
training objective rather than remain fixed references.

The branch is historically important but should not be the final method:

- it shows the project tested alternatives to fixed centroids;
- it belongs in attempted variants or ablations;
- it did not become the strongest or cleanest thesis result.

### 4. Projection Drift, Reprojecting Anchors, And Decoupling

The next central problem was temporal mismatch:

- anchors are generated in the DINO feature space;
- the projection head starts random;
- if the projection changes, projected samples and projected anchors can drift;
- fixed pseudo-labels can become stale under that drift.

Two responses emerged:

- **Reproject anchors:** keep anchors in semantic DINO space and pass them through
  the current projection head at each forward pass.
- **Decouple assignment and geometry:** use semantic anchors for assignment, then
  train toward fixed geometric targets in projected space.

This period produced some of the strongest historical trained-image results:

- `reproject_k1_early_trainable_backbone_stage2_2`: image AUROC `0.8551`;
- `regfix_k1`: image AUROC `0.8539`;
- `dual_bottleneck_k1_tight`: image AUROC `0.8540`.

Thesis role: these are important historical upper bounds for the trained anchor
idea, but they should be framed carefully because they are older, mostly
trainable-backbone, less cleanly documented runs.

### 5. Two-Stage Reconstruction And Fusion

Reconstruction and divergence were added because anchor distance alone might miss
local pathology. The two-stage idea was:

1. Stage 1 learns the anchor/projection detector.
2. Stage 2 adds reconstruction and compares bottleneck behavior.
3. Image-level scores are fused from anchor, reconstruction/pixel aggregation,
   and divergence.

The dual-bottleneck phase is the best debugging story:

- raw reconstruction aggregation often measured slice complexity rather than
  pathology;
- pixel aggregation was accidentally gated by `return_maps`;
- patch divergence was poorly aligned with a CLS-trained projection;
- equal fusion weights diluted the strong anchor signal;
- tuned fusion improved `dual_bottleneck_k1` to fused AUROC `0.8520`.

Thesis role: this is a strong mixed-result case study. It shows that auxiliary
signals can help, but only after careful normalization, bug fixes, and
conservative weighting.

### 6. Clean Global Redesign

The redesign made the global-anchor method more controlled:

- frozen DINOv3 in late clean configs;
- k-means centroids in semantic space;
- fixed/capacitated pseudo-label assignment;
- reprojected anchors;
- longer Stage 2;
- diagnostics for cluster usage and nearest-vs-second-nearest separation.

The key finding is that K controls different regimes:

- `K=1` gives the best clean global raw-image score:
  `full_redesign_stage2e70_k1`, image AUROC `0.8057`.
- `K=1024` gives the best clean global fused score:
  `full_redesign_stage2e70_k1024`, fused AUROC `0.8260`.
- `K=4` can look structurally clean in diagnostics, but does not win final
  AUROC.

Thesis interpretation: good clustering, good raw anomaly scoring, and good fused
scoring are not the same objective. The model trades off interpretability,
coverage, and auxiliary-signal usefulness.

### 7. Patch Mode

Patch mode was the locality turn. Instead of scoring only a global CLS
representation against global anchors, the method matched dense patch tokens
against patch-level normal references.

This fits the medical problem better because tumors are local. The strongest
clean patch result is:

- `patch_stage2e70_k32`: image AUROC `0.8119`, fused AUROC `0.8282`,
  pixel AUROC `0.9298`.

The patch family has a different K story from global redesign:

- global redesign's best raw image score is K=1;
- patch mode benefits from moderate K, especially K=16 and K=32;
- patch anchors behave more like a local reference bank than semantic clusters.

Thesis role: patch mode is the most natural improvement of the original
hypothesis because it preserves the anchor idea while making normality local.

### 8. Location-KMeans Patch Banks

Location-kmeans made patch references spatially specific: patches are compared
against normal references from the same spatial location or local pool. This
tested whether location-aware normal structure improves dense anomaly detection.

The best late fused run is:

- `patch_location_kmeans_stage2recon_cosine_k32`: image AUROC `0.7958`,
  fused AUROC `0.8295`, pixel AUROC `0.9350`.

This narrowly beats `patch_stage2e70_k32` on fused AUROC (`0.8295` vs `0.8282`),
but not on raw image AUROC (`0.7958` vs `0.8119`).

Thesis role: location-kmeans is a useful late variant, not a sweeping win. It
shows that local metric choice and reconstruction/fusion complementarity matter,
but the raw anchor detector remains limited.

### 9. PatchCore Baseline

The final reality check is PatchCore with frozen DINOv3 features:

- `patchcore_dinov3_vitsmall_2`: image AUROC `0.8837`, pixel AUROC `0.9612`.

This is stronger than every adapted CAM-anchor variant in the local evidence.

Thesis role: PatchCore should be central, not hidden. It proves that DINOv3
features are useful on this dataset, while the CAM-anchor training/fusion
adaptation does not exploit them as effectively as a simple frozen-feature memory
bank.

## What Worked

- DINOv3 features are strong enough to support meaningful anomaly detection.
- K=1 global anchor scoring is a reliable simple baseline.
- Reproject/trainable K=1 variants reached strong historical trained results
  around `0.854-0.855` image AUROC.
- Tuned fusion can improve over anchor-only in specific cases, especially
  `dual_bottleneck_k1`.
- Patch mode is a real improvement over global multi-anchor scoring in moderate-K
  regimes.
- Location-aware patch banks can slightly improve late fused AUROC.
- Pixel-level AUROC is consistently high for reconstruction-heavy late variants.

## What Did Not Work Cleanly

- The adapted CAM objective did not beat PatchCore.
- Multi-anchor global clustering did not produce a clean monotonic K improvement.
- Good cluster diagnostics did not guarantee best anomaly AUROC.
- Stage-1 checkpoint selection remained fragile, especially for multi-anchor
  runs whose best checkpoint often appeared very early.
- Divergence was frequently weak, noisy, or anti-correlated.
- Fixed fusion weights were often hard to justify.
- Pixel AUROC is not enough to claim strong localization because BMAD warns that
  brain MRI background can inflate pixel-level scores.

## Main Scientific Explanation

The supervised CAM loss transfers only partially because its geometry assumes
real classes. In supervised retrieval, the repeller separates meaningful class
anchors. In one-class anomaly detection, the anchors are artificial partitions of
normal anatomy. Separating them can create an over-fragmented normal manifold and
make normal samples look anomalous if they fall between arbitrary anchor regions.

At the same time, reconstruction and divergence do not fully solve the problem.
Reconstruction helps pixel maps and can add a weak image-level correction, but it
is not a strong standalone image detector. Divergence is even less reliable
because its difference signal is not consistently tied to pathology.

Patch methods work better because they make the anchor idea local. But the
strongest baseline, PatchCore, suggests that storing and comparing frozen dense
DINOv3 patch features directly is still more effective than training the
CAM-style anchor/fusion stack.

## Recommended Thesis Claim

The thesis can safely claim:

> This work adapts a supervised class-anchor loss to one-class brain MRI anomaly
> detection and studies its behavior with DINOv3 features. The adapted method
> achieves meaningful detection and localization performance, with clean
> late-stage variants around `0.81` image AUROC, `0.83` fused AUROC, and
> `0.93-0.94` pixel AUROC, and older trainable variants reaching about `0.855`
> image AUROC. However, it does not outperform a frozen DINOv3 PatchCore
> baseline (`0.8837` image, `0.9612` pixel). The main contribution is therefore
> an empirical analysis of why CAM-style anchor geometry only partially transfers
> to one-class medical anomaly detection.

## What To Avoid Claiming

- Do not claim the method is SOTA.
- Do not claim the final method beats PatchCore.
- Do not claim the local test split is balanced.
- Do not imply every final run fully fine-tuned DINOv3.
- Do not overstate location-kmeans as a decisive improvement over patch mode.
- Do not treat pixel AUROC alone as proof of precise tumor localization.

## Suggested Chapter Logic

1. **Introduction:** motivate one-class brain MRI AD; state the CAM+DINOv3
   hypothesis; frame the work as empirical investigation.
2. **Background:** DINOv3 frozen dense features, feature-based AD, CAM loss,
   BMAD/BraTS, AUROC and PRO caveats.
3. **Method:** global anchor adaptation, two-stage reconstruction/fusion, patch
   mode, location-kmeans, PatchCore baseline.
4. **Experiments:** local split counts, evaluation metrics, implementation
   regimes, headline table.
5. **Results:** show PatchCore, older historical upper bound, clean redesign,
   patch, location-kmeans, and dual-bottleneck fusion.
6. **Analysis:** explain K regimes, repeller/fragmentation, projection drift,
   fusion fragility, patch locality, and baseline gap.
7. **Conclusion:** decent results, below baseline, useful negative/mixed finding,
   future work around frozen features, simpler local memory banks, tuned fusion,
   PRO, and better checkpoint selection.

## Final Thesis Tone

The strongest tone is not defensive. It is:

- clear about the original hypothesis;
- transparent about failed or fragile branches;
- precise about which numbers are clean versus historical;
- honest that the baseline wins;
- confident that explaining why the adaptation underperforms is a real research
  contribution.
