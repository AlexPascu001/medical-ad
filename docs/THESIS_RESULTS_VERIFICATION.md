# Thesis Results Verification

This note verifies which experiment numbers are safe to use in the thesis and
how comparable the likely headline runs are.

## Local Dataset Split

The current local BMAD/BraTS slice folder contains:

| Split | Normal path | Normal count | Anomaly path | Anomaly count |
| --- | --- | ---: | --- | ---: |
| Train | `data/BraTS2021_slice/train/good` | `7500` | n/a | `0` |
| Validation | `data/BraTS2021_slice/valid/good/img` | `39` | `data/BraTS2021_slice/valid/Ungood/img` | `44` |
| Test | `data/BraTS2021_slice/test/good/img` | `640` | `data/BraTS2021_slice/test/Ungood/img` | `3075` |

This matches the full-test `evaluation_metrics.json` files, which consistently
report `num_normal=640` and `num_anomaly=3075`.

Thesis implication: avoid describing the local test set as balanced. It has
`3715` images total, but the local class ratio is `640 / 3075`, not `1:1`.

## Headline Candidate Verification

All candidates below use the same local test split (`640/3075`) and the same
nominal input size (`240 x 240`). They differ substantially in training regime,
backbone freezing, anchor mode, and scoring/fusion.

| Candidate | Image AUROC | Fused AUROC | Pixel AUROC | Backbone | Frozen | Mode | K | Stage 2 | Normalization | Thesis role |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- | --- |
| `patchcore_dinov3_vitsmall_2` | `0.8837` | n/a | `0.9612` | DINOv3 ViT-S | yes | patch memory bank | n/a | no | `zscore_only` | Strong frozen-feature baseline; honest ceiling. |
| `reproject_k1_early_trainable_backbone_stage2_2` | `0.8551` | n/a | n/a | DINOv3 ViT-S | no | global anchor | `1` | yes | code default, effectively `zscore_only` | Strongest trained image score; historically important, but not as cleanly documented as late families. |
| `regfix_k1` | `0.8539` | n/a | n/a | DINOv3 ViT-S | no | global anchor | `1` | no | `zscore_only` | Confirms older trainable K=1/regfix line was strong. |
| `dual_bottleneck_k1_tight` | `0.8540` | `0.7519` | `0.9061` | DINOv3 ViT-S | no | global anchor | `1` | yes | code default, effectively `zscore_only` | Strong raw image score, weak fused score; not a main final method. |
| `dual_bottleneck_k1` | `0.8338` | `0.8520` | `0.9210` | DINOv3 ViT-S | no | global anchor | `1` | yes | code default, effectively `zscore_only` | Best tuned fused trained score; good debugging/fusion case study. |
| `full_redesign_stage2e70_k1` | `0.8057` | `0.7718` | `0.9332` | DINOv3 ViT-S | yes | global anchor | `1` | yes | `zscore_only` | Clean late global-anchor K=1 result. |
| `full_redesign_stage2e70_k1024` | `0.7685` | `0.8260` | `0.9195` | DINOv3 ViT-S | yes | global anchor | `1024` | yes | `zscore_only` | Shows large-K can improve fusion despite weak raw anchor score. |
| `patch_stage2e70_k32` | `0.8119` | `0.8282` | `0.9298` | DINOv3 ViT-S | yes | patch anchor | `32` | yes | `zscore_only` | Best clean patch-mode result. |
| `patch_location_kmeans_stage2recon_cosine_k32` | `0.7958` | `0.8295` | `0.9350` | DINOv3 ViT-S | yes | location patch anchor | `32` | yes | `zscore_only` | Best late location-kmeans fused result; narrowly beats patch-mode fused score. |

## Comparability Conclusions

### Safe To Headline

Use these as the main thesis comparison:

| Role | Run | Why |
| --- | --- | --- |
| Frozen-feature baseline | `patchcore_dinov3_vitsmall_2` | Strongest result, training-free, same local test split. |
| Best clean global CAM-anchor result | `full_redesign_stage2e70_k1` and `full_redesign_stage2e70_k1024` | Shows K=1 raw-image regime versus large-K fusion regime. |
| Best clean patch CAM-anchor result | `patch_stage2e70_k32` | Best documented patch detector by both image and fused AUROC. |
| Best late local-centroid variant | `patch_location_kmeans_stage2recon_cosine_k32` | Best late fused AUROC, but only narrowly above patch-mode fusion. |
| Best fused trained historical result | `dual_bottleneck_k1` | Valuable as a fusion/debugging case study, not necessarily the clean final method. |

### Use As Historical Evidence, Not Main Final Method

The older `reproject`, `regfix`, `dual_bottleneck_tight`, `seed_cmp`, and
`vitbase_comparison` runs are comparable at the level of split size and AUROC
calculation, but they differ from the cleaner late families in important ways:

- many use `freeze_backbone: false`;
- several are K=1 global-anchor runs rather than patch/local variants;
- some have no fused or pixel metrics saved;
- some omit explicit `data.normalization`, relying on the code default
  `zscore_only`;
- the exact checkpoint/model artifacts were affected by the April large-file
  cleanup, so rerunning exact checkpoint-level analysis may not be possible from
  the current repository alone.

The strongest older image numbers should therefore be presented as historical
evidence that the anchor idea could reach the mid-`0.85` AUROC range, while the
main final comparison should use the cleaner late-stage families and PatchCore.

## Recommended Main Results Table

For the dissertation body, a compact table like this is defensible:

| Method / Variant | Image AUROC | Fused AUROC | Pixel AUROC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| PatchCore + frozen DINOv3 ViT-S | `0.8837` | n/a | `0.9612` | Strong baseline; method to beat. |
| Historical trainable reproject K=1 | `0.8551` | n/a | n/a | Best older trained image score, kept as historical upper bound. |
| Dual bottleneck K=1, tuned fusion | `0.8338` | `0.8520` | `0.9210` | Best fused trained score, but fusion/debugging-heavy. |
| Clean global Stage2E70 K=1 | `0.8057` | `0.7718` | `0.9332` | Best clean global raw-image run. |
| Clean global Stage2E70 K=1024 | `0.7685` | `0.8260` | `0.9195` | Large-K improves fusion, not raw anchor score. |
| Patch Stage2E70 K=32 | `0.8119` | `0.8282` | `0.9298` | Best clean patch-anchor run. |
| Location-KMeans recon cosine K=32 | `0.7958` | `0.8295` | `0.9350` | Best late local-centroid fused run, narrowly above patch baseline. |

This table supports the honest thesis conclusion: the adapted CAM-anchor family
reaches decent results, but does not beat a strong frozen-feature PatchCore
baseline on the same local test split.

## Claims To Update In Thesis Draft

- Replace any statement that the local test set is balanced with the actual local
  counts: `640` normal and `3075` anomalous test images.
- Avoid saying the final method fully fine-tunes DINOv3. Some older strong runs
  do, but the clean late families and PatchCore use frozen DINOv3 features.
- Phrase the `~0.80 / ~0.92` result as the clean late-method regime, while also
  acknowledging the stronger historical trained runs around `0.85` image AUROC.
- Keep the PatchCore comparison central: it is the strongest local baseline and
  the clearest evidence for the final negative/mixed result.
