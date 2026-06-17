# Fusion Candidate Experiment Results

These are post-hoc exploratory fusions on existing test-set score CSVs. They are useful for candidate selection, but final fusion weights should be selected on validation data.

Grid step: `0.05`.

## Clean Candidate Metrics

| Candidate | Role | Image AUROC | Fused AUROC | Pixel AUROC | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `global_k1_clean` | Clean global one-anchor | 0.8057 | 0.7718 | 0.9332 | Primary CLS/image-level K=1 candidate. |
| `global_k1024_clean` | Clean global multi-anchor | 0.7685 | 0.8260 | 0.9195 | Primary global multi-anchor candidate. |
| `patch_k32_clean` | Clean patch multi-anchor | 0.8119 | 0.8282 | 0.9298 | Primary patch-level candidate. |
| `patch_loc_cos_k32` | Location-aware patch | 0.7958 | 0.8295 | 0.9350 | Secondary local patch-bank variant. |
| `dual_bottleneck_k1_hist` | Historical fused K=1 | 0.8338 | 0.8520 | 0.9210 | Historical upper-bound/reference; less cleanly comparable. |
| `patchcore_baseline` | PatchCore baseline | 0.8837 | n/a | 0.9612 | External frozen-feature baseline; not fused into CAM-anchor candidates. |

## Ablation Candidate Metrics

| Candidate | Role | Image AUROC | Fused AUROC | Pixel AUROC | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `global_k1_nofuser` | Global K=1 ablation | 0.8149 | 0.7830 | 0.9290 | Higher raw K=1 ablation; not the clean headline run. |

## Clean Fusion Results

| Experiment | Group | Kind | Fusion AUROC | Strongest Component AUROC | Delta | Best Weights |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `four_way_clean_global_patch_and_location` | one_anchor_multi_anchor_patch | secondary | 0.8589 | 0.8295 | +0.0293 | `global_k1_clean__image_score`=0.30, `global_k1024_clean__recomputed_fused`=0.35, `patch_k32_clean__recomputed_fused`=0.25, `patch_loc_cos_k32__recomputed_fused`=0.10 |
| `three_way_clean_global_patch` | one_anchor_multi_anchor_patch | primary | 0.8582 | 0.8282 | +0.0299 | `global_k1_clean__image_score`=0.35, `global_k1024_clean__recomputed_fused`=0.35, `patch_k32_clean__recomputed_fused`=0.30 |
| `three_way_clean_global_location_patch` | one_anchor_multi_anchor_patch | secondary | 0.8561 | 0.8295 | +0.0266 | `global_k1_clean__image_score`=0.40, `global_k1024_clean__recomputed_fused`=0.35, `patch_loc_cos_k32__recomputed_fused`=0.25 |
| `one_multi_clean` | one_anchor_plus_multi_anchor | primary | 0.8491 | 0.8260 | +0.0231 | `global_k1_clean__image_score`=0.55, `global_k1024_clean__recomputed_fused`=0.45 |
| `cls_patch_location_recomputed_fused` | image_cls_plus_patch | secondary | 0.8445 | 0.8295 | +0.0150 | `global_k1_clean__image_score`=0.50, `patch_loc_cos_k32__recomputed_fused`=0.50 |
| `cls_patch_clean_patch_recomputed_fused` | image_cls_plus_patch | primary | 0.8398 | 0.8282 | +0.0115 | `global_k1_clean__image_score`=0.45, `patch_k32_clean__recomputed_fused`=0.55 |
| `patch_k32_plus_location_recomputed_fused` | patch_plus_location_patch | secondary | 0.8385 | 0.8295 | +0.0090 | `patch_k32_clean__recomputed_fused`=0.50, `patch_loc_cos_k32__recomputed_fused`=0.50 |
| `cls_patch_location_image` | image_cls_plus_patch | secondary | 0.8282 | 0.8057 | +0.0225 | `global_k1_clean__image_score`=0.70, `patch_loc_cos_k32__image_score`=0.30 |
| `cls_patch_clean_image` | image_cls_plus_patch | primary | 0.8277 | 0.8119 | +0.0158 | `global_k1_clean__image_score`=0.55, `patch_k32_clean__image_score`=0.45 |

## Ablation Fusion Results

These include `nofuser_k1`, which is a useful architectural ablation but a messier comparison than the clean Stage2E70 family.

| Experiment | Group | Kind | Fusion AUROC | Strongest Component AUROC | Delta | Best Weights |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `three_way_ablation_global_patch` | one_anchor_multi_anchor_patch | ablation | 0.8608 | 0.8282 | +0.0326 | `global_k1_nofuser__image_score`=0.35, `global_k1024_clean__recomputed_fused`=0.35, `patch_k32_clean__recomputed_fused`=0.30 |
| `one_multi_ablation_nofuser` | one_anchor_plus_multi_anchor | ablation | 0.8536 | 0.8260 | +0.0276 | `global_k1_nofuser__image_score`=0.55, `global_k1024_clean__recomputed_fused`=0.45 |
| `cls_patch_ablation_nofuser_patch_image` | image_cls_plus_patch | ablation | 0.8318 | 0.8149 | +0.0169 | `global_k1_nofuser__image_score`=0.60, `patch_k32_clean__image_score`=0.40 |

## Notes

- Rows are aligned by lowercase-normalized image path and label.
- `recomputed_fused` uses anchor + best non-anticorrelated divergence + pixel aggregation, matching the evaluation policy.
- PatchCore is reported as a baseline only and is not included in the CAM-anchor fusion grids.
- `nofuser_k1` is separated as an ablation because it removes anchor-conditioned reconstruction and also comes from a slightly different run family.
