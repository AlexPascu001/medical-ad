# Experiment Inventory

Generated from `evaluation_metrics.json` files in the current worktree. This is
a thesis-facing ledger, not a replacement for the raw artifacts.

Scope:

- JSON metric files scanned: `312`
- Full-test metric files retained: `237`
- Curated thesis-relevant full-test files retained: `185`
- Full-test means `num_normal=640`, `num_anomaly=3075`, and final
  `evaluation` or `test_evaluation`, not `validation` or `evaluation_stage1`.

Dates below are filesystem modification dates of metric files. They are useful
for correlating docs and runs, but they are not guaranteed training start dates.

## Family Summary

| Family | Runs | First metric | Last metric | Best image AUROC | Best fused AUROC | Best pixel AUROC |
| --- | ---: | --- | --- | --- | --- | --- |
| `early_bmad_anchor` | 38 | 2025-10-29 | 2026-01-12 | `bmad_random_k8_l2_trial3` `0.8358` | n/a | `bmad_random_k8_l2` `0.8932` |
| `solution_a_and_expert` | 22 | 2026-01-19 | 2026-03-01 | `solution_a_decoupled_k1` `0.8084` | n/a | n/a |
| `reproject` | 28 | 2026-02-01 | 2026-03-29 | `reproject_k1_early_trainable_backbone_stage2_2` `0.8551` | n/a | `reproject_k1_early_trainable_backbone_stage2_3` `0.9204` |
| `regfix` | 6 | 2026-03-29 | 2026-03-30 | `regfix_k1` `0.8539` | n/a | `regfix_k512_stage2` `0.9088` |
| `dual_bottleneck` | 3 | 2026-03-02 | 2026-03-16 | `dual_bottleneck_k1_tight` `0.8540` | `dual_bottleneck_k1` `0.8520` | `dual_bottleneck_k1` `0.9210` |
| `seed_cmp` | 8 | 2026-03-30 | 2026-03-30 | `seed_cmp_stage1_s42_1` `0.8537` | n/a | `seed_cmp_stage2_s789_1` `0.9218` |
| `full_redesign` | 8 | 2026-04-19 | 2026-05-03 | `full_redesign_k1` `0.8053` | `full_k1` `0.7963` | `full_k16` `0.9218` |
| `full_redesign_stage2e70` | 16 | 2026-05-03 | 2026-06-03 | `full_redesign_stage2e70_k1` `0.8057` | `full_redesign_stage2e70_k1024` `0.8260` | `full_redesign_stage2e70_k1` `0.9332` |
| `patch_stage2e70` | 32 | 2026-05-04 | 2026-05-17 | `patch_stage2e70_k32` `0.8119` | `patch_stage2e70_k32` `0.8282` | `patch_stage2e70_vitbase_p256_k1` `0.9351` |
| `patch_location_kmeans` | 8 | 2026-05-31 | 2026-05-31 | `patch_location_kmeans_k128` `0.7940` | n/a | `patch_location_kmeans_k128` `0.9338` |
| `patch_location_kmeans_stage2match` | 4 | 2026-05-31 | 2026-05-31 | `patch_location_kmeans_stage2match_k16` `0.8073` | n/a | `patch_location_kmeans_stage2match_k4` `0.9394` |
| `patch_location_kmeans_stage2recon` | 4 | 2026-05-31 | 2026-06-01 | `patch_location_kmeans_stage2recon_k16` `0.8056` | `patch_location_kmeans_stage2recon_k16` `0.8182` | `patch_location_kmeans_stage2recon_k8` `0.9385` |
| `patch_location_kmeans_stage2recon_cosine` | 4 | 2026-06-01 | 2026-06-02 | `patch_location_kmeans_stage2recon_cosine_k32` `0.7958` | `patch_location_kmeans_stage2recon_cosine_k32` `0.8295` | `patch_location_kmeans_stage2recon_cosine_k4` `0.9397` |
| `patchcore` | 1 | 2026-06-03 | 2026-06-03 | `patchcore_dinov3_vitsmall_2` `0.8837` | n/a | `patchcore_dinov3_vitsmall_2` `0.9612` |
| `vitbase_comparison` | 3 | 2026-03-30 | 2026-03-30 | `vitbase_k1` `0.8491` | n/a | `vitbase_k512` `0.9207` |

## Top Image AUROC

| Rank | Run | Family | Image | Fused | Pixel |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `patchcore_dinov3_vitsmall_2` | `patchcore` | `0.8837` | n/a | `0.9612` |
| 2 | `reproject_k1_early_trainable_backbone_stage2_2` | `reproject` | `0.8551` | n/a | n/a |
| 3 | `dual_bottleneck_k1_tight` | `dual_bottleneck` | `0.8540` | `0.7519` | `0.9061` |
| 4 | `regfix_k1` | `regfix` | `0.8539` | n/a | n/a |
| 5 | `seed_cmp_stage1_s42_1` | `seed_cmp` | `0.8537` | n/a | n/a |
| 6 | `reproject_k1_early_trainable_backbone_stage2_3` | `reproject` | `0.8532` | n/a | `0.9204` |
| 7 | `vitbase_k1` | `vitbase_comparison` | `0.8491` | n/a | `0.9113` |
| 8 | `regfix_k1_stage2` | `regfix` | `0.8395` | n/a | `0.8982` |
| 9 | `bmad_random_k8_l2_trial3` | `early_bmad_anchor` | `0.8358` | n/a | `0.8742` |
| 10 | `seed_cmp_stage2_s42_1` | `seed_cmp` | `0.8348` | n/a | `0.9215` |
| 11 | `reproject_k1_early_trainable_backbone` | `reproject` | `0.8346` | n/a | n/a |
| 12 | `dual_bottleneck_k1` | `dual_bottleneck` | `0.8338` | `0.8520` | `0.9210` |

## Top Fused AUROC

| Rank | Run | Family | Image | Fused | Pixel |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `dual_bottleneck_k1` | `dual_bottleneck` | `0.8338` | `0.8520` | `0.9210` |
| 2 | `patch_location_kmeans_stage2recon_cosine_k32` | `patch_location_kmeans_stage2recon_cosine` | `0.7958` | `0.8295` | `0.9350` |
| 3 | `patch_stage2e70_k32` | `patch_stage2e70` | `0.8119` | `0.8282` | `0.9298` |
| 4 | `full_redesign_stage2e70_k1024` | `full_redesign_stage2e70` | `0.7685` | `0.8260` | `0.9195` |
| 5 | `full_redesign_stage2e70_vitbase_p512_k1024` | `full_redesign_stage2e70` | `0.7518` | `0.8188` | `0.9224` |
| 6 | `patch_location_kmeans_stage2recon_k16` | `patch_location_kmeans_stage2recon` | `0.8056` | `0.8182` | `0.9276` |
| 7 | `patch_stage2e70_k16` | `patch_stage2e70` | `0.7932` | `0.8173` | `0.9304` |
| 8 | `patch_stage2e70_vitbase_p384_k1` | `patch_stage2e70` | `0.7518` | `0.8165` | `0.9222` |
| 9 | `patch_stage2e70_k4` | `patch_stage2e70` | `0.7879` | `0.8131` | `0.9283` |
| 10 | `patch_location_kmeans_stage2recon_k8` | `patch_location_kmeans_stage2recon` | `0.7843` | `0.8107` | `0.9385` |

## Top Pixel AUROC

| Rank | Run | Family | Image | Fused | Pixel |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `patchcore_dinov3_vitsmall_2` | `patchcore` | `0.8837` | n/a | `0.9612` |
| 2 | `patch_location_kmeans_stage2recon_cosine_k4` | `patch_location_kmeans_stage2recon_cosine` | `0.7470` | `0.7892` | `0.9397` |
| 3 | `patch_location_kmeans_stage2match_k4` | `patch_location_kmeans_stage2match` | `0.7626` | n/a | `0.9394` |
| 4 | `patch_location_kmeans_stage2recon_k8` | `patch_location_kmeans_stage2recon` | `0.7843` | `0.8107` | `0.9385` |
| 5 | `patch_location_kmeans_stage2recon_cosine_k8` | `patch_location_kmeans_stage2recon_cosine` | `0.7535` | `0.8010` | `0.9384` |
| 6 | `patch_stage2e70_vitbase_p256_k1` | `patch_stage2e70` | `0.7877` | `0.7566` | `0.9351` |
| 7 | `patch_location_kmeans_stage2recon_cosine_k32` | `patch_location_kmeans_stage2recon_cosine` | `0.7958` | `0.8295` | `0.9350` |
| 8 | `patch_location_kmeans_stage2recon_k4` | `patch_location_kmeans_stage2recon` | `0.7626` | `0.8031` | `0.9346` |
| 9 | `patch_stage2e70_vitsmall_p64_k32` | `patch_stage2e70` | `0.7630` | `0.7900` | `0.9343` |
| 10 | `patch_stage2e70_k8` | `patch_stage2e70` | `0.7473` | `0.7144` | `0.9343` |

## Interpretation Notes

- PatchCore is the strongest available baseline and should be presented as a
  baseline reality check, not as part of the CAM-anchor method.
- Older `reproject`, `regfix`, `seed_cmp`, and `dual_bottleneck` runs contain the
  strongest trained-model image scores. They are historically important but need
  config/split verification before becoming headline thesis claims.
- Later `stage2e70`, patch, and location-kmeans runs are cleaner and better
  documented, but generally lower on raw image AUROC.
- Fused AUROC can improve over the anchor score, especially for
  `dual_bottleneck_k1` and late patch/location-kmeans variants, but it depends on
  weighting and normalization choices.
- Pixel AUROC is consistently high for later reconstruction-heavy variants, but
  should be interpreted cautiously and ideally paired with PRO because of the
  BMAD brain-MRI background caveat.
