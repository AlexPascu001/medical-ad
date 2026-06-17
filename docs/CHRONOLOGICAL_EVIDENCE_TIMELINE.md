# Chronological Evidence Timeline

This file correlates commits, documentation timestamps, and experiment metric
artifact dates. It is meant to answer the historical question: "what was I
working on, when, and what evidence supports that?"

## Evidence Types

| Evidence | What It Proves Well | Caveat |
| --- | --- | --- |
| GitHub/main commits | Remote-backed project milestones on `main`. | Commit messages are sparse and often bundle many changes. |
| Local backup branch commits | Work that existed locally around February-April 2026. | GitHub search did not surface this branch; treat as local evidence unless pushed elsewhere. |
| Current docs timestamps | When analysis notes were last written/copied into their current form. | Refactors can rewrite timestamps, especially on 2026-06-03. |
| Metric file timestamps | When `evaluation_metrics.json` artifacts were last written/copied. | Not necessarily training start time. |
| Deleted docs recovered from git | Early reasoning and bugs that no longer appear in current docs. | They may reflect an intermediate understanding rather than final interpretation. |

## Git Topology

The history is not perfectly linear. Current `main` has the clean June refactor,
while a local backup branch preserves a February-April side path:

```text
56fe1ef 2026-06-03 main/origin/main refactor project
6ace823 2026-05-20 add analysis
97d41f1 2026-05-06 add patch mode
9e18ed6 2026-04-08 add docs and experiment updates without artifacts
| 2bc2974 2026-04-08 backup/pre-packfix-20260408 add architecture and pipeline docs
| 4cfd1ae 2026-04-02 untrack remaining large model files
| 25d2111 2026-04-02 untrack large .pt/.pth files
| a7ad132 2026-04-02 remove large files from git
| b7d3d3a 2026-04-02 latest version
| f13ab00 2026-02-16 add decoder for 2-stage training
08389db 2026-01-22 add new experiments
6977c8c 2026-01-11 update project
2aca1e0 2025-12-19 add learnable anchors, multiple trials etc
f074756 2025-11-20 add next phase
2258eaf 2025-11-04 add full pipeline
3e41ee2 2025-10-16 add todo
c3282cb 2025-10-16 init project
b057b33 2025-10-16 first commit
```

GitHub commit search confirms the main-line milestones:

| Date | Commit | Remote URL |
| --- | --- | --- |
| 2025-10-16 | `3e41ee2` add todo | <https://github.com/AlexPascu001/medical-ad/commit/3e41ee279b8fb3c9364210754d621c1ac50dcf0f> |
| 2025-11-04 | `2258eaf` add full pipeline | <https://github.com/AlexPascu001/medical-ad/commit/2258eaf160be4fb35952a3b98b6e2183dd2e389d> |
| 2025-11-20 | `f074756` add next phase | <https://github.com/AlexPascu001/medical-ad/commit/f07475636ae8c7654fa5a0c96bd7fd3341b62d24> |
| 2025-12-19 | `2aca1e0` add learnable anchors, multiple trials etc | <https://github.com/AlexPascu001/medical-ad/commit/2aca1e0b2afc158c2ac48325432cb683127c6067> |
| 2026-01-22 | `08389db` add new experiments | <https://github.com/AlexPascu001/medical-ad/commit/08389db726aec76fa950973ad93d8ad55cc20a64> |
| 2026-04-08 | `9e18ed6` add docs and experiment updates without artifacts | <https://github.com/AlexPascu001/medical-ad/commit/9e18ed6adebf4690bed2405bf4c0d8d40873278e> |
| 2026-05-06 | `97d41f1` add patch mode | <https://github.com/AlexPascu001/medical-ad/commit/97d41f105c5fd72eaca954d287818145fc30ef22> |
| 2026-05-20 | `6ace823` add analysis | <https://github.com/AlexPascu001/medical-ad/commit/6ace82367f172cdbc8da34dba478a467a1d59c93> |
| 2026-06-03 | `56fe1ef` refactor project | <https://github.com/AlexPascu001/medical-ad/commit/56fe1efc991cf5d9fcf3f7151436125e10c9696f> |

GitHub search did not find a `backup` branch, so commits `f13ab00` through
`2bc2974` should be treated as local backup-branch evidence.

## Timeline

| Period | Commit / Docs / Metrics | What Was Happening | Confidence |
| --- | --- | --- | --- |
| 2025-10-16 | `b057b33`, `c3282cb`, `3e41ee2` | Project skeleton and thesis TODOs. The idea was still being scoped. | High for repository genesis; low for method details. |
| 2025-11-04 | `2258eaf` | First complete training/evaluation pipeline. This is the earliest point where the project looks like an end-to-end CAM-style anomaly detector. | High. |
| 2025-11-20 | `f074756`; recovered `project/docs/*` | First major documented research phase: anchor strategies, distance metrics, pixel AUROC fixes, architecture walkthroughs. | High. |
| 2025-10-29 to 2026-01-12 | `early_bmad_anchor` metric files | Random/k-means/eigenface anchor experiments, K sweeps, cosine/L2 variants, learnable/fixed variants. Best full-test image result in this family is `bmad_random_k8_l2_trial3` at `0.8358`. | High for metrics; medium for exact run order. |
| 2025-12-19 | `2aca1e0` | Learnable anchors and repeated trials. This is the main historical evidence that the project tried to move beyond fixed prototypes. | High. |
| 2026-01-11 to 2026-01-22 | `6977c8c`, `08389db`; docs written 2026-01-20 | Anchor collapse analysis, pretraining implementation, Solution A vs expert/decoupled reasoning. | High. |
| 2026-01-19 to 2026-03-01 | `solution_a_and_expert` metric files | Reprojecting anchors through the projection head versus decoupled geometric targets. `solution_a_decoupled_k1` reaches `0.8084` image AUROC. | High for metrics; medium for final interpretation. |
| 2026-02-01 | `LARGE_ANCHOR_EXPERIMENTS.md`, `CONFIG_FIX_SUMMARY.md` | Large-K sweeps were planned and config generation bugs were fixed. The hypothesis was that reproject may collapse at high K while decoupled targets may be more stable. | High. |
| 2026-02-01 to 2026-03-29 | `reproject` metric files | Reproject variants became some of the strongest trained-model image results, with `reproject_k1_early_trainable_backbone_stage2_2` at `0.8551`. | High for result; medium for comparability until configs/splits are rechecked. |
| 2026-02-16 | local `f13ab00` | Decoder and two-stage training branch added many `reproject`, `decoupled`, and `two_stage` configs/artifacts. This is the local evidence for the transition from pure anchor scoring toward reconstruction/fusion. | High locally; not remote-confirmed. |
| 2026-03-02 to 2026-03-16 | `dual_bottleneck` metric files; `PERFORMANCE_ANALYSIS.md` last written 2026-03-09 | Dual-bottleneck/fusion debugging. Key findings: raw reconstruction aggregation was weak, pixel aggregation was accidentally gated by `return_maps`, patch divergence was poorly aligned, and tuned fusion was needed. | Very high. |
| 2026-03-29 to 2026-03-30 | `regfix`, `seed_cmp`, `vitbase_comparison` metric files | Regression/fix validation, seed comparisons, and ViT-base comparison. These runs contain several `~0.85` image AUROC results but are less clean as thesis centerpieces than later families. | High for metrics; medium for thesis use. |
| 2026-04-02 | local `b7d3d3a` then `a7ad132`, `25d2111`, `4cfd1ae` | Local backup branch records a large "latest version" snapshot followed by model-weight cleanup. This explains why many configs/CSVs/JSONs remain while `.pt/.pth` files were removed from git. | High locally. |
| 2026-04-08 | `9e18ed6`; local `2bc2974` | Main absorbed docs and experiment updates without large artifacts; backup branch retained architecture/pipeline docs before the package/refactor cleanup. | High. |
| 2026-04-19 to 2026-05-03 | `full_redesign` metric files; `SCORING_ANALYSIS.md` | Start of cleaner redesign family. K=1 remains best for raw image AUROC, while multi-anchor variants start showing fusion/auxiliary-signal behavior. | High. |
| 2026-05-03 to 2026-06-03 | `full_redesign_stage2e70` metric files | Stage2E70 full redesign. Best raw image: `full_redesign_stage2e70_k1` at `0.8057`; best fused: `full_redesign_stage2e70_k1024` at `0.8260`. | High. |
| 2026-05-04 to 2026-05-17 | `patch_stage2e70` metric files; `PATCH_RETROSPECTIVE.md` 2026-05-06 | Patch mode turns the method local. Best patch result: `patch_stage2e70_k32` at `0.8119` image / `0.8282` fused. | High. |
| 2026-05-20 | `6ace823`; `STAGE2E70_FAMILY_ANALYSIS.md` | Comparative analysis of original stage2e70, ViT-small p64, and ViT-base p256/p384/p512. Main conclusion: bigger backbone/projection changes internal behavior more than final AUROC under fixed settings. | High. |
| 2026-05-31 to 2026-06-02 | location-kmeans metrics; `PATCH_LOCATION_KMEANS_FAMILY_ANALYSIS.md` | Same-location patch centroid banks, stage2match, stage2recon, and cosine matching. Best fused result: `patch_location_kmeans_stage2recon_cosine_k32` at `0.8295`. | Very high. |
| 2026-06-03 | `56fe1ef`; PatchCore metrics | Project refactor plus PatchCore baseline. PatchCore `patchcore_dinov3_vitsmall_2` reaches `0.8837` image / `0.9612` pixel and becomes the honest baseline ceiling. | Very high. |

## Deleted Docs That Matter

Recovered from `project/docs/*` in early commits:

| Deleted doc | Historical value |
| --- | --- |
| `ANALYSIS_REPORT.md` | Records early implementation/measurement failures: increasing loss, class imbalance, ROC mismatch, inverted anomaly scoring, dense loss not used, no validation tracking. |
| `ANCHOR_EXPERIMENTS.md` | Shows the initial anchor matrix: random, k-means, eigenface, K sweeps. |
| `DISTANCE_METRICS.md` | Records early uncertainty around cosine vs L2 and normalization of anchors/embeddings. |
| `PIXEL_AUROC_FIX.md` | Shows that pixel metrics had a concrete bug/fix history rather than being clean from the start. |
| `ARCHITECTURE_WALKTHROUGH.md`, `DIMENSIONS_CHEATSHEET.md` | Useful for reconstructing the early architecture and tensor-shape assumptions. |

## Thesis-Relevant Reading Of The Timeline

The broad historical arc is not "one method failed." It is:

1. A supervised CAM-style anchor loss was adapted to one-class brain MRI anomaly
   detection.
2. Early work was dominated by making the pipeline measurable and debugging
   anchor/pixel scoring.
3. Fixed, learnable, and large-K anchor systems were explored because normal
   brain slices do not provide real semantic classes.
4. Projection drift and anchor collapse forced a choice between reprojecting
   anchors through the head and decoupling semantic assignment from geometric
   targets.
5. Reconstruction and divergence were added because anchor distance alone missed
   local evidence, but fusion proved fragile.
6. Patch mode and location-kmeans were the locality turn: tumors are local, so
   global CLS anchoring is structurally limited.
7. PatchCore showed the baseline reality: frozen DINOv3 features are excellent,
   but the CAM-style training objective did not surpass a strong memory-bank
   method.

This is a defensible dissertation story because it connects the hypothesis,
failures, fixes, and final negative comparison into one empirical investigation.
