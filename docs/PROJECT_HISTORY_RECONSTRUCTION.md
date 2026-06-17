# Project History Reconstruction

This document reconstructs the broad research history of `medical-ad` from the
current worktree, git history, recovered/deleted Markdown docs, and experiment
artifacts. It is intended as thesis scaffolding: not a polished chapter, but an
evidence-backed map of what was tried, what failed, what improved, and why.

## Evidence Used

- Local git history, read with `git -c safe.directory=... log/show`, including
  the local backup branch `backup/pre-packfix-20260408`.
- GitHub repository metadata and commit search for `AlexPascu001/medical-ad`.
  The GitHub connector found no PRs or issues, so the useful remote evidence is
  mainly the commit list, which matches local history.
- Current `docs/*.md` files and their filesystem timestamps.
- Deleted/moved docs recovered from git, especially the old `project/docs/*`
  files deleted in commit `2aca1e0`.
- Experiment folders under `experiments/`, `runs/`, `output/`,
  `project/experiments/`, and `project/runs/`, especially
  `evaluation_metrics.json`, `evaluation_image_scores.csv`, `config.yaml`, and
  training summaries where available.
- The companion ledger in `docs/EXPERIMENT_INVENTORY.md`, generated from the
  current JSON metrics after filtering to full-test evaluations.
- The companion chronology in `docs/CHRONOLOGICAL_EVIDENCE_TIMELINE.md`, which
  separates GitHub/main evidence from local backup-branch evidence.
- The thesis-facing synthesis in `docs/THESIS_RESEARCH_STORY.md`, which condenses
  the evidence into a chapter-level research narrative.

Important caveat: many committed run folders contain per-image CSVs but not
`evaluation_metrics.json`. The metric inventory below therefore emphasizes runs
with JSON metrics available in the current worktree. Some older runs can still be
analyzed from CSVs in a follow-up pass.

## Git Spine

The commit history is sparse but lines up well with research phases.

| Date | Commit | Message | Historical meaning |
| --- | --- | --- | --- |
| 2025-10-16 | `b057b33`, `c3282cb`, `3e41ee2` | first/init/todo | Initial project skeleton and early thesis TODOs. |
| 2025-11-04 | `2258eaf` | add full pipeline | First complete training/evaluation pipeline appears. |
| 2025-11-20 | `f074756` | add next phase | Anchor strategy and distance-metric phase: random/k-means/eigenface, L2/cosine, pixel AUROC fixes, first serious docs. |
| 2025-12-19 | `2aca1e0` | add learnable anchors, multiple trials etc | Learnable anchors, repeated trials, many early model artifacts; old `project/docs` removed later. |
| 2026-01-11 | `6977c8c` | update project | More anchor strategies, learnable/fixed variants, anchor debugging scripts. |
| 2026-01-22 | `08389db` | add new experiments | Projection pretraining, anchor collapse analysis, Solution A vs decoupled approach, expert configs. |
| 2026-02-16 to 2026-04-02 | `f13ab00`, `b7d3d3a`, `a7ad132`, `25d2111`, `4cfd1ae` local backup branch | decoder/two-stage branch, latest snapshot, large-file cleanup | Local evidence for decoder/two-stage training, reproject/decoupled sweeps, dual-bottleneck/regfix/seed comparisons, and later removal of large `.pt/.pth` artifacts. |
| 2026-04-08 | `2bc2974` branch, `9e18ed6` main | architecture/pipeline docs and experiment updates | Documentation consolidation after cleanup; main keeps docs and experiment updates without large artifacts. |
| 2026-05-06 | `97d41f1` | add patch mode | Patch-mode detector, full redesign family, stage2e70, retrospective docs and large committed results. |
| 2026-05-20 | `6ace823` | add analysis | Stage2E70 family analysis, backbone/projection variants, no-divergence sweep. |
| 2026-06-03 | `56fe1ef` | refactor project | Refactor into cleaner docs/test layout, PatchCore baseline, location-kmeans families, no-repeller configs. |

There is also a local branch `backup/pre-packfix-20260408` at `2bc2974`, which
preserves the point where `docs/ARCHITECTURE.md` and `docs/PIPELINE.md` were
first added before the April main-branch cleanup.

## Phase 1: Initial CAM-Anchor System

Approximate period: October to November 2025.

The first complete direction was the straightforward hypothesis:

1. Use DINO features for normal brain MRI slices.
2. Define anchors as normal prototypes.
3. Train normal samples to be close to anchors.
4. Use distance to nearest anchor as the anomaly score.

The early docs recovered from git show the first major problems:

- `ANALYSIS_REPORT.md` explicitly flags increasing training loss, severe class
  imbalance, ROC mismatch, inverted anomaly scoring, dense loss not being used,
  and missing validation loss tracking.
- `PIXEL_AUROC_FIX.md` documents a bug/shape issue in pixel metric computation.
- `DISTANCE_METRICS.md` shows that cosine versus L2 distance was an early axis of
  uncertainty, with special attention to whether anchors and embeddings were
  normalized consistently.
- `ANCHOR_EXPERIMENTS.md` lays out the early anchor strategy matrix:
  random, k-means, eigenface, and varying K.

This phase is important for the thesis because it shows the project did not
start with a polished method. It started with a plausible CAM transfer idea, then
hit implementation and measurement problems that had to be diagnosed before any
scientific conclusion was possible.

## Phase 2: Anchor Strategy, Repeller, and Collapse

Approximate period: November 2025 to January 2026.

Commit `f074756` introduced configs for:

- `random_baseline`
- `kmeans`
- `cosine_distance`
- `l2_distance`
- temporary random/k-means/eigenface sweeps

By January, `ANCHOR_COLLAPSE_ANALYSIS.md` appears. Its headings show the central
diagnosis at that time: one anchor dominated, performance could peak at epoch 0,
and the repeller loss was considered a high-priority fix. This is historically
interesting because the thesis-level interpretation later becomes more nuanced:
repeller can prevent collapse in a multi-anchor geometry, but in one-class
medical AD it can also fragment an artificial normal manifold.

This is the first major conceptual turn:

- Early engineering view: repeller may be necessary to stop anchor collapse.
- Later scientific view: repeller is inherited from supervised CAM and may be
  harmful when anchors are not real classes.

Both statements can be true in different failure modes, and the thesis should
present that tension honestly.

## Phase 3: Learnable Anchors and Repeated Trials

Approximate period: December 2025 to January 2026.

Commit `2aca1e0` added learnable anchors and many trials. The later refactor moved
these scripts under `project/test/legacy_learnable` and related legacy folders.

The main idea was to go beyond fixed anchors:

- fixed anchors: generated once from normal data, then used as reference points;
- learnable anchors: optimized jointly, with pseudo-label assignment and CAM-like
  attraction/repulsion;
- dynamic reassignment was explored as a way to avoid stale pseudo-labels.

The current docs in `docs/LEARNABLE_ANCHORS.md` preserve this branch, but it is no
longer the main successful path. For the thesis, this belongs in "attempted
variants" or an ablation/history subsection rather than the central method.

## Phase 4: Projection Pretraining and Solution A vs Expert Decoupling

Approximate period: January to February 2026.

Commit `08389db` added:

- `PRETRAIN_IMPLEMENTATION.md`
- `QUICKSTART_PRETRAIN.md`
- `SOLUTION_A_VS_EXPERT.md`
- pretraining configs and caches
- expert 100-epoch configs
- `solution_a_*` configs

The historical problem was temporal mismatch:

- anchors are generated in DINO semantic space;
- the projection head starts random;
- samples and anchors can drift through the same projection during training;
- fixed pseudo-labels may become stale when the encoder/projection changes.

Two broad solutions emerged:

- **Solution A / reproject anchors:** keep semantic anchors in 384D and project
  them through the current projection head each forward pass.
- **Expert decoupled approach:** use semantic anchors only for assignment, but
  train toward fixed geometric targets in projected space.

From the full-test JSON inventory, this era produced some strong image-level
scores:

- `solution_a_decoupled_k1`: image AUROC about `0.8084`.
- `reproject_k1_early_trainable_backbone_stage2_2`: image AUROC about `0.8551`.
- `regfix_k1`: image AUROC about `0.8539`.

These older results are stronger than the later global redesign image score, but
less cleanly documented and not necessarily the best thesis centerpiece unless
their exact configs and scoring conditions are revalidated.

## Phase 5: Dual Bottleneck and Fusion Debugging

Approximate period: March 2026.

The `dual_bottleneck_k1` line is one of the most important "worked, but only
after debugging" episodes.

`docs/PERFORMANCE_ANALYSIS.md` records the story:

- the pixel-aggregation signal was initially weak because raw top-5% reconstruction
  error mostly measured slice complexity/anatomy rather than pathology;
- pixel aggregation was accidentally gated by `return_maps`, so it could disappear
  from CSV/evaluation passes;
- weak divergence and pixel signals diluted the strong anchor signal under equal
  fusion weights;
- patch divergence was poorly aligned because a CLS-trained projection was applied
  patch-wise;
- self-normalized pixel aggregation helped by subtracting each image's baseline;
- fusion improved after selecting the better divergence signal and tuning weights.

Current full-test metrics:

| Run | Image AUROC | Fused AUROC | Pixel AUROC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `dual_bottleneck_k1` | `0.8338` | `0.8520` | `0.9210` | Best documented tuned fusion run; post-hoc fusion weights `0.72/0.16/0.12`. |
| `dual_bottleneck_k1_equal_weights` | `0.8233` | `0.7851` | `0.9138` | Shows equal weighting hurt. |
| `dual_bottleneck_k1_tight` | `0.8540` | lower/not primary | not primary | Strong image score, but needs closer validation before thesis use. |

This phase is a strong thesis story because it shows a rigorous negative/mixed
finding: the anchor score was the useful signal; reconstruction and divergence
could help, but only after careful normalization and conservative weighting.

## Phase 6: Redesign and Stage2E70

Approximate period: May 2026.

Commit `97d41f1` introduced the big redesign/patch result dump and
`REDESIGN_RETROSPECTIVE.md`. The redesign family tried to make the global anchor
pipeline cleaner:

- frozen DINOv3 in many current configs;
- K-means semantic centroids;
- fixed/capacitated pseudo-labels;
- Stage 2 with 70-epoch budget and reconstruction pixel maps;
- diagnostics for cluster usage, margin, entropy, ratio.

Current full-test landmarks:

| Family | Best image | Best fused | Main lesson |
| --- | --- | --- | --- |
| `full_redesign` | `full_redesign_k1` `0.8053` | `full_redesign_k256` `0.7957` | Simpler global redesign did not recover older `~0.85` image scores. |
| `full_redesign_stage2e70` | `full_redesign_stage2e70_k1` `0.8057` | `full_redesign_stage2e70_k1024` `0.8260` | K=1 best anchor/image; large K helps fusion through auxiliary signals. |
| no-repeller subset | `full_redesign_stage2e70_norepel_k16` image `0.7978`, fused `0.8118` | Suggests removing repeller helps some mid-K behavior but does not solve the ceiling alone. |

This is the phase that most directly supports the thesis-level analysis of K:

- K=1 is strong because there is no inter-anchor fragmentation.
- Mid-K and high-K have different behavior: more coverage can help Stage 2/fusion
  even if the raw anchor geometry is weaker.
- Pixel AUROC remains relatively high, but BMAD's black-background caveat means
  pixel AUROC should not be overclaimed without PRO.

## Phase 7: Patch Mode

Approximate period: May 2026.

Patch mode changed the problem from "one CLS token near a global anchor" to "dense
patch tokens matched against patch-level normal references." This is closer in
spirit to PatchCore and to the fact that tumors are local.

Current full-test landmarks:

| Run | Image AUROC | Fused AUROC | Pixel AUROC |
| --- | ---: | ---: | ---: |
| `patch_stage2e70_k32` | `0.8119` | `0.8282` | `0.9298` |
| `patch_stage2e70_k16` | `0.7932` | `0.8173` | `0.9304` |
| `patch_stage2e70_k4` | `0.7879` | `0.8131` | `0.9283` |

`docs/PATCH_RETROSPECTIVE.md` and `docs/STAGE2E70_FAMILY_ANALYSIS.md` explain the
mechanism:

- patch mode's best runs are not merely rescued by fusion; the local anchor score
  itself becomes competitive;
- original ViT-S / projection 128 remained strongest or close to strongest;
- ViT-B and projection changes altered training dynamics more than the final
  leaderboard;
- divergence often remained weak or was dropped.

For the thesis, patch mode is the natural "improved variant" after the global
CAM adaptation struggled: it preserves the anchor hypothesis but makes normality
local instead of purely global.

## Phase 8: Location-KMeans Patch Banks

Approximate period: May 31 to June 2, 2026.

This is the best-documented late-stage research branch, captured in
`docs/PATCH_LOCATION_KMEANS_FAMILY_ANALYSIS.md`.

Four subfamilies were tested:

1. `patch_location_kmeans`: same-location local centroid bank, Stage 1 only.
2. `patch_location_kmeans_stage2match`: stronger Stage-1 recipe, no Stage 2.
3. `patch_location_kmeans_stage2recon`: adds reconstruction Stage 2.
4. `patch_location_kmeans_stage2recon_cosine`: changes local matching from
   Euclidean to cosine inside the stage-2-capable recipe.

Full-test landmarks:

| Family | Best image | Best fused | Best pixel | Main lesson |
| --- | --- | --- | --- | --- |
| `patch_location_kmeans` | `k128` `0.7940` | n/a | `k128` `0.9338` | Original local-centroid idea saturates near `0.79` image. |
| `stage2match` | `k16` `0.8073` | n/a | `k4` `0.9394` | Stronger Stage 1 improves raw image score but can hurt local maps at higher K. |
| `stage2recon` | `k16` `0.8056` | `k16` `0.8182` | `k8` `0.9385` | Stage 2 helps fusion and pixel maps, not raw anchor score. |
| `stage2recon_cosine` | `k32` `0.7958` | `k32` `0.8295` | `k4` `0.9397` | Cosine hurts raw anchor score but helps high-K fusion at `k32`. |

The key scientific conclusion is subtle:

- location-kmeans did not clearly dominate the older patch baseline on raw image
  AUROC;
- it narrowly beat `patch_stage2e70_k32` on fused AUROC (`0.8295` vs `0.8282`);
- that improvement came from stage-2/fusion complementarity, not a universally
  better anchor detector.

## Phase 9: PatchCore Baseline

Approximate period: June 3, 2026.

Commit `56fe1ef` added `project/patchcore_baseline.py`,
`project/run_patchcore_baseline.py`, configs, and results.

Full-test PatchCore result:

| Run | Image AUROC | Pixel AUROC | Meaning |
| --- | ---: | ---: | --- |
| `patchcore_dinov3_vitsmall_2` | `0.8837` | `0.9612` | Training-free frozen DINOv3 patch memory bank beats the trained anchor variants. |

This should be a central honesty point in the thesis. The project did not beat
the benchmark-style feature baseline, but it did produce a meaningful empirical
study showing why adapting supervised CAM to one-class medical AD is difficult.

## Broad Performance Map

Filtering to full-test metrics (`640` normal, `3075` anomalous), the strongest
available JSON-backed landmarks are:

| Category | Best representative | Image AUROC | Fused AUROC | Pixel AUROC |
| --- | --- | ---: | ---: | ---: |
| Frozen DINOv3 PatchCore baseline | `patchcore_dinov3_vitsmall_2` | `0.8837` | n/a | `0.9612` |
| Older reproject/regfix line | `reproject_k1_early_trainable_backbone_stage2_2` / `regfix_k1` | `~0.855` | n/a | `~0.91-0.92` |
| Dual bottleneck tuned | `dual_bottleneck_k1` | `0.8338` | `0.8520` | `0.9210` |
| Global redesign | `full_redesign_stage2e70_k1` | `0.8057` | `0.7718` | `0.9332` |
| Global redesign large-K fusion | `full_redesign_stage2e70_k1024` | `0.7685` | `0.8260` | `0.9195` |
| Patch baseline | `patch_stage2e70_k32` | `0.8119` | `0.8282` | `0.9298` |
| Location-kmeans | `patch_location_kmeans_stage2recon_cosine_k32` | `0.7958` | `0.8295` | `0.9350` |

This supports the thesis framing the user requested:

- the hypothesis was reasonable;
- the project got decent results;
- the method did not beat a strong frozen-feature baseline;
- the most valuable contribution is the empirical and mechanistic explanation of
  why the CAM adaptation only partially transfers.

## Recommended Thesis Framing

The thesis should not present the work as a failed attempt to reach SOTA. It
should present it as an empirical investigation with a clear research arc:

1. **Hypothesis:** CAM-style anchors can model normal brain MRI structure on top
   of DINOv3 features; distance to anchors can become an anomaly score.
2. **Initial implementation:** fixed anchors and CAM-like attraction/repulsion;
   early implementation bugs and metric issues had to be fixed.
3. **Anchor geometry lessons:** anchor count, assignment, repeller, and projection
   drift matter as much as model capacity.
4. **Auxiliary signals:** reconstruction and divergence can add small corrections,
   but are fragile and require careful normalization/fusion.
5. **Locality turn:** patch-level anchors and location-specific centroid banks
   work better than purely global anchors because tumors are local.
6. **Baseline reality check:** frozen DINOv3 PatchCore remains stronger, proving
   that DINOv3 features are valuable but the CAM training objective/fusion stack
   does not fully exploit them.
7. **Main scientific takeaway:** supervised CAM's attractive geometry does not
   transfer cleanly to one-class AD because artificial normal clusters are not
   true classes; the repeller/assignment machinery can fragment or miscalibrate
   normality.

## Open Threads For Follow-Up

- Normalize duplicated output/evaluation folders and, where per-image CSVs are
  available, recompute AUROC as a spot-check against the JSON metrics.
- Verify whether the strongest older image scores (`reproject/regfix/dual tight`)
  used exactly the same split and scoring pipeline as the final PatchCore and
  patch/location-kmeans results.
- Decide whether `dual_bottleneck_k1` belongs as a main result or as a tuned
  post-hoc variant. It is strong, but the story is less clean than the later
  patch/location-kmeans families.
- Add PRO if possible, because BMAD warns that pixel AUROC can be inflated by
  black background in brain MRI.
- If new fusion experiments combine K=1 and larger-K runs, place them after the
  history above as a final "score fusion of complementary regimes" phase.
