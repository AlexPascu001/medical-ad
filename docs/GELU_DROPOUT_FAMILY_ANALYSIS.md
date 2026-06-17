# GELU/Dropout Family Analysis

Scope:

- `gelu_dropout_global_*`
- `gelu_dropout_patch_location_*`

This note analyzes the GELU/dropout experiment sweep, which adds an explicit projection-head change on top of the late stage2e70-style recipe:

- `projection_activation: gelu`
- `projection_dropout: 0.2`
- `projection_dim: 128`
- stage-2 reconstruction and score fusion enabled

The sweep was intended to cross three axes:

- anchor family: global CLS/redesign vs patch `location_kmeans`
- training policy: frozen backbone with timm transforms vs trainable backbone with legacy transforms
- K: `1`, `64`, `1024`

The practical questions are:

- did GELU/dropout improve the late-stage CAM-anchor recipes?
- is the better setting frozen+timm or finetune+legacy?
- does large K still help once the projection head is regularized?
- did the patch-location branch remain competitive with the stronger patch/location-kmeans family?

The conclusions here are based on:

- `project/configs/gelu_dropout_*.yaml`
- `run_gelu_dropout_sweep.ps1`
- `runs/gelu_dropout_*/.../evaluation/evaluation_metrics.json`
- `runs/gelu_dropout_*/.../evaluation_stage1/evaluation_metrics.json`
- `runs/gelu_dropout_*/.../training_summary.json`
- `runs/gelu_dropout_sweep_timings.json`
- `runs/gelu_dropout_sweep_logs/gelu_dropout_patch_location_k1024_frozen_timm.log`
- `runs/gelu_dropout_sweep_logs/gelu_dropout_patch_location_k1024_finetune.log`
- comparison baselines in `docs/STAGE2E70_FAMILY_ANALYSIS.md` and `docs/PATCH_LOCATION_KMEANS_FAMILY_ANALYSIS.md`

## 1. Experiment Matrix

There were `12` planned GELU/dropout runs.

| Bucket | Planned Runs | Completed Final Evaluations | K Values | Backbone Policy | Anchor Mode | Stage 2 |
| --- | ---: | ---: | --- | --- | --- | --- |
| Global frozen+timm | 3 | 3 | `1, 64, 1024` | frozen backbone, timm transforms | global CLS/redesign | enabled |
| Global finetune+legacy | 3 | 3 | `1, 64, 1024` | trainable backbone, legacy transforms | global CLS/redesign | enabled |
| Patch-location frozen+timm | 3 | 2 | `1, 64, 1024` | frozen backbone, timm transforms | patch/location_kmeans | enabled |
| Patch-location finetune+legacy | 3 | 2 | `1, 64, 1024` | trainable backbone, legacy transforms | patch/location_kmeans | enabled |

The two missing final evaluations are both patch-location `K=1024` runs.

- `gelu_dropout_patch_location_k1024_frozen_timm` started, built the local centroid bank, and failed during epoch `0` with CUDA OOM.
- `gelu_dropout_patch_location_k1024_finetune` was attempted manually after the sweep stop and failed the same way.

So this is a complete global sweep, but only a partial patch-location sweep.

## 2. What Changed

The family keeps the late stage2e70-style training envelope:

- DINOv3 `vit_small_patch16_dinov3.lvd1689m`
- `projection_dim: 128`
- `loss.beta: 0.5`
- `loss.delta: 0.0`
- stage-2 reconstruction enabled
- stage-2 frozen bottleneck enabled
- pixel map type `reconstruction_l2`
- pixel aggregation `top_k_percentile` at percentile `95`
- fixed score fusion weights `0.4 / 0.3 / 0.3`

The new explicit projection-head settings are:

| Setting | GELU/Dropout Value | Why It Matters |
| --- | --- | --- |
| `model.projection_activation` | `gelu` | changes the nonlinearity inside the projection head |
| `model.projection_dropout` | `0.2` | regularizes the learned projection space |
| `model.freeze_backbone` | swept | isolates frozen projection-head training from full backbone finetuning |
| `data.use_timm_transforms` | swept with freeze policy | tests timm eval-style preprocessing against legacy transforms |

The frozen+timm and finetune+legacy axes are coupled in these configs. That means the sweep cannot cleanly isolate whether a difference comes from freezing, transform choice, or their interaction. It can only compare the two complete recipes.

## 3. How The Two Branches Work

### 3.1 Global Branch

The global branch uses centroid anchors in CLS/global embedding space.

- `anchor.strategy: kmeans`
- `anchor.representation: centroids`
- `anchor.use_embedding_space: true`
- `anchor.reproject_anchors: true`
- `training.fixed_pseudo_labels: true`
- `training.pseudo_label_assignment: capacitated`
- `stage2.alignment_target: anchor`

This is closest to the redesign/global side of the stage2e70 family.

### 3.2 Patch-Location Branch

The patch branch uses same-location local centroid banks.

- `anchor.mode: patch`
- `anchor.patch.variant: location_kmeans`
- `anchor.representation: centroids`
- `anchor.patch.local_score_reduction: percentile`
- `anchor.patch.local_score_percentile: 95`
- `anchor.patch.local_distance_metric: euclidean`
- `training.fixed_pseudo_labels: false`
- `stage2.alignment_target: local_anchor_pool`

This is closest to the location-kmeans stage2recon family, except this sweep uses the GELU/dropout projection head and tests the frozen+timm vs finetune+legacy policy directly.

## 4. Best-Run Summary

| Objective | Winner | Image AUROC | Fused AUROC | Pixel AUROC | Comment |
| --- | --- | ---: | ---: | ---: | --- |
| Best GELU/dropout image AUROC | `gelu_dropout_patch_location_k1_frozen_timm` | `0.8429` | `0.7469` | `0.8898` | strongest raw detector in this family |
| Best GELU/dropout fused AUROC | `gelu_dropout_global_k1_finetune` | `0.7974` | `0.7724` | `0.9193` | best stored fusion, but below older fused leaders |
| Best GELU/dropout pixel AUROC | `gelu_dropout_global_k64_finetune` | `0.6860` | `0.7609` | `0.9257` | strong pixel map but weak anchor score |
| Best frozen+timm global image AUROC | `gelu_dropout_global_k1_frozen_timm_2` | `0.7966` | `0.6956` | `0.8926` | raw image close to finetune `k1`, fusion poor |
| Best finetune global image AUROC | `gelu_dropout_global_k1_finetune` | `0.7974` | `0.7724` | `0.9193` | best global result |
| Best frozen+timm patch image AUROC | `gelu_dropout_patch_location_k1_frozen_timm` | `0.8429` | `0.7469` | `0.8898` | large raw gain, weak fusion |
| Best finetune patch image AUROC | `gelu_dropout_patch_location_k1_finetune` | `0.7179` | `0.6884` | `0.9214` | much weaker raw detector |

The headline result is:

- GELU/dropout produced one very strong raw image detector: `gelu_dropout_patch_location_k1_frozen_timm` at `0.8429`
- the same run did not produce a strong fused result
- the best fused result in the family is only `0.7724`
- the global branch becomes worse as K increases
- the patch branch strongly favors frozen+timm over finetune+legacy at both completed K values
- patch-location `K=1024` did not complete because of GPU memory pressure

So this sweep improved one raw image score, but it did not improve the thesis-facing fused leaderboard.

## 5. Evaluation Results

### 5.1 Completed Runs

| Run | Image AUROC | Fused AUROC | Pixel AUROC | Reconstruction AUROC | Divergence AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gelu_dropout_global_k1_finetune` | `0.7974` | `0.7724` | `0.9193` | `0.7195` | `0.5104` |
| `gelu_dropout_global_k1_frozen_timm_2` | `0.7966` | `0.6956` | `0.8926` | `0.6746` | `0.3807` |
| `gelu_dropout_global_k64_finetune` | `0.6860` | `0.7609` | `0.9257` | `0.7314` | `0.6321` |
| `gelu_dropout_global_k64_frozen_timm` | `0.7474` | `0.7125` | `0.8858` | `0.7053` | `0.7015` |
| `gelu_dropout_global_k1024_finetune` | `0.7057` | `0.7445` | `0.9135` | `0.6055` | `0.6198` |
| `gelu_dropout_global_k1024_frozen_timm` | `0.7770` | `0.7249` | `0.8797` | `0.7088` | `0.6869` |
| `gelu_dropout_patch_location_k1_finetune` | `0.7179` | `0.6884` | `0.9214` | `0.6078` | `0.6387` |
| `gelu_dropout_patch_location_k1_frozen_timm` | `0.8429` | `0.7469` | `0.8898` | `0.6810` | `0.4908` |
| `gelu_dropout_patch_location_k64_finetune` | `0.5870` | `0.6886` | `0.9253` | `0.6225` | `0.4438` |
| `gelu_dropout_patch_location_k64_frozen_timm` | `0.8391` | `0.7657` | `0.8916` | `0.7368` | `0.2445` |

### 5.2 Global Branch

| K | Frozen Image | Finetune Image | Frozen Fused | Finetune Fused | Frozen Pixel | Finetune Pixel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `0.7966` | `0.7974` | `0.6956` | `0.7724` | `0.8926` | `0.9193` |
| `64` | `0.7474` | `0.6860` | `0.7125` | `0.7609` | `0.8858` | `0.9257` |
| `1024` | `0.7770` | `0.7057` | `0.7249` | `0.7445` | `0.8797` | `0.9135` |

The global branch is mixed.

- raw image AUROC is essentially tied at `K=1`
- frozen+timm has better raw image AUROC at `K=64` and `K=1024`
- finetune+legacy has better fused and pixel AUROC at every K
- increasing K hurts the finetune raw image score badly
- increasing K also does not improve frozen+timm beyond the `K=1` raw score

So the global branch did not recover the old large-K redesign pattern where `K=1024` gave the best fused score. Under GELU/dropout, global `K=1` finetune is the most useful completed global run.

### 5.3 Patch-Location Branch

| K | Frozen Image | Finetune Image | Frozen Fused | Finetune Fused | Frozen Pixel | Finetune Pixel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `0.8429` | `0.7179` | `0.7469` | `0.6884` | `0.8898` | `0.9214` |
| `64` | `0.8391` | `0.5870` | `0.7657` | `0.6886` | `0.8916` | `0.9253` |
| `1024` | failed OOM | failed OOM | failed OOM | failed OOM | failed OOM | failed OOM |

The patch-location branch has the clearest result in the sweep.

- frozen+timm strongly beats finetune+legacy on raw image AUROC
- the frozen advantage is `+0.1249` at `K=1`
- the frozen advantage is `+0.2521` at `K=64`
- finetune+legacy gives higher pixel AUROC, but this does not translate into better image-level detection
- `K=64` barely changes frozen raw image AUROC relative to `K=1`

This is the main practical finding: for GELU/dropout patch-location, the trainable-backbone recipe is actively harmful to the anchor detector, while frozen+timm is strong and stable at the completed K values.

### 5.4 Fusion Behavior

The stored fused score often underperforms the raw anchor/image score.

| Run | Image AUROC | Fused AUROC | Fused Delta |
| --- | ---: | ---: | ---: |
| `gelu_dropout_patch_location_k1_frozen_timm` | `0.8429` | `0.7469` | `-0.0959` |
| `gelu_dropout_patch_location_k64_frozen_timm` | `0.8391` | `0.7657` | `-0.0734` |
| `gelu_dropout_global_k1_frozen_timm_2` | `0.7966` | `0.6956` | `-0.1010` |
| `gelu_dropout_global_k1024_frozen_timm` | `0.7770` | `0.7249` | `-0.0521` |
| `gelu_dropout_global_k1_finetune` | `0.7974` | `0.7724` | `-0.0250` |
| `gelu_dropout_global_k64_finetune` | `0.6860` | `0.7609` | `+0.0749` |
| `gelu_dropout_patch_location_k64_finetune` | `0.5870` | `0.6886` | `+0.1016` |

This is the same broad warning seen in other stage-2 families, but more severe for the strongest GELU/dropout raw runs.

- fusion can rescue weak anchor runs when the pixel/reconstruction/divergence branches are better aligned
- fusion can damage strong anchor runs when auxiliary signals are weaker or misweighted
- the fixed `0.4 / 0.3 / 0.3` fusion rule is not appropriate for all runs

For the best raw run, fusion is not a value add. It hides the most important result.

## 6. Training Behavior

### 6.1 Stage-1 Validation vs Test

| Run | Best Val Image | Best Epoch | Actual Epochs | Test Image | Val-Test Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gelu_dropout_global_k1_finetune` | `0.9071` | `34` | `44` | `0.7974` | `0.1096` |
| `gelu_dropout_global_k1_frozen_timm_2` | `0.8240` | `98` | `100` | `0.7966` | `0.0274` |
| `gelu_dropout_global_k64_finetune` | `0.8304` | `5` | `20` | `0.6860` | `0.1444` |
| `gelu_dropout_global_k64_frozen_timm` | `0.7552` | `1` | `20` | `0.7474` | `0.0078` |
| `gelu_dropout_global_k1024_finetune` | `0.8368` | `5` | `20` | `0.7057` | `0.1311` |
| `gelu_dropout_global_k1024_frozen_timm` | `0.8019` | `1` | `20` | `0.7770` | `0.0249` |
| `gelu_dropout_patch_location_k1_finetune` | `0.8566` | `42` | `52` | `0.7179` | `0.1387` |
| `gelu_dropout_patch_location_k1_frozen_timm` | `0.8642` | `13` | `23` | `0.8429` | `0.0214` |
| `gelu_dropout_patch_location_k64_finetune` | `0.7652` | `25` | `35` | `0.5870` | `0.1782` |
| `gelu_dropout_patch_location_k64_frozen_timm` | `0.8800` | `67` | `77` | `0.8391` | `0.0408` |

The training behavior is one of the clearest diagnostics.

- finetune+legacy runs have large validation-to-test gaps
- frozen+timm runs have much smaller validation-to-test gaps
- the strongest raw run, `patch_location_k1_frozen_timm`, has both high validation AUROC and a small test gap
- `patch_location_k64_frozen_timm` trains much longer and also transfers well

This strongly suggests that the finetune+legacy recipe is overfitting or miscalibrating the projection space under GELU/dropout, especially for patch-location.

### 6.2 Runtime And Failure

The recorded sweep runtime was `1238.52` minutes before failure.

| Run | Status | Runtime (min) |
| --- | --- | ---: |
| `gelu_dropout_global_k1_frozen_timm` | completed | `232.66` |
| `gelu_dropout_global_k1_finetune` | completed | `107.56` |
| `gelu_dropout_patch_location_k1_frozen_timm` | completed | `75.55` |
| `gelu_dropout_patch_location_k1_finetune` | completed | `109.33` |
| `gelu_dropout_global_k64_frozen_timm` | completed | `68.02` |
| `gelu_dropout_global_k64_finetune` | completed | `73.65` |
| `gelu_dropout_patch_location_k64_frozen_timm` | completed | `160.54` |
| `gelu_dropout_patch_location_k64_finetune` | completed | `99.49` |
| `gelu_dropout_global_k1024_frozen_timm` | completed | `68.66` |
| `gelu_dropout_global_k1024_finetune` | completed | `58.19` |
| `gelu_dropout_patch_location_k1024_frozen_timm` | failed | `184.88` |
| `gelu_dropout_patch_location_k1024_finetune` | failed | `459.27` |

Both failed runs built:

- local centroid bank: `(1024, 15, 15, 384)`
- summary anchors: `(1024, 384)`

It then failed during the first training epoch:

- error: CUDA out of memory
- attempted allocation: `7.03 GiB`
- GPU capacity: `15.92 GiB`

That failure is methodologically important. Patch-location `K=1024` is not just "missing"; under the current batch size and implementation, both frozen and finetune variants are beyond available memory.

## 7. Comparison To Existing Baselines

| Run | Image AUROC | Fused AUROC | Pixel AUROC | Comment |
| --- | ---: | ---: | ---: | --- |
| `gelu_dropout_patch_location_k1_frozen_timm` | `0.8429` | `0.7469` | `0.8898` | best GELU/dropout raw run |
| `gelu_dropout_patch_location_k64_frozen_timm` | `0.8391` | `0.7657` | `0.8916` | nearly same raw image score |
| `gelu_dropout_global_k1_finetune` | `0.7974` | `0.7724` | `0.9193` | best GELU/dropout fused run |
| `patch_location_kmeans_stage2recon_cosine_k32` | `0.7958` | `0.8295` | `0.9350` | best location-kmeans fused baseline |
| `patch_stage2e70_k32` | `0.8119` | `0.8282` | `0.9298` | best original patch stage2e70 baseline |
| `full_redesign_stage2e70_k1024` | `0.7685` | `0.8260` | `0.9195` | best redesign fused baseline |
| `patchcore_dinov3_vitsmall_2` | `0.8837` | n/a | `0.9612` | strongest non-CAM-anchor baseline |

The comparison is nuanced.

1. `gelu_dropout_patch_location_k1_frozen_timm` is a strong raw image detector. It beats the late patch/location-kmeans and stage2e70 raw image baselines listed here.
2. It does not beat the strongest historical raw-image or PatchCore results in `docs/EXPERIMENT_INVENTORY.md`.
3. It does not produce a competitive fused score.
4. Pixel AUROC is weaker than the late reconstruction-heavy patch/location baselines.

So the GELU/dropout sweep gives a potentially useful raw anchor baseline, not a new best overall fused detector.

## 8. What Happened, And Why

### 8.1 GELU/Dropout Helped The Frozen Patch Anchor Path

The strongest result is not global and not large-K. It is:

- patch-location
- `K=1`
- frozen backbone
- timm transforms
- GELU/dropout projection head

That result reached `0.8429` image AUROC with a small validation-to-test gap. `K=64` stayed close at `0.8391`.

This suggests the projection-head regularization can make the frozen local-anchor detector more robust, but it does not need many anchors to do so.

### 8.2 Finetuning Was Usually Harmful For Image AUROC

The finetune+legacy recipe was especially poor in patch-location mode.

- `K=1`: `0.8429 -> 0.7179` when moving from frozen+timm to finetune+legacy
- `K=64`: `0.8391 -> 0.5870`

The validation-test gaps support the interpretation that this is not just undertraining. Finetune runs often looked much better on validation than on test.

### 8.3 Pixel AUROC And Image AUROC Split Apart

Finetune+legacy often had higher pixel AUROC but lower image AUROC.

That means the reconstruction maps remained locally useful, but the image-level anchor ranking became worse. This is another reminder that pixel AUROC is not a substitute for image-level detection quality in this project.

### 8.4 Fixed Fusion Weights Are A Problem Here

The best raw runs are damaged by stored fusion.

For `gelu_dropout_patch_location_k1_frozen_timm`, fused AUROC is almost `0.096` below raw image AUROC. For `gelu_dropout_patch_location_k64_frozen_timm`, it is about `0.073` lower.

That means the current fusion rule should not be used uncritically for this family. If this family is cited, the raw image score and fused score need to be reported separately.

### 8.5 Large K Did Not Help In The Completed Runs

In the global branch:

- `K=1` is best for finetune raw image and fused AUROC
- `K=1` is also effectively best for frozen raw image
- `K=1024` does not recover a large-K advantage

In the patch branch:

- `K=64` does not improve raw image AUROC over `K=1`
- `K=1024` fails on memory before a result exists

So GELU/dropout does not currently support a "larger K is better" conclusion.

## 9. Practical Verdict

The GELU/dropout family is valuable, but for a narrower reason than originally hoped.

### 9.1 What It Proves

1. A GELU/dropout projection head can produce a strong frozen patch-location raw detector.
2. Frozen+timm is much safer than finetune+legacy for patch-location image AUROC.
3. Finetuning with this recipe creates large validation-test gaps.
4. The fixed stage-2 fusion rule can hide strong raw anchor performance.
5. Patch-location `K=1024` needs memory-oriented changes before it can be evaluated.

### 9.2 What It Does Not Prove

1. It does not prove GELU/dropout is a universal improvement.
2. It does not produce a new best fused CAM-anchor detector.
3. It does not show that large K helps.
4. It does not give a completed patch-location `K=1024` comparison.
5. It does not isolate freezing from transform choice, because those settings are coupled.

## 10. Recommendations

### 10.1 Keep As A Raw-Image Baseline

Keep `gelu_dropout_patch_location_k1_frozen_timm` as a serious raw image AUROC baseline:

- image AUROC: `0.8429`
- fused AUROC: `0.7469`
- pixel AUROC: `0.8898`

It is not the best overall method, but it is one of the cleaner late-family raw anchor results.

### 10.2 Do Not Use Stored Fusion As The Main Claim

For this family, stored fusion is often misleading. Any thesis table should include raw image AUROC alongside fused AUROC.

The most honest presentation is:

- GELU/dropout improves a frozen patch-location anchor score
- the current stage-2 fusion recipe is not calibrated for that improved anchor score

### 10.3 Rerun Only Focused Follow-Ups

The most useful follow-ups are small and targeted:

1. Retune fusion weights for `gelu_dropout_patch_location_k1_frozen_timm` and `k64_frozen_timm`, especially reducing or removing divergence.
2. Decouple the two recipe axes:
   - frozen + legacy transforms
   - frozen + timm transforms
   - finetune + legacy transforms
   - finetune + timm transforms
3. Retry patch-location high K only after lowering memory pressure:
   - smaller batch size
   - gradient accumulation
   - fewer local centroids, such as `K=128` or `K=256`
   - chunked local distance computation
4. Search the lower K region first:
   - `K=1`, `K=4`, `K=16`, `K=32`, `K=64`

### 10.4 Avoid Broad Claims About Finetuning

The sweep shows that this finetune+legacy recipe is harmful here. It does not prove that all finetuning is harmful.

The right claim is narrower:

- under GELU/dropout, with these transforms and training settings, finetuning generalizes poorly compared with frozen+timm, especially for patch-location.

## 11. Bottom Line

The GELU/dropout sweep produced one important result:

- `gelu_dropout_patch_location_k1_frozen_timm` reached `0.8429` raw image AUROC

That is a meaningful late-stage raw detector result. But the broader family is not a new overall winner.

- the best fused result is only `0.7724`
- the strongest raw runs are damaged by fixed fusion
- finetune+legacy often overfits
- larger K does not help in the completed runs
- patch-location `K=1024` is currently blocked by memory

So the current interpretation is:

- GELU/dropout is useful for the frozen patch-location anchor path
- frozen+timm is the better policy for image-level detection in this sweep
- the stage-2 fusion recipe needs retuning before this family can be fairly judged as a fused detector
- the family should be cited as a promising raw-anchor ablation, not as a replacement for `patch_location_kmeans_stage2recon_cosine_k32`, `patch_stage2e70_k32`, or PatchCore
