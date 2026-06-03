# Patch Location-KMeans Family Analysis

Scope:

- `patch_location_kmeans*`
- `patch_location_kmeans_stage2match*`
- `patch_location_kmeans_stage2recon*`
- `patch_location_kmeans_stage2recon_cosine*`

This note analyzes the four `location_kmeans` experiment families, explains what actually changed from one sweep to the next, and compares their behavior to the original non-`k=1` patch baselines from `patch_stage2e70*`.

The practical questions are now:

- did the idea of pulling patches closer to their same-location matches help?
- did enabling stage-2 help once the first stage-2-capable `location_kmeans` path existed?
- did switching the active local patch distance from `euclidean` to `cosine` help, and if so at what K?
- if the family is still below `0.83+`, is that mostly a recipe issue or a limit of the current method definition?

The conclusions here are based on:

- `runs/patch_location_kmeans*/.../evaluation/evaluation_metrics.json`
- `runs/patch_location_kmeans*/.../training_summary.json`
- `runs/patch_location_kmeans_stage2match*/.../evaluation/evaluation_metrics.json`
- `runs/patch_location_kmeans_stage2match*/.../training_summary.json`
- `runs/patch_location_kmeans_stage2recon*/.../evaluation/evaluation_metrics.json`
- `runs/patch_location_kmeans_stage2recon*/.../training_summary.json`
- `runs/patch_location_kmeans_stage2recon_cosine*/.../evaluation/evaluation_metrics.json`
- `runs/patch_location_kmeans_stage2recon_cosine*/.../training_summary.json`
- `runs/patch_location_kmeans_k_timings.json`
- `runs/patch_location_kmeans_stage2match_timings.json`
- `runs/patch_location_kmeans_stage2recon_timings.json`
- `runs/patch_location_kmeans_stage2recon_cosine_timings.json`
- representative configs in `project/configs/`
- implementation constraints in `project/patch_mode.py`, `project/train.py`, and `project/model.py`
- the original patch baseline analysis in `docs/STAGE2E70_FAMILY_ANALYSIS.md`

## 1. Experiment Matrix

There are `18` `location_kmeans` runs in scope.

| Bucket | Runs | Backbone | Projection Dim | K Values | Mode | Representation | Stage 2 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Original `location_kmeans` | 6 | `vit_small` | `64` | `4, 8, 16, 32, 64, 128` | `patch/location_kmeans` | `centroids` | disabled |
| `stage2match` `location_kmeans` | 4 | `vit_small` | `128` | `4, 8, 16, 32` | `patch/location_kmeans` | `centroids` | disabled |
| `stage2recon` `location_kmeans` | 4 | `vit_small` | `128` | `4, 8, 16, 32` | `patch/location_kmeans` | `centroids` | enabled |
| `stage2recon_cosine` `location_kmeans` | 4 | `vit_small` | `128` | `4, 8, 16, 32` | `patch/location_kmeans` | `centroids` | enabled |

For interpretation, the main external references are the original patch baselines:

| Reference Family | Backbone | Projection Dim | K Values | Representation | Stage 2 |
| --- | --- | --- | --- | --- | --- |
| `patch_stage2e70*` | `vit_small` | `128` | `4, 8, 16, 32` | `closest_samples` | enabled |

The important sequencing point is:

- the original `location_kmeans` family tested the local-centroid idea with a weaker stage-1 recipe
- `stage2match` transplanted the strongest non-`k=1` patch *stage-1* recipe into the same `location_kmeans` structure
- `stage2recon` added actual stage-2 support on top of that stronger stage-1 recipe with the default `euclidean` local metric
- `stage2recon_cosine` kept that stage-2-capable recipe fixed and changed only `anchor.patch.local_distance_metric` to `cosine`

So the third sweep is the first one that can answer whether stage-2 actually helps this family, and the fourth is the first one that cleanly isolates the local metric choice inside that same stage-2-capable recipe.

Throughout this note, plain `stage2recon` refers to the original Euclidean local-distance sweep unless explicitly labeled `cosine`.

## 2. How The Location-KMeans Families Work

All four families share the same stage-1 core mechanism.

- `anchor.mode: patch`
- `anchor.patch.variant: location_kmeans`
- same-location local centroid banks built from frozen DINO patch tokens
- one local bank per spatial position on the `15 x 15` patch grid
- image anchor scores derived from local nearest-centroid distances

Operationally, the method is trying to learn a projected feature space where each patch becomes closer to the normal centroid bank of its own spatial location.

The first two families stop there. They are stage-1-only detectors.

The `stage2recon` family keeps the exact same stage-1 anchor path and adds a new stage-2 branch that:

- reconstructs the full image again
- builds stage-2 guidance from per-location local assignments, mean-pooled per image
- produces reconstruction pixel maps instead of relying only on upsampled dense patch distances
- exposes reconstruction scores, divergence signals, pixel aggregation, and fused image scores

That still leaves `location_kmeans` structurally different from the original `patch_stage2e70` family in two important ways.

1. `location_kmeans` uses `representation: centroids`, while `patch_stage2e70` uses `representation: closest_samples`.
2. `location_kmeans` stage-2 guidance is pooled from local assignments, while the original patch family was built around a legacy image-level anchor assignment path.

So the right comparison is not “did `location_kmeans` reproduce `patch_stage2e70` exactly?” The right comparison is:

- how far the pure local-centroid stage-1 idea can go on its own
- what the stronger stage-1 recipe changes
- what the first stage-2-capable version adds
- where the remaining gap now looks structural rather than merely tuning-related

## 3. What The Later Sweeps Actually Changed

### 3.1 What Stage2Match Changed

The second sweep was motivated by an apples-to-apples comparison with the strongest non-`k=1` original patch baseline, especially `patch_stage2e70_k32`.

However, only some of the copied settings were active under the current `location_kmeans` code path.

| Setting | Original `location_kmeans` | `stage2match` | Active Under Current Code? | Why It Matters |
| --- | --- | --- | --- | --- |
| `model.projection_dim` | `64` | `128` | yes | materially changed the learned stage-1 feature space |
| `loss.beta` | `1.0` | `0.5` | yes | reduced repulsion pressure between anchors |
| `loss.delta` | `0.1` | `0.0` | yes | removed the diversity regularizer |
| `training.pseudo_label_assignment` | `nearest` | `capacitated` | no | only relevant when fixed pseudo-labels are enabled |
| `training.capacity_multiplier` | `1.25` | `2.0` | no | same reason |
| `training.fixed_pseudo_labels` | `false` | `false` | yes | keeps pseudo-label precomputation disabled |
| `stage2.enabled` | `false` | `false` | yes | `location_kmeans` remained stage-1 only |
| `representation` | `centroids` | `centroids` | yes | structural constraint of the method |
| `local_score_reduction` | `percentile` | `percentile` | yes | image score still comes from a high-percentile local reducer |

In practice, the meaningful changes were mostly:

- `projection_dim: 64 -> 128`
- `loss.beta: 1.0 -> 0.5`
- `loss.delta: 0.1 -> 0.0`

So the second family should be read as a stronger stage-1 `location_kmeans` recipe, not as a true stage2-style `location_kmeans` variant.

### 3.2 What Stage2Recon Changed

The third sweep kept the stronger `stage2match` stage-1 recipe and enabled the first real stage-2 path for `location_kmeans`.

| Setting | `stage2match` | `stage2recon` | Active Under Current Code? | Why It Matters |
| --- | --- | --- | --- | --- |
| `stage2.enabled` | `false` | `true` | yes | enables reconstruction, divergence, pixel aggregation, and score fusion |
| `stage2.alignment_target` | n/a | `local_anchor_pool` | yes | aligns the stage-2 bottleneck to pooled local guidance instead of legacy image-level anchor semantics |
| `stage2.frozen_bottleneck` | n/a | `true` | yes | creates the new divergence signal |
| `stage2.pixel_map.enabled` | n/a | `true` | yes | uses reconstruction pixel maps as the main pixel signal |
| `stage2.pixel_aggregation` | n/a | `top_k_percentile@95` | yes | turns the reconstruction pixel map into an image-level score |
| `stage2.score_fusion.enabled` | n/a | `true` | yes | combines anchor, divergence, and pixel signals |
| `stage2.score_fusion` weights | n/a | `0.4 / 0.3 / 0.3` | yes | determines how much stage-2 can move the final detector |

One subtle but important interpretation detail follows from the implementation.

- raw `image_auroc` in final evaluation still tracks the anchor score path
- the stage-2 gains appear mainly in `fused_image_auroc` and in reconstruction-based pixel metrics

So if `stage2recon` looks only marginally different on raw image AUROC, that does **not** mean stage-2 was useless. It means the benefit is showing up in the added signals, not in the unchanged anchor score itself.

### 3.3 What Stage2ReconCosine Changed

The fourth sweep kept the entire stage-2-capable recipe fixed and changed only one active setting.

| Setting | `stage2recon` | `stage2recon_cosine` | Active Under Current Code? | Why It Matters |
| --- | --- | --- | --- | --- |
| `anchor.patch.local_distance_metric` | `euclidean` | `cosine` | yes | changes same-location centroid matching for both raw anchor scoring and local stage-2 guidance assignments |
| all other stage-1 and stage-2 settings | baseline | unchanged | yes | makes the cosine sweep a clean local-metric ablation rather than a recipe rewrite |

That means the cosine sweep is not testing a different architecture. It is testing whether the same `location_kmeans` + stage-2 stack behaves differently when local matching depends on angular similarity instead of Euclidean distance.

## 4. Best-Run Summary

| Objective | Winner | Primary AUROC | Pixel AUROC | Comment |
| --- | --- | ---: | ---: | --- |
| Best original `location_kmeans` raw image AUROC | `patch_location_kmeans_k128` | `0.7940` | `0.9338` | best result in the original p64 family |
| Best raw image AUROC across tuned `location_kmeans` variants | `patch_location_kmeans_stage2match_k16` | `0.8073` | `0.9040` | strongest anchor/image ranking overall |
| Best `stage2recon` fused AUROC | `patch_location_kmeans_stage2recon_k16` | `0.8182` | `0.9276` | best Euclidean stage-2 run |
| Best `stage2recon_cosine` fused AUROC | `patch_location_kmeans_stage2recon_cosine_k32` | `0.8295` | `0.9350` | new overall leader through stage-2 fusion |
| Best original patch baseline overall | `patch_stage2e70_k32` | `0.8282` | `0.9298` | now narrowly behind cosine `location_kmeans` on fused AUROC |

The headline result is now:

- `stage2match` improved the stage-1 image AUROC ceiling
- Euclidean `stage2recon` did **not** materially change the raw anchor/image ceiling, but it did improve the best overall `location_kmeans` detector through stage-2 fusion
- cosine `stage2recon` hurts raw image AUROC at most K, but `k32` improves the auxiliary stage-2 signals enough to reach `0.8295` fused AUROC
- the best `location_kmeans` run is now `patch_location_kmeans_stage2recon_cosine_k32` at `0.8295` fused AUROC
- that result now edges past the old overall patch leader `patch_stage2e70_k32` (`0.8295` vs `0.8282`)

So the family improved again, and this time it narrowly took the overall non-`k=1` patch lead on fused AUROC, even though it still did not produce the strongest raw anchor/image score.

## 5. Evaluation Results

### 5.1 Original Location-KMeans Family

| Run | Image AUROC | Pixel AUROC | Best Val Image | Best Epoch | Actual Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `patch_location_kmeans_k4` | `0.7390` | `0.9184` | `0.7908` | `5` | `15` |
| `patch_location_kmeans_k8` | `0.7719` | `0.9297` | `0.8607` | `8` | `18` |
| `patch_location_kmeans_k16` | `0.7884` | `0.9246` | `0.8601` | `14` | `24` |
| `patch_location_kmeans_k32` | `0.7921` | `0.9243` | `0.8782` | `30` | `40` |
| `patch_location_kmeans_k64` | `0.7917` | `0.9318` | `0.8584` | `14` | `24` |
| `patch_location_kmeans_k128` | `0.7940` | `0.9338` | `0.8566` | `17` | `27` |

The original family tells a stable story.

- image AUROC improves strongly from `k4` to `k16`
- then mostly saturates from `k32` onward
- pixel AUROC keeps improving more than image AUROC at larger K

So the first family already suggested that larger local banks help local scoring more than image-level ranking.

### 5.2 Stage2Match Location-KMeans Family

| Run | Image AUROC | Pixel AUROC | Best Val Image | Best Epoch | Actual Epochs | Runtime (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch_location_kmeans_stage2match_k4` | `0.7626` | `0.9394` | `0.8100` | `1` | `11` | `31.75` |
| `patch_location_kmeans_stage2match_k8` | `0.7832` | `0.9129` | `0.8631` | `7` | `17` | `41.87` |
| `patch_location_kmeans_stage2match_k16` | `0.8073` | `0.9040` | `0.8776` | `8` | `18` | `43.87` |
| `patch_location_kmeans_stage2match_k32` | `0.8013` | `0.8732` | `0.8875` | `8` | `18` | `63.00` |

This family behaves differently.

- image AUROC improves cleanly through `k16`
- `k32` still has the best validation AUROC, but not the best test AUROC
- pixel AUROC drops as K increases beyond `k4`

So the stronger stage-1 recipe helped image-level ranking, but it did not create a uniformly better local anomaly map.

### 5.3 Stage2Recon Location-KMeans Family

| Run | Anchor/Image AUROC | Reconstruction AUROC | Fused AUROC | Pixel AUROC | Runtime (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `patch_location_kmeans_stage2recon_k4` | `0.7626` | `0.7288` | `0.8031` | `0.9346` | `52.38` |
| `patch_location_kmeans_stage2recon_k8` | `0.7843` | `0.7376` | `0.8107` | `0.9385` | `67.63` |
| `patch_location_kmeans_stage2recon_k16` | `0.8056` | `0.7173` | `0.8182` | `0.9276` | `67.25` |
| `patch_location_kmeans_stage2recon_k32` | `0.8011` | `0.7165` | `0.8122` | `0.9288` | `83.37` |

This family answers a more specific question than the first two.

- raw image AUROC stays very close to `stage2match`, because the raw image score is still the anchor score
- fused AUROC improves over raw image AUROC at every K
- `k16` is the best overall `location_kmeans` run once fusion is allowed
- pixel AUROC rebounds into the `0.928` to `0.939` range because stage-2 uses reconstruction maps rather than upsampled dense distances

The weak part of the stage-2 stack is the divergence branch.

- divergence image AUROC ranges only from about `0.295` to `0.479`
- patch-divergence aggregated AUROC is similarly poor
- so the useful stage-2 gain is coming mostly from the reconstruction pixel map and the fusion it enables, not from divergence

### 5.4 Stage2Recon Cosine Location-KMeans Family

| Run | Anchor/Image AUROC | Reconstruction AUROC | Fused AUROC | Pixel AUROC | Runtime (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `patch_location_kmeans_stage2recon_cosine_k4` | `0.7470` | `0.7284` | `0.7892` | `0.9397` | `47.65` |
| `patch_location_kmeans_stage2recon_cosine_k8` | `0.7535` | `0.7345` | `0.8010` | `0.9384` | `47.15` |
| `patch_location_kmeans_stage2recon_cosine_k16` | `0.7607` | `0.7339` | `0.8062` | `0.9326` | `48.65` |
| `patch_location_kmeans_stage2recon_cosine_k32` | `0.7958` | `0.7361` | `0.8295` | `0.9350` | `46.74` |

This family behaves differently from the Euclidean sweep.

- raw image AUROC is worse than Euclidean at every K
- fused AUROC is also worse at `k4`, `k8`, and `k16`
- `k32` is the exception: it becomes the best fused run in the entire `location_kmeans` family at `0.8295`
- runtimes are much shorter and flatter than the Euclidean sweep

So cosine is not a blanket replacement for Euclidean. It is a high-K variant that only becomes compelling once fusion can exploit the stronger auxiliary signals.

### 5.5 Direct Euclidean-vs-Cosine Stage2Recon Comparison

| K | Euclidean Image | Cosine Image | Delta | Euclidean Fused | Cosine Fused | Delta | Euclidean Pixel | Cosine Pixel | Delta | Runtime Delta (Cos - Euc min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `0.7626` | `0.7470` | `-0.0156` | `0.8031` | `0.7892` | `-0.0139` | `0.9346` | `0.9397` | `+0.0051` | `-4.73` |
| `8` | `0.7843` | `0.7535` | `-0.0309` | `0.8107` | `0.8010` | `-0.0097` | `0.9385` | `0.9384` | `-0.0001` | `-20.48` |
| `16` | `0.8056` | `0.7607` | `-0.0449` | `0.8182` | `0.8062` | `-0.0120` | `0.9276` | `0.9326` | `+0.0050` | `-18.60` |
| `32` | `0.8011` | `0.7958` | `-0.0053` | `0.8122` | `0.8295` | `+0.0173` | `0.9288` | `0.9350` | `+0.0062` | `-36.63` |

This is the cleanest way to read the metric choice.

- cosine reduces raw anchor/image AUROC at every K, so it is not the better default stage-1 metric
- that anchor loss also hurts fused AUROC at `k4`, `k8`, and `k16`
- only `k32` flips sign, where cosine improves the reconstruction, divergence, and pixel branches enough to overcome the weaker anchor score
- the `k32` gain is therefore a stage-2/fusion win, not a raw anchor win

More specifically at `k32`, cosine improved reconstruction AUROC by about `+0.0195`, divergence AUROC by about `+0.0699`, and pixel AUROC by about `+0.0062`, which is why fusion crossed the old overall patch leader despite the slightly lower anchor score.

### 5.6 Direct Stage2Match-vs-Stage2Recon Comparison

| K | Stage2Match Image | Stage2Recon Image | Delta | Stage2Recon Fused | Fused Gain vs Stage2Match | Stage2Match Pixel | Stage2Recon Pixel | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `0.7626` | `0.7626` | `+0.0000` | `0.8031` | `+0.0406` | `0.9394` | `0.9346` | `-0.0048` |
| `8` | `0.7832` | `0.7843` | `+0.0011` | `0.8107` | `+0.0275` | `0.9129` | `0.9385` | `+0.0256` |
| `16` | `0.8073` | `0.8056` | `-0.0017` | `0.8182` | `+0.0109` | `0.9040` | `0.9276` | `+0.0236` |
| `32` | `0.8013` | `0.8011` | `-0.0002` | `0.8122` | `+0.0109` | `0.8732` | `0.9288` | `+0.0556` |

This is the clearest way to read what the new stage-2 path actually did.

- it barely changed the raw image AUROC relative to stage2match
- it provides consistent fused-image gains
- it materially strengthens pixel AUROC for `k8`, `k16`, and `k32`

So stage-2 is not fixing the raw anchor detector. It is adding a complementary reconstruction-based signal on top of it.

The cosine family does not overturn that baseline reading. It only adds one new wrinkle: at `k32`, changing the local metric made the auxiliary stage-2 signals strong enough that fusion improved much more than it did under Euclidean.

### 5.7 Direct Old-vs-New Stage-1 Comparison

| K | Original Image | Stage2Match Image | Delta | Original Pixel | Stage2Match Pixel | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `0.7390` | `0.7626` | `+0.0236` | `0.9184` | `0.9394` | `+0.0210` |
| `8` | `0.7719` | `0.7832` | `+0.0113` | `0.9297` | `0.9129` | `-0.0168` |
| `16` | `0.7884` | `0.8073` | `+0.0189` | `0.9246` | `0.9040` | `-0.0206` |
| `32` | `0.7921` | `0.8013` | `+0.0091` | `0.9243` | `0.8732` | `-0.0511` |

The main result is that the second family improved image AUROC at every overlapping K, but those gains came with weaker pixel AUROC at all but the smallest K.

That is a strong clue about what the extra optimization was doing.

- it improved the image-level ranking signal
- it did **not** improve the local distance maps in a uniformly useful way

### 5.8 Comparison To Original Patch Baselines

| Run | Image AUROC | Pixel AUROC | Fused AUROC |
| --- | ---: | ---: | ---: |
| `patch_stage2e70_k4` | `0.7879` | `0.9283` | `0.8131` |
| `patch_stage2e70_k8` | `0.7473` | `0.9343` | `0.7144` |
| `patch_stage2e70_k16` | `0.7932` | `0.9304` | `0.8173` |
| `patch_stage2e70_k32` | `0.8119` | `0.9298` | `0.8282` |
| `patch_location_kmeans_stage2match_k16` | `0.8073` | `0.9040` | `-` |
| `patch_location_kmeans_stage2recon_k16` | `0.8056` | `0.9276` | `0.8182` |
| `patch_location_kmeans_stage2recon_k32` | `0.8011` | `0.9288` | `0.8122` |
| `patch_location_kmeans_stage2recon_cosine_k32` | `0.7958` | `0.9350` | `0.8295` |

Three points matter here.

1. Euclidean `stage2recon_k16` was the first `location_kmeans` run that became clearly competitive with the strongest original patch baselines.
2. `patch_location_kmeans_stage2recon_cosine_k32` is now the strongest `location_kmeans` detector overall and narrowly beats `patch_stage2e70_k32` on fused AUROC (`0.8295` vs `0.8282`).
3. That win is **not** a raw image/anchor win. `patch_stage2e70_k32` (`0.8119`) and `stage2match_k16` (`0.8073`) still beat cosine `k32` on raw image AUROC.
4. So the new overall lead is coming from stage-2 complementarity, not from a universally stronger anchor detector.

Pixel AUROC should still be compared cautiously across families, because `stage2match` uses dense-patch-upsampled maps while `stage2recon` and the original patch stage-2 family use reconstruction maps. But even with that caution, the current picture is clear:

- stage-2 support materially strengthened `location_kmeans`
- the best `location_kmeans` run is now slightly ahead of the old strongest patch baseline on fused AUROC
- the margin is narrow enough that the new lead should still be treated as real but fragile rather than decisive

## 6. Training Behavior

### 6.1 Validation-Test Gaps

| Run | Best Val Image | Test Image | Gap |
| --- | ---: | ---: | ---: |
| `patch_location_kmeans_k4` | `0.7908` | `0.7390` | `0.0518` |
| `patch_location_kmeans_k8` | `0.8607` | `0.7719` | `0.0888` |
| `patch_location_kmeans_k16` | `0.8601` | `0.7884` | `0.0718` |
| `patch_location_kmeans_k32` | `0.8782` | `0.7921` | `0.0861` |
| `patch_location_kmeans_k64` | `0.8584` | `0.7917` | `0.0667` |
| `patch_location_kmeans_k128` | `0.8566` | `0.7940` | `0.0627` |
| `patch_location_kmeans_stage2match_k4` | `0.8100` | `0.7626` | `0.0474` |
| `patch_location_kmeans_stage2match_k8` | `0.8631` | `0.7832` | `0.0798` |
| `patch_location_kmeans_stage2match_k16` | `0.8776` | `0.8073` | `0.0703` |
| `patch_location_kmeans_stage2match_k32` | `0.8875` | `0.8013` | `0.0863` |
| `patch_location_kmeans_stage2recon_k4` | `0.8100` | `0.7626` | `0.0474` |
| `patch_location_kmeans_stage2recon_k8` | `0.8759` | `0.7843` | `0.0915` |
| `patch_location_kmeans_stage2recon_k16` | `0.8834` | `0.8056` | `0.0778` |
| `patch_location_kmeans_stage2recon_k32` | `0.8852` | `0.8011` | `0.0841` |
| `patch_location_kmeans_stage2recon_cosine_k4` | `0.8234` | `0.7470` | `0.0764` |
| `patch_location_kmeans_stage2recon_cosine_k8` | `0.8048` | `0.7535` | `0.0513` |
| `patch_location_kmeans_stage2recon_cosine_k16` | `0.7809` | `0.7607` | `0.0202` |
| `patch_location_kmeans_stage2recon_cosine_k32` | `0.8322` | `0.7958` | `0.0363` |

The main methodological point is still that the generalization gap is not unique to any one sweep.

- all four families show validation-to-test drops on the raw anchor/image score
- the cosine runs have smaller and flatter gaps than the Euclidean `stage2recon` sweep, especially at `k16` and `k32`, but they also have a lower raw-image ceiling
- the biggest new gains still show up by adding better stage-2 signals on top, not by fixing stage-1 transfer directly

So the story is not simply "the stronger recipe overfit more." The stronger recipe lifted raw performance, and stage-2 then added complementary information, but the same underlying stage-1 transfer problem remained.

### 6.2 Learning-Speed And Runtime Differences

There is still an important training-dynamics difference.

- the original `k32` run peaked at epoch `30`
- the `stage2match k32` run peaked at epoch `8`
- the `stage2recon k32` run also peaked at stage-1 epoch `8`, then used `11` extra stage-2 epochs before stage-2 early stopping

The Euclidean `stage2recon` runtimes were:

- `k4`: `52.38` min
- `k8`: `67.63` min
- `k16`: `67.25` min
- `k32`: `83.37` min
- total sweep runtime: `270.63` min

The cosine `stage2recon` runtimes were:

- `k4`: `47.65` min
- `k8`: `47.15` min
- `k16`: `48.65` min
- `k32`: `46.74` min
- total sweep runtime: `190.19` min

Compared with Euclidean, cosine was faster at every K and shorter by `80.44` minutes overall.

That combination suggests:

- the stronger stage-1 recipe made the anchor problem much better conditioned than the original p64 family
- cosine also produces a noticeably shorter training path, especially at higher K
- but that speedup is not free: at `k4`, `k8`, and `k16`, the faster cosine path also produces worse raw and fused image AUROC
- the only place where the faster cosine path also improves final detector quality is `k32`

## 7. Anchor Usage And K Saturation

At the best stage-1 validation epoch, the saved training histories still show limited effective anchor usage.

| Run | Nominal K | Effective Anchors Used | Largest Share | Normalized Entropy |
| --- | ---: | ---: | ---: | ---: |
| `patch_location_kmeans_k16` | `16` | `7` | `0.4872` | `0.4849` |
| `patch_location_kmeans_k32` | `32` | `12` | `0.3846` | `0.5770` |
| `patch_location_kmeans_k64` | `64` | `6` | `0.3846` | `0.3550` |
| `patch_location_kmeans_k128` | `128` | `11` | `0.5128` | `0.3596` |
| `patch_location_kmeans_stage2match_k16` | `16` | `7` | `0.4359` | `0.5294` |
| `patch_location_kmeans_stage2match_k32` | `32` | `10` | `0.4103` | `0.5034` |

This is still one of the clearest explanations for the weak high-K returns.

- larger K values create many more local centroids
- validation normals still use only a minority of them
- usage becomes especially poor at `k64` and `k128`

The Euclidean `stage2recon` family did not overturn this diagnosis, and the cosine sweep only partly changes it.

- raw image AUROC still does not make `k32` the best anchor detector
- the new best overall run is `stage2recon_cosine_k32`, but only after stage-2 fusion
- so higher K is still not turning into clearly better raw stage-1 capacity; the extra gain appears in the auxiliary reconstruction/divergence signals instead

So the method is still not fully converting nominal K into effective stage-1 capacity, even though stage-2 can sometimes rescue high-K runs after fusion.

## 8. What Happened, And Why

### 8.1 Why Stage2Match Helped

The second family improved image AUROC consistently because the active changes made stage-1 learning easier.

1. `projection_dim=128` gave the projection head more room than the original `64`-dimensional setup.
2. `loss.beta=0.5` and `loss.delta=0.0` relaxed the earlier regularization, which appears to have been too restrictive for image-level discrimination.
3. The gains were strongest at `k16`, where the method had enough centroid capacity to be expressive without fragmenting too far.

That is why the best new raw run reached `0.8073` rather than stalling in the high `0.79` range.

### 8.2 What The Stage2Recon Sweeps Proved

Taken together, the two stage-2-capable sweeps established three useful things.

1. Stage-2 is genuinely worth having for `location_kmeans`.
2. Euclidean `k16` was the first strong fused winner, but cosine `k32` is now the best fused run overall at `0.8295`.
3. The helpful stage-2 signal is still mostly the reconstruction-based pixel path, not the divergence branch.

That last point matters because it explains the metric pattern.

- reconstruction image AUROC is weaker than the anchor score
- divergence AUROC is much weaker still, although cosine `k32` made it less harmful than before
- reconstruction pixel maps are strong
- fusion helps because the pixel signal, and occasionally the divergence signal, add complementary information to the anchor score

So stage-2 is helping here, but it is helping in a narrower way than the original patch stage-2 family did.

### 8.3 What Cosine Actually Changed

The cosine sweep did not behave like a uniform upgrade.

1. It reduced raw anchor/image AUROC at every K.
2. At `k4`, `k8`, and `k16`, that raw drop also reduced fused AUROC.
3. At `k32`, however, cosine improved the reconstruction, divergence, and pixel branches enough to lift fused AUROC from `0.8122` to `0.8295`.

So cosine is not the new default metric. It is a high-K stage-2 interaction that became useful only once the family had enough centroid capacity and fusion could exploit the extra signals.

### 8.4 Why The Ceiling Still Held

The answer is not that the idea failed completely. The answer is that the current form of the idea still leaves several bottlenecks in place.

#### A. The raw anchor path is still the limiting base detector

Even with stage-2 enabled, raw image AUROC still sits in roughly the same `0.76` to `0.81` band as the stage-1-only family because that score is still coming from the anchor path.

So stage-2 is complementing the detector rather than replacing its main bottleneck. Even the new overall best run gets there with a `0.7958` raw anchor/image AUROC rather than by dominating stage-1.

#### B. Reconstruction helps, divergence mostly does not

The first stage-2-capable sweep added a real second signal, but the new divergence branch is weak.

- divergence image AUROC falls roughly in the `0.29` to `0.48` range
- patch-divergence aggregated AUROC is similarly poor

That suggests the current pooled-local-guidance stage-2 design is not yet turning divergence into a useful anomaly signal.

#### C. Pulling patches toward centroids is still not the same as matching against rich local exemplars

The original patch family uses `closest_samples`, which preserves real local reference structure. `location_kmeans` uses centroids, which compress that structure into synthetic local prototypes.

That compression is not obviously bad, but it can smooth away useful fine-grained variation that a local anomaly detector might want to keep.

This is still a plausible reason the best `location_kmeans` run only *narrowly* surpassed the best original patch baseline on fused AUROC rather than clearly dominating it.

#### D. Exact same-location matching is powerful, but also rigid

The method only lets a patch compete with centroids from the same spatial location. That preserves locality, but it also assumes that the correct local normal bank is tied to exact patch coordinates.

That can work well for moderate K, but as K grows it starts to spend capacity modeling local nuisance variation rather than building a more useful detector.

The K-saturation results still fit that interpretation.

#### E. Fixed fusion weights may still be leaving performance on the table

The current fused score uses a fixed `0.4 / 0.3 / 0.3` split across anchor, divergence, and pixel signals.

If divergence is weak or partly anti-helpful, a static `0.3` divergence weight is unlikely to be optimal. So even the new stage-2 path may still be under-optimized in the way it combines signals.

## 9. Method Limitation Or Parameter Limitation?

The evidence still points to a mixed answer.

### 9.1 Why This Is Not Just A Bad Idea

The idea clearly has value.

- the second family improved the first one at every overlapping K
- Euclidean `stage2recon k16` reached `0.8182` fused AUROC
- cosine `stage2recon k32` reached `0.8295` fused AUROC and narrowly beat `patch_stage2e70_k32` (`0.8295` vs `0.8282`)

So the core idea of a local-centroid `location_kmeans` detector is not empty.

### 9.2 Why This Is Not Just A Simple Tuning Problem Either

The current ceiling still looks structural.

- the raw anchor path still saturates around `0.80`
- the new overall win comes from fusion despite a weaker raw anchor score than the strongest stage-1 runs
- the local banks are still centroid banks rather than exemplar banks
- exact same-location matching remains rigid
- high K still under-utilizes a large fraction of nominal anchors
- the new divergence branch is currently weak

Those are not small hyperparameter details. They are part of the current method definition.

### 9.3 Practical Verdict

The most defensible conclusion is:

1. the original p64 `location_kmeans` recipe was under-tuned for image AUROC
2. the `stage2match` sweep fixed part of that and proved the idea had clear stage-1 headroom
3. the Euclidean `stage2recon` sweep proved that stage-2 is worth having for this family
4. the cosine `k32` result proved that the local metric is an active lever, but only in a narrow high-K fusion regime
5. beating the new `0.8295` fused result by a convincing margin, and especially getting clearly beyond the low-`0.83` band, probably still needs either a stronger stage-2 recipe or a structural stage-1 change

The current evidence makes the low-`0.83` band look real. Going materially beyond that still looks unlikely from repeating the same sweep style unchanged.

## 10. How To Improve From Here

### 10.1 Cheap Retunes Inside The Current Formulation

These are the most defensible next sweeps because they affect active parts of the method.

1. Sweep score-fusion weights, especially trying a much lower `divergence_weight` or even `0.0`.
2. Treat `local_distance_metric` as K-dependent rather than globally fixed:
   - keep `euclidean` as the default reference
   - retest `cosine` specifically around `k24` to `k32`, where it helped fused AUROC
3. Sweep stage-2 alignment and consistency strengths instead of keeping the first-pass defaults fixed.
4. Sweep the image reducer:
   - `local_score_reduction: mean`
   - `local_score_reduction: max`
   - percentiles such as `90`, `95`, `97.5`
5. Search the K region that now looks most promising instead of pushing upward automatically:
   - Euclidean: `k12`, `k16`, `k24`
   - Cosine: `k24`, `k32`
6. Sweep intermediate regularization values instead of only the extremes already tested:
   - `beta: 0.75`
   - `delta: 0.05` or `0.1`

What should still *not* be prioritized inside the current code path:

- `pseudo_label_assignment`
- `capacity_multiplier`

Those settings still do not matter while `fixed_pseudo_labels=false` remains enforced.

### 10.2 Structural Changes More Likely To Raise The Ceiling

1. Rework the divergence branch for pooled local guidance, or drop it from fusion if it keeps underperforming.
2. Add a `closest_samples` or hybrid centroid-plus-exemplar local bank instead of centroids only.
3. Relax the exact same-location rule into a small spatial neighborhood so the method can absorb mild anatomical shifts.
4. Replace static fusion weights with a learned or at least validation-tuned fusion scheme.

These changes are more work, but they are also the ones most likely to move the ceiling beyond the current `0.8295` band.

## 11. Bottom Line

The later `location_kmeans` stage-2 sweeps were worth doing.

- `stage2match` showed that the earlier family was not using the best stage-1 recipe
- Euclidean `stage2recon` showed that stage-2 can materially improve the overall `location_kmeans` detector
- cosine `stage2recon` showed that the local metric can change the overall winner even when the raw anchor score gets worse
- the best `location_kmeans` run is now `patch_location_kmeans_stage2recon_cosine_k32` at `0.8295` fused AUROC

But it also clarified the current limit.

- the best raw image/anchor run is still `stage2match_k16` at `0.8073`
- the new overall lead over `patch_stage2e70_k32` is narrow (`0.8295` vs `0.8282`) rather than decisive
- the divergence branch is still the weakest part of the stage-2 stack overall

So the most honest interpretation is now this:

- the idea works
- stage-2 helps
- the local metric matters, but in a K-dependent way rather than as a universal upgrade
- the current ceiling is no longer just "stage-1 only," but it is still structurally constrained by centroid banks, rigid local matching, and a weak divergence branch
- the next gains are more likely to come from stage-2 / fusion redesign or richer local reference structure than from repeating the same sweep pattern again
