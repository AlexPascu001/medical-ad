# GELU/Dropout Family Analysis

Scope:

- original `gelu_dropout_global_*` experiments;
- original `gelu_dropout_patch_location_*` experiments;
- staged `gelu_dropout_*_finetune_timm` follow-ups.

The family adds an explicitly regularized projection head to the late stage2e70-style recipe:

- `projection_activation: gelu`;
- `projection_dropout: 0.2`;
- `projection_dim: 128`;
- stage-2 reconstruction and fixed score fusion enabled.

The original sweep compared two coupled recipes:

- frozen backbone + timm eval preprocessing;
- full-backbone finetuning + legacy preprocessing.

That design could not distinguish backbone policy from input preprocessing. The follow-up holds timm preprocessing fixed and introduces a gentler finetuning schedule:

- epochs `0-4`: train the projection head only;
- epoch `5+`: also train DINOv3 blocks `10-11` and final norm;
- projection-head LR `1e-4`;
- backbone LR `1e-6`;
- earlier DINOv3 blocks remain frozen.

This analysis asks:

1. Did staged finetuning recover the pretrained anchor geometry lost by the original full-finetune recipe?
2. Did it improve on frozen+timm, or merely avoid degradation?
3. Did global and patch-location branches respond differently?
4. Did better Stage-1 behavior improve reconstruction, pixel evidence, or fixed fusion?

The conclusions are based on:

- `project/configs/gelu_dropout_*.yaml`;
- `run_gelu_dropout_sweep.ps1`;
- `run_gelu_dropout_timm_warmup_sweep.ps1`;
- `runs/gelu_dropout_*/.../evaluation/evaluation_metrics.json`;
- `runs/gelu_dropout_*/.../evaluation_stage1/evaluation_metrics.json`;
- `runs/gelu_dropout_*/.../training_history.json`;
- `runs/gelu_dropout_*/.../training_summary.json`;
- both sweep timing files and logs;
- comparison baselines in `docs/STAGE2E70_FAMILY_ANALYSIS.md` and `docs/PATCH_LOCATION_KMEANS_FAMILY_ANALYSIS.md`.

## 1. Experiment Matrix

The original sweep planned `12` runs and completed `10`. The staged follow-up planned and completed `5`.

| Bucket | Planned | Completed | K Values | Input Pipeline | Stage-1 Backbone |
|---|---:|---:|---|---|---|
| Global frozen+timm | 3 | 3 | `1, 64, 1024` | timm eval | frozen |
| Global full finetune+legacy | 3 | 3 | `1, 64, 1024` | legacy | all blocks, LR `1e-4` |
| Global staged finetune+timm | 3 | 3 | `1, 64, 1024` | timm eval | head warm-up, then last 2 blocks at `1e-6` |
| Patch-location frozen+timm | 3 | 2 | `1, 64, 1024` | timm eval | frozen |
| Patch-location full finetune+legacy | 3 | 2 | `1, 64, 1024` | legacy | all blocks, LR `1e-4` |
| Patch-location staged finetune+timm | 2 | 2 | `1, 64` | timm eval | head warm-up, then last 2 blocks at `1e-6` |

Both original patch-location `K=1024` runs failed with CUDA OOM during the first Stage-1 backward pass. No staged `K=1024` patch run was attempted.

Across both sweeps:

- `17` configurations were attempted;
- `15` produced final evaluations;
- `2` failed because of patch-bank memory pressure.

## 2. What Changed

All completed experiments retain:

- DINOv3 `vit_small_patch16_dinov3.lvd1689m`;
- `projection_dim: 128`;
- `loss.beta: 0.5`;
- `loss.delta: 0.0`;
- stage-2 reconstruction;
- frozen Stage-1 bottleneck for divergence;
- reconstruction-L2 pixel maps;
- 95th-percentile pixel aggregation;
- fixed fusion weights `0.4 / 0.3 / 0.3`.

The follow-up changes only the finetune-side recipe:

| Setting | Original Full Finetune | Staged Finetune |
|---|---|---|
| preprocessing | legacy | timm eval |
| initial backbone state | trainable from epoch 0 | frozen for epochs 0-4 |
| trainable depth | entire backbone | final 2/12 blocks + final norm |
| backbone LR | `1e-4` | `1e-6` |
| head LR | `1e-4` | `1e-4` |

This is a controlled comparison against frozen+timm, because both sides use timm preprocessing. It is not a component-level ablation of warm-up, trainable depth, and LR; those remain bundled.

## 3. Branch Definitions

### 3.1 Global Branch

The global branch uses centroid anchors in CLS embedding space:

- `anchor.strategy: kmeans`;
- `anchor.representation: centroids`;
- `anchor.use_embedding_space: true`;
- `anchor.reproject_anchors: true`;
- `training.fixed_pseudo_labels: true`;
- `training.pseudo_label_assignment: capacitated`;
- `stage2.alignment_target: anchor`.

Anchors are extracted in the pretrained 384D DINOv3 space and re-projected through the current head. When the backbone is updated, sample embeddings can drift relative to those stored raw anchor embeddings.

### 3.2 Patch-Location Branch

The patch branch uses same-location local centroid banks:

- `anchor.mode: patch`;
- `anchor.patch.variant: location_kmeans`;
- `anchor.representation: centroids`;
- 95th-percentile local score reduction;
- Euclidean local distance;
- dynamic per-patch nearest-centroid assignment;
- `stage2.alignment_target: local_anchor_pool`.

This branch is especially sensitive to changes in pretrained patch geometry because each stored centroid is tied to a spatial patch location.

## 4. Best-Run Summary

| Objective | Winner | Image AUROC | Fused AUROC | Pixel AUROC | Interpretation |
|---|---|---:|---:|---:|---|
| Best family raw image AUROC | `patch_location_k1_finetune_timm` | **`0.8479`** | `0.6948` | `0.8961` | staged result, but only `+0.0050` over frozen |
| Best family fused AUROC | `global_k1_finetune` | `0.7974` | **`0.7724`** | `0.9193` | old full-finetune recipe |
| Best family pixel AUROC | `global_k64_finetune` | `0.6860` | `0.7609` | **`0.9257`** | strong pixels, weak anchor score |
| Best staged fused AUROC | `patch_location_k64_finetune_timm` | `0.8354` | **`0.7303`** | `0.8843` | fusion still loses `0.1051` |
| Best reconstruction AUROC | `patch_location_k64_finetune_timm` | `0.8354` | `0.7303` | `0.8843` | reconstruction AUROC `0.7447` |
| Best frozen raw AUROC | `patch_location_k1_frozen_timm` | `0.8429` | `0.7469` | `0.8898` | statistically/practically near staged K=1 |

The headline changed slightly: the best raw score now comes from staged finetuning, but the gain is too small and inconsistent across K to establish finetuning as the preferred default.

## 5. Evaluation Results

### 5.1 All Completed Runs

| Family | K | Recipe | Image AUROC | Fused AUROC | Pixel AUROC | Recon AUROC | Divergence AUROC |
|---|---:|---|---:|---:|---:|---:|---:|
| Global | 1 | Frozen+timm | `0.7966` | `0.6956` | `0.8926` | `0.6746` | `0.3807` |
| Global | 1 | Full finetune+legacy | `0.7974` | `0.7724` | `0.9193` | `0.7195` | `0.5104` |
| Global | 1 | Staged finetune+timm | **`0.8104`** | `0.7068` | `0.8916` | `0.6831` | `0.3719` |
| Global | 64 | Frozen+timm | **`0.7474`** | `0.7125` | `0.8858` | `0.7053` | `0.7015` |
| Global | 64 | Full finetune+legacy | `0.6860` | **`0.7609`** | **`0.9257`** | **`0.7314`** | `0.6321` |
| Global | 64 | Staged finetune+timm | **`0.7474`** | `0.7121` | `0.8939` | `0.6879` | `0.6999` |
| Global | 1024 | Frozen+timm | **`0.7770`** | `0.7249` | `0.8797` | **`0.7088`** | **`0.6869`** |
| Global | 1024 | Full finetune+legacy | `0.7057` | **`0.7445`** | **`0.9135`** | `0.6055` | `0.6198` |
| Global | 1024 | Staged finetune+timm | **`0.7770`** | `0.7198` | `0.8866` | `0.6729` | `0.6823` |
| Patch-location | 1 | Frozen+timm | `0.8429` | **`0.7469`** | `0.8898` | **`0.6810`** | `0.4908` |
| Patch-location | 1 | Full finetune+legacy | `0.7179` | `0.6884` | **`0.9214`** | `0.6078` | **`0.6387`** |
| Patch-location | 1 | Staged finetune+timm | **`0.8479`** | `0.6948` | `0.8961` | `0.6727` | `0.5123` |
| Patch-location | 64 | Frozen+timm | **`0.8391`** | **`0.7657`** | `0.8916` | `0.7368` | `0.2445` |
| Patch-location | 64 | Full finetune+legacy | `0.5870` | `0.6886` | **`0.9253`** | `0.6225` | **`0.4438`** |
| Patch-location | 64 | Staged finetune+timm | `0.8354` | `0.7303` | `0.8843` | **`0.7447`** | `0.2360` |

### 5.2 Global Branch

| K | Frozen Image | Full-FT Image | Staged Image | Frozen Fused | Full-FT Fused | Staged Fused |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.7966` | `0.7974` | **`0.8104`** | `0.6956` | **`0.7724`** | `0.7068` |
| 64 | **`0.7474`** | `0.6860` | **`0.7474`** | `0.7125` | **`0.7609`** | `0.7121` |
| 1024 | **`0.7770`** | `0.7057` | **`0.7770`** | `0.7249` | **`0.7445`** | `0.7198` |

The staged recipe recovers the raw image AUROC lost by full finetuning:

- `+0.0614` at `K=64`;
- `+0.0713` at `K=1024`.

But these are not successful post-unfreeze models. Both selected epoch `1`, during the head-only warm-up. Their best post-unfreeze validation AUROCs were lower:

- global `K=64`: `0.7069` post-unfreeze versus `0.7552` during warm-up;
- global `K=1024`: `0.7319` post-unfreeze versus `0.8019` during warm-up.

Global `K=1` behaves differently. Its selected epoch is `96`, and raw test AUROC improves from `0.7966` frozen to `0.8104` staged. This is the clearest evidence that low-K global features can benefit from gentle adaptation.

### 5.3 Patch-Location Branch

| K | Frozen Image | Full-FT Image | Staged Image | Frozen Fused | Full-FT Fused | Staged Fused |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.8429` | `0.7179` | **`0.8479`** | **`0.7469`** | `0.6884` | `0.6948` |
| 64 | **`0.8391`** | `0.5870` | `0.8354` | **`0.7657`** | `0.6886` | `0.7303` |

The staged schedule removes the catastrophic degradation seen under full finetuning:

- K=1 improves by `+0.1300` over full finetune;
- K=64 improves by `+0.2484`.

Both staged patch runs selected checkpoints after unfreezing:

- K=1 selected epoch `18`;
- K=64 selected epoch `22`.

Their validation scores improved after unfreezing by `+0.0093` and `+0.0181` over the best warm-up values. On test, however, the differences relative to frozen+timm are only `+0.0050` and `-0.0037`.

The correct conclusion is that gentle adaptation is safe for patch-location at these K values, not that it materially outperforms freezing.

### 5.4 Fusion Behavior

The strongest raw staged runs are damaged severely by fixed fusion:

| Run | Image AUROC | Fused AUROC | Delta |
|---|---:|---:|---:|
| Global K=1 staged | `0.8104` | `0.7068` | `-0.1036` |
| Global K=64 staged | `0.7474` | `0.7121` | `-0.0353` |
| Global K=1024 staged | `0.7770` | `0.7198` | `-0.0572` |
| Patch K=1 staged | `0.8479` | `0.6948` | `-0.1531` |
| Patch K=64 staged | `0.8354` | `0.7303` | `-0.1051` |

The staged K=1 patch detector has pixel-aggregated AUROC `0.4609`. Its fixed positive pixel weight therefore adds a weak or misaligned signal to an excellent anchor ranker.

The old full-finetune recipe often produces the opposite pattern: weak anchor scores but stronger reconstruction and pixel signals, allowing fusion to rescue the result. That does not make its anchor representation better.

## 6. Training Behavior

### 6.1 Validation Versus Test

| Run | Best Val | Best Epoch | Actual Epochs | Test Image | Gap |
|---|---:|---:|---:|---:|---:|
| Global K=1 frozen | `0.8240` | `98` | `100` | `0.7966` | `0.0274` |
| Global K=1 full-FT | `0.9071` | `34` | `44` | `0.7974` | `0.1096` |
| Global K=1 staged | `0.8193` | `96` | `100` | `0.8104` | **`0.0089`** |
| Global K=64 frozen | `0.7552` | `1` | `20` | `0.7474` | `0.0078` |
| Global K=64 full-FT | `0.8304` | `5` | `20` | `0.6860` | `0.1444` |
| Global K=64 staged | `0.7552` | `1` | `20` | `0.7474` | `0.0078` |
| Global K=1024 frozen | `0.8019` | `1` | `20` | `0.7770` | `0.0249` |
| Global K=1024 full-FT | `0.8368` | `5` | `20` | `0.7057` | `0.1311` |
| Global K=1024 staged | `0.8019` | `1` | `20` | `0.7770` | `0.0249` |
| Patch K=1 frozen | `0.8642` | `13` | `23` | `0.8429` | `0.0214` |
| Patch K=1 full-FT | `0.8566` | `42` | `52` | `0.7179` | `0.1387` |
| Patch K=1 staged | `0.8666` | `18` | `28` | `0.8479` | `0.0186` |
| Patch K=64 frozen | `0.8800` | `67` | `77` | `0.8391` | `0.0408` |
| Patch K=64 full-FT | `0.7652` | `25` | `35` | `0.5870` | `0.1782` |
| Patch K=64 staged | `0.8409` | `22` | `32` | `0.8354` | **`0.0056`** |

The staged runs generalize much more cleanly than the old full-finetune runs. Their validation-test gaps are in the same small range as frozen+timm.

The old full-finetune validation scores were misleadingly optimistic. This is especially severe for patch K=64: validation `0.7652`, test `0.5870`.

### 6.2 Warm-Up And Unfreeze Transition

| Staged Run | Best Warm-up Val | Best Post-Unfreeze Val | Selected Phase |
|---|---:|---:|---|
| Global K=1 | `0.7529` | **`0.8193`** | partial unfreeze |
| Global K=64 | **`0.7552`** | `0.7069` | warm-up |
| Global K=1024 | **`0.8019`** | `0.7319` | warm-up |
| Patch K=1 | `0.8572` | **`0.8666`** | partial unfreeze |
| Patch K=64 | `0.8228` | **`0.8409`** | partial unfreeze |

The schedule executed as configured in all five logs:

- epochs `0-4`: `stage1_backbone_trainable = false`;
- epoch `5+`: `stage1_backbone_trainable = true`;
- epoch-5 head LR approximately `9.94e-5`;
- epoch-5 backbone LR approximately `9.94e-7`.

Checkpoint selection is essential to interpreting the result. A configuration can execute partial unfreezing while its final reported detector still comes from the earlier head-only phase.

### 6.3 Runtime And Memory

The staged sweep completed in `960.19` minutes:

| Run | Runtime (min) |
|---|---:|
| Global K=1 staged | `389.29` |
| Patch K=1 staged | `162.91` |
| Global K=64 staged | `129.07` |
| Patch K=64 staged | `162.98` |
| Global K=1024 staged | `115.93` |

The patch-location `K=1024` memory failure remains:

- local bank `(1024, 15, 15, 384)`;
- summary anchors `(1024, 384)`;
- failure during first backward pass;
- allocation request about `7.03 GiB` on a 16 GB GPU.

Freezing or partially unfreezing DINOv3 does not remove the dense projection graph for `1024 × 15 × 15` local anchors.

## 7. Comparison To Existing Baselines

| Run | Image AUROC | Fused AUROC | Pixel AUROC | Comment |
|---|---:|---:|---:|---|
| `gelu_dropout_patch_location_k1_finetune_timm` | **`0.8479`** | `0.6948` | `0.8961` | new GELU/dropout raw leader |
| `gelu_dropout_patch_location_k1_frozen_timm` | `0.8429` | `0.7469` | `0.8898` | simpler near-tied detector |
| `gelu_dropout_patch_location_k64_finetune_timm` | `0.8354` | `0.7303` | `0.8843` | best family reconstruction AUROC |
| `patch_location_kmeans_stage2recon_cosine_k32` | `0.7958` | `0.8295` | `0.9350` | stronger fused location-kmeans baseline |
| `patch_stage2e70_k32` | `0.8119` | `0.8282` | `0.9298` | stronger fused original patch baseline |
| `full_redesign_stage2e70_k1024` | `0.7685` | `0.8260` | `0.9195` | stronger fused global baseline |
| `patchcore_dinov3_vitsmall_2` | `0.8837` | n/a | `0.9612` | strongest listed non-CAM-anchor baseline |

The staged run improves the GELU/dropout raw leaderboard but does not improve the thesis-facing fused leaderboard. PatchCore remains stronger on both image and pixel AUROC.

## 8. What Happened, And Why

### 8.1 The Staged Recipe Preserved Pretrained Geometry

The old full-finetune setup exposed all DINOv3 blocks to gradients from a new stochastic projection head at the same `1e-4` LR. It also used a different preprocessing path from the frozen runs.

The staged recipe reduces all three risks:

- the projection head learns for five epochs before DINOv3 receives gradients;
- only the final two blocks and final norm are trainable;
- their LR is 100 times lower than the head LR;
- timm preprocessing matches the frozen control.

The large raw-AUROC recovery, especially in patch-location, is consistent with preservation of pretrained anchor geometry.

### 8.2 Preservation Is Not The Same As Beneficial Adaptation

Four of five staged-vs-frozen raw comparisons are within `0.005`, except global K=1 at `+0.0138`.

For global K=64 and K=1024, the selected checkpoints precede unfreezing entirely. Those experiments actively argue against adapting the backbone for global multi-anchor scoring under the current objective.

### 8.3 Patch Adaptation Is Stable But Test-Neutral

Both patch runs improve validation after epoch 5 and select post-unfreeze checkpoints. This means the optimization is not simply falling back to a frozen solution.

Nevertheless:

- K=1 test delta versus frozen is `+0.0050`;
- K=64 test delta is `-0.0037`.

The adaptation is safe and well-regularized, but its benefit does not survive as a material test improvement.

### 8.4 Anchor And Reconstruction Objectives Still Pull Differently

Full finetuning often improves reconstruction and pixel AUROC while damaging anchor AUROC. Staged finetuning restores the anchor score but usually gives up those auxiliary gains.

Patch K=64 shows that a middle ground is possible: reconstruction AUROC rises to `0.7447` while raw image AUROC stays near frozen. Its fixed fusion still drops to `0.7303` because divergence remains anti-informative (`0.2360`).

### 8.5 GELU/Dropout Still Does Not Favor Large K

Global K=1 staged is the best global raw result (`0.8104`). K=64 and K=1024 prefer epoch-1 warm-up checkpoints and do not benefit from partial unfreezing.

Patch K=64 does not improve raw AUROC over K=1, and patch K=1024 remains infeasible.

Nothing in the extended sweep supports a “larger K is better” conclusion.

## 9. Practical Verdict

### 9.1 What The Extended Family Shows

1. The original full-finetune+legacy recipe was unnecessarily destructive to anchor geometry.
2. Warm-up, partial unfreezing, low backbone LR, and timm preprocessing form a much safer combined recipe.
3. Staged finetuning produces small validation-test gaps and avoids the severe overfitting of full finetuning.
4. Global K=1 can gain modestly from gentle adaptation.
5. Global K=64 and K=1024 still prefer head-only checkpoints.
6. Patch-location can tolerate gentle adaptation, but test AUROC remains effectively tied with frozen+timm.
7. Fixed fusion remains unsuitable for the strongest raw detectors.

### 9.2 What It Does Not Show

1. It does not isolate whether warm-up, timm preprocessing, partial depth, or lower LR caused the recovery.
2. It does not establish staged finetuning as superior to freezing.
3. It does not rescue global multi-anchor adaptation.
4. It does not make GELU/dropout the strongest fused CAM-anchor family.
5. It does not solve patch-location K=1024 memory usage.

## 10. Recommendations

### 10.1 Keep Two Raw-Image References

Report both:

- `patch_location_k1_finetune_timm`: image AUROC `0.8479`;
- `patch_location_k1_frozen_timm`: image AUROC `0.8429`.

The staged run is numerically best; the frozen run is simpler and nearly tied.

### 10.2 Treat Frozen+Timm As The Default

Frozen+timm remains the best default when:

- simplicity and reproducibility matter;
- the raw score difference is negligible;
- global K is greater than 1;
- preserving pretrained patch geometry is the primary goal.

Use staged finetuning as a focused low-K option, not as the universal training policy.

### 10.3 Separate The Staged Components

Before another broad sweep, run K=1 pilots for:

1. timm + one unfrozen block;
2. timm + two unfrozen blocks;
3. timm + low LR without head warm-up;
4. timm + head-only throughout.

This would determine whether the gain comes from actual backbone adaptation or simply from avoiding the original aggressive recipe.

### 10.4 Fix Fusion Before Expanding K

For strong raw patch runs:

- allow anchor-only selection;
- tune fusion weights on validation;
- permit zero weight for divergence or pixel aggregation;
- report per-signal AUROC alongside fused AUROC.

The fixed `0.4 / 0.3 / 0.3` rule should not be the primary endpoint when an auxiliary signal has AUROC below `0.5`.

### 10.5 Keep Patch K=1024 Out Of Training Sweeps

Retry only after:

- chunking local distance computation;
- pre-projecting or caching dense anchors;
- reducing batch size;
- or reducing K to `128` or `256`.

## 11. Bottom Line

The follow-up changes the interpretation of the original sweep.

It is no longer accurate to say that finetuning itself is categorically harmful. Aggressive full-backbone finetuning at the head LR, combined with legacy preprocessing, was harmful. A five-epoch head warm-up followed by two-block finetuning at `1e-6` with timm preprocessing preserves the pretrained detector far more effectively.

But the staged recipe mostly converges back to frozen+timm performance:

- modest gain for global K=1;
- near-tie for both patch runs;
- head-only checkpoint selection for global K=64 and K=1024.

The family’s strongest claim is therefore:

> Gentle staged finetuning is a viable way to avoid destroying DINOv3 anchor geometry, but it provides limited evidence of improvement over simply freezing the backbone.
