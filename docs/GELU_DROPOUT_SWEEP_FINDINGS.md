# GELU/Dropout Sweep Findings

This note summarizes both GELU/dropout sweeps:

1. the original `run_gelu_dropout_sweep.ps1` grid, which coupled frozen backbones with timm eval preprocessing and full-backbone finetuning with legacy preprocessing;
2. the follow-up `run_gelu_dropout_timm_warmup_sweep.ps1` grid, which held timm preprocessing fixed and used staged finetuning:
   - epochs `0-4`: projection head only;
   - epoch `5+`: projection head plus DINOv3 blocks `10-11` and final norm;
   - projection-head LR `1e-4`;
   - backbone LR `1e-6`.

All runs use:

- DINOv3 `vit_small_patch16_dinov3.lvd1689m`;
- `projection_activation: gelu`;
- `projection_dropout: 0.2`;
- `projection_dim: 128`;
- stage-2 reconstruction and fixed `0.4 / 0.3 / 0.3` score fusion.

The original grid completed `10/12` runs. Both patch-location `K=1024` runs failed with CUDA OOM. The staged follow-up completed all `5/5` planned runs in `960.19` minutes.

## 1. Recipes Compared

| Recipe | Input Pipeline | Stage-1 Backbone Policy | Backbone LR |
|---|---|---|---:|
| Frozen+timm | timm eval preprocessing | frozen for all epochs | n/a |
| Full finetune+legacy | legacy preprocessing | all blocks trainable from epoch 0 | `1e-4` |
| Staged finetune+timm | timm eval preprocessing | 5 head-only epochs, then final 2 blocks + norm | `1e-6` |

The staged comparison fixes the largest confound in the original sweep: frozen and staged runs now use the same timm input pipeline. It does not isolate warm-up, partial unfreezing, and lower LR from one another; those three changes form one combined recipe.

## 2. Completed Metrics

| Family | K | Recipe | Image AUROC | Fused AUROC | Pixel AUROC | Recon AUROC | Pixel-agg AUROC |
|---|---:|---|---:|---:|---:|---:|---:|
| Global | 1 | Frozen+timm | `0.7966` | `0.6956` | `0.8926` | `0.6746` | `0.4584` |
| Global | 1 | Full finetune+legacy | `0.7974` | **`0.7724`** | `0.9193` | `0.7195` | `0.7200` |
| Global | 1 | Staged finetune+timm | **`0.8104`** | `0.7068` | `0.8916` | `0.6831` | `0.4632` |
| Global | 64 | Frozen+timm | **`0.7474`** | `0.7125` | `0.8858` | `0.7053` | `0.4890` |
| Global | 64 | Full finetune+legacy | `0.6860` | **`0.7609`** | **`0.9257`** | **`0.7314`** | **`0.7365`** |
| Global | 64 | Staged finetune+timm | **`0.7474`** | `0.7121` | `0.8939` | `0.6879` | `0.4899` |
| Global | 1024 | Frozen+timm | **`0.7770`** | `0.7249` | `0.8797` | **`0.7088`** | `0.4748` |
| Global | 1024 | Full finetune+legacy | `0.7057` | **`0.7445`** | **`0.9135`** | `0.6055` | **`0.6421`** |
| Global | 1024 | Staged finetune+timm | **`0.7770`** | `0.7198` | `0.8866` | `0.6729` | `0.4559` |
| Patch-location | 1 | Frozen+timm | `0.8429` | `0.7469` | `0.8898` | **`0.6810`** | `0.4620` |
| Patch-location | 1 | Full finetune+legacy | `0.7179` | `0.6884` | **`0.9214`** | `0.6078` | **`0.6336`** |
| Patch-location | 1 | Staged finetune+timm | **`0.8479`** | `0.6948` | `0.8961` | `0.6727` | `0.4609` |
| Patch-location | 64 | Frozen+timm | **`0.8391`** | **`0.7657`** | `0.8916` | `0.7368` | `0.5469` |
| Patch-location | 64 | Full finetune+legacy | `0.5870` | `0.6886` | **`0.9253`** | `0.6225` | **`0.6484`** |
| Patch-location | 64 | Staged finetune+timm | `0.8354` | `0.7303` | `0.8843` | **`0.7447`** | `0.5603` |

The new family-level raw-image leader is `patch_location_k1_finetune_timm` at `0.8479`. The gain over its frozen+timm control is only `+0.0050`, so it should be treated as a near-tie rather than a decisive improvement.

The old full-finetune+legacy global `K=1` run remains the family fused-score leader at `0.7724`, and the old global `K=64` run remains the pixel-AUROC leader at `0.9257`.

## 3. What The Staged Recipe Changed

### 3.1 It Prevented The Large Anchor-Score Collapse

Relative to full finetune+legacy, staged finetune+timm changed raw image AUROC by:

| Family | K | Full Finetune+Legacy | Staged Finetune+Timm | Delta |
|---|---:|---:|---:|---:|
| Global | 1 | `0.7974` | `0.8104` | `+0.0130` |
| Global | 64 | `0.6860` | `0.7474` | `+0.0614` |
| Global | 1024 | `0.7057` | `0.7770` | `+0.0713` |
| Patch-location | 1 | `0.7179` | `0.8479` | `+0.1300` |
| Patch-location | 64 | `0.5870` | `0.8354` | `+0.2484` |

This is the clearest positive result. Gentle partial finetuning with timm preprocessing no longer destroys the patch-location anchor detector.

That improvement cannot be assigned to warm-up alone. The staged recipe simultaneously changes preprocessing, trainable depth, and backbone LR relative to the old full-finetune runs.

### 3.2 It Usually Matched, Rather Than Beat, Frozen+Timm

Relative to the controlled frozen+timm baselines:

| Family | K | Frozen+Timm | Staged Finetune+Timm | Delta |
|---|---:|---:|---:|---:|
| Global | 1 | `0.7966` | `0.8104` | `+0.0138` |
| Global | 64 | `0.7474` | `0.7474` | `+0.0000` |
| Global | 1024 | `0.7770` | `0.7770` | `+0.0000` |
| Patch-location | 1 | `0.8429` | `0.8479` | `+0.0050` |
| Patch-location | 64 | `0.8391` | `0.8354` | `-0.0037` |

The practical interpretation is preservation, not a broad finetuning gain. Staged finetuning is much safer than the original full-finetune recipe, but frozen+timm remains equally good or nearly equally good in four of five comparisons.

### 3.3 Global Multi-Anchor Checkpoint Selection Rejected Unfreezing

The strongest diagnostic comes from the validation trajectory:

| Run | Best Warm-up Val | Best Post-Unfreeze Val | Selected Epoch | Test Image AUROC |
|---|---:|---:|---:|---:|
| Global K=1 staged | `0.7529` | `0.8193` | `96` | `0.8104` |
| Global K=64 staged | **`0.7552`** | `0.7069` | `1` | `0.7474` |
| Global K=1024 staged | **`0.8019`** | `0.7319` | `1` | `0.7770` |
| Patch-location K=1 staged | `0.8572` | **`0.8666`** | `18` | `0.8479` |
| Patch-location K=64 staged | `0.8228` | **`0.8409`** | `22` | `0.8354` |

For global `K=64` and `K=1024`, the selected checkpoint is epoch `1`, inside the head-only warm-up and before the epoch-`5` unfreeze. Their post-unfreeze validation maxima were lower by `0.0484` and `0.0699`, respectively.

Therefore these two final models do not demonstrate a successful partial-backbone adaptation. They demonstrate that early checkpoint selection protected the frozen representation from a harmful later phase.

Global `K=1` is the only global case where the partially unfrozen model was selected and produced a modest test improvement. Both patch-location runs also selected post-unfreeze checkpoints, but their test scores remained within `0.005` of frozen+timm.

## 4. Generalization Behavior

| Run | Best Val Image | Best Epoch | Actual Epochs | Test Image | Val-Test Gap |
|---|---:|---:|---:|---:|---:|
| Global K=1 frozen | `0.8240` | `98` | `100` | `0.7966` | `0.0274` |
| Global K=1 full finetune | `0.9071` | `34` | `44` | `0.7974` | `0.1096` |
| Global K=1 staged | `0.8193` | `96` | `100` | `0.8104` | **`0.0089`** |
| Global K=64 frozen | `0.7552` | `1` | `20` | `0.7474` | `0.0078` |
| Global K=64 full finetune | `0.8304` | `5` | `20` | `0.6860` | `0.1444` |
| Global K=64 staged | `0.7552` | `1` | `20` | `0.7474` | `0.0078` |
| Global K=1024 frozen | `0.8019` | `1` | `20` | `0.7770` | `0.0249` |
| Global K=1024 full finetune | `0.8368` | `5` | `20` | `0.7057` | `0.1311` |
| Global K=1024 staged | `0.8019` | `1` | `20` | `0.7770` | `0.0249` |
| Patch K=1 frozen | `0.8642` | `13` | `23` | `0.8429` | `0.0214` |
| Patch K=1 full finetune | `0.8566` | `42` | `52` | `0.7179` | `0.1387` |
| Patch K=1 staged | `0.8666` | `18` | `28` | `0.8479` | `0.0186` |
| Patch K=64 frozen | `0.8800` | `67` | `77` | `0.8391` | `0.0408` |
| Patch K=64 full finetune | `0.7652` | `25` | `35` | `0.5870` | `0.1782` |
| Patch K=64 staged | `0.8409` | `22` | `32` | `0.8354` | **`0.0056`** |

The staged runs have small validation-test gaps (`0.0056-0.0249`). The old full-finetune runs have much larger gaps (`0.1096-0.1782`). This supports the claim that the staged recipe is substantially better regularized.

It does not prove that partial unfreezing is intrinsically better: timm preprocessing and reduced trainable capacity are part of the same intervention.

## 5. Reconstruction, Pixel Evidence, And Fusion

The old full-finetune+legacy recipe often damaged anchor ranking while improving reconstruction-derived signals. The staged recipe reverses most of that trade:

- anchor AUROC returns to the frozen+timm band;
- pixel and pixel-aggregated AUROC usually return toward the frozen values;
- the old auxiliary-signal gains mostly disappear.

Patch-location `K=64` is the useful exception: staged finetuning reaches the best reconstruction AUROC in this family (`0.7447`), slightly above frozen+timm (`0.7368`), while retaining nearly the same raw anchor score.

Fixed fusion remains poorly calibrated. For the staged runs:

| Run | Image AUROC | Fused AUROC | Fused Delta |
|---|---:|---:|---:|
| Global K=1 staged | `0.8104` | `0.7068` | `-0.1036` |
| Global K=64 staged | `0.7474` | `0.7121` | `-0.0353` |
| Global K=1024 staged | `0.7770` | `0.7198` | `-0.0572` |
| Patch K=1 staged | `0.8479` | `0.6948` | `-0.1531` |
| Patch K=64 staged | `0.8354` | `0.7303` | `-0.1051` |

The new raw-image leader is also the run most damaged by fusion. Its pixel-aggregated AUROC is `0.4609`, so adding that signal to a strong anchor score is counterproductive.

## 6. Runtime And Failure Status

| Staged Run | Runtime (min) | Status |
|---|---:|---|
| Global K=1 | `389.29` | completed |
| Patch-location K=1 | `162.91` | completed |
| Global K=64 | `129.07` | completed |
| Patch-location K=64 | `162.98` | completed |
| Global K=1024 | `115.93` | completed |

The original patch-location `K=1024` failures remain unresolved. Both frozen and full-finetune variants built a `(1024, 15, 15, 384)` local centroid bank and then failed during the first backward pass with a roughly `7.03 GiB` allocation request on a 16 GB GPU.

No staged patch-location `K=1024` run was attempted. Warm-up and partial unfreezing do not address the dominant dense-anchor projection memory cost.

## 7. Practical Interpretation

The combined evidence supports these conclusions:

1. The old statement “finetuning is harmful” was too broad. Full-backbone, same-LR finetuning with legacy preprocessing was harmful to anchor geometry; staged finetuning with timm preprocessing is much safer.
2. The staged recipe mainly preserves frozen+timm performance. It does not establish a general benefit from adapting DINOv3.
3. Global multi-anchor models still prefer the head-only representation. For `K=64` and `K=1024`, checkpoint selection explicitly rejected every partially unfrozen checkpoint.
4. Global `K=1` is the clearest positive adaptation case: `0.8104` versus `0.7966` frozen.
5. Patch-location adaptation is stable but nearly neutral on test: `+0.0050` at `K=1`, `-0.0037` at `K=64`.
6. Lower validation-test gaps show that the staged recipe avoids the severe overfitting seen in the original full-finetune runs.
7. Fixed fusion remains the main reason the strongest raw detectors do not become strong final fused detectors.

## 8. Recommended Follow-Ups

The next experiments should be narrower:

1. Use `K=1` pilots to separate the staged recipe:
   - timm + full finetune at low LR, no warm-up;
   - timm + 5-epoch warm-up + one unfrozen block;
   - timm + 5-epoch warm-up + two unfrozen blocks;
   - timm + head-only for the full run.
2. Do not expand this to another full K grid until the one-block/two-block question is resolved.
3. Retune or disable fusion for the strong patch-location runs. Divergence and pixel aggregation should not receive fixed positive weight when their AUROC is near or below `0.5`.
4. Continue reporting raw and fused image AUROC separately.
5. Avoid patch-location `K=1024` until dense anchor projection is chunked, cached, or otherwise made memory-safe.

## 9. Bottom Line

Warm-up plus low-LR partial unfreezing successfully prevents the catastrophic anchor degradation seen with the old full-finetune recipe. It does not, however, make finetuning broadly superior to freezing.

The best staged result is `patch_location_k1_finetune_timm` at `0.8479` raw image AUROC, only `0.0050` above frozen+timm. For global `K=64` and `K=1024`, the final selected models come from the head-only warm-up, not the partially unfrozen phase.

The strongest defensible claim is therefore:

> Gentle staged finetuning can preserve DINOv3 anchor geometry and occasionally improve a low-K detector, but frozen+timm remains the simpler and equally competitive default for this GELU/dropout family.
