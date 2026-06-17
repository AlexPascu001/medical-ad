# GELU/Dropout Sweep Findings

Generated after running `run_gelu_dropout_sweep.ps1` on the 12-config grid:

- `K in {1, 64, 1024}`
- family in `{global CLS/full_redesign, patch location_kmeans}`
- mode in `{frozen backbone + timm eval transforms, trainable backbone + legacy transforms}`
- projection head set to `projection_activation: gelu`, `projection_dropout: 0.2`

The sweep completed 10 of 12 configs. Both `patch_location_k1024` variants were attempted and failed with CUDA OOM during the first Stage-1 backward pass.

## Run Status

| Run | Status | Runtime (min) | Notes |
|---|---:|---:|---|
| `gelu_dropout_global_k1_frozen_timm` | completed | 232.66 | First successful full run; earlier failed attempts created suffixed output dirs. |
| `gelu_dropout_global_k1_finetune` | completed | 107.56 | Completed cleanly. |
| `gelu_dropout_patch_location_k1_frozen_timm` | completed | 75.55 | Completed cleanly. |
| `gelu_dropout_patch_location_k1_finetune` | completed | 109.33 | Completed cleanly. |
| `gelu_dropout_global_k64_frozen_timm` | completed | 68.02 | Completed cleanly. |
| `gelu_dropout_global_k64_finetune` | completed | 73.65 | Completed cleanly. |
| `gelu_dropout_patch_location_k64_frozen_timm` | completed | 160.54 | Completed cleanly. |
| `gelu_dropout_patch_location_k64_finetune` | completed | 99.49 | Completed cleanly. |
| `gelu_dropout_global_k1024_frozen_timm` | completed | 68.66 | Completed cleanly. |
| `gelu_dropout_global_k1024_finetune` | completed | 58.19 | Completed cleanly. |
| `gelu_dropout_patch_location_k1024_frozen_timm` | failed | 184.88 | OOM in first Stage-1 backward pass. |
| `gelu_dropout_patch_location_k1024_finetune` | failed | 459.27 | OOM in first Stage-1 backward pass after manual attempt. |

## Metrics

| Run | Image AUROC | Fused AUROC | Pixel AUROC | Recon AUROC | Pixel-agg AUROC |
|---|---:|---:|---:|---:|---:|
| `global_k1_frozen_timm` | 0.7966 | 0.6956 | 0.8926 | 0.6746 | 0.4584 |
| `global_k1_finetune` | 0.7974 | 0.7724 | 0.9193 | 0.7195 | 0.7200 |
| `patch_location_k1_frozen_timm` | **0.8429** | 0.7469 | 0.8898 | 0.6810 | 0.4620 |
| `patch_location_k1_finetune` | 0.7179 | 0.6884 | 0.9214 | 0.6078 | 0.6336 |
| `global_k64_frozen_timm` | 0.7474 | 0.7125 | 0.8858 | 0.7053 | 0.4890 |
| `global_k64_finetune` | 0.6860 | 0.7609 | **0.9257** | 0.7314 | 0.7365 |
| `patch_location_k64_frozen_timm` | 0.8391 | 0.7657 | 0.8916 | **0.7368** | 0.5469 |
| `patch_location_k64_finetune` | 0.5870 | 0.6886 | 0.9253 | 0.6225 | 0.6484 |
| `global_k1024_frozen_timm` | 0.7770 | 0.7249 | 0.8797 | 0.7088 | 0.4748 |
| `global_k1024_finetune` | 0.7057 | 0.7445 | 0.9135 | 0.6055 | 0.6421 |

## What Went Well

The strongest image-level result in this sweep is `patch_location_k1_frozen_timm` at `0.8429` image AUROC. This is a real improvement over the older `patch_stage2e70_k1` result (`0.7411`) and over the clean global `full_redesign_stage2e70_k1` result (`0.8057`). The likely reason is that the frozen timm-preprocessed DINO geometry is useful when the patch-location bank is small and stable: with `K=1`, the model gets a single local normal reference per patch position, so the anchor score remains simple rather than over-fragmented.

The frozen-timm patch-location runs also preserve strong raw anchor scores at both completed K values: `0.8429` for `K=1` and `0.8391` for `K=64`. That suggests the combination of timm-style inputs plus a frozen backbone is good for patch-bank matching. It protects the pretrained representation from being distorted by the anomaly objective.

Finetuning helped the reconstruction and pixel branches in the global family. For example, `global_k64_finetune` reaches `0.7365` pixel-aggregated image AUROC and `0.9257` pixel AUROC, both clearly above `global_k64_frozen_timm` (`0.4890`, `0.8858`). This pattern repeats at `K=1` and `K=1024`: the trainable backbone tends to improve reconstruction-derived signals even when the anchor score weakens.

The sweep script itself now handles the project environment better than the first draft: it forces UTF-8 output, uses Matplotlib `Agg`, disables Albumentations update checks, and writes per-run logs plus `runs/gelu_dropout_sweep_timings.json`.

## What Did Not Improve Much

The global GELU/dropout configs did not beat the clean global baselines. The closest comparisons:

| Comparison | Baseline | GELU/dropout result |
|---|---:|---:|
| Global `K=1` image AUROC | `full_redesign_stage2e70_k1`: 0.8057 | best new global `K=1`: 0.7974 |
| Global `K=1024` fused AUROC | `full_redesign_stage2e70_k1024`: 0.8260 | best new global `K=1024`: 0.7445 |

The likely reason is that dropout in the projection head regularizes the very bottleneck that anchors depend on. In the global family, the anchor score is already fragile as K grows; adding stochastic projection noise during training appears to weaken the stable geometry more than it helps generalization.

Fusion often underperformed the raw anchor score in frozen-timm patch-location runs. `patch_location_k1_frozen_timm` has `0.8429` image AUROC but only `0.7469` fused AUROC. `patch_location_k64_frozen_timm` has `0.8391` image AUROC but only `0.7657` fused AUROC. The auxiliary branches are the reason: pixel-aggregated AUROC is weak (`0.4620` and `0.5469`), and divergence is poor or anti-correlated, especially at `K=64` (`0.2445`). Fusion is adding weak signals to a strong anchor score.

Finetuning was not a free improvement. In patch-location, it badly hurt the raw image score: `K=1` drops from `0.8429` frozen-timm to `0.7179` finetune, and `K=64` drops from `0.8391` to `0.5870`. This is consistent with the idea that local patch banks depend on preserving pretrained patch geometry. Updating the backbone changes the feature space while the patch-location anchors remain a fixed reference structure.

## What Failed

Both `patch_location_k1024` configs failed with CUDA OOM in the first training epoch:

- `gelu_dropout_patch_location_k1024_frozen_timm`
- `gelu_dropout_patch_location_k1024_finetune`

Both reached successful location-kmeans bank construction:

```text
Local centroid bank: (1024, 15, 15, 384)
Summary anchors    : (1024, 384)
```

The failure happens during backward after the first batch. The immediate allocation request was about `7.03 GiB`, with the 16 GB GPU already saturated. The likely mechanism is the dense local bank scale: `1024 * 15 * 15` anchor tokens are re-projected and compared during training. Because the projection head is trainable, this path retains enough graph state that memory blows up. Freezing the backbone does not solve this, because the trainable projection head and dense anchor comparisons still dominate.

## Interpretation

The main positive result is not "GELU/dropout globally improves the method." It is narrower and more useful:

1. Frozen-timm preprocessing plus patch-location anchors can produce strong raw image ranking, especially at `K=1` and `K=64`.
2. Trainable-backbone variants strengthen reconstruction/pixel evidence but tend to damage anchor geometry.
3. GELU/dropout does not rescue the global multi-anchor family and likely weakens the projection geometry for anchor scoring.
4. Patch-location `K=1024` is not currently viable with the dense raw-anchor re-projection implementation.

The best completed detector by raw image AUROC is `gelu_dropout_patch_location_k1_frozen_timm` (`0.8429`). The best completed detector by fused AUROC is `gelu_dropout_global_k1_finetune` (`0.7724`), but this is still below prior clean patch/global fused baselines around `0.826-0.830`. The best completed detector by pixel AUROC is `gelu_dropout_global_k64_finetune` (`0.9257`), close to but not above the strongest older patch/location pixel results.

## Follow-Up

For this specific branch, the most promising next run is not the full 12-grid again. It is a focused ablation around `patch_location_k1_frozen_timm` and `patch_location_k64_frozen_timm`:

- keep frozen backbone and timm transforms;
- compare `projection_dropout: 0.0`, `0.05`, `0.1`, `0.2`;
- compare `relu` vs `gelu`;
- disable fixed fusion or report anchor-only as the primary score when auxiliary signals are below `0.5`;
- avoid `patch_location K=1024` unless dense anchor projection is chunked or pre-projected.

For `patch_location K=1024`, feasible fixes would be reducing batch size, pre-projecting/fixing dense anchors, chunking local-distance computation, or lowering K before trying to train. As implemented, both frozen and trainable backbone variants exceed available GPU memory.
