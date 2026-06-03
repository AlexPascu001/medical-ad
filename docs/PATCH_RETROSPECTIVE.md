# Patch Retrospective

Scope: `patch_stage2e70_*`

This note summarizes what changed in the patch-mode stage-2 e70 family, what the experiments actually showed, what is working, what is not working, why the observed behavior makes sense technically, how the patch results compare with `full_redesign_stage2e70`, and what the next steps should be.

The conclusions here are based on:

- `runs/patch_stage2e70_*/.../evaluation/evaluation_metrics.json`
- `runs/patch_stage2e70_*/.../training_summary.json`
- `runs/patch_stage2e70_*/.../training_history.json`
- representative family configs such as `project/configs/patch_stage2e70_k16.yaml`
- the comparison baseline in `REDESIGN_RETROSPECTIVE.md`

## 1. What Changed

The patch family kept the same broad two-stage training structure as the redesign family, but changed the semantics of the anchor detector.

Core patch-family choices:

- `anchor.mode: patch`
- anchors are still prepared from frozen DINOv3 features with `strategy: kmeans`
- anchors use `representation: closest_samples`, not centroids
- anchor scoring is patch-to-anchor and then reduced spatially with `anchor.patch.score_reduction: mean`
- anchors stay in embedding space with `use_embedding_space: true` and `reproject_anchors: true`
- training uses `train_augment_mode: none` to preserve patch-location meaning
- stage-1 pseudo-labels are fixed and capacitated with `pseudo_label_assignment: capacitated` and `capacity_multiplier: 2.0`
- stage-2 is enabled with a 70-epoch budget, frozen encoder, frozen anchors, frozen anchor target, and `alignment_target: anchor`
- stage-2 early stopping still monitors `pixel_aggregated_image_auroc`
- score fusion uses the same fixed weights as redesign: anchor `0.4`, divergence `0.3`, pixel `0.3`
- pixel maps come from the reconstruction branch; `use_pixel_decoder: false`

Important differences from `full_redesign_stage2e70`:

- patch mode uses local dense anchor matching rather than global centroid prototypes
- patch mode uses nearest real samples (`closest_samples`) rather than centroid anchors
- the patch family does not enforce a stage-1 minimum-epoch floor; `min_epochs_before_early_stopping: 0`
- the patch family was tested on `K = 1, 2, 4, 8, 16, 32`, not on the redesign scale of `K = 1, 4, 16, 256, 1024`

Important implementation consequences:

- `training_summary.json` is still stage-1-only and does not summarize stage-2
- final evaluation prefers the best stage-2 checkpoint when it exists
- patch-mode image scores come from reduced dense patch-to-anchor distances rather than a pure CLS-only path
- patch-mode assignment statistics are still useful, but they should be read as summaries of reduced dense behavior rather than as clean global semantic clusters

## 2. Headline Findings

1. Patch mode is currently the stronger practical anomaly detector in its tested K range once `K >= 4`, with the best overall result at `K = 32`.
2. Stage-1 checkpoint selection is still a real problem at larger K: the best stage-1 checkpoint falls back to epoch 1 for `K = 16` and `K = 32`.
3. Stage-2 uses only a small fraction of the nominal 70-epoch budget and behaves mostly like a light calibrator, not a deep refinement phase.
4. Reconstruction-driven pixel metrics are strong and stable across the whole patch family, roughly `0.928` to `0.934` pixel AUROC.
5. The divergence branches are weak to strongly anti-correlated in most runs, so the fixed `0.3` divergence fusion weight is difficult to justify.
6. The tested patch runs do not show the earlier one-anchor collapse failure mode; larger K still use many anchors, but patch mode has a weaker global clustering story than the redesign family because its semantics are spatial first and image-level second.

## 3. Final Results

### 3.1 `patch_stage2e70`

| K | Image AUROC | Pixel AUROC | Pixel-Agg AUROC | Fused AUROC |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7411 | 0.9281 | 0.7186 | 0.7643 |
| 2 | 0.7568 | 0.9287 | 0.7171 | 0.7387 |
| 4 | 0.7879 | 0.9283 | 0.7609 | 0.8131 |
| 8 | 0.7473 | 0.9343 | 0.7501 | 0.7144 |
| 16 | 0.7932 | 0.9304 | 0.7173 | 0.8173 |
| 32 | 0.8119 | 0.9298 | 0.7226 | 0.8282 |

Immediate pattern:

- `K = 32` is the best current patch run for both image AUROC and fused AUROC
- `K = 16` is also strong and only slightly behind `K = 32`
- `K = 4` is already clearly competitive
- `K = 8` is the unstable middle case and is the main negative outlier
- pixel AUROC is very stable regardless of K

### 3.2 Shared-K Comparison vs `full_redesign_stage2e70`

For the directly shared K values:

| K | Patch Image | Redesign Image | Patch Fused | Redesign Fused |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7411 | 0.8057 | 0.7643 | 0.7718 |
| 4 | 0.7879 | 0.7050 | 0.8131 | 0.7386 |
| 16 | 0.7932 | 0.6981 | 0.8173 | 0.7435 |

This is the cleanest direct comparison.

- redesign is better only at `K = 1`
- patch mode is substantially better at `K = 4`
- patch mode is substantially better at `K = 16`

### 3.3 Best-Run Comparison and Interpretation

The broader best-run comparison is also notable:

- best patch run in the tested patch family: `K = 32`, fused AUROC `0.8282`
- best redesign run in `full_redesign_stage2e70`: `K = 1024`, fused AUROC `0.8260`

That does not mean `patch K = 32` and `redesign K = 1024` are equivalent structures. It means that, as practical anomaly detectors in their own tested regimes, the current patch family is at least as strong and slightly stronger at its best point.

The patch family therefore does not tell the same story as the redesign family.

- redesign's most interpretable structure was at small K, especially `K = 4`
- patch mode's practical optimum inside the tested window shifts upward toward `K = 16` and `K = 32`
- the patch detector appears to benefit from a richer local anchor bank without needing the redesign family's very large K regime

## 4. Training Behavior

### 4.1 Stage-1 Behavior

`training_summary.json` shows the stage-1 stopping pattern.

| K | Stage-1 Epochs Trained | Best Stage-1 Epoch | Best Stage-1 Image AUROC |
| --- | ---: | ---: | ---: |
| 1 | 24 | 14 | 0.9103 |
| 2 | 18 | 8 | 0.7879 |
| 4 | 13 | 3 | 0.8287 |
| 8 | 14 | 4 | 0.8124 |
| 16 | 11 | 1 | 0.8456 |
| 32 | 11 | 1 | 0.8945 |

The important pattern is the same one seen in the redesign family, but it appears at a different point.

- small-to-mid K still improve for a few epochs
- larger patch K values (`16`, `32`) select the very first checkpoint
- because this family has no 20-epoch floor, it reveals that behavior directly instead of hiding it behind extra forced training time

That leads to a useful comparison with redesign:

- `full_redesign_stage2e70` forced stage-1 to continue to epoch 20 for `K > 1`, but the best checkpoint still stayed at epoch 1
- `patch_stage2e70` reaches the same failure mode for larger K without paying that extra runtime cost

So the underlying issue is the same: the monitored stage-1 image metric is not reliably aligned with the best longer-horizon multi-anchor solution.

### 4.2 Stage-2 Behavior

Stage-2 behavior must be read from `training_history.json`, not from `training_summary.json`.

| K | Logged Stage-2 Epochs | Best Stage-2 Epoch | Best Val Pixel-Agg AUROC |
| --- | ---: | ---: | ---: |
| 1 | 16 | 5 | 0.9202 |
| 2 | 14 | 3 | 0.9441 |
| 4 | 21 | 10 | 0.9505 |
| 8 | 17 | 6 | 0.9382 |
| 16 | 13 | 2 | 0.9341 |
| 32 | 14 | 3 | 0.9295 |

Important interpretation:

- the nominal 70-epoch stage-2 budget is mostly theoretical
- every run stops far earlier
- `K = 16` and `K = 32` peak very early
- `K = 4` is the clearest case where stage-2 keeps helping for a meaningful fraction of the run
- the larger-K patch runs do not look like cases where stage-2 is discovering a substantially better second representation over time

The simplest reading is that stage-2 is mostly calibrating a representation that was already decided by stage-1, especially at `K >= 16`.

## 5. Patch Assignment Signals: What They Mean

The patch family does not have the same cluster-diagnostics surface as the redesign retrospective, but `training_history.json` still records useful assignment statistics during stage-1 validation.

The most useful ones are:

- `effective_anchors_used`: how many anchors receive at least one validation assignment
- `assignment_entropy_normalized`: how spread the validation assignment histogram is
- `largest_anchor_share`: fraction of validation samples assigned to the busiest anchor

How to read them in patch mode:

- higher `effective_anchors_used` means the model is not collapsing to a tiny subset of anchors
- higher normalized entropy means assignments are more spread out overall
- lower `largest_anchor_share` means no single anchor dominates the validation split

But there is an important caveat:

- these are summaries of reduced dense patch-to-anchor behavior
- they are not the same as redesign's global semantic cluster story
- good assignment spread in patch mode is useful evidence against collapse, but it does not automatically mean the model discovered clean image-level semantic clusters

## 6. What the Available Signals Say

Representative stage-1 validation assignment traces look like this:

| K | Effective Anchors Used | Normalized Entropy | Largest Anchor Share |
| --- | ---: | ---: | ---: |
| 4 | 4 / 4 throughout | `0.91 - 0.97` | `0.34 - 0.42` |
| 16 | `12 - 15 / 16` | `0.80 - 0.89` | `0.14 - 0.20` |
| 32 | `21 - 26 / 32` | `0.80 - 0.86` | `0.12 - 0.19` |

This supports three conclusions.

1. The patch family is not suffering from one-anchor collapse.
2. Larger K does broaden anchor usage in the sense of a wider reference bank.
3. The signal is more consistent with a dense reference-bank detector than with a crisp global clustering story.

That is the key interpretive difference from `REDESIGN_RETROSPECTIVE.md`.

- redesign used diagnostics to reason about cluster geometry and nearest-vs-second-nearest separation
- patch mode's current signals are better for identifying collapse versus coverage than for proving clean cluster semantics

## 7. What Is Working

### 7.1 Patch mode is operationally viable

The patch family works as a real end-to-end experiment family.

- the runs complete successfully
- fixed pseudo-labels, capacitated assignment, and stage-2 anchor alignment all work together
- reconstruction-based pixel maps are available and stable
- the family produces strong image-level anomaly results without needing the redesign family's very large K regime

### 7.2 Moderate-to-larger K is useful in patch mode

Within the tested window, the practical best region is not tiny K.

- `K = 4` is already strong
- `K = 16` improves further
- `K = 32` is the best current patch result

That is a meaningful contrast with redesign, where the structural clustering success story was small-K while the best fused AUROC appeared only at very large K.

### 7.3 Pixel reconstruction is consistently strong

Pixel AUROC is stable across the family:

- `K = 1`: `0.9281`
- `K = 2`: `0.9287`
- `K = 4`: `0.9283`
- `K = 8`: `0.9343`
- `K = 16`: `0.9304`
- `K = 32`: `0.9298`

This means the reconstruction branch is not the limiting factor in the patch family. The main variation comes from image-level discrimination and from how well the anchor and fusion signals behave.

### 7.4 The tested patch runs do not show anchor-collapse behavior

The validation anchor-usage signals remain broad enough that the patch family is not collapsing into one dominant anchor.

- `K = 4` uses all anchors throughout validation
- `K = 16` typically uses `12 - 15` anchors during stage-1 validation
- `K = 32` typically uses `21 - 26` anchors during stage-1 validation

This is not the old collapse failure mode. Even at `K = 16` and `K = 32`, validation assignments are spread across many anchors rather than concentrating into one.

## 8. What Is Not Working

### 8.1 Stage-1 checkpoint selection is still the main unresolved issue

For `K = 16` and `K = 32`, the best stage-1 checkpoint is epoch 1.

That means:

- patch mode did not solve the stage-1 model-selection problem
- it only delayed the onset of the failure to a somewhat larger K than the redesign family's shared-K cases
- the current stage-1 monitored metric still does not capture the best patch-mode objective cleanly

### 8.2 The divergence signals are poor

The divergence numbers are the weakest part of the patch family.

| K | Divergence AUROC | Patch-Divergence AUROC |
| --- | ---: | ---: |
| 1 | 0.5403 | 0.5463 |
| 2 | 0.4731 | 0.5268 |
| 4 | 0.3115 | 0.2872 |
| 8 | 0.5140 | 0.4760 |
| 16 | 0.2970 | 0.2506 |
| 32 | 0.3809 | 0.2574 |

This is not a minor weakness. In several runs the divergence branch is actively anti-correlated.

That has two implications:

- the current fusion design is carrying a branch that is frequently near-random or harmful
- the fixed `divergence_weight: 0.3` is hard to defend as a general policy for patch mode

### 8.3 `K = 8` is unstable

`K = 8` is the main counterexample to a simple “more anchors are better” story.

- image AUROC falls to `0.7473`
- fused AUROC falls sharply to `0.7144`
- reconstruction and pixel behavior remain good
- divergence is weak and patch-divergence is below chance

That suggests the failure is not in the reconstruction branch. It is more likely in the interaction between the anchor score, the selected checkpoint, and the fixed fusion rule.

### 8.4 Stage-2 saturates early

Stage-2 helps, but it rarely behaves like a genuinely long second training phase.

- every run stops far before the 70-epoch ceiling
- larger-K runs peak after only a few stage-2 epochs
- the current frozen setup looks more like score calibration than deep refinement

This does not mean stage-2 is useless. It means the present stage-2 recipe is too constrained to exploit the full budget.

### 8.5 Patch mode has a weaker interpretability story than redesign

The redesign family could tell a cleaner cluster-geometry story in global anchor space. Patch mode cannot do that as cleanly.

The reason is structural:

- patch semantics begin with per-patch matching
- image-level scores are reduced from spatial behavior
- an image-level “assigned anchor” is therefore a summary statistic, not a clean global cluster label in the redesign sense

So patch mode can be the stronger detector while still being the weaker clustering narrative.

## 9. Why the Main Outcomes Make Sense

### 9.1 Why patch mode improves at `K = 16` and `K = 32`

Patch mode uses anchors more like a local reference bank than a global prototype set.

As K grows within a moderate range:

- the model gets finer local coverage of normal patterns
- samples can find better local matches without needing a huge global bank
- image-level anomaly scoring improves even when pixel AUROC stays mostly flat

This is a natural fit for patch semantics. The anchor bank is not being used to define a few crisp global modes. It is being used to cover normal local structure.

### 9.2 Why stage-2 saturates quickly

Stage-2 is heavily constrained:

- encoder is frozen
- anchors are frozen
- anchor target is frozen
- early stopping watches pixel-aggregated AUROC

Under that setup, stage-2 is not free to reshape the representation dramatically. It is mostly learning a reconstruction-side correction and score calibration. That explains why the larger-K runs peak early rather than improving steadily for dozens of epochs.

### 9.3 Why divergence is weak in patch mode

The divergence branch is the least natural fit in this family.

A plausible reason is that the main useful signal in patch mode is local matching plus reconstruction, while the frozen-bottleneck divergence path is not well aligned with the patch detector's real decision boundary. The data support that reading:

- divergence rarely helps strongly on its own
- patch-divergence is often worse than the plain divergence score
- the branch is weak exactly where the anchor branch is strongest

### 9.4 Why patch mode and redesign peak in different K regimes

The redesign family uses global centroid-style anchors and explicit cluster geometry. Patch mode uses local dense matching reduced to image-level scores.

Those are not the same operating mode.

- redesign needs very large K before the global reference bank becomes strong enough to maximize fused AUROC
- patch mode gains practical value earlier because local reference coverage improves already at moderate K

That is why `patch K = 32` can compete with or slightly beat `redesign K = 1024` without implying that they discovered equivalent structure.

## 10. Which Redesign Conclusions Still Hold

Several conclusions from `REDESIGN_RETROSPECTIVE.md` still hold under patch mode.

1. Stage-1 model selection is still the main unresolved training problem.
2. A 70-epoch stage-2 budget does not mean 70 useful epochs in practice because early stopping ends runs much sooner.
3. Good anomaly AUROC and clean clustering are not the same objective.
4. Keeping the repeller active and using fixed capacitated labels is operationally viable.

But some redesign conclusions do not transfer directly.

1. The “small-K is the main structural success story” conclusion does not describe the patch family well.
2. In patch mode, practical performance improves again at moderate-to-larger K inside the tested window.
3. The redesign family remains the stronger interpretability story, while patch mode is currently the stronger practical detector in the tested range.

## 11. Current Best Baselines

There is not one single “best” result across both families. It depends on the objective.

If the goal is best practical image-level anomaly detection in the current patch family:

- current best is `patch_stage2e70_k32`
- image AUROC: `0.8119`
- fused AUROC: `0.8282`

If the goal is best shared-K comparison against redesign:

- patch mode wins clearly at `K = 4` and `K = 16`
- redesign wins only at `K = 1`

If the goal is the cleaner cluster-geometry story:

- redesign still has the better narrative, especially at small K
- patch mode is better understood as a local reference-bank detector than as a global clustering method

## 12. Recommended Next Steps

### Priority 1: Change stage-1 checkpoint selection for patch mode

Do not continue using stage-1 `image_auroc` alone as the main selection signal for larger patch runs.

Recommended direction:

- monitor a patch-aware assignment quality signal alongside image AUROC
- candidates: effective anchors used, normalized entropy, largest-anchor share, or a small composite score
- keep image AUROC as a secondary guard rather than the only decision signal

### Priority 2: Revisit fusion when divergence is weak

The current divergence branch is often near-random or anti-correlated.

Recommended direction:

- gate or zero the divergence term when its validation AUROC is below `0.5`
- explicitly test anchor-plus-pixel fusion without the divergence branch
- keep the divergence path only if it earns its weight on validation

### Priority 3: Loosen stage-2 for larger-K patch runs

The current stage-2 setup looks too frozen to use the full 70-epoch budget meaningfully.

Recommended direction:

- test less-frozen stage-2 variants for `K >= 16`
- consider lower alignment pressure or a different early stopping signal
- evaluate whether `pixel_aggregated_image_auroc` is still the right selection metric if final fused image AUROC is the real target

### Priority 4: Probe the `K = 16` to `K = 32` region more densely

The practical optimum seems to be in the moderate-to-larger K range already tested.

Recommended direction:

- test `K = 24`, `K = 48`, and `K = 64`
- do not assume the current power-of-two ladder is optimal
- use the new runs to determine whether `K = 32` is a local optimum or just the current best checked point

### Priority 5: Add patch-specific diagnostics rather than borrowing the redesign lens unchanged

Patch mode needs analysis tools that respect per-patch assignment semantics.

Recommended direction:

- keep the current assignment spread metrics
- add patch-aware diagnostics that focus on dense reference coverage and spatial consistency
- avoid treating patch-mode image assignments as if they were global semantic clusters by default

## 13. Bottom Line

The patch stage-2 e70 family is a real success, but it is not a success for the same reasons as the redesign family.

It is strong because it behaves like an effective local reference-bank detector.

- image-level AUROC improves into the `K = 16` to `K = 32` range
- fused AUROC reaches `0.8282`, which is slightly above the best `full_redesign_stage2e70` result
- pixel-level reconstruction remains consistently strong

But the central training question is still open.

The current patch pipeline can achieve strong anomaly AUROC without showing a healthy stage-1 checkpoint-selection story and without getting much reliable value from the divergence branch. The strongest evidence for that is the combination of:

- epoch-1 best stage-1 checkpoints for `K = 16` and `K = 32`
- weak or anti-correlated divergence signals
- stage-2 runs that peak early and use only a small part of the nominal 70-epoch budget

So the next iteration should not just ask whether the score can go up. It should ask what the patch variant is supposed to be optimized for:

- best anomaly detection quality
- best patch-reference coverage with stable assignments
- or a deliberate compromise between practical detection and interpretability

Right now, the evidence favors patch mode as the stronger practical detector and redesign mode as the stronger interpretability baseline.