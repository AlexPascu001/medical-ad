# Thesis Alignment Notes

This file connects the reconstructed project history to the current thesis draft
under `D:\Documents\FMI\Disertatie\thesis\paper`.

## Current Thesis State

The current LaTeX draft has:

- `0-abstract.tex`: bilingual abstract already framing the work as an empirical
  investigation with results around `0.80 / 0.92` and PatchCore around
  `0.88 / 0.96`.
- `1-introduction.tex`: a strong introduction that states the research question,
  hypothesis, and contributions.
- `2-background.tex`: a mostly complete background chapter covering DINOv3,
  anomaly detection paradigms, CAM, BMAD, and objectives.
- `3-concluzii.tex`: only `\chapter{Concluzii}`.

Important mechanical flags:

- `main.tex` currently includes `\include{3-conclusion}`, but the present source
  file is `3-concluzii.tex`. There is a `3-conclusion.aux` from an earlier build,
  but no matching `3-conclusion.tex` source file in the listing.
- `bibliography.bib` still contains a placeholder Knuth citation and does not
  define the keys already used in the thesis draft: `dinov3`, `cam_loss`,
  `bmad`, `brats2021`, `dino`, `dinov2`, `deepsvdd`, `padim`, `patchcore`,
  `rd4ad`.
- The current draft/background describes the BMAD test set as balanced, but the
  local data and all full-test JSON metrics show `640` normal and `3075`
  anomalous test images.
- The introduction promises chapters for method, experimental setup, results,
  critical analysis, and conclusions, but only introduction/background/conclusion
  source files exist right now.

## Alignment With Project Evidence

The thesis framing is broadly right:

1. The work should be presented as an empirical investigation rather than a
   leaderboard/SOTA claim.
2. The core hypothesis was to adapt a CAM-style anchor geometry to one-class
   brain MRI anomaly detection using DINOv3 features.
3. The method produced meaningful but sub-baseline results.
4. The scientific contribution is the explanation: CAM's supervised anchor
   geometry only partially transfers to one-class AD.

The main refinement needed is historical precision:

- Some older high-scoring runs (`reproject`, `regfix`, `dual_bottleneck`) used
  trainable-backbone or two-stage variants and reached image AUROC around
  `0.854-0.855`.
- Later cleaner stage2e70, patch, and location-kmeans families are more
  defensible/documented but have lower raw image AUROC around `0.80-0.812`, with
  fused scores around `0.826-0.8295`.
- PatchCore with frozen DINOv3 is the strongest local baseline at `0.8837`
  image AUROC and `0.9612` pixel AUROC.
- The "fully fine-tuning DINOv3 degrades features" interpretation should be
  phrased carefully. It is plausible and supported by DINOv3's frozen-feature
  premise, but the later clean configs increasingly use frozen backbones. A safer
  thesis claim is: early trainable/reproject variants and later frozen variants
  suggest that the bottleneck/anchor objective, not just backbone size, is the
  limiting factor; fully fine-tuning remains a risk and a historical design
  choice to analyze, not the sole explanation for every final result.

## Suggested Thesis Body

Use the reconstruction artifacts as source scaffolding:

- `PROJECT_HISTORY_RECONSTRUCTION.md`: narrative research arc.
- `CHRONOLOGICAL_EVIDENCE_TIMELINE.md`: date/evidence chronology.
- `EXPERIMENT_INVENTORY.md`: tables of full-test metrics.
- `THESIS_RESULTS_VERIFICATION.md`: verified headline table and comparability
  notes for thesis results.
- `THESIS_RESEARCH_STORY.md`: concise thesis-ready story of the hypothesis,
  research path, results, explanations, and final claim.

Recommended body structure:

| Chapter | Purpose | Evidence To Use |
| --- | --- | --- |
| Method | Describe global anchor CAM adaptation, two-stage/fusion, patch variant, and location-kmeans. | `PROJECT_OVERVIEW.md`, `ARCHITECTURE.md`, `PIPELINE.md`, current code/configs. |
| Experimental Setup | BMAD/BraTS split, DINOv3 variants, metrics, full-test filtering, PatchCore baseline. | `EXPERIMENT_INVENTORY.md`, configs, BMAD paper. |
| Results | Main results table: PatchCore, best older trained runs, clean redesign, patch, location-kmeans. | `EXPERIMENT_INVENTORY.md`. |
| Analysis | Explain anchor count, repeller, projection drift, reconstruction/fusion fragility, locality turn, PatchCore gap. | `PROJECT_HISTORY_RECONSTRUCTION.md`, `PERFORMANCE_ANALYSIS.md`, `STAGE2E70_FAMILY_ANALYSIS.md`, `PATCH_LOCATION_KMEANS_FAMILY_ANALYSIS.md`. |
| Conclusions | Honest outcome: decent results, below baseline, useful lessons for one-class AD with SSL features. | Thesis notes and reconstruction summary. |

## Claims That Are Safe

- The adapted anchor method did not beat the frozen DINOv3 PatchCore baseline.
- K=1 is consistently a strong raw image detector in global-anchor families.
- Multi-anchor/high-K regimes often help fusion or auxiliary signals more than
  they help raw anchor AUROC.
- Reconstruction-derived pixel maps give high pixel AUROC, but BMAD's brain-MRI
  background caveat means pixel AUROC should not be overclaimed without PRO.
- Divergence is fragile: it is often weak, below chance, or useful only in
  carefully selected fusion regimes.
- Patch-level/local methods fit the tumor-locality assumption better than a
  pure CLS/global anchor score, but they still do not surpass PatchCore.

## Claims To Handle Carefully

- Do not imply all final runs fully fine-tuned DINOv3. The clean late-stage
  families appear to use frozen backbones in many configs.
- Do not present the older `~0.855` reproject/regfix runs as the final main
  method until their configs/splits/scoring are verified against the final
  evaluation path.
- Do not treat high pixel AUROC as proof of precise localization without PRO or
  qualitative masks.
- Do not claim location-kmeans clearly dominates patch mode. It narrowly improves
  fused AUROC in one late configuration, while raw anchor/image AUROC remains
  lower than `patch_stage2e70_k32`.

## Immediate Thesis TODOs

1. Fix the `main.tex` include/file-name mismatch for the conclusion chapter.
2. Replace the placeholder bibliography with real entries for all cited works.
3. Correct the experimental setup text to use the local split counts
   (`7500` train normal, `39/44` validation, `640/3075` test).
4. Add source files for method, experimental setup, results, and analysis.
5. Choose one headline results table and keep the rest in annexes.
6. Add a short limitations section that explicitly discusses PRO, frozen-feature
   baselines, and why a negative/mixed result is still a valid research outcome.
