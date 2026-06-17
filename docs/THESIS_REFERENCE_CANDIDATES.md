# Thesis Reference Candidates

Date: 2026-06-14

Scope: Chapters 1--5 of `D:\Documents\FMI\Disertatie\thesis\paper`.

This file proposes real references that could be added to the thesis bibliography and inserted into specific places in the text. It does not apply the citations yet. The goal is to let you approve each candidate before we change the paper.

## Validation Protocol

- Current bibliography method: external `bibliography.bib` via `biblatex`.
- Current citation keys were already cross-referenced against `bibliography.bib`; no missing cited keys were found in the previous bibliography pass.
- Candidate references below were checked against arXiv, DOI pages, publisher pages, or an independent validation agent.
- A separate validation judge checked the main candidate set for fabrication risk and relevance. It did not reject any as fake, but it marked several as narrower-use references.
- Paperpile validation from the `bib-validate` skill was not run because the local Paperpile CLI/library was not available in this workspace.

## High-Confidence Core Additions

These are the references I would most confidently add because they directly ground claims already present in the thesis.

| Proposed key | Reference | Suggested placement | Why it fits | Validation |
|---|---|---|---|---|
| `chandola2009survey` | Varun Chandola, Arindam Banerjee, Vipin Kumar. "Anomaly Detection: A Survey." ACM Computing Surveys, 2009. DOI: `10.1145/1541880.1541882`. | Chapter 1, lines 18--34; Chapter 2, lines 61--66. | Grounds the general anomaly detection definition, taxonomy, and normal-behaviour modelling framing. | Judge: strong. DOI confirmed. |
| `pang2021deepad` | Guansong Pang, Chunhua Shen, Longbing Cao, Anton van den Hengel. "Deep Learning for Anomaly Detection: A Review." ACM Computing Surveys, 2021. DOI: `10.1145/3439950`; arXiv: `2007.02500`. | Chapter 2, lines 59--100; Chapter 5, lines 147--153. | Grounds the taxonomy of deep anomaly detection families and their assumptions/limitations. | Judge: strong. arXiv and ACM DOI confirmed. |
| `fernando2021medicalad` | Tharindu Fernando, Harshala Gammulle, Simon Denman, Sridha Sridharan, Clinton Fookes. "Deep Learning for Medical Anomaly Detection -- A Survey." arXiv: `2012.02364`, 2021. | Chapter 1, lines 22--34; Chapter 2, lines 59--100; Chapter 5, lines 147--153. | Medical-domain survey for the broad medical AD framing. Useful if you want a domain-specific citation in addition to Pang/Chandola. | arXiv confirmed. Scope is survey/preprint. |
| `shurrab2022sslmedical` | Saeed Shurrab, Rehab Duwairi. "Self-supervised learning methods and applications in medical imaging analysis: A survey." PeerJ Computer Science, 2022. DOI: `10.7717/peerj-cs.1045`; arXiv: `2109.08685`. | Chapter 1, lines 51--55; Chapter 2, lines 13--19 and 53--57. | Supports the claim that annotated medical data are scarce and SSL is useful in medical imaging. | Judge: strong. arXiv and DOI confirmed. |
| `zhou2019modelsgenesis` | Zongwei Zhou et al. "Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis." MICCAI, 2019. arXiv: `1908.06912`. | Chapter 2, lines 53--57; possibly Chapter 5, lines 171--176. | Grounds SSL/autodidactic medical representation learning and the natural-image-to-medical domain-shift concern, especially for 3D medical imaging. | Judge: strong. arXiv confirmed; use narrowly for medical SSL/domain transfer, not DINO specifically. |
| `dosovitskiy2021vit` | Alexey Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR, 2021. arXiv: `2010.11929`. | Chapter 2, lines 30--35; Chapter 3, lines 35--42. | Grounds ViT patch-token/CLS-token architecture before discussing DINOv3 tokens. | Judge: strong. arXiv confirmed. Do not use as DINOv3 evidence. |
| `baur2020autoencoders` | Christoph Baur, Stefan Denner, Benedikt Wiestler, Shadi Albarqouni, Nassir Navab. "Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study." arXiv: `2004.03271`, 2020. | Chapter 2, lines 70--79; Chapter 3, lines 128--131; Chapter 5, lines 90--103. | Very relevant: brain MRI UAD, reconstruction of normal anatomy, comparison difficulties, domain shift, and reconstruction challenges. | Judge: strong. arXiv confirmed. |
| `scholkopf2001support` | Bernhard Schoelkopf, John C. Platt, John Shawe-Taylor, Alex J. Smola, Robert C. Williamson. "Estimating the Support of a High-Dimensional Distribution." Neural Computation, 2001. DOI: `10.1162/089976601750264965`. | Chapter 2, lines 83--89; Chapter 3, lines 28--31. | Canonical one-class support estimation reference; grounds the idea of learning support/normality from normal examples. | Judge: strong. DOI confirmed. |
| `fawcett2006roc` | Tom Fawcett. "An Introduction to ROC Analysis." Pattern Recognition Letters, 2006. DOI: `10.1016/j.patrec.2005.10.010`. | Chapter 2, lines 170--173; Chapter 4, lines 21--24. | Grounds ROC/AUROC definitions and use for score-based evaluation. | Judge: strong. DOI/publisher page confirmed. |
| `saito2015pr` | Takaya Saito, Marc Rehmsmeier. "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." PLOS ONE, 2015. DOI: `10.1371/journal.pone.0118432`. | Chapter 4, lines 21--29; optionally Chapter 2, lines 170--180. | Grounds AUPR and the caution around imbalanced evaluation. It is directly useful because BMAD test labels are imbalanced. | PLOS DOI page confirmed. This may be more immediately useful than Davis & Goadrich for the thesis wording. |

## Useful but Narrower Additions

These are real and relevant, but I would add them only where the text makes the narrower point.

| Proposed key | Reference | Suggested placement | Why it fits | Validation |
|---|---|---|---|---|
| `schlegl2017anogan` | Thomas Schlegl, Philipp Seeboeck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs. "Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery." IPMI, 2017. arXiv: `1703.05921`. | Chapter 2, lines 70--79. | Classic medical UAD/generative normal-manifold reference. | Judge: strong, but retina/OCT rather than brain MRI. Use for medical UAD history, not brain-specific claims. |
| `tian2023memmcmae` | Yu Tian et al. "Unsupervised Anomaly Detection in Medical Images with a Memory-augmented Multi-level Cross-attentional Masked Autoencoder." MICCAI MLMI, 2023. arXiv: `2203.11725`. | Chapter 2, lines 75--79; Chapter 5, lines 90--103. | Directly supports the statement that reconstruction methods can give low reconstruction error even on anomalous images, and that masking/memory are mitigation strategies. | Judge: usable. arXiv confirmed. Not brain MRI; use narrowly. |
| `lloyd1982kmeans` | Stuart P. Lloyd. "Least squares quantization in PCM." IEEE Transactions on Information Theory, 1982. DOI: `10.1109/TIT.1982.1056489`. | Chapter 3, lines 57--60; Chapter 4, lines 54--58. | Grounds Lloyd/k-means centroids as data-driven prototypes. | Judge: usable. DOI confirmed. For the exact term "k-means", MacQueen 1967 is historically earlier, but Lloyd is a standard algorithm citation. |
| `turk1991eigenfaces` | Matthew Turk, Alex Pentland. "Eigenfaces for Recognition." Journal of Cognitive Neuroscience, 1991. DOI: `10.1162/jocn.1991.3.1.71`. | Chapter 4, lines 54--58 and 75--79, around the eigenface/PCA-style anchor baseline. | Grounds the eigenface/PCA-style anchor strategy historically. | Judge: usable. DOI confirmed. Not anomaly-specific. |
| `fastflow2021` | Jiawei Yu et al. "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows." arXiv: `2111.07677`, 2021. | Chapter 2, lines 94--96. | Grounds the normalizing-flow method family mentioned in the background. | Judge: usable. arXiv confirmed. Industrial/MVTec context, not medical brain MRI. |
| `chalapathy2019survey` | Raghavendra Chalapathy, Sanjay Chawla. "Deep Learning for Anomaly Detection: A Survey." arXiv: `1901.03407`, 2019. | Chapter 2, lines 59--100, if you want another survey. | Broad deep AD survey; useful, but mostly redundant if `pang2021deepad` is included. | Judge: usable. arXiv confirmed. |
| `davis2006prroc` | Jesse Davis, Mark Goadrich. "The Relationship Between Precision-Recall and ROC Curves." ICML, 2006. DOI: `10.1145/1143844.1143874`. | Chapter 4, lines 21--29, as an alternative or companion to `saito2015pr`. | Foundational PR/ROC relationship reference. | Judge: strong. For the thesis's imbalanced-data wording, `saito2015pr` is more direct. |

## Candidate Insertions by Thesis Location

The snippets below show the kind of citation placement I would suggest. They avoid changing the wording much.

### Chapter 1

- `1-introduction.tex:18--26`
  - Current claim: anomaly detection identifies deviations from normality; medical anomalies are heterogeneous, rare, and open-ended.
  - Candidate citations: `chandola2009survey`, `fernando2021medicalad`, optionally `pang2021deepad`.
  - Suggested citation placement: after "normal distribution" or after "open-ended".

- `1-introduction.tex:28--34`
  - Current claim: supervised classifiers are poorly suited when the negative class cannot be enumerated.
  - Candidate citations: `chandola2009survey`, `fernando2021medicalad`.
  - Reason: grounds the motivation for one-class/unsupervised AD.

- `1-introduction.tex:38--44`
  - Current claim: one-class training uses only normal images and scores deviations at inference.
  - Candidate citations: `scholkopf2001support`, `chandola2009survey`, `baur2020autoencoders`.
  - Reason: `scholkopf2001support` grounds support estimation; `baur2020autoencoders` grounds the medical brain MRI version.

- `1-introduction.tex:51--55`
  - Current claim: annotated medical data is scarce; SSL features are attractive.
  - Candidate citations: `shurrab2022sslmedical`, `zhou2019modelsgenesis`.
  - Reason: both support SSL for medical-image representation learning and annotation scarcity.

### Chapter 2

- `2-background.tex:13--19`
  - Current claim: SSL learns from unlabelled data and transfers to downstream tasks.
  - Candidate citations: `shurrab2022sslmedical`, `zhou2019modelsgenesis`, optionally `dosovitskiy2021vit` for transformer transfer.

- `2-background.tex:30--35`
  - Current claim: DINOv3/ViT produces CLS and patch tokens.
  - Candidate citation: `dosovitskiy2021vit`.
  - Reason: DINOv3 is already cited, but ViT is the architectural source for patch-token framing.

- `2-background.tex:53--57`
  - Current claim: medical anomaly detection benefits from transferable features but natural-image features introduce domain shift.
  - Candidate citations: `shurrab2022sslmedical`, `zhou2019modelsgenesis`.

- `2-background.tex:61--66`
  - Current claim: one-class AD learns normality from normal samples and groups into reconstruction/feature-based families.
  - Candidate citations: `chandola2009survey`, `pang2021deepad`, `fernando2021medicalad`.

- `2-background.tex:70--79`
  - Current claim: reconstruction-based methods reconstruct normal images but expressive models may reconstruct anomalies too.
  - Candidate citations: `baur2020autoencoders`, `tian2023memmcmae`, optionally `schlegl2017anogan`.

- `2-background.tex:83--96`
  - Current claim: one-class classifiers, memory-bank methods, teacher-student methods, and normalizing flows.
  - Candidate citations: `scholkopf2001support` for one-class support estimation; `fastflow2021` for normalizing flows. PaDiM/PatchCore/RD4AD are already cited.

- `2-background.tex:104--107`
  - Current claim: patch-score aggregation should avoid diluting small anomalies.
  - Candidate citation: probably keep with existing PatchCore/PaDiM context unless adding a specific aggregation paper. I would not force a new citation here without a stronger direct source.

- `2-background.tex:170--180`
  - Current claim: AUROC/pixel AUROC/PRO and localisation metric caution.
  - Candidate citations: `fawcett2006roc`, `saito2015pr`, existing `bmad`.

### Chapter 3

- `3-method.tex:28--31`
  - Current claim: no tumour images are used; normality is learned from geometry of normal examples.
  - Candidate citation: `scholkopf2001support` or `chandola2009survey`.

- `3-method.tex:35--42`
  - Current claim: ViT CLS/patch/register-token feature extraction.
  - Candidate citation: `dosovitskiy2021vit` for CLS/patch token basics, with DINOv3 left as the register-token/model-specific citation.

- `3-method.tex:57--60`
  - Current claim: k-means centroids define normal anchors.
  - Candidate citation: `lloyd1982kmeans`.

- `3-method.tex:128--131`
  - Current claim: decoder trained only on normal anatomy may reconstruct anomalies poorly.
  - Candidate citations: `baur2020autoencoders`, `tian2023memmcmae`.

### Chapter 4

- `4-experiments.tex:21--29`
  - Current claim: AUROC/AUPR/pixel metrics and imbalance/localisation caution.
  - Candidate citations: `fawcett2006roc`, `saito2015pr`, existing `bmad`.

- `4-experiments.tex:54--58`
  - Current claim: random, eigenface/PCA-style, and k-means anchors.
  - Candidate citations: `turk1991eigenfaces`, `lloyd1982kmeans`.

- `4-experiments.tex:124--129`
  - Current claim: tumours are spatially local, motivating patch-level scoring.
  - Candidate citations: existing `bmad`/BraTS context may be enough. I would avoid adding a generic citation unless we add a tumour segmentation/localisation paper.

### Chapter 5

- `5-discussion.tex:13--19`
  - Current claim: PatchCore with frozen dense features is a strong reference.
  - Candidate citations: existing `patchcore`, existing `dinov3`; no extra citation needed unless adding DINOv2/PatchCore variants.

- `5-discussion.tex:90--103`
  - Current claim: reconstruction can fail because decoders reconstruct anomalies or respond to anatomy/edges rather than pathology.
  - Candidate citations: `baur2020autoencoders`, `tian2023memmcmae`.

- `5-discussion.tex:128--132`
  - Current claim: local tumours may not dominate global representations, motivating patch features.
  - Candidate citations: existing BMAD/BraTS context is likely sufficient; consider `patchcore` only if phrased as patch-level nearest-neighbour anomaly scoring.

- `5-discussion.tex:149--153`
  - Current claim: feature-based methods are strong while reconstruction-heavy branches are less reliable.
  - Candidate citations: `bmad`, `pang2021deepad`, `baur2020autoencoders`, possibly `fernando2021medicalad`.

## Suggested Priority Order

If you want a clean thesis rather than a maximal bibliography, I would add these first:

1. `chandola2009survey`
2. `pang2021deepad`
3. `fernando2021medicalad`
4. `shurrab2022sslmedical`
5. `zhou2019modelsgenesis`
6. `dosovitskiy2021vit`
7. `baur2020autoencoders`
8. `scholkopf2001support`
9. `fawcett2006roc`
10. `saito2015pr`
11. `lloyd1982kmeans`
12. `turk1991eigenfaces`

Then add narrower references only if you want more complete method-family coverage:

- `tian2023memmcmae`
- `schlegl2017anogan`
- `fastflow2021`
- `davis2006prroc`
- `chalapathy2019survey`

## Notes Before Adding

- Do not cite `dosovitskiy2021vit` as evidence for DINOv3 quality; use it only for ViT architecture and patch-token representation.
- Do not cite `fastflow2021` as PatchCore evidence; it is for normalizing-flow background only.
- `baur2020autoencoders` is probably the most thesis-relevant new medical reconstruction reference because it is directly about brain MRI UAD.
- `saito2015pr` is more directly tied to imbalanced evaluation than Davis and Goadrich, although both are real and useful.
- When these are added to `bibliography.bib`, the file should keep `sorting=none` and entries should be inserted in first-citation order so numbering remains by first appearance.
