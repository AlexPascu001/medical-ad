# Thesis Contrastive Constructions Review

This note audits the current thesis prose for repeated antithesis-style
constructions: "not X, but Y", "while", "whereas", "however", "rather than",
"instead of", "no longer", and related patterns. The goal is not to remove every
contrast. Many are scientifically useful. The goal is to reduce the repeated
defensive rhythm and make the dissertation read more directly and confidently.

Line numbers refer to the current LaTeX files in
`D:\Documents\FMI\Disertatie\thesis\paper` at the time of this review.

## Highest-Priority Edits

These are the places most likely to affect the tone of the thesis.

| Location | Current construction | Why revise | Suggested reformulation |
| --- | --- | --- | --- |
| `0-abstract.tex:25` | "not only to rank methods, but also to understand..." | The abstract starts to sound like it is justifying the project. | "The comparison ranks the methods and clarifies how anchor, reconstruction, divergence, and patch-based signals contribute to detection." |
| `0-abstract.tex:29` | "Reconstruction is useful..., while divergence..." | The contrast is valid, but the abstract can state the finding more smoothly. | "Reconstruction provides useful localisation evidence; divergence is less consistent and is best interpreted as an indirect auxiliary signal." |
| `0-abstract.tex:65` | "nu doar..., ci si..." | The Romanian abstract mirrors the slightly stiff English contrast. | "Comparatia ierarhizeaza metodele si clarifica rolul semnalelor bazate pe ancore, reconstructie, divergenta si patch-uri." |
| `0-abstract.tex:70` | "in timp ce divergenta..." | Sounds translated and a bit mechanical. | "Divergenta este mai putin consecventa si necesita o interpretare mai prudenta, ca semnal auxiliar indirect." |
| `1-introduction.tex:25` | "normal anatomy... whereas abnormalities..." | Good idea, but the "whereas" pattern appears often later too. | "Normal anatomy is comparatively well defined and abundant. Abnormal findings are heterogeneous, rare, and effectively open-ended." |
| `1-introduction.tex:34` | "not to classify..., but to raise..." | Strong "not X, but Y" framing in the opening motivation. | "The objective of medical anomaly detection is to raise an alert when an image departs from the learned notion of normality; closed-set disease labels are outside this formulation." |
| `1-introduction.tex:46` | "easy to obtain while..." | Another broad contrast in the same motivation arc. | "Disease-free examples are relatively easy to obtain. A comprehensive abnormal training set is much harder to assemble, because pathology is diverse and long-tailed." |
| `1-introduction.tex:73` | "and, where they do not, what..." | The research question ends with a negative premise. | "The experiments also analyse the conditions that shape their performance on brain MRI anomaly detection and localisation." |
| `2-background.tex:55` | "At the same time, the features are learned..." | Correct caveat, but it can sound like a warning label. | "Because the features are learned on natural-image data, their behaviour on medical images remains an empirical question for this thesis." |
| `2-background.tex:74` | "reconstruct normal inputs accurately but fail..." | Standard reconstruction premise, but the "but fail" phrasing is abrupt. | "A model trained on normal data is expected to reconstruct normal inputs accurately; anomalous structures should produce larger reconstruction errors when they are poorly represented by the learned normal manifold." |
| `2-background.tex:182` | "does not make pixel AUROC uninformative, but..." | Defensive phrasing around evaluation. | "Pixel AUROC remains useful, and the BMAD discussion motivates pairing it with qualitative maps and, where available, region-level metrics such as PRO." |
| `2-background.tex:191` | "DINOv3 provides..., while CAM..." | The objectives-in-context section leans heavily on parallel contrast. | "The thesis combines two ingredients: DINOv3 as a source of dense visual representations and CAM as a prototype-based metric-learning objective." |
| `2-background.tex:195` | "whereas medical anomaly detection..." | Strong domain mismatch wording. | "This transfer changes the role of the anchors, because the BMAD setting provides only normal training images and no semantic tumour classes." |
| `3-method.tex:8` | "anchors cannot represent disease categories; instead..." | Technically true, but it repeats the limitation-first framing. | "Consequently, anchors represent prototypes of the normal feature distribution, obtained by clustering DINOv3 features from normal training images." |
| `3-method.tex:12` | "empirical adaptation rather than direct transfer" | Good framing, but slightly defensive. | "The method is an empirical adaptation of CAM to a one-class setting, with design choices chosen to test whether DINOv3 features can be organised around normal anchors." |
| `3-method.tex:89` | "normal sample should..., whereas an anomalous..." | Fine scientifically, but a repeated rhetorical contrast. | "In this formulation, normal samples are expected to lie near at least one normal prototype; larger nearest-anchor distances indicate stronger evidence of abnormality." |
| `3-method.tex:110` | "instead of separating semantic categories..." | Another direct negation of the original CAM role. | "In the one-class setting, the term acts as a geometric regulariser between normal prototypes." |
| `3-method.tex:174` | "Fusion does not guarantee improvement..." | Accurate, but phrased as a warning. | "Fusion is therefore evaluated empirically, because each branch can contribute differently after normalisation." |
| `3-method.tex:181` | "However, this approach may be insufficient..." | Starts the patch section by weakening the global model. | "Local anomalies motivate a patch-level variant of the global centroid-anchor model." |
| `3-method.tex:199` | "while PatchCore remains..." | The method chapter closes by re-centering PatchCore superiority. | "The results position PatchCore as the strongest reference point and the trained variants as a structured analysis of anchor, reconstruction, fusion, and locality." |
| `4-experiments.tex:4` | "not only to identify..., but also..." | Results chapter opens with a stock antithesis. | "This chapter identifies the best-performing configurations and analyses how each design choice changes system behaviour." |
| `4-experiments.tex:79` | "slightly higher mean..., while also..." | Fine content; use simpler cumulative phrasing. | "Across these trials, k-means achieved the highest single run, a slightly higher mean image AUROC, and the lowest variance across seeds." |
| `4-experiments.tex:109` | "$K=1$..., while $K=1024$..." | The contrast is useful; make it a direct result statement. | "The sweep is not monotonic: $K=1$ gives the strongest raw image score, and $K=1024$ gives the strongest fused score." |
| `4-experiments.tex:210` | "However, it lowers..." | The caveat is fair, but the phrasing flattens the positive result. | "At the same time, the raw image AUROC is lower than the best shared patch-anchor run, so the gain is clearest in fused and pixel-level metrics." |
| `4-experiments.tex:212` | "not... decisive improvement, but..." | Very explicit downplaying. | "The result supports spatially constrained patch references as a promising direction whose benefit depends on scoring and fusion." |
| `4-experiments.tex:272` | "PatchCore remains..., while..." | Useful comparison, but the figure caption can be more affirmative. | "PatchCore is the strongest reference baseline; the trained anchor variants provide complementary image-level and pixel-level evidence, especially after moving from global to patch-based scoring." |
| `5-discussion.tex:6` | "At the same time, PatchCore..." | The opening paragraph quickly pivots to the stronger baseline. | "PatchCore provides the strongest reference result in this experiment set, and the following analysis uses it to understand the behaviour of the trained variants." |
| `5-discussion.tex:21` | "no longer merely reads...; it reshapes..." | Dramatic contrast. | "When the backbone is trainable, the loss reshapes the DINOv3 space according to normal-data anchor assignments." |
| `5-discussion.tex:24` | "can enforce..., but it cannot..." | Valid caveat, but this pattern recurs often. | "The objective enforces compactness and anchor structure on normal examples; tumour-specific separation is therefore inferred indirectly at test time." |
| `5-discussion.tex:29` | "PatchCore reaches..., while..." | Important result, but it reads as a deficit statement. | "PatchCore reaches $0.8837$ image AUROC. The main trained anchor families reach fused image AUROC values around $0.83$, showing useful signal with a remaining gap to the frozen-feature reference." |
| `5-discussion.tex:42` | "not to true classes, but to..." | Strong antithesis in the CAM repeller explanation. | "When $K>1$, the anchors correspond to clusters within the normal distribution rather than semantic disease classes." |
| `5-discussion.tex:88` | "However, reconstruction is not..." | Could sound like a correction after praise. | "Its strongest role is localisation and auxiliary fusion, rather than standalone image-level ranking." |
| `5-discussion.tex:101` | "localisation, while the anchor..." | Valid, but smooth it into a hierarchy. | "Reconstruction supports localisation and improves some fused results; the anchor score remains the more direct image-level detector." |
| `5-discussion.tex:123` | "reconstruction-derived signals are useful, while divergence..." | Good summary, but repeated two-column contrast. | "The main patch and location-aware families support a measured interpretation: reconstruction-derived signals are useful, and divergence is less consistent." |
| `5-discussion.tex:138` | "but the gains are measured rather than dramatic" | Slightly apologetic wording. | "The patch experiments support this reasoning with measured gains in the final structured families." |
| `5-discussion.tex:163` | "While the method does not achieve state-of-the-art..." | This is the most visible "not SOTA" sentence. | "Although PatchCore remains stronger on this benchmark, this work contributes a systematic empirical analysis of CAM-style one-class anchor learning with DINOv3 features." |
| `5-discussion.tex:171` | "Several limitations should be stated explicitly." | Sounds like a formal warning. | "The interpretation of the results has several boundaries." |
| `5-discussion.tex:187` | "preserve... while adding..." | Reasonable, but can be more recommendation-like. | "Future anchor methods should preserve as much pretrained patch geometry as possible and add task-specific structure where the experiments show a benefit." |
| `6-conclusion.tex:19` | "analytical rather than leaderboard-oriented" | True, but it can still read as a defensive thesis label. | "The main conclusion is analytical: DINOv3 features are highly valuable for brain MRI anomaly detection, and class-anchor learning provides an interpretable way to organise normal-data prototypes." |
| `6-conclusion.tex:22` | "However, the assumptions..." | Strong pivot in the conclusion. | "The experiments also show that CAM assumptions change in the one-class setting." |

## Additional Candidates

These can stay if the surrounding paragraph needs explicit contrast, but they
are worth smoothing if you do a final prose pass.

| Location | Current construction | Suggested lighter formulation |
| --- | --- | --- |
| `2-background.tex:43` | "may deteriorate even while global benchmarks keep improving" | "may deteriorate despite improvements on global benchmarks" |
| `2-background.tex:101` | "high-level structure rather than pixel fidelity" | "high-level structure, with less emphasis on exact pixel fidelity" |
| `2-background.tex:154` | "of the representation itself rather than..." | "of the representation itself, independently of a specific supervised task" |
| `3-method.tex:49` | "CLS token..., while the patch variant..." | "The global variant uses the CLS token; the patch variant uses the dense patch-token grid." |
| `3-method.tex:155` | "Reconstruction provides..., while divergence..." | "Reconstruction provides useful pixel-level evidence and can improve fused scores. Divergence is more indirect and configuration-dependent." |
| `3-method.tex:182` | "while still using anchors..." | "and retains anchors prepared from whole-image clustering" |
| `3-method.tex:191` | "preserve some spatial context while retaining..." | "preserve spatial context and retain the feature-distance principle" |
| `4-experiments.tex:36` | "whether... sensitive..., while the main..." | "These runs check sensitivity to backbone capacity; the main reported families remain..." |
| `4-experiments.tex:47` | "whereas the patch families..." | "The patch families use earlier stopping without this minimum." |
| `4-experiments.tex:48` | "while keeping the encoder..." | "with the encoder, anchors, anchor targets, and bottleneck kept frozen" |
| `4-experiments.tex:116` | "while keeping this protocol fixed" | "under the same two-stage training protocol" |
| `4-experiments.tex:134` | "rather than global changes" | "because tumours are spatially local changes to a slice" |
| `4-experiments.tex:157` | "$K=16$ close behind, while..." | "$K=16$ is close behind; $K=8$ is a clear negative outlier." |
| `4-experiments.tex:159` | "rather than asking a global CLS..." | "which avoids relying only on a global CLS representation for small lesions" |
| `4-experiments.tex:167` | "while fusion combines..." | "and fusion combines the useful cues" |
| `4-experiments.tex:177` | "while also showing some response..." | "with additional responses on anatomical boundaries and high-contrast structures" |
| `4-experiments.tex:233` | "does not train... and does not use..." | "uses frozen DINOv3 features and nearest-neighbour memory-bank scoring, without the adapted class-anchor objective" |
| `5-discussion.tex:68` | "may no longer describe..." | "may become misaligned with the current feature geometry" |
| `5-discussion.tex:74` | "In this one-class clustered setting, however..." | "In this one-class clustered setting, the pseudo-labels are geometric artefacts." |
| `5-discussion.tex:93` | "anatomy rather than pathology" | "anatomical complexity more strongly than pathology" |
| `5-discussion.tex:103` | "rather than as the sole measure..." | "as part of a broader localisation assessment" |
| `5-discussion.tex:110` | "below chance, while..." | "below chance; anchor and reconstruction-related scores remain useful" |
| `5-discussion.tex:117` | "do not consistently align..." | "align inconsistently with anomaly labels" |
| `5-discussion.tex:152` | "methods are very competitive, while..." | "methods are very competitive; reconstruction-heavy branches are less reliable in these experiments" |
| `5-discussion.tex:160` | "rather than as true semantic..." | "as geometric structure inside the normal class, not as semantic class separation" |
| `5-discussion.tex:172` | "although BMAD recommends..." | "BMAD recommends PRO for localisation in domains with large uniform backgrounds, so this remains a useful future addition." |
| `5-discussion.tex:175` | "rather than full three-dimensional..." | "on two-dimensional slices, leaving inter-slice context for future work" |
| `5-discussion.tex:190` | "rather than introduce..." | "and avoid artificial separation that fragments the normal manifold" |
| `6-conclusion.tex:44` | "while removing..." | "and address the mismatched assumptions identified by the experiments" |

## Patterns To Watch In The Final Prose Pass

1. **Repeated caveat pivots.** Sentences beginning with "However", "At the same
   time", and "While..." are useful sparingly, but several consecutive sections
   use them to move from a positive result to a limitation.

2. **Negative-first explanations.** Phrases such as "cannot", "does not",
   "not to X, but to Y", and "not state-of-the-art" often state a true boundary,
   but the thesis usually reads better when the positive formulation comes
   first.

3. **PatchCore comparisons.** PatchCore should remain the strongest reference
   point, but it does not need to appear as a contrast in every interpretive
   paragraph. Some comparisons can be grouped in the results table, the overall
   comparison paragraph, and the discussion opening.

4. **"While" overload.** Many "while" clauses are harmless technical
   comparisons. When they cluster, split them into two sentences or replace them
   with "and", "with", or a direct cumulative result statement.

5. **Romanian abstract translation feel.** The Romanian abstract should avoid
   mirroring English antitheses too literally. Shorter Romanian sentences with
   direct verbs will sound more natural.

