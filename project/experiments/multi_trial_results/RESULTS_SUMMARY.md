# Multiple Trial Experiment Results

**Total Trials:** 15
**Strategies:** eigenface, kmeans, random
**Trials per Strategy:** 5

## Summary Statistics

### Image-Level AUROC

| Strategy | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| Eigenface  | 0.7893 | 0.0192 | 0.7658 | 0.8162 |
| Kmeans     | 0.8118 | 0.0112 | 0.7967 | 0.8214 |
| Random     | 0.8149 | 0.0154 | 0.7979 | 0.8358 |

### Pixel-Level AUROC

| Strategy | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| Eigenface  | 0.5092 | 0.0501 | 0.4498 | 0.5631 |
| Kmeans     | 0.6413 | 0.0769 | 0.5920 | 0.7759 |
| Random     | 0.8065 | 0.0697 | 0.7030 | 0.8742 |

## Best Models (by Image AUROC)

### Eigenface
- **Experiment:** `bmad_eigenface_k8_l2_trial3`
- **Trial ID:** 3
- **Seed:** 45
- **Image AUROC:** 0.8162

### Kmeans
- **Experiment:** `bmad_kmeans_k8_l2_trial1`
- **Trial ID:** 1
- **Seed:** 48
- **Image AUROC:** 0.8214

### Random
- **Experiment:** `bmad_random_k8_l2_trial3`
- **Trial ID:** 3
- **Seed:** 55
- **Image AUROC:** 0.8358


## Individual Trial Results

| Strategy | Trial | Seed | Image AUROC | Pixel AUROC |
|----------|-------|------|-------------|-------------|
| eigenface  |     0 |    42 | 0.7766 | 0.5599 |
| eigenface  |     1 |    43 | 0.7658 | 0.4855 |
| eigenface  |     2 |    44 | 0.7935 | 0.5631 |
| eigenface  |     3 |    45 | 0.8162 | 0.4876 |
| eigenface  |     4 |    46 | 0.7942 | 0.4498 |
| kmeans     |     0 |    47 | 0.8030 | 0.7759 |
| kmeans     |     1 |    48 | 0.8214 | 0.6025 |
| kmeans     |     2 |    49 | 0.8181 | 0.6019 |
| kmeans     |     3 |    50 | 0.8198 | 0.6344 |
| kmeans     |     4 |    51 | 0.7967 | 0.5920 |
| random     |     0 |    52 | 0.8088 | 0.7775 |
| random     |     1 |    53 | 0.8062 | 0.8633 |
| random     |     2 |    54 | 0.7979 | 0.8145 |
| random     |     3 |    55 | 0.8358 | 0.8742 |
| random     |     4 |    56 | 0.8255 | 0.7030 |
