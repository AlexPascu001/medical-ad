"""
Generate large-scale anchor experiment configs (K=64,128,256,512,1024)
matching the structure of solution_a_reproject_k*.yaml and solution_a_decoupled_k*.yaml
"""

import os

def create_reproject_config(k, early_stopping, output_dir):
    """Create reproject config matching solution_a_reproject_k*.yaml structure"""
    patience = 10 if early_stopping else 1000
    es_label = "early" if early_stopping else "noearly"
    
    content = f"""# SOLUTION A: Re-project Anchors Each Forward Pass - K={k}
# Generate anchors in 384D DINOv2 space, then re-project through projection head each forward.
# This allows anchors to "move" with the projection head during training.
{'# Early stopping enabled (patience=10)' if early_stopping else '# No early stopping (patience=1000)'}

seed: 42
output_dir: './experiments/reproject_k{k}_{es_label}'

# Data configuration
data:
  data_root: './data/BraTS2021_slice'
  target_size: [240, 240]

# Anchor configuration - SOLUTION A
anchor:
  strategy: 'kmeans'        # K-means in 384D DINOv2 space (semantic clustering)
  n_components: 50          # Only for eigenface strategy
  n_anchors: {k}             # Number of semantic clusters
  max_images_for_pca: null  # Use ALL training images (7500)
  learnable: false          # Fixed semantic anchors (not trainable)
  init_from: null
  use_embedding_space: true # ✓ Generate anchors in 384D DINOv2 space
  reproject_anchors: true   # ✓ CRITICAL: Re-project anchors each forward pass (Solution A)
  # geometric_init: DO NOT SET - Solution A doesn't use geometric targets

# Model configuration
model:
  backbone: 'vit_small_patch16_dinov3.lvd1689m'
  freeze_backbone: true     # ✓ Keep DINOv2 frozen (semantic features stable)
  projection_dim: 128       # ✓ Trainable projection head: 384D → 128D

# Loss configuration - SOLUTION A with diversity regularization
loss:
  margin: 1.0
  alpha: 1.0                # Attractor weight (pull samples to nearest anchor)
  beta: 0.5                 # Repeller weight (push anchors apart)
  gamma: 0.0                # Min-norm (not needed for fixed anchors)
  delta: 0.1                # ✓ DIVERSITY LOSS: Prevent anchor collapse via entropy regularization
  min_norm: 0.5
  diversity_temperature: 0.1  # ✓ Temperature for soft assignments (lower = sharper)
  distance_metric: 'euclidean'
  use_dense: false
  global_weight: 1.0
  dense_weight: 0.5
  spatial_reduction: 'mean'

# Pre-training configuration (DISABLED)
pretraining:
  enabled: false
  epochs: 0
  lr: 0.001
  batch_size: 64
  temp_anchors: 8
  loss_alpha: 1.0
  loss_beta: 0.0
  distance_metric: 'euclidean'

# Training configuration
training:
  epochs: 100               # Extended training to test convergence
  batch_size: 64
  num_workers: 4
  lr: 0.0001                # Conservative LR for projection head
  weight_decay: 0.0001
  use_amp: true             # Mixed precision training
  log_interval: 50          # Log every N batches
  val_interval: 1           # Validate every N epochs
  early_stopping_patience: {patience}  # {'Early stopping enabled' if early_stopping else 'Effectively disabled (no early stopping)'}
  fixed_pseudo_labels: true # ✓ Compute pseudo-labels in 384D space ONCE at start
  save_checkpoints: false   # Only save best and final models (saves ~144GB total)

# Evaluation configuration
evaluation:
  eval_every: 1
  save_best: true
  metric: 'pixel_auroc'
  save_predictions: true
  visualize_embeddings: true  # ✓ Visualize to verify no collapse
"""
    
    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write(content)


def create_decoupled_config(k, early_stopping, output_dir):
    """Create decoupled config matching solution_a_decoupled_k*.yaml structure"""
    patience = 10 if early_stopping else 1000
    es_label = "early" if early_stopping else "noearly"
    
    content = f"""# Expert's Decoupled Approach - K={k}
# Semantic anchors (384D) for labeling, geometric targets (128D) for training
{'# Early stopping enabled (patience=10)' if early_stopping else '# No early stopping (patience=1000)'}

seed: 42
output_dir: './experiments/decoupled_k{k}_{es_label}'

# Data configuration
data:
  data_root: './data/BraTS2021_slice'
  target_size: [240, 240]

# Anchor configuration - EXPERT'S APPROACH
anchor:
  strategy: 'kmeans'        # K-means in 384D DINOv2 space (semantic clustering)
  n_components: 50          # Only for eigenface strategy
  n_anchors: {k}             # Number of semantic clusters
  max_images_for_pca: null  # Use ALL training images (7500)
  learnable: false          # Fixed anchors (not trainable)
  init_from: null
  use_embedding_space: true # ✓ CRITICAL: Generate anchors in 384D DINOv2 space (not pixel space)
  geometric_init: 'random_orthogonal'  # How to init 128D geometric targets: 'random_orthogonal' or 'project_once'

# Model configuration
model:
  backbone: 'vit_small_patch16_dinov3.lvd1689m'
  freeze_backbone: true     # ✓ Keep DINOv2 frozen (semantic features stable)
  projection_dim: 128       # ✓ Trainable projection head (anchors re-projected each forward)

# Loss configuration - SOLUTION A with diversity regularization
loss:
  margin: 1.0
  alpha: 1.0                # Attractor weight
  beta: 0.5                 # Repeller weight (push anchors apart)
  gamma: 0.0                # Min-norm (not needed for fixed anchors)
  delta: 0.1                # ✓ DIVERSITY LOSS: Prevent collapse via entropy regularization
  min_norm: 0.5
  diversity_temperature: 0.1  # ✓ Temperature for soft assignments (lower = sharper)
  distance_metric: 'euclidean'
  use_dense: false
  global_weight: 1.0
  dense_weight: 0.5
  spatial_reduction: 'mean'

# Pre-training configuration (DISABLED for Solution A)
pretraining:
  enabled: false            # No pre-training needed (anchors in semantic space)
  epochs: 0
  lr: 0.001
  batch_size: 64
  temp_anchors: 8
  loss_alpha: 1.0
  loss_beta: 0.0
  distance_metric: 'euclidean'

# Training configuration
training:
  epochs: 100               # Extended training to see convergence
  batch_size: 64
  num_workers: 4
  lr: 0.0001                # Conservative LR for projection head
  weight_decay: 0.0001
  use_amp: true             # Mixed precision training
  log_interval: 50          # Log every N batches
  val_interval: 1           # Validate every N epochs
  early_stopping_patience: {patience}  # {'Early stopping enabled' if early_stopping else 'Effectively disabled (no early stopping)'}
  fixed_pseudo_labels: true
  save_checkpoints: false   # Only save best and final models (saves ~144GB total)

# Evaluation configuration
evaluation:
  eval_every: 1
  save_best: true
  metric: 'pixel_auroc'
  save_predictions: true
  visualize_embeddings: true  # ✓ Visualize to verify no collapse
"""
    
    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write(content)


# Generate configs
configs_dir = './configs'
k_values = [64, 128, 256, 512, 1024]

count = 0
for k in k_values:
    for early_stopping in [True, False]:
        es_label = "early" if early_stopping else "noearly"
        
        # Reproject approach
        reproject_path = os.path.join(configs_dir, f'reproject_k{k}_{es_label}.yaml')
        create_reproject_config(k, early_stopping, reproject_path)
        print(f"Created: reproject_k{k}_{es_label}.yaml")
        count += 1
        
        # Decoupled approach
        decoupled_path = os.path.join(configs_dir, f'decoupled_k{k}_{es_label}.yaml')
        create_decoupled_config(k, early_stopping, decoupled_path)
        print(f"Created: decoupled_k{k}_{es_label}.yaml")
        count += 1

print(f"\nTotal configs created: {count}")
