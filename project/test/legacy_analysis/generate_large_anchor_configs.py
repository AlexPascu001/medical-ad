"""
Generate config files for large anchor count experiments
Creates 20 configs: 5 anchor counts × 2 approaches × 2 early stopping settings
"""

import yaml
from pathlib import Path

# Base configuration template
base_config = {
    'seed': 42,
    'data': {
        'data_root': './data/BraTS2021_slice',
        'target_size': [240, 240]
    },
    'model': {
        'backbone': 'vit_small_patch16_dinov3.lvd1689m',
        'freeze_backbone': True,
        'projection_dim': 128
    },
    'loss': {
        'margin': 1.0,
        'alpha': 1.0,
        'beta': 0.5,
        'gamma': 0.0,
        'delta': 0.1,
        'min_norm': 0.5,
        'diversity_temperature': 0.1,
        'distance_metric': 'euclidean',
        'use_dense': False,
        'global_weight': 1.0,
        'dense_weight': 0.5,
        'spatial_reduction': 'mean'
    },
    'pretraining': {
        'enabled': False,
        'epochs': 0,
        'lr': 0.001,
        'batch_size': 64,
        'temp_anchors': 8,
        'loss_alpha': 1.0,
        'loss_beta': 0.0,
        'distance_metric': 'euclidean'
    },
    'training': {
        'epochs': 100,
        'batch_size': 64,
        'num_workers': 4,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'use_amp': True,
        'log_interval': 50,
        'val_interval': 1,
        'fixed_pseudo_labels': True,
        'save_checkpoints': False  # Only save best and final
    },
    'evaluation': {
        'eval_every': 1,
        'save_best': True,
        'metric': 'pixel_auroc',
        'save_predictions': True,
        'visualize_embeddings': True
    }
}

# Experiment parameters
anchor_counts = [64, 128, 256, 512, 1024]
approaches = [
    {'name': 'reproject', 'reproject_anchors': True, 'use_decoupled': False},
    {'name': 'decoupled', 'reproject_anchors': False, 'use_decoupled': True}
]
early_stopping_configs = [
    {'name': 'early', 'patience': 10, 'description': 'with early stopping'},
    {'name': 'noearly', 'patience': 1000, 'description': 'no early stopping'}
]

# Output directory
configs_dir = Path(__file__).parent / 'configs'
configs_dir.mkdir(exist_ok=True)

# Generate all configs
configs_created = []
for k in anchor_counts:
    for approach in approaches:
        for es in early_stopping_configs:
            # Create config
            config = base_config.copy()
            config['output_dir'] = f'./experiments/{approach["name"]}_k{k}_{es["name"]}'
            
            # Anchor configuration
            config['anchor'] = {
                'strategy': 'kmeans',
                'n_components': 50,
                'n_anchors': k,
                'max_images_for_pca': None,
                'learnable': False,
                'init_from': None,
                'use_embedding_space': True
            }
            
            # Approach-specific settings
            if approach['reproject_anchors']:
                config['anchor']['reproject_anchors'] = True
            else:
                config['anchor']['geometric_init'] = 'random_orthogonal'
            
            # Early stopping
            config['training']['early_stopping_patience'] = es['patience']
            
            # Save config
            filename = f"{approach['name']}_k{k}_{es['name']}.yaml"
            filepath = configs_dir / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            configs_created.append(filename)
            print(f"Created: {filename}")

print(f"\nTotal configs created: {len(configs_created)}")
print(f"Saved to: {configs_dir}")
