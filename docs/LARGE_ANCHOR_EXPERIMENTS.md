# Large Anchor Count Experiments

## Overview
Testing both re-projection and decoupled approaches with high anchor counts to determine optimal K value and compare approaches at scale.

## Experiment Matrix
- **Anchor counts**: 64, 128, 256, 512, 1024
- **Approaches**: Re-projection vs Decoupled
- **Early stopping**: With (patience=10) vs Without (patience=1000)
- **Total experiments**: 5 × 2 × 2 = 20

## Configuration Changes

### 1. Checkpoint Saving
**New parameter**: `save_checkpoints: false` in training config
- **Only saves**: `best_model.pth` and `final_model.pth`
- **Skips**: Periodic checkpoints (`checkpoint_epoch_5.pth`, etc.)
- **Reason**: Save disk space for large-scale experiments

### 2. Training Summary
**New file**: `training_summary.json` created after each experiment

Contains:
```json
{
  "training_completed": true,
  "total_epochs_configured": 100,
  "actual_epochs_trained": <number>,
  "early_stopping_enabled": <true/false>,
  "early_stopping_patience": <10 or 1000>,
  "early_stopped": <true/false>,
  "best_model": {
    "epoch": <epoch_number>,
    "image_auroc": <auroc_value>,
    "saved_as": "best_model.pth",
    "description": "Best model achieved at epoch X with image AUROC Y"
  },
  "final_model": {
    "epoch": <final_epoch>,
    "saved_as": "final_model.pth",
    "description": "Final model after X epochs"
  },
  "checkpoints_saved": false,
  "model_files": ["best_model.pth", "final_model.pth"]
}
```

## Running Experiments

### Run All 20 Experiments
```powershell
.\run_large_anchor_experiments.ps1
```

This will:
- Activate virtual environment
- Run all 20 experiments sequentially
- Track timing for each experiment
- Continue on failure
- Display summary at the end

### Run Single Experiment
```powershell
.\venv\Scripts\Activate.ps1
python .\project\main.py --config .\project\configs\reproject_k64_early.yaml
```

## Expected Results

### Re-projection Approach
- **Hypothesis**: May suffer from collapse or instability with very high K
- **Watch for**: Anchor distribution balance, training stability
- **Expected AUROC**: May plateau or degrade with K > 256

### Decoupled Approach
- **Hypothesis**: More stable with high K due to fixed geometric targets
- **Watch for**: Whether random targets still work well at scale
- **Expected AUROC**: Should remain stable or improve with higher K

### Early Stopping Impact
- **With early stopping**: May stop before full convergence
- **Without early stopping**: Full 100 epochs, better final performance
- **Comparison**: Which epoch typically achieves best validation AUROC?

## Files Per Experiment

### Always Created
- `best_model.pth` - Best checkpoint based on validation AUROC
- `final_model.pth` - Model after final epoch
- `training_history.json` - Full training metrics per epoch
- `training_summary.json` - High-level summary with best epoch info
- `training_curves.png` - Loss and AUROC plots
- `anchor_images.png` - Visualization of anchor images
- Various t-SNE visualizations

### NOT Created
- `checkpoint_epoch_5.pth`, `checkpoint_epoch_10.pth`, etc. (disabled via `save_checkpoints: false`)

## Disk Space Savings

### Per Experiment
- **Without save_checkpoints=false**: ~20 checkpoint files × ~400MB = ~8GB
- **With save_checkpoints=false**: 2 model files × ~400MB = ~800MB
- **Savings**: ~7.2GB per experiment

### Total (20 experiments)
- **Without optimization**: ~160GB
- **With optimization**: ~16GB
- **Total savings**: ~144GB

## Analysis After Running

### Metrics to Compare
1. **Best validation AUROC** across all K values
2. **Epoch at which best model was achieved** (early vs late training)
3. **Impact of early stopping** (did it help or hurt?)
4. **Training stability** (loss curves, anchor distribution)
5. **Optimal K value** for both approaches

### Questions to Answer
1. What is the optimal number of anchors (K)?
2. Does re-projection collapse at high K?
3. Are decoupled anchors more stable at scale?
4. Is early stopping beneficial or detrimental?
5. Do the approaches converge to similar performance with enough anchors?

## Code Changes Summary

### train.py
- Added `save_checkpoints` parameter to `Trainer.__init__`
- Modified checkpoint saving logic to skip periodic checkpoints when `save_checkpoints=False`
- Added `_create_training_summary()` method to generate summary JSON

### main.py
- Pass `save_checkpoints` from config to `Trainer`

### Config Files
- All new configs have `save_checkpoints: false` in training section
- Two variants per K: early stopping (patience=10) and no early stopping (patience=1000)
