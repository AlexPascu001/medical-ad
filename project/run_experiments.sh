#!/bin/bash
# Batch runner for BMAD experiments
# This script runs multiple experiments with different hyperparameters

set -e  # Exit on error

echo "======================================"
echo "BMAD Experiment Suite"
echo "======================================"
echo ""

# Activate conda environment (modify as needed)
# conda activate bmad

# Base directory for experiments
BASE_DIR="./experiments"

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local exp_name=$2
    
    echo "Running experiment: $exp_name"
    echo "Config: $config_name"
    echo "--------------------------------------"
    
    python main.py --config "$config_name"
    
    echo "✓ Completed: $exp_name"
    echo ""
}

# ===== EXPERIMENT GROUP 1: Number of Anchors (K) =====
echo "GROUP 1: Varying Number of Anchors (K)"
echo "======================================"

# K=4
run_experiment "configs/grid_search.yaml:experiment_k4" "K=4 anchors"

# K=8 (baseline)
run_experiment "configs/grid_search.yaml:experiment_k8" "K=8 anchors (baseline)"

# K=16
run_experiment "configs/grid_search.yaml:experiment_k16" "K=16 anchors"

# ===== EXPERIMENT GROUP 2: PCA Components (M) =====
echo "GROUP 2: Varying PCA Components (M)"
echo "======================================"

# M=20
run_experiment "configs/grid_search.yaml:experiment_m20" "M=20 components"

# M=100
run_experiment "configs/grid_search.yaml:experiment_m100" "M=100 components"

# ===== EXPERIMENT GROUP 3: Model Variants =====
echo "GROUP 3: Model Variants"
echo "======================================"

# Finetuned backbone
run_experiment "configs/grid_search.yaml:experiment_finetune" "Finetuned backbone"

# Dense features
run_experiment "configs/grid_search.yaml:experiment_dense" "Dense features"

# Small backbone
run_experiment "configs/grid_search.yaml:experiment_vits" "ViT-S backbone"

# ===== EXPERIMENT GROUP 4: Loss Margins =====
echo "GROUP 4: Loss Margins"
echo "======================================"

# Tight margins
run_experiment "configs/grid_search.yaml:experiment_tight_margins" "Tight margins"

# Wide margins
run_experiment "configs/grid_search.yaml:experiment_wide_margins" "Wide margins"

# ===== SUMMARY =====
echo ""
echo "======================================"
echo "All Experiments Complete!"
echo "======================================"
echo ""

# Generate summary
echo "Generating summary..."
python utils.py summary \
    experiments/bmad_k4 \
    experiments/bmad_k8 \
    experiments/bmad_k16 \
    experiments/bmad_m20 \
    experiments/bmad_m100 \
    experiments/bmad_finetune \
    experiments/bmad_dense \
    experiments/bmad_vits \
    experiments/bmad_tight_margins \
    experiments/bmad_wide_margins \
    --output experiments/summary.csv

# Generate comparison plots
echo "Generating comparison plots..."
python utils.py compare \
    experiments/bmad_k4 \
    experiments/bmad_k8 \
    experiments/bmad_k16 \
    experiments/bmad_m20 \
    experiments/bmad_m100 \
    experiments/bmad_finetune \
    experiments/bmad_dense \
    experiments/bmad_vits \
    experiments/bmad_tight_margins \
    experiments/bmad_wide_margins \
    --metric image_auroc

echo ""
echo "Results saved to:"
echo "  - experiments/summary.csv"
echo "  - comparison.png"
echo ""
echo "Done!"