# Run all Expert's Decoupled Anchor experiments
# This script runs 7 configurations sequentially

$ErrorActionPreference = "Stop"

# Activate virtual environment
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host "ACTIVATING VIRTUAL ENVIRONMENT" -ForegroundColor Cyan
Write-Host "================================================================================================" -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Array of config files to run
$configs = @(
    ".\project\configs\solution_a_384d.yaml",              # Expert's approach: K=8, decoupled, no early stopping
    ".\project\configs\solution_a_reproject.yaml",         # Solution A: K=8, re-project each forward, no early stopping
    ".\project\configs\expert_100ep_k4_early.yaml",        # K=4, WITH early stopping
    ".\project\configs\expert_100ep_k4_noearly.yaml",      # K=4, NO early stopping
    ".\project\configs\expert_100ep_k6_early.yaml",        # K=6, WITH early stopping
    ".\project\configs\expert_100ep_k6_noearly.yaml",      # K=6, NO early stopping
    ".\project\configs\expert_100ep_k12_early.yaml",       # K=12, WITH early stopping
    ".\project\configs\expert_100ep_k12_noearly.yaml"      # K=12, NO early stopping
)

$total = $configs.Count
$current = 0

foreach ($config in $configs) {
    $current++
    
    Write-Host ""
    Write-Host "================================================================================================" -ForegroundColor Green
    Write-Host "EXPERIMENT $current/$total" -ForegroundColor Green
    Write-Host "Config: $config" -ForegroundColor Green
    Write-Host "================================================================================================" -ForegroundColor Green
    Write-Host ""
    
    # Run experiment
    python .\project\main.py --config $config
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "================================================================================================" -ForegroundColor Red
        Write-Host "ERROR: Experiment failed with exit code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Config: $config" -ForegroundColor Red
        Write-Host "================================================================================================" -ForegroundColor Red
        Write-Host ""
        # Continue to next experiment instead of stopping
        Write-Host "Continuing to next experiment..." -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "================================================================================================" -ForegroundColor Green
        Write-Host "SUCCESS: Experiment $current/$total completed" -ForegroundColor Green
        Write-Host "================================================================================================" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host ""
Write-Host "================================================================================================" -ForegroundColor Cyan
Write-Host "ALL EXPERIMENTS COMPLETE" -ForegroundColor Cyan
Write-Host "Completed $total experiments" -ForegroundColor Cyan
Write-Host "================================================================================================" -ForegroundColor Cyan
