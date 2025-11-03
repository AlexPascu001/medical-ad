"""
Simple test to check pixel AUROC computation during validation
"""
import sys
import torch
from pathlib import Path

# Run as: python -m test_pixel_simple
# This ensures main.py can be imported properly

if __name__ == "__main__":
    # Run training with evaluation
    from main import main
    import argparse
    
    # Create args for evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/bmad_fixed/config.yaml')
    parser.add_argument('--resume', type=str, default='experiments/bmad_fixed/best_model.pth')
    parser.add_argument('--eval-only', action='store_true', default=True)
    
    args = parser.parse_args([
        '--config', 'experiments/bmad_fixed/config.yaml',
        '--resume', 'experiments/bmad_fixed/best_model.pth',
        '--eval-only'
    ])
    
    print("="*60)
    print("RUNNING VALIDATION WITH PIXEL AUROC")
    print("="*60)
    
    main(args)
