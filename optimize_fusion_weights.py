"""
Post-hoc fusion weight grid search.
Reads per-image scores from CSV and finds optimal fusion weights.
Usage: python optimize_fusion_weights.py <csv_path>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def minmax_normalize(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def compute_fused_auroc(df, wa, wd, wp, wr=0.0):
    """Compute AUROC for a specific (anchor, divergence, pixel, reconstruction) weight triple."""
    labels = df['label'].values
    anchor = minmax_normalize(df['anchor_score'].values)
    
    # Choose best divergence signal (bottleneck vs patch) by AUROC
    div = None
    bdiv_auroc = None
    pdiv_auroc = None
    if 'bottleneck_divergence' in df.columns:
        bdiv = minmax_normalize(df['bottleneck_divergence'].values)
        bdiv_auroc = roc_auc_score(labels, bdiv)
    if 'patch_divergence_aggregated' in df.columns:
        pdiv = minmax_normalize(df['patch_divergence_aggregated'].values)
        pdiv_auroc = roc_auc_score(labels, pdiv)
    
    # Pick better signal
    if bdiv_auroc is not None and pdiv_auroc is not None:
        if bdiv_auroc >= pdiv_auroc and bdiv_auroc >= 0.5:
            div = minmax_normalize(df['bottleneck_divergence'].values)
        elif pdiv_auroc >= 0.5:
            div = minmax_normalize(df['patch_divergence_aggregated'].values)
    elif bdiv_auroc is not None and bdiv_auroc >= 0.5:
        div = minmax_normalize(df['bottleneck_divergence'].values)
    elif pdiv_auroc is not None and pdiv_auroc >= 0.5:
        div = minmax_normalize(df['patch_divergence_aggregated'].values)
    
    # Pixel aggregated
    pix = None
    if 'pixel_aggregated_score' in df.columns:
        pix = minmax_normalize(df['pixel_aggregated_score'].values)
        pix_auroc = roc_auc_score(labels, pix)
        if pix_auroc < 0.5:
            pix = None  # anti-correlated, drop
    
    # Reconstruction
    rec = None
    if 'reconstruction_score' in df.columns and wr > 0:
        rec = minmax_normalize(df['reconstruction_score'].values)
        rec_auroc = roc_auc_score(labels, rec)
        if rec_auroc < 0.5:
            rec = None
    
    # Build fused score
    fused = wa * anchor
    total_w = wa
    
    if div is not None:
        fused = fused + wd * div
        total_w += wd
    
    if pix is not None:
        fused = fused + wp * pix
        total_w += wp
    
    if rec is not None:
        fused = fused + wr * rec
        total_w += wr
    
    if total_w < 1e-8:
        return 0.5  # degenerate case
    fused = fused / total_w
    return roc_auc_score(labels, fused)

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/dual_bottleneck_k1/evaluation/evaluation_image_scores.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} images, {df['label'].sum()} anomalous")
    print("\nPer-signal AUROCs:")
    labels = df['label'].values
    # Select best divergence signal
    bdiv_auroc = roc_auc_score(labels, df['bottleneck_divergence'].values) if 'bottleneck_divergence' in df.columns else None
    pdiv_auroc = roc_auc_score(labels, df['patch_divergence_aggregated'].values) if 'patch_divergence_aggregated' in df.columns else None
    best_div_auroc = max(x for x in [bdiv_auroc, pdiv_auroc] if x is not None)
    
    for col in ['anchor_score', 'reconstruction_score', 'bottleneck_divergence', 'patch_divergence_aggregated', 'pixel_aggregated_score']:
        if col in df.columns:
            auroc = roc_auc_score(labels, df[col].values)
            flag = " [INVERTED]" if auroc < 0.5 else ""
            star = " [best-div]" if (col in ['bottleneck_divergence', 'patch_divergence_aggregated'] and auroc == best_div_auroc) else ""
            print(f"  {col}: {auroc:.4f}{flag}{star}")
    
    print("\nGrid searching fusion weights (anchor, divergence, pixel, reconstruction)...")
    best_auroc = -1
    best_weights = None
    
    step = 0.05
    results = []
    weights = np.arange(0, 1.01, step)
    # Also try including reconstruction as 4th signal
    recon_weights = [0.0, 0.05, 0.10, 0.15]
    
    for wr in recon_weights:
        for wa in weights:
            for wd in weights:
                for wp in weights:
                    total = wa + wd + wp + wr
                    if total < 0.01:
                        continue
                    wa_n = wa / total
                    wd_n = wd / total
                    wp_n = wp / total
                    wr_n = wr / total
                    auroc = compute_fused_auroc(df, wa_n, wd_n, wp_n, wr_n)
                    results.append((auroc, wa_n, wd_n, wp_n, wr_n))
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_weights = (wa_n, wd_n, wp_n, wr_n)
    
    results.sort(reverse=True)
    print(f"\nTop 10 fusion weight combinations:")
    print(f"{'Rank':<5} {'AUROC':<8} {'Anchor':<8} {'Div':<8} {'Pixel':<8} {'Recon':<8}")
    for i, (auroc, wa, wd, wp, wr) in enumerate(results[:10]):
        print(f"{i+1:<5} {auroc:.4f}   {wa:.3f}   {wd:.3f}   {wp:.3f}   {wr:.3f}")
    
    print(f"\nBest: AUROC={best_auroc:.4f}, anchor={best_weights[0]:.3f}, div={best_weights[1]:.3f}, pixel={best_weights[2]:.3f}, recon={best_weights[3]:.3f}")
    
    # Also check pure anchor
    a_only_auroc = roc_auc_score(labels, df['anchor_score'].values)
    print(f"\nSingle-signal anchor: {a_only_auroc:.4f}")
    
    # Show current config weights result
    current_auroc = compute_fused_auroc(df, 0.72, 0.16, 0.12)
    print(f"Current config (a=0.72, d=0.16, p=0.12, r=0.0): {current_auroc:.4f}")

if __name__ == '__main__':
    main()
