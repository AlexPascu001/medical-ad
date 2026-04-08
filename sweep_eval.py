"""
Quick sweep: test multiple configurations on the existing checkpoint.
Tests: best vs final model, fusion weights, percentiles, normalization modes.
Outputs a summary table sorted by best fused AUROC.
"""
import sys, os, yaml, torch, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, 'project')
from model import AnomalyDetector, DINOv3Backbone
from pixel_aggregation import aggregate_pixel_scores_torch
from data import create_dataloaders

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = 'project/configs/two_stage_dual_bottleneck_k1.yaml'
EXP_DIR = Path('experiments/dual_bottleneck_k1')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# ─── Data ─────────────────────────────────────────────────────────────────────
print("Loading data...")
_, _, test_loader = create_dataloaders(config)

# ─── Build model shell ────────────────────────────────────────────────────────
def build_model(config):
    backbone = DINOv3Backbone(
        model_name=config['model']['backbone'],
        freeze_backbone=True,
        projection_dim=config['model'].get('projection_dim', 128),
    )
    
    # Load anchors
    anchor_path = EXP_DIR / 'anchor_embeddings.pt'
    anchor_data = torch.load(anchor_path, map_location='cpu', weights_only=False)
    semantic_anchors = anchor_data.get('semantic_anchors', anchor_data.get('anchors'))
    
    s2_cfg = config.get('stage2', {})
    pm_cfg = s2_cfg.get('pixel_map', {})
    
    model = AnomalyDetector(
        backbone=backbone,
        anchors=semantic_anchors,
        anchors_already_projected=False,
        distance_metric=config['model'].get('distance_metric', 'euclidean'),
    )
    
    # Enable reconstruction branch
    model.enable_reconstruction(
        latent_dim=config['model'].get('projection_dim', 128),
        target_size=(config['data'].get('image_size', 240), config['data'].get('image_size', 240)),
        freeze_anchor_target=s2_cfg.get('freeze_anchor_target', True),
        out_channels=3,
        pixel_map_enabled=pm_cfg.get('enabled', True),
        pixel_map_type=pm_cfg.get('type', 'reconstruction_l2'),
        use_frozen_bottleneck=s2_cfg.get('frozen_bottleneck', False),
    )
    
    return model

# ─── Collect raw outputs once per checkpoint ──────────────────────────────────
def collect_raw(model, loader, device):
    """Run inference, return raw per-sample tensors (no aggregation yet)."""
    model.eval()
    all_anchor = []
    all_recon = []
    all_div = []
    all_patch_div_map = []
    all_recon_pixel_map = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Inference'):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            outputs = model.forward(images, return_dense=True)
            
            # Anchor distance
            dists = outputs['global_distances']  # (B, K)
            all_anchor.append(dists.min(dim=1)[0].cpu())
            
            # Reconstruction error
            if 'reconstruction_error' in outputs:
                all_recon.append(outputs['reconstruction_error'].cpu())
            
            # CLS divergence
            if outputs.get('bottleneck_divergence') is not None:
                all_div.append(outputs['bottleneck_divergence'].cpu())
            
            # Reconstruction pixel map (B, H, W)
            if outputs.get('reconstruction_pixel_map') is not None:
                all_recon_pixel_map.append(outputs['reconstruction_pixel_map'].cpu())
            
            # Patch divergence map (B, H, W)
            if outputs.get('patch_divergence_map') is not None:
                all_patch_div_map.append(outputs['patch_divergence_map'].cpu())
            
            all_labels.append(labels)
    
    result = {
        'anchor': torch.cat(all_anchor),
        'labels': np.concatenate(all_labels),
    }
    if all_recon:
        result['recon'] = torch.cat(all_recon)
    if all_div:
        result['div'] = torch.cat(all_div)
    if all_recon_pixel_map:
        result['recon_pixel_map'] = torch.cat(all_recon_pixel_map)
    if all_patch_div_map:
        result['patch_div_map'] = torch.cat(all_patch_div_map)
    
    return result


def evaluate_config(raw, method, percentile, norm_mode, w_a, w_d, w_p):
    """Given raw outputs + config, compute all AUROCs."""
    labels = raw['labels']
    anchor = raw['anchor'].numpy()
    
    metrics = {
        'anchor_auroc': roc_auc_score(labels, anchor),
    }
    
    # Pixel-aggregated score
    pix_agg = None
    if 'recon_pixel_map' in raw:
        pix_agg = aggregate_pixel_scores_torch(
            raw['recon_pixel_map'], method=method, percentile=percentile
        ).numpy()
        metrics['pixel_agg_auroc'] = roc_auc_score(labels, pix_agg)
    
    # CLS divergence
    div = None
    if 'div' in raw:
        div = raw['div'].numpy()
        metrics['div_auroc'] = roc_auc_score(labels, div)
    
    # Patch divergence aggregated
    patch_div_agg = None
    if 'patch_div_map' in raw:
        patch_div_agg = aggregate_pixel_scores_torch(
            raw['patch_div_map'], method=method, percentile=percentile
        ).numpy()
        metrics['patch_div_agg_auroc'] = roc_auc_score(labels, patch_div_agg)
    
    # Recon image-level
    if 'recon' in raw:
        recon = raw['recon'].numpy()
        metrics['recon_auroc'] = roc_auc_score(labels, recon)
    
    # Fusion
    def _norm(values, mode):
        if mode == 'minmax':
            lo, hi = values.min(), values.max()
            return (values - lo) / max(hi - lo, 1e-12)
        elif mode == 'rank':
            from scipy.stats import rankdata
            return rankdata(values) / len(values)
        elif mode == 'robust':
            q25, q50, q75 = np.percentile(values, [25, 50, 75])
            iqr = max(q75 - q25, 1e-12)
            return (values - q50) / iqr
        elif mode == 'zscore':
            std = values.std()
            if std < 1e-12:
                return np.zeros_like(values)
            return (values - values.mean()) / std
        return values
    
    a_n = _norm(anchor, norm_mode)
    
    # Pick best divergence (guard anti-correlated)
    best_div = None
    best_div_auroc = 0
    if div is not None and metrics.get('div_auroc', 0) >= 0.5:
        best_div = div
        best_div_auroc = metrics['div_auroc']
    if patch_div_agg is not None and metrics.get('patch_div_agg_auroc', 0) > best_div_auroc and metrics.get('patch_div_agg_auroc', 0) >= 0.5:
        best_div = patch_div_agg
    
    # Build fusion
    if best_div is not None and pix_agg is not None:
        d_n = _norm(best_div, norm_mode)
        p_n = _norm(pix_agg, norm_mode)
        fused = w_a * a_n + w_d * d_n + w_p * p_n
    elif pix_agg is not None:
        total = w_a + w_p
        p_n = _norm(pix_agg, norm_mode)
        fused = (w_a / total) * a_n + (w_p / total) * p_n
    elif best_div is not None:
        total = w_a + w_d
        d_n = _norm(best_div, norm_mode)
        fused = (w_a / total) * a_n + (w_d / total) * d_n
    else:
        fused = a_n
    
    metrics['fused_auroc'] = roc_auc_score(labels, fused)
    
    # Also try anchor + pixel only (no divergence)
    if pix_agg is not None:
        p_n = _norm(pix_agg, norm_mode)
        fused_2sig = 0.5 * a_n + 0.5 * p_n
        metrics['anchor_pixel_50_50'] = roc_auc_score(labels, fused_2sig)
    
    return metrics


# ─── Main sweep ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = []
    
    checkpoints = [
        ('best_stage2', EXP_DIR / 'best_stage2_model.pth'),
        ('final_stage2', EXP_DIR / 'final_stage2_model.pth'),
    ]
    
    # Only include checkpoints that exist
    checkpoints = [(name, path) for name, path in checkpoints if path.exists()]
    
    for ckpt_name, ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name} ({ckpt_path.name})")
        print(f"{'='*60}")
        
        model = build_model(config)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"  Loaded epoch {epoch}")
        model = model.to(DEVICE)
        
        raw = collect_raw(model, test_loader, DEVICE)
        
        # Sweep configurations
        for method in ['self_normalized', 'top_k_percentile']:
            for percentile in [90, 95, 99]:
                for norm_mode in ['minmax', 'rank', 'robust']:
                    for w_a, w_d, w_p in [(0.4, 0.3, 0.3), (0.5, 0.0, 0.5), (0.6, 0.0, 0.4), (0.7, 0.0, 0.3), (0.3, 0.0, 0.7)]:
                        m = evaluate_config(raw, method, percentile, norm_mode, w_a, w_d, w_p)
                        results.append({
                            'ckpt': ckpt_name,
                            'epoch': epoch,
                            'method': method,
                            'pct': percentile,
                            'norm': norm_mode,
                            'w': f'{w_a}/{w_d}/{w_p}',
                            'anchor': m['anchor_auroc'],
                            'pix_agg': m.get('pixel_agg_auroc', 0),
                            'div': m.get('div_auroc', 0),
                            'recon': m.get('recon_auroc', 0),
                            'fused': m['fused_auroc'],
                            'a+p_50': m.get('anchor_pixel_50_50', 0),
                        })
        
        # Cleanup
        del model, raw
        torch.cuda.empty_cache()
    
    # Sort by fused AUROC
    results.sort(key=lambda x: x['fused'], reverse=True)
    
    # Print table
    print(f"\n{'='*120}")
    print(f"TOP 25 CONFIGURATIONS (sorted by fused AUROC)")
    print(f"{'='*120}")
    print(f"{'Rank':<5} {'Ckpt':<15} {'Method':<18} {'Pct':<5} {'Norm':<8} {'Weights':<12} {'Anchor':<8} {'PixAgg':<8} {'Div':<8} {'Recon':<8} {'Fused':<8} {'A+P50':<8}")
    print('-' * 120)
    for i, r in enumerate(results[:25]):
        print(f"{i+1:<5} {r['ckpt']:<15} {r['method']:<18} {r['pct']:<5} {r['norm']:<8} {r['w']:<12} {r['anchor']:.4f}  {r['pix_agg']:.4f}  {r['div']:.4f}  {r['recon']:.4f}  {r['fused']:.4f}  {r['a+p_50']:.4f}")
    
    # Also print the single best anchor+pixel 50/50
    results_ap = sorted(results, key=lambda x: x['a+p_50'], reverse=True)
    print(f"\n{'='*120}")
    print(f"TOP 10 by anchor+pixel 50/50")
    print(f"{'='*120}")
    for i, r in enumerate(results_ap[:10]):
        print(f"{i+1:<5} {r['ckpt']:<15} {r['method']:<18} {r['pct']:<5} {r['norm']:<8} {r['a+p_50']:.4f}  pix_agg={r['pix_agg']:.4f}")
    
    print(f"\nTotal configs tested: {len(results)}")
