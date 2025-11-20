"""Quick test to get actual dimensions used in the model"""
import torch
from model import DINOv3Backbone

# Initialize backbone with projection head
backbone = DINOv3Backbone(
    model_name='vit_small_patch16_dinov3.lvd1689m',
    projection_dim=128,
    freeze_backbone=True
)

# Test with actual BraTS image size
x = torch.randn(8, 3, 240, 240)  # Batch of 8 images

# Forward pass
out = backbone(x)

print("\n" + "="*80)
print("DIMENSION TEST WITH ACTUAL CONFIG")
print("="*80)
print(f"\nInput shape: {tuple(x.shape)}")
print(f"  -> (batch_size, channels, height, width)")
print(f"  -> (8, 3, 240, 240)")

print(f"\nBackbone info:")
print(f"  Model: vit_small_patch16_dinov3.lvd1689m")
print(f"  Embedding dimension: {backbone.embed_dim}")
print(f"  Patch size: {backbone.patch_size}x{backbone.patch_size}")
print(f"  Projection dimension: {backbone.projection_dim}")

print(f"\nPatch grid calculation:")
h_patches = 240 // backbone.patch_size
w_patches = 240 // backbone.patch_size
print(f"  Image size: 240x240")
print(f"  Patch size: {backbone.patch_size}x{backbone.patch_size}")
print(f"  Grid: {h_patches}x{w_patches} = {h_patches * w_patches} patches")

print(f"\nOutput shapes:")
print(f"  Global features: {tuple(out['global'].shape)}")
print(f"    -> (batch_size, projection_dim)")
print(f"    -> (8, {backbone.projection_dim})")

print(f"\n  Dense features: {tuple(out['dense'].shape)}")
print(f"    -> (batch_size, h_patches, w_patches, projection_dim)")
print(f"    -> (8, {h_patches}, {w_patches}, {backbone.projection_dim})")

print("\n" + "="*80)
