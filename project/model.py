"""
DINOv3 Backbone Wrapper with Global, Dense, and Multi-Scale Feature Extraction
Supports frozen and finetunable modes with optional pixel-level decoder
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Optional, List, Tuple

from pixel_aggregation import aggregate_pixel_scores_torch


class FeaturePyramidDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) style decoder for pixel-level embeddings.
    
    Takes multi-scale features from DINOv3 intermediate layers and produces
    dense pixel-level embeddings through lateral connections and progressive upsampling.
    
    Architecture:
        - Lateral connections: 1x1 conv to reduce channel dims
        - Top-down pathway: upsample + add for feature fusion
        - Progressive upsampling: 15x15 -> 30 -> 60 -> 120 -> 240
        - Output: pixel embeddings (B, output_dim, H, W)
    """
    
    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_scales: int = 4,
        target_size: Tuple[int, int] = (240, 240)
    ):
        """
        Args:
            in_dim: Input feature dimension from backbone (384 for ViT-S)
            hidden_dim: Hidden dimension for FPN layers
            output_dim: Output pixel embedding dimension
            num_scales: Number of multi-scale features (default 4)
            target_size: Target output spatial size (H, W)
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_scales = num_scales
        self.target_size = target_size
        
        # Lateral connections (1x1 conv to reduce dims)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
            for _ in range(num_scales)
        ])
        
        # Smooth convs after feature fusion (3x3 conv)
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_scales)
        ])
        
        # Progressive upsampling blocks: 15 -> 30 -> 60 -> 120 -> 240
        # Each block does 2x upsampling with refinement
        self.upsample_blocks = nn.ModuleList([
            self._make_upsample_block(hidden_dim, hidden_dim)
            for _ in range(4)  # 4 blocks for 16x total upsampling
        ])
        
        # Final projection to output dimension
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, output_dim, kernel_size=1)
        )
        
        self._init_weights()
    
    def _make_upsample_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create an upsampling block with ConvTranspose + residual refinement"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize weights with kaiming normal"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through FPN decoder.
        
        Args:
            multi_scale_features: List of (B, H', W', D) features from different layers
                                  Ordered from shallow to deep (e.g., layers [2, 5, 8, 11])
        
        Returns:
            pixel_embeddings: (B, output_dim, H, W) pixel-level embeddings
        """
        assert len(multi_scale_features) == self.num_scales, \
            f"Expected {self.num_scales} features, got {len(multi_scale_features)}"
        
        # Convert from (B, H', W', D) to (B, D, H', W') for conv operations
        features = [f.permute(0, 3, 1, 2) for f in multi_scale_features]
        
        # Apply lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway with feature fusion (from deepest to shallowest)
        # Start from deepest feature
        fused = laterals[-1]
        for i in range(self.num_scales - 2, -1, -1):
            # Upsample deeper feature to match shallower
            upsampled = F.interpolate(fused, size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            # Add lateral connection
            fused = laterals[i] + upsampled
            # Smooth
            fused = self.smooth_convs[i](fused)
        
        # Progressive upsampling to target size
        x = fused
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        
        # Ensure exact target size
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Project to output dimension
        pixel_embeddings = self.output_proj(x)  # (B, output_dim, H, W)
        
        return pixel_embeddings


class ReconstructionDecoder(nn.Module):
    """Lightweight latent-to-image decoder used in stage-2 reconstruction training."""

    def __init__(self, latent_dim: int, out_channels: int = 3, target_size: Tuple[int, int] = (240, 240)):
        super().__init__()
        self.target_size = target_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 15 * 15),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.view(latent.shape[0], 256, 15, 15)
        x = self.decoder(x)
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x


class DINOv3Backbone(nn.Module):
    """
    DINOv3 feature extractor with global, dense, and multi-scale outputs
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone: bool = True,
        projection_dim: Optional[int] = None,
        pretrained: bool = True,
        multi_scale_indices: Optional[List[int]] = None,
        projection_hidden_dims: Optional[List[int]] = None
    ):
        """
        Args:
            model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', etc.)
            freeze_backbone: Whether to freeze backbone weights
            projection_dim: If set, add trainable projection head to this dimension
            pretrained: Load pretrained weights
            multi_scale_indices: Block indices to extract features from (e.g., [2, 5, 8, 11])
            projection_hidden_dims: If set, build a multi-layer projection head with these output
                                    dims (e.g. [192, 128] gives 384→192→128). Overrides projection_dim.
        """
        super().__init__()
        
        # projection_hidden_dims overrides projection_dim
        if projection_hidden_dims is not None:
            projection_dim = projection_hidden_dims[-1]
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.projection_dim = projection_dim
        self.projection_hidden_dims = projection_hidden_dims
        self.multi_scale_indices = multi_scale_indices or []

        # Load DINOv2 model from timm
        print(f"Loading {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained) 

        # Get embedding dimension
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        
        # Get number of blocks for validation
        self.num_blocks = len(self.backbone.blocks)
        
        # Detect register tokens: timm Eva models use reg_token (1, N_reg, D)
        if hasattr(self.backbone, 'reg_token') and self.backbone.reg_token is not None:
            self.num_register_tokens = self.backbone.reg_token.shape[1]
        elif hasattr(self.backbone, 'num_prefix_tokens'):
            self.num_register_tokens = self.backbone.num_prefix_tokens - 1  # subtract CLS
        else:
            self.num_register_tokens = 0
        
        print(f"Backbone embed_dim: {self.embed_dim}, patch_size: {self.patch_size}, num_blocks: {self.num_blocks}, register_tokens: {self.num_register_tokens}")
        
        if self.multi_scale_indices:
            print(f"Multi-scale feature extraction enabled at blocks: {self.multi_scale_indices}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("Backbone frozen")
        
        # Trainable projection head for learning better embeddings
        self.projection = None
        if projection_hidden_dims is not None:
            # Configurable multi-layer projection: embed_dim -> [hidden_dims...]
            dims = [self.embed_dim] + list(projection_hidden_dims)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:  # ReLU between all layers except the last
                    layers.append(nn.ReLU())
            self.projection = nn.Sequential(*layers)
            self._init_projection_head()
            dims_str = ' -> '.join(str(d) for d in dims)
            print(f"Added trainable projection head: {dims_str}")
            print(f"  Trainable parameters: {sum(p.numel() for p in self.projection.parameters() if p.requires_grad):,}")
            print(f"  Initialized with orthogonal weights (gain=1.0)")
        elif projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 2, projection_dim)
            )
            # Initialize with orthogonal weights for better semantic preservation
            self._init_projection_head()
            print(f"Added trainable projection head: {self.embed_dim} -> {self.embed_dim // 2} -> {projection_dim}")
            print(f"  Trainable parameters: {sum(p.numel() for p in self.projection.parameters() if p.requires_grad):,}")
            print(f"  Initialized with orthogonal weights (gain=1.0)")
    
    def _init_projection_head(self):
        """Initialize projection head with orthogonal weights to preserve DINOv3 semantic structure."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal init with gain=1.0 for better initial separation
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _extract_multi_scale_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features from multiple intermediate layers using timm's API.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            final_features: (B, N_tokens, D) final layer output
            intermediate_features: List of (B, H', W', D) features from specified blocks
        """
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_register_tokens = self.num_register_tokens
        
        # Use forward_intermediates to get features from specific blocks
        # This returns the final output and a list of intermediate features
        final_features, intermediates = self.backbone.forward_intermediates(
            x,
            indices=self.multi_scale_indices,
            return_prefix_tokens=False,  # Don't include CLS/register tokens
            norm=True,  # Apply layer norm
            output_fmt='NLC'  # (B, N_patches, D) format
        )
        
        # Reshape intermediates to spatial format
        multi_scale_features = []
        for feat in intermediates:
            # feat shape: (B, N_patches, D)
            feat_spatial = feat.view(B, h_patches, w_patches, -1)  # (B, H', W', D)
            multi_scale_features.append(feat_spatial)
        
        return final_features, multi_scale_features
    
    def forward(self, x: torch.Tensor, return_multi_scale: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract global and dense features
        
        Args:
            x: Input images (B, C, H, W)
            return_multi_scale: Whether to return multi-scale features for decoder
            
        Returns:
            Dictionary with:
                'global': (B, D) global embedding
                'dense': (B, H', W', D) dense feature map
                'multi_scale': List of (B, H', W', D) multi-scale features (if requested)
        """
        B, C, H, W = x.shape
        
        # Multi-scale extraction path
        if return_multi_scale and self.multi_scale_indices:
            if self.freeze_backbone:
                with torch.no_grad():
                    features, multi_scale_features = self._extract_multi_scale_features(x)
            else:
                features, multi_scale_features = self._extract_multi_scale_features(x)
        else:
            # Standard single-scale extraction
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone.forward_features(x)
            else:
                features = self.backbone.forward_features(x)
            multi_scale_features = None
        
        # DINOv3 returns a tensor (B, N_tokens, D) where N_tokens = 1 (cls) + N_register + N_patches
        # Token order: [CLS, REG1, ..., REG_N, PATCH1, PATCH2, ...]
        cls_token = features[:, 0]  # (B, D)
        
        # Skip register tokens
        num_register_tokens = self.num_register_tokens
        patch_tokens = features[:, 1 + num_register_tokens:]  # (B, N_patches, D)
        
        # Global embedding (from CLS token)
        global_raw = cls_token
        global_feat = cls_token
        
        # Reshape patch tokens to spatial grid
        n_patches = patch_tokens.shape[1]
        
        # Calculate based on input image size and patch size
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        # Verify we have the expected number of patches
        expected_patches = h_patches * w_patches
        if n_patches != expected_patches:
            print(f"Warning: Expected {expected_patches} patches ({h_patches}x{w_patches}), "
                  f"but got {n_patches} patches. Input size: {H}x{W}, patch size: {self.patch_size}")
            # Try square root method as fallback
            h_patches = w_patches = int(n_patches ** 0.5)
            
        # Reshape to (B, H', W', D)
        dense_feat = patch_tokens.view(B, h_patches, w_patches, -1)
        
        # Apply projection if exists
        if self.projection is not None:
            global_feat = self.projection(global_feat)
            # Apply projection to dense features
            B, H_p, W_p, D = dense_feat.shape
            dense_feat = dense_feat.view(B, H_p * W_p, D)
            dense_feat = self.projection(dense_feat)
            dense_feat = dense_feat.view(B, H_p, W_p, -1)
        
        # Normalize global features
        global_feat = F.normalize(global_feat, dim=1)
        
        result = {
            'global_raw': global_raw,
            'global': global_feat,
            'dense': dense_feat,
            'patch_raw': patch_tokens,  # raw pre-projection 384D patch tokens (B, N_patches, D_backbone)
        }
        
        # Add multi-scale features if requested
        if return_multi_scale and multi_scale_features is not None:
            result['multi_scale'] = multi_scale_features
        
        return result
    
    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if requested"""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class AnomalyDetector(nn.Module):
    """
    Complete anomaly detection model with DINOv3 backbone, optional pixel decoder,
    and anchor-based scoring for both global and pixel-level anomaly detection.
    
    CRITICAL: When anchors_already_projected=True, anchors are stored in PROJECTED
    space and NOT re-projected during forward pass. This prevents collapse where
    the projection head learns to map everything to one point.
    """
    
    def __init__(
        self,
        backbone: DINOv3Backbone,
        anchor_global_embeddings: torch.Tensor,
        anchor_dense_embeddings: Optional[torch.Tensor] = None,
        distance_metric: str = 'cosine',
        learnable_anchors: bool = False,
        use_pixel_decoder: bool = False,
        decoder_hidden_dim: int = 256,
        target_size: Tuple[int, int] = (240, 240),
        anchors_already_projected: bool = False,
        anchor_semantic_embeddings: Optional[torch.Tensor] = None,
        anchor_geometric_targets: Optional[torch.Tensor] = None,
        use_decoupled_anchors: bool = False
    ):
        """
        EXPERT'S APPROACH: Decouple semantic anchors from geometric targets.
        
        Args:
            backbone: DINOv3Backbone model
            anchor_global_embeddings: (K, D) anchor embeddings (LEGACY compatibility)
            anchor_dense_embeddings: (K, H', W', D) dense anchor features
            distance_metric: 'cosine' or 'euclidean' for computing distances
            learnable_anchors: If True, make anchors trainable parameters
            use_pixel_decoder: If True, add FPN decoder for pixel-level predictions
            decoder_hidden_dim: Hidden dimension for decoder layers
            target_size: Target output size for pixel predictions (H, W)
            anchors_already_projected: LEGACY flag (ignored if use_decoupled_anchors=True)
            anchor_semantic_embeddings: (K, 384) SEMANTIC anchors in DINOv3 space (for pseudo-label computation)
            anchor_geometric_targets: (K, 128) GEOMETRIC targets in projection space (FIXED training targets)
            use_decoupled_anchors: If True, use EXPERT'S APPROACH with decoupled semantic/geometric anchors
        """
        super().__init__()
        
        self.backbone = backbone
        self.distance_metric = distance_metric
        self.learnable_anchors = learnable_anchors
        self.use_pixel_decoder = use_pixel_decoder
        self.target_size = target_size
        self.use_decoupled_anchors = use_decoupled_anchors
        
        # EXPERT'S APPROACH: Decouple semantic from geometric anchors
        if use_decoupled_anchors:
            # Store SEMANTIC anchors (384D) - for pseudo-label computation ONLY
            # These are FROZEN DINOv3 embeddings used to assign labels
            self.register_buffer('anchor_semantic', anchor_semantic_embeddings)
            
            # Store GEOMETRIC targets (128D) - FIXED training targets
            # These NEVER move during training - projection head learns to map samples to these
            if learnable_anchors:
                self.anchor_geometric = nn.Parameter(anchor_geometric_targets.clone())
                print(f"  ✓ DECOUPLED ANCHORS (Expert's Approach):")
                print(f"    - Semantic (384D): {anchor_semantic_embeddings.shape} [FROZEN, for pseudo-labels]")
                print(f"    - Geometric (128D): {anchor_geometric_targets.shape} [LEARNABLE, training targets]")
            else:
                self.register_buffer('anchor_geometric', anchor_geometric_targets)
                print(f"  ✓ DECOUPLED ANCHORS (Expert's Approach):")
                print(f"    - Semantic (384D): {anchor_semantic_embeddings.shape} [FROZEN, for pseudo-labels]")
                print(f"    - Geometric (128D): {anchor_geometric_targets.shape} [FIXED, never move]")
            
            # No legacy fields needed
            self.anchor_global = None
            self.anchor_global_raw = None
            self.anchor_dense = None
            self.anchor_dense_raw = None
            self.anchors_already_projected = True  # Geometric targets are in projected space
            
        elif anchors_already_projected:
            # LEGACY APPROACH: Anchors already in projected space
            self.anchors_already_projected = anchors_already_projected
            if learnable_anchors:
                self.anchor_global = nn.Parameter(anchor_global_embeddings.clone())
                if anchor_dense_embeddings is not None:
                    self.anchor_dense = nn.Parameter(anchor_dense_embeddings.clone())
                else:
                    self.anchor_dense = None
                print(f"  ✓ Anchors are LEARNABLE in PROJECTED space ({anchor_global_embeddings.shape[0]} × {anchor_global_embeddings.shape[1]}D)")
            else:
                self.register_buffer('anchor_global', anchor_global_embeddings)
                if anchor_dense_embeddings is not None:
                    self.register_buffer('anchor_dense', anchor_dense_embeddings)
                else:
                    self.anchor_dense = None
                print(f"  ✓ Anchors are FIXED in PROJECTED space ({anchor_global_embeddings.shape[0]} × {anchor_global_embeddings.shape[1]}D)")
                print(f"    They will NOT be re-projected - acting as fixed targets")
            
            self.anchor_global_raw = None
            self.anchor_dense_raw = None
            self.anchor_semantic = None
            self.anchor_geometric = None
            
        else:
            # LEGACY: Anchors in RAW space, will be re-projected each forward pass
            self.anchors_already_projected = anchors_already_projected
            # WARNING: This can cause collapse with trainable projection heads!
            if learnable_anchors:
                self.anchor_global_raw = nn.Parameter(anchor_global_embeddings.clone())
                if anchor_dense_embeddings is not None:
                    self.anchor_dense_raw = nn.Parameter(anchor_dense_embeddings.clone())
                else:
                    self.anchor_dense_raw = None
                print(f"  ✓ Anchors are LEARNABLE in RAW space ({anchor_global_embeddings.shape[0]} × {anchor_global_embeddings.shape[1]}D)")
            else:
                self.register_buffer('anchor_global_raw', anchor_global_embeddings)
                if anchor_dense_embeddings is not None:
                    self.register_buffer('anchor_dense_raw', anchor_dense_embeddings)
                else:
                    self.anchor_dense_raw = None
                print(f"  ⚠ Anchors are FIXED in RAW space ({anchor_global_embeddings.shape[0]} × {anchor_global_embeddings.shape[1]}D)")
                print(f"    WARNING: Will be re-projected each forward - may cause issues!")
            
            # No projected anchors stored
            self.anchor_global = None
            self.anchor_dense = None
        
        self.n_anchors = len(anchor_global_embeddings)
        
        # Initialize pixel decoder if requested
        self.pixel_decoder = None
        if use_pixel_decoder:
            if not backbone.multi_scale_indices:
                raise ValueError("Pixel decoder requires multi_scale_indices to be set in backbone")
            
            output_dim = backbone.projection_dim if backbone.projection_dim else backbone.embed_dim
            self.pixel_decoder = FeaturePyramidDecoder(
                in_dim=backbone.embed_dim,
                hidden_dim=decoder_hidden_dim,
                output_dim=output_dim,
                num_scales=len(backbone.multi_scale_indices),
                target_size=target_size
            )
            print(f"  ✓ Pixel decoder enabled: {backbone.embed_dim}D -> {output_dim}D at {target_size}")
            print(f"    Decoder parameters: {sum(p.numel() for p in self.pixel_decoder.parameters()):,}")
        
        print(f"Initialized detector with {self.n_anchors} anchors")
        print(f"Distance metric: {distance_metric}")

        # Stage-2 reconstruction branch (initialized on demand)
        self.reconstruction_enabled = False
        self.stage2_freeze_anchor_target = True
        self.stage2_projection = None
        self.frozen_projection = None          # frozen copy of stage-1 bottleneck
        self.stage2_fuser = None
        self.reconstruction_decoder = None
        self.stage2_pixel_map_enabled = True
        self.stage2_pixel_map_type = 'reconstruction_l2'
        self.anchor_reproject = None           # anchor_dim → recon_dim (decoupled only)
        self._recon_dim = None
        self._anchor_dim = None

        # Dual-bottleneck flag
        self.use_frozen_bottleneck = False

        # Pixel-to-image aggregation settings
        self.pixel_aggregation_method = 'top_k_percentile'
        self.pixel_aggregation_percentile = 95.0
        self.pixel_aggregation_threshold = None   # set from training stats

        # Three-signal score fusion controls
        self.score_fusion_enabled = False
        self.score_fusion_normalization = 'minmax'
        self.score_fusion_anchor_weight = 0.4
        self.score_fusion_divergence_weight = 0.3
        self.score_fusion_pixel_weight = 0.3

        # Optional score-combination controls for inference/evaluation (legacy)
        self.score_combination_enabled = False
        self.score_combination_alpha = 0.5
        self.score_combination_normalization = 'minmax'

    def configure_score_combination(self, enabled: bool = False, alpha: float = 0.5, normalization: str = 'minmax'):
        """Configure optional anchor+reconstruction score combination at inference/eval."""
        self.score_combination_enabled = bool(enabled)
        self.score_combination_alpha = float(alpha)
        self.score_combination_normalization = normalization

    def configure_score_fusion(self, enabled: bool = False, normalization: str = 'minmax',
                               anchor_weight: float = 0.4, divergence_weight: float = 0.3,
                               pixel_weight: float = 0.3, drop_anticorrelated: bool = True):
        """Configure three-signal score fusion (anchor + bottleneck divergence + pixel aggregation)."""
        self.score_fusion_enabled = bool(enabled)
        self.score_fusion_normalization = normalization
        self.score_fusion_anchor_weight = float(anchor_weight)
        self.score_fusion_divergence_weight = float(divergence_weight)
        self.score_fusion_pixel_weight = float(pixel_weight)
        self.score_fusion_drop_anticorrelated = bool(drop_anticorrelated)

    def configure_pixel_aggregation(self, method: str = 'top_k_percentile',
                                    percentile: float = 95.0, threshold: Optional[float] = None):
        """Configure pixel-to-image aggregation strategy."""
        self.pixel_aggregation_method = method
        self.pixel_aggregation_percentile = float(percentile)
        self.pixel_aggregation_threshold = threshold

    def enable_reconstruction_branch(
        self,
        freeze_anchor_target: bool = True,
        out_channels: int = 3,
        pixel_map_enabled: bool = True,
        pixel_map_type: str = 'reconstruction_l2',
        use_frozen_bottleneck: bool = False,
        recon_projection_dim: int = None
    ):
        """
        Enable stage-2 branch with optionally decoupled reconstruction projection.

        When *recon_projection_dim* differs from the anchor projection dim, a NEW
        wider projection head is built for reconstruction (stage-2) while the
        frozen divergence signal stays in anchor space.  The fuser re-projects
        anchors into recon space so concatenation is well-typed.
        """
        if self.reconstruction_enabled:
            return

        anchor_dim = self.backbone.projection_dim if self.backbone.projection is not None else self.backbone.embed_dim
        recon_dim = recon_projection_dim if recon_projection_dim is not None else anchor_dim
        self._recon_dim = recon_dim   # store for forward()
        self._anchor_dim = anchor_dim

        # --- stage-2 projection head ---
        embed_dim = self.backbone.embed_dim
        if recon_dim != anchor_dim or self.backbone.projection is None:
            # Build a fresh projection head sized for the reconstruction bottleneck
            self.stage2_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, recon_dim)
            )
            # Orthogonal init (same as backbone projection head)
            for m in self.stage2_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            print(f"  ✓ Stage-2 projection head (NEW): {embed_dim} → {embed_dim // 2} → {recon_dim}")
        else:
            # Same dim → deep-copy anchor projection as before
            self.stage2_projection = copy.deepcopy(self.backbone.projection)
            print(f"  ✓ Stage-2 projection head (cloned from anchor): → {recon_dim}")

        # --- Frozen bottleneck (stays in anchor space for divergence signal) ---
        self.use_frozen_bottleneck = use_frozen_bottleneck
        if use_frozen_bottleneck:
            if self.backbone.projection is not None:
                self.frozen_projection = copy.deepcopy(self.backbone.projection)
            else:
                self.frozen_projection = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim, anchor_dim)
                )
            for p in self.frozen_projection.parameters():
                p.requires_grad = False
            self.frozen_projection.eval()
            print(f"  ✓ Frozen bottleneck (stage-1 copy, {anchor_dim}D) created")

        # --- Anchor re-projection layer (anchor_dim → recon_dim) ---
        if recon_dim != anchor_dim:
            self.anchor_reproject = nn.Linear(anchor_dim, recon_dim)
            nn.init.orthogonal_(self.anchor_reproject.weight, gain=1.0)
            nn.init.zeros_(self.anchor_reproject.bias)
            print(f"  ✓ Anchor re-projection: {anchor_dim} → {recon_dim}")
        else:
            self.anchor_reproject = None

        # --- Fuser and decoder operate in recon_dim space ---
        self.stage2_fuser = nn.Sequential(
            nn.Linear(2 * recon_dim, recon_dim),
            nn.ReLU(inplace=True),
            nn.Linear(recon_dim, recon_dim)
        )

        self.reconstruction_decoder = ReconstructionDecoder(
            latent_dim=recon_dim,
            out_channels=out_channels,
            target_size=self.target_size
        )

        model_device = next(self.backbone.parameters()).device
        if self.stage2_projection is not None:
            self.stage2_projection = self.stage2_projection.to(model_device)
        if self.frozen_projection is not None:
            self.frozen_projection = self.frozen_projection.to(model_device)
        if self.stage2_fuser is not None:
            self.stage2_fuser = self.stage2_fuser.to(model_device)
        if self.reconstruction_decoder is not None:
            self.reconstruction_decoder = self.reconstruction_decoder.to(model_device)
        if self.anchor_reproject is not None:
            self.anchor_reproject = self.anchor_reproject.to(model_device)

        self.reconstruction_enabled = True
        self.stage2_freeze_anchor_target = freeze_anchor_target
        self.stage2_pixel_map_enabled = pixel_map_enabled
        self.stage2_pixel_map_type = pixel_map_type
        print("  ✓ Stage-2 reconstruction branch enabled")

    def prepare_stage2_training(self, freeze_encoder: bool = True,
                               freeze_anchor_parameters: bool = True,
                               freeze_encoder_mode: str = 'full',
                               unfreeze_last_n_blocks: int = 2,
                               unfreeze_lr_multiplier: float = 0.1):
        """Freeze stage-1 modules and leave stage-2 branch trainable.

        Args:
            freeze_encoder: Legacy flag – freeze the entire backbone.
            freeze_anchor_parameters: Freeze anchor-related parameters.
            freeze_encoder_mode: 'full' (freeze all), 'partial' (unfreeze last N
                blocks), 'none' (no freezing).  Overrides *freeze_encoder* when
                set to something other than 'full'.
            unfreeze_last_n_blocks: How many trailing transformer blocks to
                unfreeze when *freeze_encoder_mode='partial'*.
            unfreeze_lr_multiplier: Stored but enforced externally when building
                the optimizer param groups.
        """
        if not self.reconstruction_enabled:
            raise RuntimeError("Call enable_reconstruction_branch() before prepare_stage2_training().")

        self._unfreeze_lr_multiplier = unfreeze_lr_multiplier

        # --- encoder freezing ---
        if freeze_encoder_mode == 'partial':
            # Freeze everything first
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze last N transformer blocks + layer-norm
            total_blocks = len(self.backbone.backbone.blocks)
            for idx in range(total_blocks - unfreeze_last_n_blocks, total_blocks):
                for p in self.backbone.backbone.blocks[idx].parameters():
                    p.requires_grad = True
            if hasattr(self.backbone.backbone, 'norm'):
                for p in self.backbone.backbone.norm.parameters():
                    p.requires_grad = True
            print(f"  Partial freeze: last {unfreeze_last_n_blocks}/{total_blocks} blocks unfrozen (LR ×{unfreeze_lr_multiplier})")
        elif freeze_encoder_mode == 'none':
            # Don't freeze anything
            print("  Encoder fully trainable in stage-2 (no freeze)")
        else:  # 'full' or legacy
            if freeze_encoder:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        if freeze_anchor_parameters:
            for name, p in self.named_parameters():
                if 'anchor_' in name:
                    p.requires_grad = False

        # Ensure frozen_projection stays frozen
        if self.frozen_projection is not None:
            for p in self.frozen_projection.parameters():
                p.requires_grad = False
            self.frozen_projection.eval()

        for module in [self.stage2_projection, self.stage2_fuser, self.reconstruction_decoder]:
            if module is not None:
                module.train()
                for p in module.parameters():
                    p.requires_grad = True

    def get_stage2_trainable_parameters(self):
        """Return stage-2 trainable parameters only (excludes frozen_projection)."""
        params = []
        for module in [self.stage2_projection, self.stage2_fuser, self.reconstruction_decoder, self.anchor_reproject]:
            if module is not None:
                params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def get_stage2_param_groups(self, base_lr: float, weight_decay: float = 1e-6):
        """Return parameter groups with per-group LR for stage-2 optimizer.

        Unfrozen backbone blocks (partial freeze) get ``base_lr * multiplier``.
        Stage-2 heads get ``base_lr``.
        """
        groups = []
        # Stage-2 heads
        head_params = self.get_stage2_trainable_parameters()
        if head_params:
            groups.append({'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay})
        # Partially unfrozen backbone blocks
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        mult = getattr(self, '_unfreeze_lr_multiplier', 0.1)
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': base_lr * mult, 'weight_decay': weight_decay})
        return groups
    
    def get_semantic_anchors(self):
        """
        Get semantic anchors (384D DINOv3 space) for pseudo-label computation.
        
        EXPERT'S APPROACH: Returns frozen DINOv3 embeddings used for labeling.
        LEGACY: Falls back to anchor_global_raw if available.
        """
        if self.use_decoupled_anchors:
            return self.anchor_semantic
        elif self.anchor_global_raw is not None:
            return self.anchor_global_raw
        else:
            raise ValueError("No semantic anchors available. Use decoupled anchors approach.")
    
    def _get_projected_anchors(self):
        """
        Get anchors in projected space for distance computation.
        
        EXPERT'S APPROACH: If use_decoupled_anchors=True, returns FIXED geometric
        targets (128D) that NEVER change during training.
        
        LEGACY: If anchors_already_projected=True, returns stored anchors directly.
        If False, re-projects raw anchors (can cause collapse!).
        """
        if self.use_decoupled_anchors:
            # EXPERT'S APPROACH: Return FIXED geometric targets (128D)
            # These NEVER move - projection head learns to map samples to these fixed points
            anchor_global = self.anchor_geometric
            anchor_dense = None  # Dense not yet implemented for decoupled approach
        elif self.anchors_already_projected:
            # LEGACY: Anchors are already in projected space - use directly
            anchor_global = self.anchor_global
            anchor_dense = self.anchor_dense
        else:
            # LEGACY: Project raw anchors through current projection head
            # WARNING: This can cause collapse because anchors move with projection!
            anchor_global = self.anchor_global_raw
            
            # Project through current projection head (same as samples)
            if self.backbone.projection is not None:
                anchor_global = self.backbone.projection(anchor_global)
            
            # Normalize to unit norm (same as samples in backbone.forward)
            anchor_global = F.normalize(anchor_global, dim=1)
            
            anchor_dense = None
            if self.anchor_dense_raw is not None:
                anchor_dense = self.anchor_dense_raw
                if self.backbone.projection is not None:
                    K, H_p, W_p, D = anchor_dense.shape
                    dense_flat = anchor_dense.view(K * H_p * W_p, D)
                    dense_flat = self.backbone.projection(dense_flat)
                    anchor_dense = dense_flat.view(K, H_p, W_p, -1)
                # Normalize dense features
                anchor_dense = F.normalize(anchor_dense, dim=-1)
        
        return anchor_global, anchor_dense
    
    def forward(self, x: torch.Tensor, return_dense: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with distance computation
        
        Args:
            x: Input images (B, C, H, W)
            return_dense: Whether to compute dense features and distances
            
        Returns:
            Dictionary with embeddings and distances to anchors
        """
        # Extract features (already projected if projection head exists)
        # Request multi-scale features if we have a pixel decoder
        B, C, H, W = x.shape
        features = self.backbone(x, return_multi_scale=self.use_pixel_decoder)
        
        global_feat = features['global']  # (B, D) or (B, D_proj)
        dense_feat = features['dense']    # (B, H', W', D) or (B, H', W', D_proj)
        
        # Get projected anchors (will use projection head if it exists)
        anchor_global, anchor_dense = self._get_projected_anchors()
        
        # Compute distances to anchors
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            cosine_sim = torch.mm(global_feat, anchor_global.t())  # (B, K)
            global_distances = 1.0 - cosine_sim  # (B, K)
        else:  # euclidean
            # L2 distance
            global_distances = torch.cdist(global_feat, anchor_global, p=2)  # (B, K)
        
        output = {
            'global_feat': global_feat,
            'global_distances': global_distances,
            'dense_feat': dense_feat
        }

        if self.reconstruction_enabled and self.stage2_projection is not None:
            global_raw = features['global_raw']
            stage2_feat = self.stage2_projection(global_raw)  # (B, recon_dim)
            stage2_feat = F.normalize(stage2_feat, dim=1)

            decoupled = getattr(self, '_recon_dim', None) is not None and self._recon_dim != self._anchor_dim

            # Frozen bottleneck → divergence signal (always in anchor space)
            frozen_feat = None
            bottleneck_divergence = None
            if self.use_frozen_bottleneck and self.frozen_projection is not None:
                with torch.no_grad():
                    frozen_feat = self.frozen_projection(global_raw)  # (B, anchor_dim)
                    frozen_feat = F.normalize(frozen_feat, dim=1)
                if decoupled:
                    # Divergence in anchor space: frozen vs current backbone projection
                    bottleneck_divergence = 1.0 - F.cosine_similarity(frozen_feat, global_feat, dim=1)
                else:
                    bottleneck_divergence = 1.0 - F.cosine_similarity(frozen_feat, stage2_feat, dim=1)

            # Per-patch divergence map
            patch_divergence_map = None
            if self.use_frozen_bottleneck and self.frozen_projection is not None and 'patch_raw' in features:
                pr = features['patch_raw']  # (B, N_patches, D_backbone)
                B_p, N_p, D_p = pr.shape
                with torch.no_grad():
                    fp = self.frozen_projection(pr.reshape(B_p * N_p, D_p))
                    fp = F.normalize(fp, dim=1).view(B_p, N_p, -1)  # (B, N, anchor_dim)
                if decoupled:
                    # Compare frozen vs backbone projection (both anchor space)
                    if self.backbone.projection is not None:
                        with torch.no_grad():
                            sp = self.backbone.projection(pr.reshape(B_p * N_p, D_p))
                            sp = F.normalize(sp, dim=1).view(B_p, N_p, -1)
                    else:
                        sp = F.normalize(pr, dim=2)
                else:
                    sp = self.stage2_projection(pr.reshape(B_p * N_p, D_p))
                    sp = F.normalize(sp, dim=1).view(B_p, N_p, -1)
                per_patch_div = 1.0 - (fp * sp).sum(dim=-1)  # (B, N_patches)
                h_p = H // self.backbone.patch_size
                w_p = W // self.backbone.patch_size
                patch_divergence_map = F.interpolate(
                    per_patch_div.view(B_p, 1, h_p, w_p),
                    size=(H, W), mode='bilinear', align_corners=False
                ).squeeze(1)  # (B, H, W)

            assigned_anchor_idx = global_distances.argmin(dim=1)
            assigned_anchor_embeddings = anchor_global[assigned_anchor_idx]  # (B, anchor_dim)
            if self.stage2_freeze_anchor_target:
                assigned_anchor_embeddings = assigned_anchor_embeddings.detach()

            # Re-project anchors to recon_dim if decoupled
            if self.anchor_reproject is not None:
                assigned_anchor_embeddings = self.anchor_reproject(assigned_anchor_embeddings)

            recon_latent = self.stage2_fuser(torch.cat([assigned_anchor_embeddings, stage2_feat], dim=1))
            reconstruction = self.reconstruction_decoder(recon_latent)
            reconstruction_error = (reconstruction - x).pow(2).mean(dim=(1, 2, 3))

            reconstruction_pixel_map = None
            if self.stage2_pixel_map_enabled:
                if self.stage2_pixel_map_type == 'reconstruction_l1':
                    reconstruction_pixel_map = (reconstruction - x).abs().mean(dim=1)
                else:
                    reconstruction_pixel_map = (reconstruction - x).pow(2).mean(dim=1)

            output.update({
                'stage2_feat': stage2_feat,
                'frozen_feat': frozen_feat,
                'bottleneck_divergence': bottleneck_divergence,
                'patch_divergence_map': patch_divergence_map,
                'recon_latent': recon_latent,
                'assigned_anchor_idx': assigned_anchor_idx,
                'assigned_anchor_embeddings': assigned_anchor_embeddings,
                'reconstruction': reconstruction,
                'reconstruction_error': reconstruction_error,
                'reconstruction_pixel_map': reconstruction_pixel_map
            })
        
        # Pixel decoder path: compute pixel-level embeddings and distances
        if self.use_pixel_decoder and self.pixel_decoder is not None:
            multi_scale_features = features.get('multi_scale')
            if multi_scale_features is not None:
                # Get pixel embeddings from decoder: (B, D, H, W)
                pixel_embeddings = self.pixel_decoder(multi_scale_features)
                
                # Normalize pixel embeddings for distance computation
                if self.distance_metric == 'cosine':
                    pixel_embeddings = F.normalize(pixel_embeddings, dim=1)
                
                output['pixel_embeddings'] = pixel_embeddings
                
                # Compute pixel-level distances to each anchor
                B, D, H, W = pixel_embeddings.shape
                K = self.n_anchors
                
                # Reshape pixel embeddings for distance computation: (B, H*W, D)
                pixel_flat = pixel_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
                
                # Compute distances to each anchor
                if self.distance_metric == 'cosine':
                    # anchor_global: (K, D), pixel_flat: (B, H*W, D)
                    # Compute cosine similarity: (B, H*W, K)
                    pixel_anchor_sim = torch.bmm(
                        pixel_flat, 
                        anchor_global.t().unsqueeze(0).expand(B, -1, -1)
                    )
                    pixel_distances = 1.0 - pixel_anchor_sim  # (B, H*W, K)
                else:  # euclidean
                    # Compute L2 distance: (B, H*W, K)
                    pixel_distances = torch.cdist(pixel_flat, anchor_global.unsqueeze(0).expand(B, -1, -1), p=2)
                
                # Reshape to spatial: (B, K, H, W)
                pixel_distances = pixel_distances.permute(0, 2, 1).reshape(B, K, H, W)
                output['pixel_distances'] = pixel_distances
        
        # Legacy dense distances (per-patch to anchor patches) - kept for backwards compatibility
        elif return_dense and anchor_dense is not None:
            B, H_p, W_p, D = dense_feat.shape
            K = self.n_anchors
            
            if self.distance_metric == 'cosine':
                # Normalize dense features
                dense_feat_norm = F.normalize(dense_feat, dim=-1)  # (B, H', W', D)
                anchor_dense_norm = F.normalize(anchor_dense, dim=-1)  # (K, H', W', D)
                
                # Compute per-patch distances to each anchor
                # Reshape for batch computation
                dense_flat = dense_feat_norm.view(B, -1, D)  # (B, H'*W', D)
                anchor_flat = anchor_dense_norm.view(K, -1, D)  # (K, H'*W', D)
                
                # Compute cosine similarity for all patches
                dense_distances = torch.zeros(B, K, H_p * W_p, device=x.device)
                
                for k in range(K):
                    # (B, H'*W', D) @ (D, H'*W') -> (B, H'*W', H'*W')
                    sim = torch.bmm(dense_flat, anchor_flat[k].t().unsqueeze(0).expand(B, -1, -1))
                    # Take diagonal (corresponding patches)
                    sim_diag = sim.diagonal(dim1=1, dim2=2)  # (B, H'*W')
                    dense_distances[:, k] = 1.0 - sim_diag
            else:  # euclidean
                # Reshape for L2 distance computation
                dense_flat = dense_feat.view(B, H_p * W_p, D)  # (B, H'*W', D)
                anchor_flat = anchor_dense.view(K, H_p * W_p, D)  # (K, H'*W', D)
                
                # Compute L2 distances
                dense_distances = torch.zeros(B, K, H_p * W_p, device=x.device)
                
                for k in range(K):
                    # L2 distance for each patch to corresponding anchor patch
                    dist = torch.norm(dense_flat - anchor_flat[k].unsqueeze(0), dim=2, p=2)  # (B, H'*W')
                    dense_distances[:, k] = dist
            
            # Reshape to spatial
            dense_distances = dense_distances.view(B, K, H_p, W_p)  # (B, K, H', W')
            
            output['dense_distances'] = dense_distances
        
        return output
    
    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        return_maps: bool = True,
        target_size: Optional[tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute anomaly scores at test time
        
        Args:
            x: Input images (B, C, H, W)
            return_maps: Whether to return pixel-level anomaly maps
            target_size: Target size for upsampling maps (H, W)
            
        Returns:
            Dictionary with image-level and pixel-level scores
        """
        with torch.no_grad():
            outputs = self.forward(x, return_dense=return_maps)
            
            # Image-level score: minimum distance to any anchor
            global_distances = outputs['global_distances']  # (B, K)
            image_scores = global_distances.min(dim=1)[0]  # (B,)
            assigned_anchors = global_distances.argmin(dim=1)  # (B,)
            
            result = {
                'image_scores': image_scores,
                'anchor_scores': image_scores,
                'assigned_anchors': assigned_anchors,
                'all_distances': global_distances
            }

            if 'reconstruction_error' in outputs:
                result['reconstruction_scores'] = outputs['reconstruction_error']
                result['reconstruction'] = outputs.get('reconstruction')

            # Bottleneck divergence (CLS-level, single scalar per image)
            if outputs.get('bottleneck_divergence') is not None:
                result['bottleneck_divergence'] = outputs['bottleneck_divergence']

            # Per-patch divergence map → aggregated image-level score
            # Spatially resolved (15×15 patches → upsampled), then top-k aggregated
            if outputs.get('patch_divergence_map') is not None:
                patch_div_map = outputs['patch_divergence_map']  # (B, H, W)
                if target_size is not None and patch_div_map.shape[1:] != target_size:
                    patch_div_map = F.interpolate(
                        patch_div_map.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False
                    ).squeeze(1)
                if return_maps:
                    result['patch_divergence_map'] = patch_div_map
                patch_div_aggregated = aggregate_pixel_scores_torch(
                    patch_div_map,
                    method=self.pixel_aggregation_method,
                    percentile=self.pixel_aggregation_percentile,
                    threshold=self.pixel_aggregation_threshold,
                )
                result['patch_divergence_aggregated_score'] = patch_div_aggregated

            # Stage-2 reconstruction pixel map (preferred for stage-2 pixel metrics)
            if outputs.get('reconstruction_pixel_map') is not None:
                reconstruction_pixel_scores = outputs['reconstruction_pixel_map']  # (B, H, W)
                if target_size is not None and reconstruction_pixel_scores.shape[1:] != target_size:
                    reconstruction_pixel_scores = F.interpolate(
                        reconstruction_pixel_scores.unsqueeze(1),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)

                if return_maps:
                    result['reconstruction_pixel_scores'] = reconstruction_pixel_scores
                    result['pixel_scores'] = reconstruction_pixel_scores

                # Pixel-to-image aggregation — always computed (not gated by return_maps)
                pixel_aggregated = aggregate_pixel_scores_torch(
                    reconstruction_pixel_scores,
                    method=self.pixel_aggregation_method,
                    percentile=self.pixel_aggregation_percentile,
                    threshold=self.pixel_aggregation_threshold,
                )
                result['pixel_aggregated_score'] = pixel_aggregated
            
            # Pixel-level anomaly map from decoder (preferred)
            if return_maps and 'pixel_distances' in outputs:
                pixel_distances = outputs['pixel_distances']  # (B, K, H, W) - true pixel-level!
                
                # Min distance across anchors for each pixel
                pixel_scores, _ = pixel_distances.min(dim=1)  # (B, H, W)
                
                # Resize if needed (decoder already outputs at target_size, but allow override)
                if target_size is not None and pixel_scores.shape[1:] != target_size:
                    pixel_scores = F.interpolate(
                        pixel_scores.unsqueeze(1),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
                result['anchor_pixel_scores'] = pixel_scores
                if 'pixel_scores' not in result:
                    result['pixel_scores'] = pixel_scores
                result['pixel_embeddings'] = outputs.get('pixel_embeddings')
            
            # Fallback: patch-level anomaly map (legacy, upsampled)
            elif return_maps and 'dense_distances' in outputs:
                dense_distances = outputs['dense_distances']  # (B, K, H', W')
                
                # Min distance across anchors for each patch
                pixel_scores, _ = dense_distances.min(dim=1)  # (B, H', W')
                
                # Upsample to image size
                if target_size is not None:
                    pixel_scores = F.interpolate(
                        pixel_scores.unsqueeze(1),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)  # (B, H, W)
                
                result['anchor_pixel_scores'] = pixel_scores
                if 'pixel_scores' not in result:
                    result['pixel_scores'] = pixel_scores

            # Optional raw combination output (final AUROC combination should be dataset-level in eval)
            if self.score_combination_enabled and ('anchor_scores' in result) and ('reconstruction_scores' in result):
                alpha = self.score_combination_alpha
                result['combined_scores_raw'] = (1.0 - alpha) * result['anchor_scores'] + alpha * result['reconstruction_scores']
            
            return result