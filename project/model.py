"""
DINOv3 Backbone Wrapper with Global and Dense Feature Extraction
Supports frozen and finetunable modes
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DINOv3Backbone(nn.Module):
    """
    DINOv3 feature extractor with global and dense outputs
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone: bool = True,
        projection_dim: Optional[int] = None,
        pretrained: bool = True
    ):
        """
        Args:
            model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', etc.)
            freeze_backbone: Whether to freeze backbone weights
            projection_dim: If set, add trainable projection head to this dimension
            pretrained: Load pretrained weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.projection_dim = projection_dim

        # Load DINOv2 model from timm
        print(f"Loading {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained) 

        # Get embedding dimension
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        
        print(f"Backbone embed_dim: {self.embed_dim}, patch_size: {self.patch_size}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("Backbone frozen")
        
        # Trainable projection head for learning better embeddings
        self.projection = None
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 2, projection_dim)
            )
            print(f"Added trainable projection head: {self.embed_dim} -> {self.embed_dim // 2} -> {projection_dim}")
            print(f"  Trainable parameters: {sum(p.numel() for p in self.projection.parameters() if p.requires_grad):,}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract global and dense features
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Dictionary with:
                'global': (B, D) global embedding
                'dense': (B, H', W', D) dense feature map
        """
        B, C, H, W = x.shape
        
        # Get patch embeddings from backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.forward_features(x)
        else:
            features = self.backbone.forward_features(x)
        
        # DINOv3 returns a tensor (B, N_tokens, D) where N_tokens = 1 (cls) + N_register + N_patches
        # DINOv3 models have 4 register tokens after the CLS token
        # features shape: (B, N_tokens, D)
        # Token order: [CLS, REG1, REG2, REG3, REG4, PATCH1, PATCH2, ...]
        cls_token = features[:, 0]  # (B, D)
        
        # DINOv3 has 4 register tokens, skip them
        num_register_tokens = 4
        patch_tokens = features[:, 1 + num_register_tokens:]  # (B, N_patches, D)
        
        # Global embedding (from CLS token)
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
        
        return {
            'global': global_feat,
            'dense': dense_feat
        }
    
    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if requested"""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class AnomalyDetector(nn.Module):
    """
    Complete anomaly detection model with DINOv3 backbone and anchor-based scoring
    """
    
    def __init__(
        self,
        backbone: DINOv3Backbone,
        anchor_global_embeddings: torch.Tensor,
        anchor_dense_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Args:
            backbone: DINOv3Backbone model
            anchor_global_embeddings: (K, D) fixed anchor embeddings in backbone space
            anchor_dense_embeddings: (K, H', W', D) fixed dense anchor features in backbone space
        """
        super().__init__()
        
        self.backbone = backbone
        
        # Store original anchors in backbone space (not trainable)
        self.register_buffer('anchor_global_original', anchor_global_embeddings)
        
        if anchor_dense_embeddings is not None:
            self.register_buffer('anchor_dense_original', anchor_dense_embeddings)
        else:
            self.anchor_dense_original = None
        
        self.n_anchors = len(anchor_global_embeddings)
        
        print(f"Initialized detector with {self.n_anchors} anchors")
        if backbone.projection is not None:
            print(f"  Anchors will be projected through trainable head during forward pass")
    
    def _get_projected_anchors(self):
        """Get anchors projected through the trainable head (if exists)"""
        if self.backbone.projection is None:
            # No projection head - use original anchors
            return self.anchor_global_original, self.anchor_dense_original
        
        # Project anchors through the trainable head
        anchor_global_projected = self.backbone.projection(self.anchor_global_original)
        anchor_global_projected = F.normalize(anchor_global_projected, dim=1)
        
        # Project dense anchors if they exist
        anchor_dense_projected = None
        if self.anchor_dense_original is not None:
            K, H_p, W_p, D = self.anchor_dense_original.shape
            # Reshape and project
            dense_flat = self.anchor_dense_original.view(K, H_p * W_p, D)  # (K, H'*W', D)
            dense_projected = self.backbone.projection(dense_flat.view(-1, D))  # (K*H'*W', D_proj)
            anchor_dense_projected = dense_projected.view(K, H_p, W_p, -1)  # (K, H', W', D_proj)
        
        return anchor_global_projected, anchor_dense_projected
    
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
        features = self.backbone(x)
        
        global_feat = features['global']  # (B, D) or (B, D_proj)
        dense_feat = features['dense']    # (B, H', W', D) or (B, H', W', D_proj)
        
        # Get projected anchors (will use projection head if it exists)
        anchor_global, anchor_dense = self._get_projected_anchors()
        
        # Compute distances to anchors (cosine distance = 1 - cosine similarity)
        # Global distances
        cosine_sim = torch.mm(global_feat, anchor_global.t())  # (B, K)
        global_distances = 1.0 - cosine_sim  # (B, K)
        
        output = {
            'global_feat': global_feat,
            'global_distances': global_distances,
            'dense_feat': dense_feat
        }
        
        # Dense distances (per-patch to anchor patches)
        if return_dense and anchor_dense is not None:
            B, H_p, W_p, D = dense_feat.shape
            K = self.n_anchors
            
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
                'assigned_anchors': assigned_anchors,
                'all_distances': global_distances
            }
            
            # Pixel-level anomaly map
            if return_maps and 'dense_distances' in outputs:
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
                
                result['pixel_scores'] = pixel_scores
            
            return result