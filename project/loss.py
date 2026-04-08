"""
Anchor-Margin Loss: Attractor + Repeller Terms (paper-accurate)
Implements Class Anchor Margin Loss from https://arxiv.org/abs/2306.00630

Attractor: L_A(x_i, C) = (1/2) * ||e_i - c_{y_i}||_2^2
    Pull samples toward their assigned anchor using squared L2 distance

Repeller: L_R(C) = (1/2) * Σ_{y≠y'} max(0, 2m - ||c_y - c_{y'}||_2)^2
    Push different anchors apart to maintain margin separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AnchorMarginLoss(nn.Module):
    """
    Anchor-Margin Loss from "Class Anchor Margin Loss for Content-Based Image Retrieval"
    (arXiv:2306.00630)
    
    Attractor: Pull samples of same class toward their anchor (tight intra-class)
    Repeller: Push different class anchors apart (clear inter-class separation)
    Min-Norm: Prevent anchor collapse to zero (optional, for learnable anchors)
    
    NOTE: This version uses L2 (Euclidean) distance, not cosine distance.
    Features should be normalized if using normalized embeddings.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        delta: float = 0.0,
        min_norm: float = 0.5,
        diversity_temperature: float = 0.1,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            margin: Margin m for repeller term (anchors should be >= 2m apart)
            alpha: Weight for attractor loss (default: 1.0)
            beta: Weight for repeller loss (default: 1.0)
            gamma: Weight for min-norm loss (default: 0.0, use 0.1 for learnable anchors)
            delta: Weight for diversity loss (default: 0.0, use 0.1 to prevent collapse)
            min_norm: Minimum norm threshold for anchors (default: 0.5)
            diversity_temperature: Temperature for soft assignments (default: 0.1)
            distance_metric: 'euclidean' or 'cosine' distance
        """
        super().__init__()
        
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.min_norm = min_norm
        self.diversity_temperature = diversity_temperature
        self.distance_metric = distance_metric
        
    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        return_components: bool = False,
        fixed_assignments: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute anchor-margin loss following the paper formulation
        
        Args:
            embeddings: (B, D) feature embeddings for batch samples
            anchor_embeddings: (K, D) anchor embeddings for K anchors
            return_components: Whether to return loss components separately
            
        Returns:
            Dictionary with total loss and optional components
        """
        B, D = embeddings.shape
        K, _ = anchor_embeddings.shape
        
        # === COMPUTE DISTANCES (paper-accurate; normalize for cosine) ===
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(embeddings, anchor_embeddings, p=2)
            anchors_for_repeller = anchor_embeddings
        else:  # cosine
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            anchors_norm = F.normalize(anchor_embeddings, p=2, dim=1)
            similarities = embeddings_norm @ anchors_norm.T  # (B, K)
            distances = 1.0 - similarities
            anchors_for_repeller = anchors_norm  # keep anchors unit-norm for repeller
        
        # Find nearest anchor for each sample, or use fixed pseudo-labels if provided
        if fixed_assignments is not None:
            assigned_anchors = fixed_assignments
            min_distances = distances[torch.arange(B, device=embeddings.device), assigned_anchors]
        else:
            min_distances, assigned_anchors = distances.min(dim=1)  # (B,), (B,)
        
        # === ATTRACTOR TERM (paper) ===
        # 0.5 * ||e_i - c_{y_i}||^2, averaged over batch
        loss_attract = 0.5 * (min_distances ** 2).mean()
        
        # === REPELLER TERM ===
        # L_R(C) = (1/(K(K-1))) * Σⱼ≠ₖ max(0, m - ||c_j - c_k||_2)
        # Push different anchors apart by at least margin m
        # Skip if K=1 (no pairs to push apart)
        
        loss_repel = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        if K > 1:
            # Compute all pairwise anchor distances: (K, K)
            if self.distance_metric == 'euclidean':
                anchor_distances = torch.cdist(anchors_for_repeller, anchors_for_repeller, p=2)
            else:
                anchor_sims = anchors_for_repeller @ anchors_for_repeller.T
                anchor_distances = 1.0 - anchor_sims
            
            # Create mask to exclude diagonal (self-distances)
            mask = ~torch.eye(K, dtype=torch.bool, device=anchor_distances.device)
            
            # Paper hinge: 0.5 * (max(0, 2m - ||c_j - c_k||))^2
            violations = torch.relu(2 * self.margin - anchor_distances)
            violations_masked = violations[mask]
            loss_repel = 0.5 * (violations_masked ** 2).mean()
        else:
            anchor_distances = None
            mask = None
        
        # === MIN-NORM TERM (for learnable anchors) ===
        # L_N(C) = (1/K) * Σₖ max(0, δ - ||c_k||_2)
        # Prevent anchor collapse to zero (still meaningful for cosine if anchors drift)
        loss_norm = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        if self.gamma > 0:
            anchor_norms = torch.norm(anchor_embeddings, p=2, dim=1)  # (K,)
            norm_violations = torch.relu(self.min_norm - anchor_norms)  # (K,)
            loss_norm = norm_violations.mean()
        
        # === DIVERSITY TERM (prevent anchor collapse) ===
        # Encourage balanced anchor usage via entropy regularization
        # Skip if K=1 (nothing to balance)
        loss_diversity = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        if self.delta > 0 and K > 1:
            # Compute soft assignments using temperature-scaled softmax
            soft_assignments = torch.softmax(-distances / self.diversity_temperature, dim=1)  # (B, K)
            
            # Average assignment probabilities across batch
            avg_assignments = soft_assignments.mean(dim=0)  # (K,)
            
            # Diversity loss: negative entropy (high entropy = balanced distribution)
            # We want to MAXIMIZE entropy, so minimize negative entropy
            entropy = -(avg_assignments * torch.log(avg_assignments + 1e-8)).sum()
            
            # Normalize by max entropy: log(K)
            max_entropy = torch.log(torch.tensor(K, dtype=torch.float32, device=embeddings.device))
            loss_diversity = 1.0 - (entropy / max_entropy)  # 0 = perfect balance, 1 = total collapse
        
        # === COMBINED LOSS ===
        # L_total = λ₁ * L_attract + λ₂ * L_repel + λ₃ * L_norm + λ₄ * L_diversity
        total_loss = self.alpha * loss_attract + self.beta * loss_repel + self.gamma * loss_norm + self.delta * loss_diversity
        
        result = {
            'loss': total_loss,
            'loss_attract': loss_attract.item(),
            'loss_repel': loss_repel.item(),
            'loss_norm': loss_norm.item() if self.gamma > 0 else 0.0,
            'loss_diversity': loss_diversity.item() if self.delta > 0 else 0.0
        }
        
        if return_components:
            # Additional statistics
            result.update({
                'min_distance': min_distances.mean().item(),
                'max_distance': distances.max().item(),
                'mean_distance': distances.mean().item(),
                'assigned_anchors': assigned_anchors
            })
            
            # Only compute anchor separation stats if K > 1
            if K > 1 and anchor_distances is not None and mask is not None:
                result.update({
                    'anchor_min_separation': anchor_distances[mask].min().item(),
                    'anchor_mean_separation': anchor_distances[mask].mean().item()
                })
        
        return result


class DenseAnchorMarginLoss(nn.Module):
    """
    Anchor-Margin Loss for pixel-level embeddings from decoder.
    
    Applies the attractor loss to each pixel embedding, pulling pixels
    toward their nearest anchor (or assigned anchor via pseudo-labels).
    This enables self-supervised dense anomaly detection without GT masks.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 1.0,
        distance_metric: str = 'euclidean',
        spatial_reduction: str = 'mean'
    ):
        """
        Args:
            margin: Not used in dense attractor (kept for compatibility)
            alpha: Weight for attractor loss
            distance_metric: 'euclidean' or 'cosine' distance
            spatial_reduction: How to aggregate spatial losses ('mean', 'max')
        """
        super().__init__()
        
        self.margin = margin
        self.alpha = alpha
        self.distance_metric = distance_metric
        self.spatial_reduction = spatial_reduction
    
    def forward(
        self,
        pixel_embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dense anchor-margin loss for pixel embeddings.
        
        Args:
            pixel_embeddings: (B, D, H, W) pixel-level embeddings from decoder
            anchor_embeddings: (K, D) anchor embeddings
            return_components: Whether to return loss components
            
        Returns:
            Dictionary with loss values
        """
        B, D, H, W = pixel_embeddings.shape
        K = anchor_embeddings.shape[0]
        device = pixel_embeddings.device
        
        # Reshape pixel embeddings: (B, D, H, W) -> (B, H*W, D)
        pixel_flat = pixel_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        # Normalize if using cosine distance
        if self.distance_metric == 'cosine':
            pixel_flat = F.normalize(pixel_flat, p=2, dim=-1)
            anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1)
        else:
            anchor_norm = anchor_embeddings
        
        # Compute distances from each pixel to each anchor: (B, H*W, K)
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarities = torch.bmm(
                pixel_flat,
                anchor_norm.t().unsqueeze(0).expand(B, -1, -1)
            )  # (B, H*W, K)
            distances = 1.0 - similarities
        else:  # euclidean
            # L2 distance
            distances = torch.cdist(pixel_flat, anchor_norm.unsqueeze(0).expand(B, -1, -1), p=2)  # (B, H*W, K)
        
        # Find minimum distance to any anchor for each pixel
        min_distances, assigned_anchors = distances.min(dim=-1)  # (B, H*W)
        
        # Attractor loss: pull each pixel toward its nearest anchor
        # L_A = 0.5 * ||e_pixel - c_nearest||^2
        loss_attract = 0.5 * (min_distances ** 2)  # (B, H*W)
        
        # Spatial reduction
        if self.spatial_reduction == 'mean':
            loss_attract = loss_attract.mean()
        elif self.spatial_reduction == 'max':
            loss_attract = loss_attract.max(dim=-1)[0].mean()
        else:
            loss_attract = loss_attract.mean()
        
        total_loss = self.alpha * loss_attract
        
        result = {
            'loss': total_loss,
            'loss_attract': loss_attract.item() if isinstance(loss_attract, torch.Tensor) else loss_attract
        }
        
        if return_components:
            result.update({
                'min_distance': min_distances.mean().item(),
                'spatial_max_distance': min_distances.max().item(),
                'assigned_anchors_spatial': assigned_anchors.reshape(B, H, W)
            })
        
        return result


class CombinedAnchorLoss(nn.Module):
    """
    Combined loss for global and pixel-level features.
    
    Combines:
    - Global loss: AnchorMarginLoss on CLS token embeddings
    - Dense loss: DenseAnchorMarginLoss on pixel embeddings from decoder
    """
    
    def __init__(
        self,
        global_loss: AnchorMarginLoss,
        dense_loss: Optional[DenseAnchorMarginLoss] = None,
        global_weight: float = 1.0,
        dense_weight: float = 0.5
    ):
        """
        Args:
            global_loss: Loss for global features
            dense_loss: Loss for pixel-level features (optional)
            global_weight: Weight for global loss
            dense_weight: Weight for dense/pixel loss
        """
        super().__init__()
        
        self.global_loss = global_loss
        self.dense_loss = dense_loss
        self.global_weight = global_weight
        self.dense_weight = dense_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor], anchor_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: Dictionary from model forward pass with:
                - 'global_feat': (B, D) embeddings
                - 'pixel_embeddings': (B, D, H, W) pixel embeddings from decoder [optional]
            anchor_embeddings: (K, D) anchor embeddings
        
        Returns:
            Dictionary with losses
        """
        # Global loss (optionally with fixed pseudo-labels if provided in outputs)
        fixed_assignments = outputs.get('fixed_assignments')
        global_result = self.global_loss(
            outputs['global_feat'], 
            anchor_embeddings, 
            return_components=True,
            fixed_assignments=fixed_assignments
        )
        total_loss = self.global_weight * global_result['loss']
        
        result = {
            'loss': total_loss,
            'loss_global': global_result['loss'],
            'loss_global_attract': global_result['loss_attract'],
            'loss_global_repel': global_result['loss_repel'],
            'loss_global_norm': global_result.get('loss_norm', 0.0),
            'assigned_anchors': global_result['assigned_anchors']
        }
        
        # Dense/pixel loss if decoder outputs are available
        if self.dense_loss is not None and 'pixel_embeddings' in outputs:
            pixel_embeddings = outputs['pixel_embeddings']  # (B, D, H, W)
            dense_result = self.dense_loss(
                pixel_embeddings,
                anchor_embeddings,
                return_components=True
            )
            
            dense_loss = self.dense_weight * dense_result['loss']
            total_loss = total_loss + dense_loss
            
            result.update({
                'loss_dense': dense_result['loss'],
                'loss_dense_attract': dense_result['loss_attract'],
                'dense_min_distance': dense_result.get('min_distance', 0.0),
                'dense_max_distance': dense_result.get('spatial_max_distance', 0.0)
            })
        
        result['loss'] = total_loss
        
        return result