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
        min_norm: float = 0.5,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            margin: Margin m for repeller term (anchors should be >= 2m apart)
            alpha: Weight for attractor loss (default: 1.0)
            beta: Weight for repeller loss (default: 1.0)
            gamma: Weight for min-norm loss (default: 0.0, use 0.1 for learnable anchors)
            min_norm: Minimum norm threshold for anchors (default: 0.5)
            distance_metric: 'euclidean' or 'cosine' distance
        """
        super().__init__()
        
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_norm = min_norm
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
        
        # === MIN-NORM TERM (for learnable anchors) ===
        # L_N(C) = (1/K) * Σₖ max(0, δ - ||c_k||_2)
        # Prevent anchor collapse to zero (still meaningful for cosine if anchors drift)
        loss_norm = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        if self.gamma > 0:
            anchor_norms = torch.norm(anchor_embeddings, p=2, dim=1)  # (K,)
            norm_violations = torch.relu(self.min_norm - anchor_norms)  # (K,)
            loss_norm = norm_violations.mean()
        
        # === COMBINED LOSS ===
        # L_total = λ₁ * L_attract + λ₂ * L_repel + λ₃ * L_norm
        total_loss = self.alpha * loss_attract + self.beta * loss_repel + self.gamma * loss_norm
        
        result = {
            'loss': total_loss,
            'loss_attract': loss_attract.item(),
            'loss_repel': loss_repel.item(),
            'loss_norm': loss_norm.item() if self.gamma > 0 else 0.0
        }
        
        if return_components:
            # Additional statistics
            result.update({
                'min_distance': min_distances.mean().item(),
                'max_distance': distances.max().item(),
                'mean_distance': distances.mean().item(),
                'assigned_anchors': assigned_anchors,
                'anchor_min_separation': anchor_distances[mask].min().item(),
                'anchor_mean_separation': anchor_distances[mask].mean().item()
            })
        
        return result


class DenseAnchorMarginLoss(nn.Module):
    """
    Anchor-Margin Loss for dense features (per-patch)
    
    Note: The repeller term doesn't apply to dense features in the paper formulation
    (it operates on anchor-anchor distances). Here we only use the attractor term.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 1.0,
        beta: float = 0.0,  # No repeller for dense
        spatial_reduction: str = 'mean'
    ):
        """
        Args:
            margin: Not used in dense attractor (kept for compatibility)
            alpha: Weight for attractor loss
            beta: Weight for repeller loss (not used, kept for compatibility)
            spatial_reduction: How to aggregate spatial losses ('mean', 'max')
        """
        super().__init__()
        
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.spatial_reduction = spatial_reduction
    
    def forward(
        self,
        dense_embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dense anchor-margin loss (disabled: kept for reference)
        
        Args:
            dense_embeddings: (B, D, H', W') per-patch embeddings
            anchor_embeddings: (K, D) anchor embeddings
            return_components: Whether to return loss components
            
        Returns:
            Dictionary with loss values
        """
        # Dense loss path intentionally disabled until a decoder-based pixel head exists.
        # Returning zeros keeps the interface stable without contributing to total loss.
        device = dense_embeddings.device
        zero = torch.tensor(0.0, device=device)
        result = {
            'loss': zero,
            'loss_attract': 0.0,
            'loss_repel': 0.0
        }
        if return_components:
            result.update({
                'min_distance': 0.0,
                'spatial_max_distance': 0.0
            })
        return result


class CombinedAnchorLoss(nn.Module):
    """
    Combined loss for global and dense features
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
            dense_loss: Loss for dense features (optional)
            global_weight: Weight for global loss
            dense_weight: Weight for dense loss
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
                - 'dense_feat': (B, H', W', D) embeddings [optional]
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
        
        # Dense loss is currently disabled (per-patch, not pixel-wise). Keep stub for future decoder.
        result['loss'] = total_loss
        
        return result