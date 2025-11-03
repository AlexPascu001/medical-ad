"""
Anchor-Margin Loss: Attractor + Repeller Terms
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
    
    NOTE: This version uses L2 (Euclidean) distance, not cosine distance.
    Features should be normalized if using normalized embeddings.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            margin: Margin m for repeller term (anchors should be >= 2m apart)
            alpha: Weight for attractor loss (default: 1.0)
            beta: Weight for repeller loss (default: 1.0)
            distance_metric: 'euclidean' or 'cosine' distance
        """
        super().__init__()
        
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.distance_metric = distance_metric
        
    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        return_components: bool = False
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
        
        # === COMPUTE DISTANCES ===
        if self.distance_metric == 'euclidean':
            # Compute pairwise L2 distances: ||e_i - c_k||_2
            # Shape: (B, K)
            distances = torch.cdist(embeddings, anchor_embeddings, p=2)
        else:  # cosine
            # Normalize and compute cosine distance
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            anchors_norm = F.normalize(anchor_embeddings, p=2, dim=1)
            # Cosine distance = 1 - cosine similarity
            similarities = embeddings_norm @ anchors_norm.T  # (B, K)
            distances = 1.0 - similarities
        
        # Find nearest anchor for each sample
        min_distances, assigned_anchors = distances.min(dim=1)  # (B,), (B,)
        
        # === ATTRACTOR TERM ===
        # L_A(x_i, C) = (1/2) * ||e_i - c_{y_i}||_2^2
        # Pull samples toward their assigned anchor
        loss_attract = 0.5 * (min_distances ** 2).mean()
        
        # === REPELLER TERM ===
        # L_R(C) = (1/2) * Σ_{y≠y'} max(0, 2m - ||c_y - c_{y'}||_2)^2
        # Push different anchors apart (operates on anchor-anchor distances)
        
        # Compute all pairwise anchor distances: (K, K)
        if self.distance_metric == 'euclidean':
            anchor_distances = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)
        else:
            anchor_sims = anchors_norm @ anchors_norm.T
            anchor_distances = 1.0 - anchor_sims
        
        # Create mask to exclude diagonal (self-distances)
        mask = ~torch.eye(K, dtype=torch.bool, device=anchor_distances.device)
        
        # Compute hinge loss: max(0, 2m - ||c_y - c_{y'}||_2)
        violations = torch.relu(2.0 * self.margin - anchor_distances)  # (K, K)
        
        # Square and sum over all pairs (excluding diagonal)
        violations_masked = violations * mask.float()
        loss_repel = 0.5 * (violations_masked ** 2).sum() / (K * (K - 1))  # Average over pairs
        
        # === COMBINED LOSS ===
        total_loss = self.alpha * loss_attract + self.beta * loss_repel
        
        result = {
            'loss': total_loss,
            'loss_attract': loss_attract.item(),
            'loss_repel': loss_repel.item()
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
        Compute dense anchor-margin loss
        
        Args:
            dense_embeddings: (B, D, H', W') per-patch embeddings
            anchor_embeddings: (K, D) anchor embeddings
            return_components: Whether to return loss components
            
        Returns:
            Dictionary with loss values
        """
        B, D, H, W = dense_embeddings.shape
        K, _ = anchor_embeddings.shape
        
        # Reshape: (B, D, H, W) -> (B, H*W, D)
        embeddings_flat = dense_embeddings.permute(0, 2, 3, 1).reshape(B * H * W, D)
        
        # Compute distances to all anchors: (B*H*W, K)
        distances = torch.cdist(embeddings_flat, anchor_embeddings, p=2)
        
        # Find nearest anchor per patch
        min_distances, assigned = distances.min(dim=1)  # (B*H*W,)
        
        # === ATTRACTOR TERM ===
        # L_A = (1/2) * ||e_patch - c_nearest||_2^2
        loss_attract = 0.5 * (min_distances ** 2)
        
        if self.spatial_reduction == 'mean':
            loss_attract = loss_attract.mean()
        else:  # max
            loss_attract = loss_attract.view(B, H * W).max(dim=1)[0].mean()
        
        # No repeller term for dense features (operates on anchors, not patches)
        loss_repel = torch.tensor(0.0, device=dense_embeddings.device)
        
        # === COMBINED ===
        total_loss = self.alpha * loss_attract
        
        result = {
            'loss': total_loss,
            'loss_attract': loss_attract.item(),
            'loss_repel': loss_repel.item()
        }
        
        if return_components:
            result.update({
                'min_distance': min_distances.mean().item(),
                'spatial_max_distance': distances.max().item()
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
        # Global loss
        global_result = self.global_loss(
            outputs['global_feat'], 
            anchor_embeddings, 
            return_components=True
        )
        total_loss = self.global_weight * global_result['loss']
        
        result = {
            'loss': total_loss,
            'loss_global': global_result['loss'],
            'loss_global_attract': global_result['loss_attract'],
            'loss_global_repel': global_result['loss_repel'],
            'assigned_anchors': global_result['assigned_anchors']
        }
        
        # Dense loss if available
        if self.dense_loss is not None and 'dense_feat' in outputs:
            # Reshape dense features from (B, H', W', D) to (B, D, H', W')
            dense_feat = outputs['dense_feat'].permute(0, 3, 1, 2)
            dense_result = self.dense_loss(
                dense_feat,
                anchor_embeddings,
                return_components=True
            )
            total_loss = total_loss + self.dense_weight * dense_result['loss']
            
            result.update({
                'loss': total_loss,
                'loss_dense': dense_result['loss'],
                'loss_dense_attract': dense_result['loss_attract'],
                'loss_dense_repel': dense_result['loss_repel']
            })
        
        return result