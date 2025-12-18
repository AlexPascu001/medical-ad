"""
Learnable Anchors with Class Anchor Margin (CAM) Loss

Based on: "Class Anchor Margin Loss for Content-Based Image Retrieval"
         Ghita & Ionescu, arXiv:2306.00630

This module implements:
1. Learnable anchor embeddings initialized from fixed anchors (eigenface/kmeans/random)
2. CAM Loss with three components:
   - Attractor: Pulls embeddings toward their assigned anchor
   - Repeller: Pushes anchors apart by a margin
   - Min-Norm: Prevents anchor collapse to zero
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LearnableAnchors(nn.Module):
    """
    Learnable anchor embeddings for anomaly detection
    
    Args:
        initial_anchors: (K, D) tensor of initial anchor embeddings
        freeze: If True, anchors are not learnable (for baseline comparison)
    """
    
    def __init__(self, initial_anchors: torch.Tensor, freeze: bool = False):
        super().__init__()
        
        self.n_anchors, self.dim = initial_anchors.shape
        
        # Create learnable parameter
        self.anchors = nn.Parameter(initial_anchors.clone(), requires_grad=not freeze)
        
        if freeze:
            print(f"  Anchors frozen (not learnable)")
        else:
            print(f"  Anchors are learnable: {self.n_anchors} anchors × {self.dim}D")
    
    def forward(self) -> torch.Tensor:
        """Return current anchor embeddings"""
        return self.anchors
    
    def get_anchor_norms(self) -> torch.Tensor:
        """Get L2 norms of all anchors"""
        return torch.norm(self.anchors, p=2, dim=1)
    
    def get_pairwise_distances(self) -> torch.Tensor:
        """Get pairwise distances between all anchors (K × K matrix)"""
        # anchors: (K, D)
        # Compute pairwise L2 distances
        diff = self.anchors.unsqueeze(0) - self.anchors.unsqueeze(1)  # (K, K, D)
        distances = torch.norm(diff, p=2, dim=2)  # (K, K)
        return distances


class CAMLoss(nn.Module):
    """
    Class Anchor Margin (CAM) Loss for anomaly detection with learnable anchors
    
    Loss = λ₁ * L_attractor + λ₂ * L_repeller + λ₃ * L_norm
    
    Components:
    1. Attractor Loss: Pulls embeddings toward assigned anchor
       L_attractor = (1/N) Σᵢ ||z_i - c_{y_i}||²
    
    2. Repeller Loss: Pushes anchors apart by margin m
       L_repeller = (1/(K(K-1))) Σⱼ≠ₖ max(0, m - ||c_j - c_k||)
    
    3. Min-Norm Loss: Prevents anchor collapse
       L_norm = (1/K) Σₖ max(0, δ - ||c_k||)
    
    Args:
        lambda_attractor: Weight for attractor loss (default: 1.0)
        lambda_repeller: Weight for repeller loss (default: 1.0)
        lambda_norm: Weight for min-norm loss (default: 0.1)
        margin: Minimum distance between anchors (default: 1.0)
        min_norm: Minimum anchor norm threshold (default: 0.5)
        distance_metric: 'euclidean' or 'cosine'
    """
    
    def __init__(
        self,
        lambda_attractor: float = 1.0,
        lambda_repeller: float = 1.0,
        lambda_norm: float = 0.1,
        margin: float = 1.0,
        min_norm: float = 0.5,
        distance_metric: str = 'euclidean'
    ):
        super().__init__()
        
        self.lambda_attractor = lambda_attractor
        self.lambda_repeller = lambda_repeller
        self.lambda_norm = lambda_norm
        self.margin = margin
        self.min_norm = min_norm
        self.distance_metric = distance_metric
        
        print(f"\nCAM Loss Configuration:")
        print(f"  λ_attractor: {lambda_attractor}")
        print(f"  λ_repeller: {lambda_repeller}")
        print(f"  λ_norm: {lambda_norm}")
        print(f"  Margin (m): {margin}")
        print(f"  Min-Norm (δ): {min_norm}")
        print(f"  Distance: {distance_metric}")
    
    def compute_attractor_loss(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
        anchor_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Attractor loss: Pull embeddings toward their assigned anchor
        
        Args:
            embeddings: (N, D) embeddings
            anchors: (K, D) anchor embeddings
            anchor_assignments: (N,) indices of assigned anchors
        
        Returns:
            Scalar loss
        """
        # Get assigned anchors for each embedding
        assigned_anchors = anchors[anchor_assignments]  # (N, D)
        
        # Compute distances
        if self.distance_metric == 'euclidean':
            # Squared L2 distance
            distances = torch.sum((embeddings - assigned_anchors) ** 2, dim=1)  # (N,)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            anchors_norm = F.normalize(assigned_anchors, p=2, dim=1)
            distances = 1 - torch.sum(embeddings_norm * anchors_norm, dim=1)  # (N,)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Mean over batch
        loss = torch.mean(distances)
        return loss
    
    def compute_repeller_loss(
        self,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Repeller loss: Push anchors apart by at least margin m
        
        Args:
            anchors: (K, D) anchor embeddings
        
        Returns:
            Scalar loss
        """
        K = anchors.shape[0]
        
        if K == 1:
            # No repeller loss for single anchor
            return torch.tensor(0.0, device=anchors.device)
        
        # Compute pairwise distances (K × K)
        if self.distance_metric == 'euclidean':
            diff = anchors.unsqueeze(0) - anchors.unsqueeze(1)  # (K, K, D)
            distances = torch.norm(diff, p=2, dim=2)  # (K, K)
        elif self.distance_metric == 'cosine':
            anchors_norm = F.normalize(anchors, p=2, dim=1)
            # Cosine similarity matrix
            sim_matrix = torch.mm(anchors_norm, anchors_norm.t())  # (K, K)
            distances = 1 - sim_matrix  # Convert to distance
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Create mask to exclude diagonal (distance to self)
        mask = ~torch.eye(K, dtype=torch.bool, device=anchors.device)
        
        # Get off-diagonal distances
        off_diag_distances = distances[mask]  # (K*(K-1),)
        
        # Apply margin: max(0, m - distance)
        violations = F.relu(self.margin - off_diag_distances)
        
        # Mean over all pairs
        loss = torch.mean(violations)
        return loss
    
    def compute_norm_loss(
        self,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Min-Norm loss: Prevent anchor collapse to zero
        
        Args:
            anchors: (K, D) anchor embeddings
        
        Returns:
            Scalar loss
        """
        # Compute L2 norms
        norms = torch.norm(anchors, p=2, dim=1)  # (K,)
        
        # Apply threshold: max(0, δ - norm)
        violations = F.relu(self.min_norm - norms)
        
        # Mean over anchors
        loss = torch.mean(violations)
        return loss
    
    def forward(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
        anchor_assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total CAM loss
        
        Args:
            embeddings: (N, D) image embeddings
            anchors: (K, D) learnable anchor embeddings
            anchor_assignments: (N,) indices of assigned anchors for each embedding
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        # Compute individual components
        attractor_loss = self.compute_attractor_loss(embeddings, anchors, anchor_assignments)
        repeller_loss = self.compute_repeller_loss(anchors)
        norm_loss = self.compute_norm_loss(anchors)
        
        # Weighted sum
        total_loss = (
            self.lambda_attractor * attractor_loss +
            self.lambda_repeller * repeller_loss +
            self.lambda_norm * norm_loss
        )
        
        # Return loss and components for logging
        loss_dict = {
            'total': total_loss.item(),
            'attractor': attractor_loss.item(),
            'repeller': repeller_loss.item(),
            'norm': norm_loss.item()
        }
        
        return total_loss, loss_dict


def assign_to_nearest_anchor(
    embeddings: torch.Tensor,
    anchors: torch.Tensor,
    distance_metric: str = 'euclidean'
) -> torch.Tensor:
    """
    Assign each embedding to its nearest anchor
    
    Args:
        embeddings: (N, D) embeddings
        anchors: (K, D) anchors
        distance_metric: 'euclidean' or 'cosine'
    
    Returns:
        assignments: (N,) tensor of anchor indices
    """
    N = embeddings.shape[0]
    K = anchors.shape[0]
    
    if distance_metric == 'euclidean':
        # Compute pairwise L2 distances: (N, K)
        # ||x - c||² = ||x||² + ||c||² - 2<x, c>
        emb_sq = torch.sum(embeddings ** 2, dim=1, keepdim=True)  # (N, 1)
        anc_sq = torch.sum(anchors ** 2, dim=1, keepdim=True)  # (K, 1)
        cross = torch.mm(embeddings, anchors.t())  # (N, K)
        
        distances = emb_sq + anc_sq.t() - 2 * cross  # (N, K)
        
    elif distance_metric == 'cosine':
        # Cosine distance = 1 - cosine_similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        anchors_norm = F.normalize(anchors, p=2, dim=1)
        
        similarities = torch.mm(embeddings_norm, anchors_norm.t())  # (N, K)
        distances = 1 - similarities
        
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Get index of nearest anchor
    assignments = torch.argmin(distances, dim=1)  # (N,)
    
    return assignments
