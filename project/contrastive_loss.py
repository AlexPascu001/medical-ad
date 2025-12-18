"""
Contrastive Anchor Loss: Center Loss / InfoNCE style approach for learnable anchors
Treats anchors as class prototypes that move toward their assigned samples.

This is more suitable for learnable anchors than the CAM loss, as it:
1. Pulls anchors toward their samples (not just samples toward anchors)
2. Uses temperature-scaled softmax for soft assignments
3. Combines contrastive learning with anchor separation

Based on:
- Center Loss (ECCV 2016): https://ydwen.github.io/papers/WenECCV16.pdf
- InfoNCE (SimCLR): https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CenterLoss(nn.Module):
    """
    Center Loss: Pull samples toward their assigned anchor AND pull anchors toward their samples.
    
    Original Center Loss: L = (1/2) * sum_i ||x_i - c_y_i||^2
    where c_y_i is the center (anchor) for sample i's class.
    
    In our unsupervised case:
    - y_i = argmin_k ||x_i - c_k||  (pseudo-labeling: assign to nearest anchor)
    - Update both samples (via backprop) and centers (via gradient or moving average)
    """
    
    def __init__(
        self,
        distance_metric: str = 'euclidean',
        lambda_center: float = 1.0,
        lambda_repel: float = 0.1,
        margin: float = 1.0
    ):
        """
        Args:
            distance_metric: 'euclidean' or 'cosine'
            lambda_center: Weight for center loss (pull samples to anchors)
            lambda_repel: Weight for anchor separation (push anchors apart)
            margin: Minimum distance between anchors
        """
        super().__init__()
        self.distance_metric = distance_metric
        self.lambda_center = lambda_center
        self.lambda_repel = lambda_repel
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) sample embeddings
            anchor_embeddings: (K, D) anchor embeddings (learnable)
        
        Returns:
            Dictionary with loss and components
        """
        B, D = embeddings.shape
        K, _ = anchor_embeddings.shape
        
        # Compute distances to all anchors
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(embeddings, anchor_embeddings, p=2)  # (B, K)
        else:  # cosine
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            anchors_norm = F.normalize(anchor_embeddings, p=2, dim=1)
            similarities = embeddings_norm @ anchors_norm.T
            distances = 1.0 - similarities
        
        # Assign each sample to nearest anchor (hard assignment)
        min_distances, assigned_anchors = distances.min(dim=1)  # (B,), (B,)
        
        # === CENTER LOSS ===
        # Pull samples toward their assigned anchor
        # This creates gradients for BOTH samples and anchors
        loss_center = (min_distances ** 2).mean()
        
        # === REPELLER LOSS ===
        # Push anchors apart to maintain diversity
        if self.lambda_repel > 0:
            if self.distance_metric == 'euclidean':
                anchor_distances = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)
            else:
                anchor_sims = anchors_norm @ anchors_norm.T
                anchor_distances = 1.0 - anchor_sims
            
            # Exclude diagonal
            mask = ~torch.eye(K, dtype=torch.bool, device=anchor_distances.device)
            violations = torch.relu(self.margin - anchor_distances)
            loss_repel = violations[mask].mean()
        else:
            loss_repel = torch.tensor(0.0, device=embeddings.device)
        
        # === TOTAL LOSS ===
        total_loss = self.lambda_center * loss_center + self.lambda_repel * loss_repel
        
        result = {
            'loss': total_loss,
            'loss_center': loss_center.item(),
            'loss_repel': loss_repel.item(),
            'assigned_anchors': assigned_anchors
        }
        
        if return_components:
            result.update({
                'min_distance': min_distances.mean().item(),
                'anchor_utilization': torch.bincount(assigned_anchors, minlength=K).float() / B
            })
        
        return result


class InfoNCEAnchorLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for anchor learning.
    
    Treats each sample's nearest anchor as the "positive" and other anchors as "negatives".
    Uses temperature-scaled softmax for soft assignment.
    
    L = -log( exp(sim(x, c+)/tau) / sum_k exp(sim(x, c_k)/tau) )
    
    where c+ is the nearest anchor to x, and tau is temperature.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_repel: float = 0.1,
        margin: float = 1.0,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            temperature: Temperature for softmax (lower = harder assignments)
            lambda_repel: Weight for anchor separation
            margin: Minimum distance between anchors
            distance_metric: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_repel = lambda_repel
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) sample embeddings
            anchor_embeddings: (K, D) anchor embeddings (learnable)
        
        Returns:
            Dictionary with loss and components
        """
        B, D = embeddings.shape
        K, _ = anchor_embeddings.shape
        
        # Normalize for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        anchors_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        
        # Compute cosine similarities: (B, K)
        logits = embeddings_norm @ anchors_norm.T / self.temperature
        
        # Find nearest anchor for each sample (positive)
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(embeddings, anchor_embeddings, p=2)
            assigned_anchors = distances.argmin(dim=1)
        else:
            assigned_anchors = logits.argmax(dim=1)
        
        # === INFONCE LOSS ===
        # Cross-entropy: pull to assigned anchor, push from others
        # The softmax naturally creates soft assignments
        loss_infonce = F.cross_entropy(logits, assigned_anchors)
        
        # === REPELLER LOSS ===
        # Push anchors apart to maintain diversity
        if self.lambda_repel > 0:
            if self.distance_metric == 'euclidean':
                anchor_distances = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)
            else:
                anchor_sims = anchors_norm @ anchors_norm.T
                anchor_distances = 1.0 - anchor_sims
            
            mask = ~torch.eye(K, dtype=torch.bool, device=anchor_distances.device)
            violations = torch.relu(self.margin - anchor_distances)
            loss_repel = violations[mask].mean()
        else:
            loss_repel = torch.tensor(0.0, device=embeddings.device)
        
        # === TOTAL LOSS ===
        total_loss = loss_infonce + self.lambda_repel * loss_repel
        
        result = {
            'loss': total_loss,
            'loss_infonce': loss_infonce.item(),
            'loss_repel': loss_repel.item(),
            'assigned_anchors': assigned_anchors
        }
        
        if return_components:
            # Compute soft assignment probabilities
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            
            result.update({
                'assignment_entropy': entropy.item(),  # Higher = more uncertain assignments
                'anchor_utilization': torch.bincount(assigned_anchors, minlength=K).float() / B
            })
        
        return result


class HybridAnchorLoss(nn.Module):
    """
    Hybrid loss combining Center Loss and InfoNCE with anchor separation.
    
    This provides:
    1. Hard pulling via Center Loss (L2 distance minimization)
    2. Soft contrastive learning via InfoNCE (temperature-scaled softmax)
    3. Anchor diversity via repeller term
    
    Best of both worlds for learnable anchors.
    """
    
    def __init__(
        self,
        lambda_center: float = 1.0,
        lambda_infonce: float = 0.5,
        lambda_repel: float = 0.1,
        temperature: float = 0.07,
        margin: float = 1.0,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            lambda_center: Weight for center loss (hard pull)
            lambda_infonce: Weight for InfoNCE loss (soft contrastive)
            lambda_repel: Weight for anchor separation
            temperature: Temperature for InfoNCE softmax
            margin: Minimum distance between anchors
            distance_metric: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_infonce = lambda_infonce
        self.lambda_repel = lambda_repel
        self.temperature = temperature
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) sample embeddings
            anchor_embeddings: (K, D) anchor embeddings (learnable)
        
        Returns:
            Dictionary with loss and components
        """
        B, D = embeddings.shape
        K, _ = anchor_embeddings.shape
        
        # Normalize for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        anchors_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        
        # Compute distances for hard assignment
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(embeddings, anchor_embeddings, p=2)  # (B, K)
        else:
            similarities = embeddings_norm @ anchors_norm.T
            distances = 1.0 - similarities
        
        min_distances, assigned_anchors = distances.min(dim=1)
        
        # === CENTER LOSS (Hard Pull) ===
        loss_center = (min_distances ** 2).mean()
        
        # === INFONCE LOSS (Soft Contrastive) ===
        logits = embeddings_norm @ anchors_norm.T / self.temperature  # (B, K)
        loss_infonce = F.cross_entropy(logits, assigned_anchors)
        
        # === REPELLER LOSS (Anchor Diversity) ===
        if self.lambda_repel > 0:
            if self.distance_metric == 'euclidean':
                anchor_distances = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)
            else:
                anchor_sims = anchors_norm @ anchors_norm.T
                anchor_distances = 1.0 - anchor_sims
            
            mask = ~torch.eye(K, dtype=torch.bool, device=anchor_distances.device)
            violations = torch.relu(self.margin - anchor_distances)
            loss_repel = violations[mask].mean()
        else:
            loss_repel = torch.tensor(0.0, device=embeddings.device)
        
        # === TOTAL LOSS ===
        total_loss = (
            self.lambda_center * loss_center +
            self.lambda_infonce * loss_infonce +
            self.lambda_repel * loss_repel
        )
        
        result = {
            'loss': total_loss,
            'loss_center': loss_center.item(),
            'loss_infonce': loss_infonce.item(),
            'loss_repel': loss_repel.item(),
            'assigned_anchors': assigned_anchors
        }
        
        if return_components:
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            
            result.update({
                'min_distance': min_distances.mean().item(),
                'assignment_entropy': entropy.item(),
                'anchor_utilization': torch.bincount(assigned_anchors, minlength=K).float() / B,
                'anchor_min_separation': anchor_distances[mask].min().item(),
                'anchor_mean_separation': anchor_distances[mask].mean().item()
            })
        
        return result


class CombinedContrastiveLoss(nn.Module):
    """
    Wrapper for using contrastive losses with both global and dense features.
    Compatible with existing training pipeline.
    """
    
    def __init__(
        self,
        global_loss: nn.Module,
        dense_loss: Optional[nn.Module] = None,
        global_weight: float = 1.0,
        dense_weight: float = 0.5
    ):
        """
        Args:
            global_loss: Loss for global features (CenterLoss, InfoNCE, or Hybrid)
            dense_loss: Loss for dense features (optional)
            global_weight: Weight for global loss
            dense_weight: Weight for dense loss
        """
        super().__init__()
        self.global_loss = global_loss
        self.dense_loss = dense_loss
        self.global_weight = global_weight
        self.dense_weight = dense_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        anchor_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dictionary from model forward pass with 'global_feat'
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
        
        # Extract loss names dynamically (center/infonce/etc)
        result = {
            'loss': total_loss,
            'loss_global': global_result['loss'],
            'assigned_anchors': global_result['assigned_anchors']
        }
        
        # Add all global loss components with prefix
        for key, value in global_result.items():
            if key.startswith('loss_') or key in ['min_distance', 'assignment_entropy', 
                                                     'anchor_min_separation', 'anchor_mean_separation']:
                # Convert tensors to scalars, skip non-scalar tensors (like anchor_utilization)
                if isinstance(value, (int, float)):
                    result[f'loss_global_{key}'] = value
                elif isinstance(value, torch.Tensor):
                    if value.numel() == 1:  # Only convert single-element tensors
                        result[f'loss_global_{key}'] = value.item()
                    # Skip multi-element tensors (like anchor_utilization which is a distribution)
                else:
                    result[f'loss_global_{key}'] = value
        
        # Dense loss if available
        if self.dense_loss is not None and 'dense_feat' in outputs:
            # Reshape dense features from (B, H', W', D) to (B, D, H', W')
            dense_feat = outputs['dense_feat'].permute(0, 3, 1, 2)
            B, D, H, W = dense_feat.shape
            
            # Flatten spatial dimensions: (B, D, H, W) -> (B*H*W, D)
            dense_flat = dense_feat.permute(0, 2, 3, 1).reshape(B * H * W, D)
            
            dense_result = self.dense_loss(
                dense_flat,
                anchor_embeddings,
                return_components=False
            )
            
            total_loss = total_loss + self.dense_weight * dense_result['loss']
            
            result.update({
                'loss': total_loss,
                'loss_dense': dense_result['loss']
            })
            
            # Add dense components
            for key, value in dense_result.items():
                if key.startswith('loss_'):
                    # Safely convert to scalar
                    if isinstance(value, (int, float)):
                        result[f'loss_dense_{key}'] = value
                    elif isinstance(value, torch.Tensor) and value.numel() == 1:
                        result[f'loss_dense_{key}'] = value.item()
        
        return result
