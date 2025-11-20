"""
Anchor Construction: Modular Strategies for Anchor Generation
1. Eigenface-based: PCA on mean-centered images, reconstruct from eigenvectors
2. KMeans Centroid-based: Direct clustering in image space
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional
import pickle
from pathlib import Path
from abc import ABC, abstractmethod


class AnchorStrategy(ABC):
    """Base class for anchor generation strategies"""
    
    @abstractmethod
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors from training images
        
        Args:
            images: Array of shape (N, H, W) - normalized training images
            
        Returns:
            anchor_images: Array of shape (K, H, W) - K anchor images
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save strategy state"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load strategy state"""
        pass


class EigenfaceAnchorStrategy(AnchorStrategy):
    """
    Eigenface-based anchor generation following classical eigenface approach:
    https://en.wikipedia.org/wiki/Eigenface#Practical_implementation
    
    1. Compute mean image μ = (1/N) Σ X_i
    2. Subtract mean: Φ_i = X_i - μ
    3. Compute covariance matrix C = (1/N) Σ Φ_i Φ_i^T
    4. Find eigenvectors (eigenfaces) of C
    5. Cluster images in eigenface coefficient space
    6. Reconstruct anchor from cluster centroids
    """
    
    def __init__(
        self,
        n_components: int = 50,
        n_anchors: int = 8,
        random_state: int = 42
    ):
        """
        Args:
            n_components: Number of eigenvectors to retain (M)
            n_anchors: Number of anchors to generate (K)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_anchors = n_anchors
        self.random_state = random_state
        
        self.mean_image = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.anchor_images = None
        self.anchor_coefficients = None
        self.kmeans = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate eigenface-based anchors
        
        Steps:
        1. Mean-center images
        2. Compute eigenvectors via PCA
        3. Project images to eigenface space
        4. Cluster coefficients with KMeans
        5. Reconstruct anchors from cluster centroids
        """
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"EIGENFACE ANCHOR GENERATION")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        # Step 1: Compute mean image and center data
        print(f"\n1. Computing mean image...")
        X = images.reshape(N, -1)  # (N, H*W)
        self.mean_image = X.mean(axis=0)  # (H*W,)
        
        # Subtract mean from all images (mean-centering)
        X_centered = X - self.mean_image  # (N, H*W)
        print(f"   Mean image computed: μ ∈ ℝ^{H*W}")
        print(f"   Centered data: Φ = X - μ ∈ ℝ^{N}×{H*W}")
        
        # Step 2: Compute eigenvectors of covariance matrix via PCA
        print(f"\n2. Computing eigenvectors (eigenfaces) via PCA...")
        print(f"   Using randomized SVD for efficiency")
        pca = PCA(
            n_components=self.n_components,
            svd_solver='randomized',
            random_state=self.random_state
        )
        
        # Fit PCA on centered data
        coefficients = pca.fit_transform(X_centered)  # (N, M)
        
        self.eigenvectors = pca.components_  # (M, H*W) - eigenfaces
        self.eigenvalues = pca.explained_variance_  # (M,)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"   Retained {self.n_components} eigenfaces")
        print(f"   Explained variance: {explained_var:.2%}")
        print(f"   Top 5 eigenvalues: {self.eigenvalues[:5]}")
        
        # Step 3: Cluster in eigenface coefficient space
        print(f"\n3. Clustering images in eigenface space...")
        print(f"   KMeans with K={self.n_anchors} clusters")
        self.kmeans = KMeans(
            n_clusters=self.n_anchors,
            n_init=10,
            random_state=self.random_state,
            max_iter=300
        )
        
        cluster_labels = self.kmeans.fit_predict(coefficients)
        centroids = self.kmeans.cluster_centers_  # (K, M)
        self.anchor_coefficients = centroids
        
        # Step 4: Reconstruct anchor images from cluster centroids
        print(f"\n4. Reconstructing anchor images...")
        anchor_images = []
        
        for k in range(self.n_anchors):
            # Reconstruct: X_anchor = μ + Σ(c_i * e_i)
            # where c_i are centroid coefficients and e_i are eigenvectors
            anchor_centered = centroids[k] @ self.eigenvectors  # (H*W,)
            anchor_flat = anchor_centered + self.mean_image  # Add mean back
            anchor_img = anchor_flat.reshape(H, W)
            anchor_images.append(anchor_img)
            
            # Count samples assigned to this anchor
            count = (cluster_labels == k).sum()
            coeff_norm = np.linalg.norm(centroids[k])
            print(f"   Anchor {k}: {count:4d} images assigned, "
                  f"coeff norm = {coeff_norm:.3f}")
        
        self.anchor_images = np.array(anchor_images)  # (K, H, W)
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} eigenface-based anchors")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        """Return anchor images"""
        return self.anchor_images
    
    def save(self, path: str):
        """Save eigenface strategy state"""
        state = {
            'strategy': 'eigenface',
            'n_components': self.n_components,
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'mean_image': self.mean_image,
            'eigenvectors': self.eigenvectors,
            'eigenvalues': self.eigenvalues,
            'anchor_images': self.anchor_images,
            'anchor_coefficients': self.anchor_coefficients,
            'kmeans': self.kmeans
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved eigenface strategy to {path}")
    
    def load(self, path: str):
        """Load eigenface strategy state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.n_components = state['n_components']
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.mean_image = state['mean_image']
        self.eigenvectors = state['eigenvectors']
        self.eigenvalues = state['eigenvalues']
        self.anchor_images = state['anchor_images']
        self.anchor_coefficients = state['anchor_coefficients']
        self.kmeans = state['kmeans']
        
        print(f"Loaded eigenface strategy from {path}")
        print(f"  {self.n_anchors} anchors, {self.n_components} eigenfaces")


class KMeansCentroidAnchorStrategy(AnchorStrategy):
    """
    Simple KMeans clustering in image space
    
    Directly cluster images and use cluster centroids as anchors.
    This is more straightforward than eigenfaces but may be less
    interpretable and more sensitive to noise.
    """
    
    def __init__(
        self,
        n_anchors: int = 8,
        random_state: int = 42,
        max_iter: int = 300
    ):
        """
        Args:
            n_anchors: Number of anchors (K)
            random_state: Random seed
            max_iter: Maximum KMeans iterations
        """
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.max_iter = max_iter
        
        self.kmeans = None
        self.anchor_images = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors by direct KMeans clustering
        
        Steps:
        1. Flatten images to vectors
        2. Run KMeans clustering
        3. Use cluster centroids as anchors
        """
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"KMEANS CENTROID ANCHOR GENERATION")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        # Flatten images
        X = images.reshape(N, -1)  # (N, H*W)
        
        # Run KMeans
        print(f"\nClustering with K={self.n_anchors}...")
        self.kmeans = KMeans(
            n_clusters=self.n_anchors,
            n_init=10,
            random_state=self.random_state,
            max_iter=self.max_iter,
            verbose=0
        )
        
        cluster_labels = self.kmeans.fit_predict(X)
        centroids = self.kmeans.cluster_centers_  # (K, H*W)
        
        # Reshape centroids to images
        anchor_images = []
        for k in range(self.n_anchors):
            anchor_img = centroids[k].reshape(H, W)
            anchor_images.append(anchor_img)
            
            count = (cluster_labels == k).sum()
            inertia = np.sum((X[cluster_labels == k] - centroids[k])**2)
            print(f"   Anchor {k}: {count:4d} images assigned, "
                  f"inertia = {inertia:.2e}")
        
        self.anchor_images = np.array(anchor_images)  # (K, H, W)
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} KMeans centroid anchors")
        print(f"  Total inertia: {self.kmeans.inertia_:.2e}")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        """Return anchor images"""
        return self.anchor_images
    
    def save(self, path: str):
        """Save KMeans strategy state"""
        state = {
            'strategy': 'kmeans',
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'kmeans': self.kmeans,
            'anchor_images': self.anchor_images
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved KMeans strategy to {path}")
    
    def load(self, path: str):
        """Load KMeans strategy state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.max_iter = state['max_iter']
        self.kmeans = state['kmeans']
        self.anchor_images = state['anchor_images']
        
        print(f"Loaded KMeans strategy from {path}")
        print(f"  {self.n_anchors} anchors")


class RandomAnchorStrategy(AnchorStrategy):
    """
    Random anchor generation - baseline strategy
    
    Simply selects random images from the training set as anchors.
    This serves as a baseline to compare against more sophisticated methods.
    """
    
    def __init__(
        self,
        n_anchors: int = 8,
        random_state: int = 42
    ):
        """
        Args:
            n_anchors: Number of anchors (K)
            random_state: Random seed
        """
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.anchor_images = None
        self.selected_indices = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors by randomly selecting images
        
        Steps:
        1. Randomly select K images from training set
        2. Use them directly as anchors
        """
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"RANDOM ANCHOR GENERATION (BASELINE)")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Randomly select K images
        print(f"\nRandomly selecting {self.n_anchors} images as anchors...")
        self.selected_indices = np.random.choice(N, size=self.n_anchors, replace=False)
        
        anchor_images = []
        for k, idx in enumerate(self.selected_indices):
            anchor_images.append(images[idx])
            print(f"   Anchor {k}: Selected image index {idx}")
        
        self.anchor_images = np.array(anchor_images)  # (K, H, W)
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} random anchors")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        """Return anchor images"""
        return self.anchor_images
    
    def save(self, path: str):
        """Save random strategy state"""
        state = {
            'strategy': 'random',
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'anchor_images': self.anchor_images,
            'selected_indices': self.selected_indices
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved random strategy to {path}")
    
    def load(self, path: str):
        """Load random strategy state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.anchor_images = state['anchor_images']
        self.selected_indices = state['selected_indices']
        
        print(f"Loaded random strategy from {path}")
        print(f"  {self.n_anchors} anchors")


class AnchorGenerator:
    """
    Factory class for anchor generation with modular strategy selection
    """
    
    def __init__(
        self,
        strategy: str = 'eigenface',
        n_components: int = 50,
        n_anchors: int = 8,
        random_state: int = 42
    ):
        """
        Args:
            strategy: 'eigenface', 'kmeans', or 'random'
            n_components: PCA components (only for eigenface)
            n_anchors: Number of anchors (K)
            random_state: Random seed
        """
        self.strategy_name = strategy
        
        if strategy == 'eigenface':
            self.strategy = EigenfaceAnchorStrategy(
                n_components=n_components,
                n_anchors=n_anchors,
                random_state=random_state
            )
        elif strategy == 'kmeans':
            self.strategy = KMeansCentroidAnchorStrategy(
                n_anchors=n_anchors,
                random_state=random_state
            )
        elif strategy == 'random':
            self.strategy = RandomAnchorStrategy(
                n_anchors=n_anchors,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Choose 'eigenface', 'kmeans', or 'random'")
    
    def fit(self, images: np.ndarray) -> np.ndarray:
        """Generate anchors using selected strategy"""
        return self.strategy.fit(images)
    
    def get_anchor_images(self) -> np.ndarray:
        """Return anchor images"""
        return self.strategy.get_anchor_images()
    
    def save(self, path: str):
        """Save anchor generator state"""
        self.strategy.save(path)
    
    @staticmethod
    def load(path: str) -> 'AnchorGenerator':
        """Load anchor generator from file"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        strategy_name = state['strategy']
        
        # Create generator with appropriate strategy
        gen = AnchorGenerator.__new__(AnchorGenerator)
        gen.strategy_name = strategy_name
        
        if strategy_name == 'eigenface':
            gen.strategy = EigenfaceAnchorStrategy.__new__(EigenfaceAnchorStrategy)
        elif strategy_name == 'kmeans':
            gen.strategy = KMeansCentroidAnchorStrategy.__new__(KMeansCentroidAnchorStrategy)
        elif strategy_name == 'random':
            gen.strategy = RandomAnchorStrategy.__new__(RandomAnchorStrategy)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        gen.strategy.load(path)
        
        return gen


def compute_anchor_embeddings(
    anchor_images: np.ndarray,
    backbone_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute DINOv3 embeddings for anchor images
    
    Args:
        anchor_images: (K, H, W) anchor images
        backbone_model: DINOv3 model
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        global_embeddings: (K, D) global embeddings
        dense_embeddings: (K, H', W', C) dense feature maps
    """
    K = len(anchor_images)
    backbone_model.eval()
    
    global_embeds = []
    dense_embeds = []
    
    with torch.no_grad():
        for i in range(0, K, batch_size):
            batch = anchor_images[i:i+batch_size]
            
            # Convert to tensor and add channel/batch dims
            batch_tensor = torch.from_numpy(batch).float().unsqueeze(1)  # (B, 1, H, W)
            
            # Repeat to 3 channels for DINOv3
            batch_tensor = batch_tensor.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            batch_tensor = batch_tensor.to(device)
            
            # Get embeddings
            outputs = backbone_model(batch_tensor)
            
            global_embeds.append(outputs['global'].cpu())
            dense_embeds.append(outputs['dense'].cpu())
    
    global_embeddings = torch.cat(global_embeds, dim=0)  # (K, D)
    dense_embeddings = torch.cat(dense_embeds, dim=0)    # (K, H', W', C)
    
    # Normalize global embeddings
    global_embeddings = torch.nn.functional.normalize(global_embeddings, dim=1)
    
    print(f"Computed anchor embeddings: global {global_embeddings.shape}, dense {dense_embeddings.shape}")
    
    return global_embeddings, dense_embeddings


def visualize_anchors(anchor_images: np.ndarray, save_path: str):
    """Visualize anchor images in a grid"""
    import matplotlib.pyplot as plt
    
    K = len(anchor_images)
    cols = min(4, K)
    rows = (K + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if K > 1 else [axes]
    
    for i in range(K):
        axes[i].imshow(anchor_images[i], cmap='gray')
        axes[i].set_title(f'Anchor {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(K, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved anchor visualization to {save_path}")