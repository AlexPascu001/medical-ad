"""
Anchor Construction: Modular Strategies for Anchor Generation
1. Eigenface-based: PCA on mean-centered images, reconstruct from eigenvectors
2. KMeans Centroid-based: Direct clustering in image space
3. K-Center (Farthest Point Sampling): Maximize minimum distance coverage
4. Density-weighted: Sample inversely proportional to local density
5. GMM-based: Gaussian Mixture Model clustering
6. Stratified: Divide space into strata and sample uniformly
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Optional
import pickle
from pathlib import Path
from abc import ABC, abstractmethod


def _grayscale_batch_to_tensor(images_np: np.ndarray, device: torch.device,
                               apply_imagenet_norm: bool = False) -> torch.Tensor:
    """Convert a batch of preprocessed grayscale images to 3-channel tensors.

    Args:
        images_np: (B, H, W) float32 array (already preprocessed by BMADPreprocessor).
        device: Target torch device.
        apply_imagenet_norm: If True, apply ImageNet normalization (use when
            preprocessor outputs [0,1] via min-max scaling).

    Returns:
        (B, 3, H, W) tensor ready for DINOv3 backbone.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ops = []
    if apply_imagenet_norm:
        ops.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ops.append(ToTensorV2())
    _tf = A.Compose(ops)
    tensors = []
    for img in images_np:
        img3 = np.stack([img, img, img], axis=-1)  # (H,W) -> (H,W,3)
        tensors.append(_tf(image=img3)['image'])
    return torch.stack(tensors).to(device)


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
        
        # Step 4: Select real samples closest to each centroid (sample-anchored)
        print(f"\n4. Selecting closest real samples to centroids...")
        anchor_images = []
        anchor_indices = []

        for k in range(self.n_anchors):
            cluster_mask = cluster_labels == k
            count = cluster_mask.sum()
            if count == 0:
                print(f"   Anchor {k}: empty cluster, falling back to global nearest")
                # Global nearest to this centroid
                dists = np.linalg.norm(coefficients - centroids[k], axis=1)
                idx = int(np.argmin(dists))
            else:
                cluster_coeffs = coefficients[cluster_mask]
                dists = np.linalg.norm(cluster_coeffs - centroids[k], axis=1)
                local_idx = int(np.argmin(dists))
                idx = np.where(cluster_mask)[0][local_idx]

            anchor_indices.append(idx)
            anchor_img = images[idx]
            anchor_images.append(anchor_img)

            coeff_norm = np.linalg.norm(centroids[k])
            print(f"   Anchor {k}: {count:4d} images assigned, coeff norm = {coeff_norm:.3f}, nearest sample idx = {idx}")

        self.anchor_images = np.array(anchor_images)  # (K, H, W)
        self.anchor_coefficients = centroids

        print(f"\n{'='*80}")
        print(f"✓ Selected {self.n_anchors} sample-based eigenface anchors")
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
            max_iter=self.max_iter
        )
        
        cluster_labels = self.kmeans.fit_predict(X)
        centroids = self.kmeans.cluster_centers_  # (K, H*W)
        self.anchor_images = []

        # Select real samples closest to each centroid
        for k in range(self.n_anchors):
            mask = cluster_labels == k
            count = mask.sum()
            if count == 0:
                print(f"   Anchor {k}: empty cluster, falling back to global nearest")
                dists = np.linalg.norm(X - centroids[k], axis=1)
                idx = int(np.argmin(dists))
            else:
                cluster_X = X[mask]
                dists = np.linalg.norm(cluster_X - centroids[k], axis=1)
                local_idx = int(np.argmin(dists))
                idx = np.where(mask)[0][local_idx]

            anchor_img = images[idx]
            self.anchor_images.append(anchor_img)
            print(f"   Anchor {k}: {count:4d} images assigned, nearest sample idx = {idx}")
        
        self.anchor_images = np.array(self.anchor_images)
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} sample-based k-means anchors")
        print(f"{'='*80}\n")
        
        return self.anchor_images

    def get_anchor_images(self) -> np.ndarray:
        return self.anchor_images
    
    def save(self, path: str):
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


class KCenterAnchorStrategy(AnchorStrategy):
    """
    K-Center (Farthest Point Sampling / Greedy Coreset) anchor generation.
    
    This strategy maximizes the minimum distance between anchors, ensuring
    maximum coverage of the embedding space. It iteratively selects the point
    that is farthest from all previously selected anchors.
    
    Why this might work better than k-means:
    - K-means finds dense clusters, but anomaly detection needs coverage
    - K-center ensures no region is too far from an anchor
    - More robust to outliers and non-spherical distributions
    """
    
    def __init__(
        self,
        n_anchors: int = 8,
        random_state: int = 42,
        use_pca: bool = True,
        n_components: int = 50
    ):
        """
        Args:
            n_anchors: Number of anchors (K)
            random_state: Random seed for first point selection
            use_pca: Whether to perform k-center in PCA space
            n_components: PCA components if use_pca=True
        """
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.use_pca = use_pca
        self.n_components = n_components
        self.anchor_images = None
        self.selected_indices = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors using farthest point sampling.
        
        Steps:
        1. Optionally reduce dimensionality with PCA
        2. Select first point randomly
        3. Iteratively select point farthest from all selected points
        """
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"K-CENTER (FARTHEST POINT SAMPLING) ANCHOR GENERATION")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        np.random.seed(self.random_state)
        
        # Flatten images
        X = images.reshape(N, -1)  # (N, H*W)
        
        # Optionally apply PCA for efficiency
        if self.use_pca and X.shape[1] > self.n_components:
            print(f"\nApplying PCA to reduce to {self.n_components} dimensions...")
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_reduced = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"   Explained variance: {explained_var:.2%}")
        else:
            X_reduced = X
        
        print(f"\nPerforming farthest point sampling...")
        
        # Initialize: select first point randomly
        selected_indices = [np.random.randint(N)]
        print(f"   Anchor 0: Selected index {selected_indices[0]} (random start)")
        
        # Track minimum distance to any selected point for each sample
        min_distances = np.full(N, np.inf)
        
        for k in range(1, self.n_anchors):
            # Update min distances with distance to last selected point
            last_selected = selected_indices[-1]
            distances_to_last = np.linalg.norm(X_reduced - X_reduced[last_selected], axis=1)
            min_distances = np.minimum(min_distances, distances_to_last)
            
            # Select point with maximum minimum distance (farthest from all selected)
            # Exclude already selected points
            min_distances_copy = min_distances.copy()
            for idx in selected_indices:
                min_distances_copy[idx] = -np.inf
            
            next_idx = int(np.argmax(min_distances_copy))
            selected_indices.append(next_idx)
            
            max_min_dist = min_distances[next_idx]
            print(f"   Anchor {k}: Selected index {next_idx}, min_dist_to_anchors = {max_min_dist:.4f}")
        
        self.selected_indices = np.array(selected_indices)
        self.anchor_images = images[self.selected_indices]
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} k-center anchors (maximum coverage)")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        return self.anchor_images
    
    def save(self, path: str):
        state = {
            'strategy': 'kcenter',
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'anchor_images': self.anchor_images,
            'selected_indices': self.selected_indices
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved k-center strategy to {path}")
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.use_pca = state['use_pca']
        self.n_components = state['n_components']
        self.anchor_images = state['anchor_images']
        self.selected_indices = state['selected_indices']
        print(f"Loaded k-center strategy from {path}")


class DensityWeightedAnchorStrategy(AnchorStrategy):
    """
    Density-weighted anchor sampling.
    
    Samples anchors inversely proportional to local density, meaning:
    - Sparse regions get more anchors (better coverage of rare patterns)
    - Dense regions get fewer anchors (avoid redundancy)
    
    This addresses the issue where k-means puts all anchors in dense clusters,
    leaving sparse regions (potentially important edge cases) uncovered.
    """
    
    def __init__(
        self,
        n_anchors: int = 8,
        random_state: int = 42,
        n_neighbors: int = 10,
        use_pca: bool = True,
        n_components: int = 50,
        temperature: float = 1.0
    ):
        """
        Args:
            n_anchors: Number of anchors (K)
            random_state: Random seed
            n_neighbors: Number of neighbors for density estimation
            use_pca: Whether to compute density in PCA space
            n_components: PCA components
            temperature: Controls sampling sharpness (higher = more uniform)
        """
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.use_pca = use_pca
        self.n_components = n_components
        self.temperature = temperature
        self.anchor_images = None
        self.selected_indices = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors by sampling inversely proportional to density.
        """
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"DENSITY-WEIGHTED ANCHOR GENERATION")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        np.random.seed(self.random_state)
        
        X = images.reshape(N, -1)
        
        # Optionally apply PCA
        if self.use_pca and X.shape[1] > self.n_components:
            print(f"\nApplying PCA to reduce to {self.n_components} dimensions...")
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        # Estimate local density using k-NN
        print(f"\nEstimating local density with {self.n_neighbors}-NN...")
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)  # +1 because point is its own neighbor
        nn.fit(X_reduced)
        distances, _ = nn.kneighbors(X_reduced)
        
        # Density ~ 1 / (mean distance to k neighbors)
        mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self
        densities = 1.0 / (mean_distances + 1e-8)
        
        # Sampling probability inversely proportional to density
        inv_density = 1.0 / (densities + 1e-8)
        inv_density = inv_density ** (1.0 / self.temperature)  # Temperature scaling
        sampling_probs = inv_density / inv_density.sum()
        
        print(f"   Density range: [{densities.min():.4f}, {densities.max():.4f}]")
        print(f"   Sampling from sparse regions preferentially")
        
        # Sample without replacement
        selected_indices = np.random.choice(
            N, size=self.n_anchors, replace=False, p=sampling_probs
        )
        
        for k, idx in enumerate(selected_indices):
            print(f"   Anchor {k}: index {idx}, density = {densities[idx]:.4f}, prob = {sampling_probs[idx]:.6f}")
        
        self.selected_indices = selected_indices
        self.anchor_images = images[selected_indices]
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} density-weighted anchors")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        return self.anchor_images
    
    def save(self, path: str):
        state = {
            'strategy': 'density',
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'n_neighbors': self.n_neighbors,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'temperature': self.temperature,
            'anchor_images': self.anchor_images,
            'selected_indices': self.selected_indices
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved density-weighted strategy to {path}")
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.n_neighbors = state['n_neighbors']
        self.use_pca = state['use_pca']
        self.n_components = state['n_components']
        self.temperature = state['temperature']
        self.anchor_images = state['anchor_images']
        self.selected_indices = state['selected_indices']
        print(f"Loaded density-weighted strategy from {path}")


class GMMCentroidAnchorStrategy(AnchorStrategy):
    """
    Gaussian Mixture Model (GMM) based anchor generation.
    
    Uses GMM instead of k-means for clustering. GMM advantages:
    - Soft clustering (probabilistic assignments)
    - Can model elliptical clusters (not just spherical)
    - Better handles clusters of different sizes/shapes
    
    This might work better if the normal data has multimodal
    distributions with different variances.
    """
    
    def __init__(
        self,
        n_anchors: int = 8,
        random_state: int = 42,
        use_pca: bool = True,
        n_components: int = 50,
        covariance_type: str = 'full'
    ):
        """
        Args:
            n_anchors: Number of anchors (K)
            random_state: Random seed
            use_pca: Apply PCA before GMM
            n_components: PCA components
            covariance_type: 'full', 'tied', 'diag', 'spherical'
        """
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.use_pca = use_pca
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.anchor_images = None
        self.selected_indices = None
        self.gmm = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors using GMM clustering.
        """
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"GMM CENTROID ANCHOR GENERATION")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        np.random.seed(self.random_state)
        
        X = images.reshape(N, -1)
        
        # Apply PCA (GMM in high dimensions is problematic)
        if self.use_pca:
            n_comp = min(self.n_components, X.shape[1], N - 1)
            print(f"\nApplying PCA to reduce to {n_comp} dimensions...")
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            X_reduced = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"   Explained variance: {explained_var:.2%}")
        else:
            X_reduced = X
        
        # Fit GMM
        print(f"\nFitting GMM with {self.n_anchors} components (covariance: {self.covariance_type})...")
        self.gmm = GaussianMixture(
            n_components=self.n_anchors,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            n_init=5
        )
        
        cluster_probs = self.gmm.fit_predict(X_reduced)  # Hard assignments for selection
        
        # Select sample closest to each GMM mean
        means = self.gmm.means_  # (K, D)
        selected_indices = []
        
        for k in range(self.n_anchors):
            mask = cluster_probs == k
            count = mask.sum()
            
            if count == 0:
                # Empty cluster - find globally nearest
                dists = np.linalg.norm(X_reduced - means[k], axis=1)
                idx = int(np.argmin(dists))
            else:
                # Find nearest within cluster
                cluster_X = X_reduced[mask]
                dists = np.linalg.norm(cluster_X - means[k], axis=1)
                local_idx = int(np.argmin(dists))
                idx = np.where(mask)[0][local_idx]
            
            selected_indices.append(idx)
            weight = self.gmm.weights_[k]
            print(f"   Anchor {k}: {count:4d} samples, weight = {weight:.3f}, nearest idx = {idx}")
        
        self.selected_indices = np.array(selected_indices)
        self.anchor_images = images[self.selected_indices]
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} GMM-based anchors")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        return self.anchor_images
    
    def save(self, path: str):
        state = {
            'strategy': 'gmm',
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'anchor_images': self.anchor_images,
            'selected_indices': self.selected_indices,
            'gmm': self.gmm
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved GMM strategy to {path}")
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.use_pca = state['use_pca']
        self.n_components = state['n_components']
        self.covariance_type = state['covariance_type']
        self.anchor_images = state['anchor_images']
        self.selected_indices = state['selected_indices']
        self.gmm = state['gmm']
        print(f"Loaded GMM strategy from {path}")


class StratifiedAnchorStrategy(AnchorStrategy):
    """
    Stratified anchor sampling.
    
    Divides the PCA space into strata (grid cells) and samples one anchor
    per stratum. This ensures uniform coverage across the embedding space.
    
    The key insight is that random sampling can accidentally cluster anchors,
    while stratified sampling guarantees spread.
    """
    
    def __init__(
        self,
        n_anchors: int = 8,
        random_state: int = 42,
        n_components: int = 2  # Use first 2 PCs for stratification
    ):
        """
        Args:
            n_anchors: Number of anchors (K)
            random_state: Random seed
            n_components: Number of PCA components for stratification (usually 2-3)
        """
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.n_components = n_components
        self.anchor_images = None
        self.selected_indices = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors using stratified sampling in PCA space.
        """
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"STRATIFIED ANCHOR GENERATION")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        np.random.seed(self.random_state)
        
        X = images.reshape(N, -1)
        
        # Apply PCA for stratification
        n_comp = min(self.n_components, X.shape[1], N - 1)
        print(f"\nApplying PCA to {n_comp} dimensions for stratification...")
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_reduced = pca.fit_transform(X)
        
        # Determine grid structure based on n_anchors
        # For 8 anchors in 2D: 3x3 grid (pick 8 from 9), or 4x2
        # For simplicity, use k-means to define strata boundaries
        n_strata = self.n_anchors
        
        print(f"\nCreating {n_strata} strata using percentile-based grid...")
        
        # Use percentiles to create roughly equal-sized strata
        # Simpler approach: use k-means but then sample randomly within each cluster
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_strata, random_state=self.random_state, n_init=10)
        strata_labels = kmeans.fit_predict(X_reduced)
        
        selected_indices = []
        for k in range(n_strata):
            mask = strata_labels == k
            stratum_indices = np.where(mask)[0]
            count = len(stratum_indices)
            
            if count == 0:
                # Empty stratum - skip (shouldn't happen with k-means)
                print(f"   Stratum {k}: EMPTY - finding nearest to centroid")
                dists = np.linalg.norm(X_reduced - kmeans.cluster_centers_[k], axis=1)
                idx = int(np.argmin(dists))
            else:
                # Random sample within stratum
                idx = np.random.choice(stratum_indices)
            
            selected_indices.append(idx)
            print(f"   Stratum {k}: {count:4d} samples, selected idx = {idx}")
        
        self.selected_indices = np.array(selected_indices)
        self.anchor_images = images[self.selected_indices]
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} stratified anchors")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        return self.anchor_images
    
    def save(self, path: str):
        state = {
            'strategy': 'stratified',
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'n_components': self.n_components,
            'anchor_images': self.anchor_images,
            'selected_indices': self.selected_indices
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved stratified strategy to {path}")
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.n_components = state['n_components']
        self.anchor_images = state['anchor_images']
        self.selected_indices = state['selected_indices']
        print(f"Loaded stratified strategy from {path}")


class EmbeddingDiverseAnchorStrategy(AnchorStrategy):
    """
    Embedding-aware anchor selection using farthest point sampling in DINO embedding space.
    
    This strategy addresses the problem where DINO embeddings for similar images
    (like brain MRI slices) can be highly correlated. It selects anchors that
    maximize diversity in the actual embedding space used by the model.
    
    Key advantages:
    - Directly optimizes for diversity in the space where anomaly detection happens
    - Works better than pixel-space selection for domain-specific images
    - Ensures anchors span the embedding manifold more effectively
    
    Steps:
    1. Compute DINO embeddings for all candidate images
    2. Apply farthest point sampling in embedding space
    3. Return images corresponding to selected diverse embeddings
    """
    
    def __init__(
        self,
        n_anchors: int = 8,
        random_state: int = 42,
        backbone_name: str = 'vit_small_patch16_dinov3.lvd1689m',
        batch_size: int = 32
    ):
        """
        Args:
            n_anchors: Number of anchors (K)
            random_state: Random seed for first point selection
            backbone_name: DINO backbone model name
            batch_size: Batch size for computing embeddings
        """
        self.n_anchors = n_anchors
        self.random_state = random_state
        self.backbone_name = backbone_name
        self.batch_size = batch_size
        self.anchor_images = None
        self.selected_indices = None
        self.embedding_diversity_score = None
        
    def fit(self, images: np.ndarray) -> np.ndarray:
        """
        Generate anchors using farthest point sampling in DINO embedding space.
        
        Steps:
        1. Compute DINO embeddings for all images
        2. Perform farthest point sampling in embedding space
        3. Select corresponding images as anchors
        """
        import torch
        import torch.nn.functional as F
        
        N, H, W = images.shape
        print(f"\n{'='*80}")
        print(f"EMBEDDING-DIVERSE ANCHOR GENERATION (DINO SPACE FPS)")
        print(f"{'='*80}")
        print(f"Input: {N} images of size {H}×{W}")
        
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # Load DINO backbone (without projection for raw embeddings)
        print(f"\nLoading DINO backbone: {self.backbone_name}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        import timm
        model = timm.create_model(
            self.backbone_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        model = model.to(device)
        model.eval()
        
        # Compute embeddings for all images
        print(f"Computing DINO embeddings for {N} images...")
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, N, self.batch_size):
                batch_imgs = images[i:i+self.batch_size]
                
                # Preprocess: convert to 3-channel tensor
                batch_tensor = _grayscale_batch_to_tensor(batch_imgs, device)
                
                # Get embeddings
                emb = model(batch_tensor)  # (B, D)
                emb = F.normalize(emb, dim=1)  # L2 normalize
                embeddings.append(emb.cpu())
                
                if (i // self.batch_size) % 10 == 0:
                    print(f"   Processed {min(i+self.batch_size, N)}/{N} images...")
        
        embeddings = torch.cat(embeddings, dim=0).numpy()  # (N, D)
        print(f"   Embedding shape: {embeddings.shape}")
        
        # Analyze embedding similarity
        print(f"\nAnalyzing embedding space diversity...")
        sample_indices = np.random.choice(N, size=min(100, N), replace=False)
        sample_emb = embeddings[sample_indices]
        cos_sim = np.dot(sample_emb, sample_emb.T)
        np.fill_diagonal(cos_sim, 0)
        avg_cos_sim = cos_sim.sum() / (len(sample_indices) * (len(sample_indices) - 1))
        print(f"   Average pairwise cosine similarity: {avg_cos_sim:.4f}")
        print(f"   (Values close to 1.0 indicate similar embeddings)")
        
        # Farthest point sampling in embedding space
        print(f"\nPerforming farthest point sampling in embedding space...")
        
        # Initialize: select first point randomly
        selected_indices = [np.random.randint(N)]
        print(f"   Anchor 0: Selected index {selected_indices[0]} (random start)")
        
        # Track minimum distance to any selected point for each sample
        min_distances = np.full(N, np.inf)
        
        for k in range(1, self.n_anchors):
            # Update min distances with distance to last selected point
            last_selected = selected_indices[-1]
            # Use cosine distance = 1 - cosine_similarity
            cosine_sims = np.dot(embeddings, embeddings[last_selected])
            distances_to_last = 1.0 - cosine_sims
            min_distances = np.minimum(min_distances, distances_to_last)
            
            # Select point with maximum minimum distance (farthest from all selected)
            # Exclude already selected points
            min_distances_copy = min_distances.copy()
            for idx in selected_indices:
                min_distances_copy[idx] = -np.inf
            
            next_idx = int(np.argmax(min_distances_copy))
            selected_indices.append(next_idx)
            
            max_min_dist = min_distances[next_idx]
            print(f"   Anchor {k}: Selected index {next_idx}, "
                  f"min_cosine_dist_to_anchors = {max_min_dist:.4f}")
        
        self.selected_indices = np.array(selected_indices)
        self.anchor_images = images[self.selected_indices]
        
        # Compute diversity score for selected anchors
        selected_embeddings = embeddings[self.selected_indices]
        anchor_cos_sim = np.dot(selected_embeddings, selected_embeddings.T)
        np.fill_diagonal(anchor_cos_sim, 0)
        self.embedding_diversity_score = 1.0 - (anchor_cos_sim.sum() / (self.n_anchors * (self.n_anchors - 1)))
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {self.n_anchors} embedding-diverse anchors")
        print(f"   Diversity score: {self.embedding_diversity_score:.4f} (higher = more diverse)")
        print(f"   Average anchor pairwise cosine similarity: {1.0 - self.embedding_diversity_score:.4f}")
        print(f"{'='*80}\n")
        
        return self.anchor_images
    
    def get_anchor_images(self) -> np.ndarray:
        return self.anchor_images
    
    def save(self, path: str):
        state = {
            'strategy': 'embedding_diverse',
            'n_anchors': self.n_anchors,
            'random_state': self.random_state,
            'backbone_name': self.backbone_name,
            'batch_size': self.batch_size,
            'anchor_images': self.anchor_images,
            'selected_indices': self.selected_indices,
            'embedding_diversity_score': self.embedding_diversity_score
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved embedding-diverse strategy to {path}")
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.n_anchors = state['n_anchors']
        self.random_state = state['random_state']
        self.backbone_name = state['backbone_name']
        self.batch_size = state['batch_size']
        self.anchor_images = state['anchor_images']
        self.selected_indices = state['selected_indices']
        self.embedding_diversity_score = state.get('embedding_diversity_score', None)
        print(f"Loaded embedding-diverse strategy from {path}")
        print(f"   Diversity score: {self.embedding_diversity_score}")


class AnchorGenerator:
    """
    Factory class for anchor generation with modular strategy selection
    
    Available strategies:
    - 'random': Random sampling (baseline)
    - 'kmeans': K-means clustering in image space
    - 'eigenface': PCA + K-means in eigenface space
    - 'kcenter': Farthest point sampling (max coverage)
    - 'density': Density-weighted sampling (sparse regions)
    - 'gmm': Gaussian Mixture Model clustering
    - 'stratified': Stratified sampling in PCA space
    - 'embedding_diverse': Farthest point sampling in DINO embedding space (RECOMMENDED)
    """
    
    AVAILABLE_STRATEGIES = ['random', 'kmeans', 'eigenface', 'kcenter', 'density', 'gmm', 'stratified', 'embedding_diverse']
    
    def __init__(
        self,
        strategy: str = 'eigenface',
        n_components: int = 50,
        n_anchors: int = 8,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            strategy: One of 'eigenface', 'kmeans', 'random', 'kcenter', 'density', 'gmm', 'stratified', 'embedding_diverse'
            n_components: PCA components (for eigenface, kcenter, density, gmm, stratified)
            n_anchors: Number of anchors (K)
            random_state: Random seed
            **kwargs: Additional strategy-specific parameters
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
        elif strategy == 'kcenter':
            self.strategy = KCenterAnchorStrategy(
                n_anchors=n_anchors,
                random_state=random_state,
                use_pca=kwargs.get('use_pca', True),
                n_components=n_components
            )
        elif strategy == 'density':
            self.strategy = DensityWeightedAnchorStrategy(
                n_anchors=n_anchors,
                random_state=random_state,
                n_neighbors=kwargs.get('n_neighbors', 10),
                use_pca=kwargs.get('use_pca', True),
                n_components=n_components,
                temperature=kwargs.get('temperature', 1.0)
            )
        elif strategy == 'gmm':
            self.strategy = GMMCentroidAnchorStrategy(
                n_anchors=n_anchors,
                random_state=random_state,
                use_pca=kwargs.get('use_pca', True),
                n_components=n_components,
                covariance_type=kwargs.get('covariance_type', 'full')
            )
        elif strategy == 'stratified':
            self.strategy = StratifiedAnchorStrategy(
                n_anchors=n_anchors,
                random_state=random_state,
                n_components=kwargs.get('stratified_components', 2)
            )
        elif strategy == 'embedding_diverse':
            self.strategy = EmbeddingDiverseAnchorStrategy(
                n_anchors=n_anchors,
                random_state=random_state,
                backbone_name=kwargs.get('backbone_name', 'vit_small_patch16_dinov3.lvd1689m'),
                batch_size=kwargs.get('batch_size', 32)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Choose from: {', '.join(self.AVAILABLE_STRATEGIES)}")
    
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
        
        strategy_classes = {
            'eigenface': EigenfaceAnchorStrategy,
            'kmeans': KMeansCentroidAnchorStrategy,
            'random': RandomAnchorStrategy,
            'kcenter': KCenterAnchorStrategy,
            'density': DensityWeightedAnchorStrategy,
            'gmm': GMMCentroidAnchorStrategy,
            'stratified': StratifiedAnchorStrategy,
            'embedding_diverse': EmbeddingDiverseAnchorStrategy
        }
        
        if strategy_name not in strategy_classes:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        gen.strategy = strategy_classes[strategy_name].__new__(strategy_classes[strategy_name])
        gen.strategy.load(path)
        
        return gen


def compute_anchor_embeddings(
    anchor_images: np.ndarray,
    backbone_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8,
    return_projected: bool = False,
    apply_imagenet_norm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute DINOv3 embeddings for anchor images.
    
    CRITICAL: When return_projected=True, this function returns embeddings that
    have been projected through the backbone's projection head (if it exists).
    These embeddings should be stored as FIXED anchors and NOT re-projected
    during training. This prevents the collapse issue.
    
    Args:
        anchor_images: (K, H, W) anchor images
        backbone_model: DINOv3 model (with or without projection head)
        device: torch device
        batch_size: Batch size for processing
        return_projected: If True, return projected embeddings (for fixed anchors).
                         The backbone's forward() method handles projection and normalization.
    
    Returns:
        global_embeddings: (K, D) or (K, D_proj) global embeddings
        dense_embeddings: (K, H', W', C) dense feature maps
    """
    K = len(anchor_images)
    backbone_model.eval()
    
    global_embeds = []
    dense_embeds = []
    
    with torch.no_grad():
        for i in range(0, K, batch_size):
            batch = anchor_images[i:i+batch_size]
            
            # Convert to 3-channel tensor for DINOv3
            batch_tensor = _grayscale_batch_to_tensor(batch, device, apply_imagenet_norm=apply_imagenet_norm)
            
            if return_projected:
                # Use backbone's forward() which applies projection + normalization
                # This gives us embeddings in the PROJECTED space (e.g., 128D)
                outputs = backbone_model(batch_tensor)
                global_embeds.append(outputs['global'].cpu())
                dense_embeds.append(outputs['dense'].cpu())
            else:
                # Get RAW embeddings without projection (legacy behavior)
                features = backbone_model.backbone.forward_features(batch_tensor)
                cls_token = features[:, 0]  # (B, 384) - TRUE raw CLS token
                
                # Get dense features
                num_register_tokens = backbone_model.num_register_tokens
                patch_tokens = features[:, 1 + num_register_tokens:]
                H, W = batch_tensor.shape[2:]
                h_patches = H // backbone_model.patch_size
                w_patches = W // backbone_model.patch_size
                dense_feat = patch_tokens.view(batch_tensor.shape[0], h_patches, w_patches, -1)
                
                global_embeds.append(cls_token.cpu())
                dense_embeds.append(dense_feat.cpu())
    
    global_embeddings = torch.cat(global_embeds, dim=0)  # (K, D) or (K, D_proj)
    dense_embeddings = torch.cat(dense_embeds, dim=0)    # (K, H', W', C)
    
    # Note: When return_projected=True, embeddings are already normalized by backbone.forward()
    # When return_projected=False (raw), we don't normalize - caller can do so if needed
    
    has_projection = hasattr(backbone_model, 'projection') and backbone_model.projection is not None
    if return_projected and has_projection:
        print(f"Computed PROJECTED anchor embeddings: global {global_embeddings.shape} (normalized)")
    else:
        print(f"Computed RAW anchor embeddings: global {global_embeddings.shape}")
    print(f"  Dense embeddings: {dense_embeddings.shape}")
    
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