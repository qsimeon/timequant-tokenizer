"""Lightweight k-means codebook for vector quantization.

Provides both pure NumPy implementation and optional scikit-learn 
MiniBatchKMeans for codebook learning.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

try:
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class KMeansCodebook:
    """K-means codebook for vector quantization.

    Learns a codebook of centroids and maps input vectors to nearest
    centroid indices. Supports both sklearn MiniBatchKMeans and a 
    simple NumPy implementation.

    Parameters
    ----------
    n_codes : int
        Number of centroids in the codebook.

    Attributes
    ----------
    n_codes : int
        Codebook size.
    centroids : np.ndarray or None
        Learned centroids, shape (n_codes, D).
    """

    def __init__(self, n_codes: int):
        self.n_codes = int(n_codes)
        self.centroids: Optional[np.ndarray] = None
        self._km_model = None  # For sklearn model if used

    def fit(
        self, 
        X: np.ndarray, 
        seed: int = 0, 
        max_iter: int = 100,
        batch_size: int = 1024,
        use_sklearn: bool = True
    ) -> None:
        """Fit codebook on input vectors.

        Parameters
        ----------
        X : np.ndarray
            Input vectors, shape (N, D).
        seed : int
            Random seed.
        max_iter : int
            Maximum iterations for k-means.
        batch_size : int
            Batch size for mini-batch k-means (sklearn only).
        use_sklearn : bool
            Whether to use sklearn MiniBatchKMeans if available.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        N, D = X.shape
        if N < self.n_codes:
            warnings.warn(f"Number of samples ({N}) < n_codes ({self.n_codes})")

        np.random.seed(seed)

        if use_sklearn and HAS_SKLEARN:
            self._fit_sklearn(X, seed, max_iter, batch_size)
        else:
            self._fit_numpy(X, seed, max_iter)

    def _fit_sklearn(
        self, 
        X: np.ndarray, 
        seed: int, 
        max_iter: int, 
        batch_size: int
    ) -> None:
        """Fit using sklearn MiniBatchKMeans."""
        self._km_model = MiniBatchKMeans(
            n_clusters=self.n_codes,
            random_state=seed,
            max_iter=max_iter,
            batch_size=min(batch_size, X.shape[0]),
            n_init=3
        )
        self._km_model.fit(X)
        self.centroids = self._km_model.cluster_centers_.copy()

    def _fit_numpy(self, X: np.ndarray, seed: int, max_iter: int) -> None:
        """Simple k-means implementation using NumPy."""
        N, D = X.shape
        
        # Initialize centroids with k-means++
        centroids = self._kmeans_plus_plus_init(X, self.n_codes, seed)
        
        for _ in range(max_iter):
            # Assign to nearest centroids
            distances = np.linalg.norm(
                X[:, None, :] - centroids[None, :, :], axis=2
            )
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_codes):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        self.centroids = centroids

    def _kmeans_plus_plus_init(
        self, X: np.ndarray, k: int, seed: int
    ) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm."""
        np.random.seed(seed)
        N, D = X.shape
        centroids = np.zeros((k, D))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(N)]
        
        # Choose remaining centroids
        for c_id in range(1, k):
            # Compute distances to nearest centroid
            distances = np.array([
                min([np.linalg.norm(x - c) ** 2 for c in centroids[:c_id]])
                for x in X
            ])
            
            # Choose next centroid with probability proportional to squared distance
            probs = distances / distances.sum()
            cumprobs = probs.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumprobs):
                if r < p:
                    centroids[c_id] = X[j]
                    break
                    
        return centroids

    def predict(self, x: np.ndarray) -> int:
        """Predict cluster for single vector.

        Parameters
        ----------
        x : np.ndarray
            Input vector, shape (D,).

        Returns
        -------
        int
            Nearest centroid index.
        """
        if not self.is_fitted:
            raise ValueError("Codebook not fitted. Call fit() first.")

        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"Expected shape (D,) or (1, D), got {x.shape}")

        if self._km_model is not None:
            return int(self._km_model.predict(x)[0])
        else:
            distances = np.linalg.norm(self.centroids - x, axis=1)
            return int(np.argmin(distances))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict clusters for batch of vectors.

        Parameters
        ----------
        X : np.ndarray
            Batch of vectors, shape (N, D).

        Returns
        -------
        np.ndarray
            Nearest centroid indices, shape (N,).
        """
        if not self.is_fitted:
            raise ValueError("Codebook not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        if self._km_model is not None:
            return self._km_model.predict(X)
        else:
            distances = np.linalg.norm(
                X[:, None, :] - self.centroids[None, :, :], axis=2
            )
            return np.argmin(distances, axis=1)

    def decode(self, code: int) -> np.ndarray:
        """Get centroid vector for given code.

        Parameters
        ----------
        code : int
            Centroid index.

        Returns
        -------
        np.ndarray
            Centroid vector, shape (D,).
        """
        if not self.is_fitted:
            raise ValueError("Codebook not fitted. Call fit() first.")

        if not (0 <= code < self.n_codes):
            raise ValueError(f"Code {code} out of range [0, {self.n_codes})")

        return self.centroids[code].copy()

    def decode_batch(self, codes: np.ndarray) -> np.ndarray:
        """Get centroid vectors for batch of codes.

        Parameters
        ----------
        codes : np.ndarray
            Centroid indices, shape (N,).

        Returns
        -------
        np.ndarray
            Centroid vectors, shape (N, D).
        """
        if not self.is_fitted:
            raise ValueError("Codebook not fitted. Call fit() first.")

        codes = np.asarray(codes)
        if codes.ndim != 1:
            raise ValueError(f"Expected 1D array, got {codes.ndim}D")

        return self.centroids[codes]

    def get_distortion(self, X: np.ndarray) -> float:
        """Compute average squared distance to nearest centroids.

        Parameters
        ----------
        X : np.ndarray
            Input vectors, shape (N, D).

        Returns
        -------
        float
            Average squared distance (distortion).
        """
        if not self.is_fitted:
            raise ValueError("Codebook not fitted. Call fit() first.")

        labels = self.predict_batch(X)
        distances_sq = np.sum(
            (X - self.centroids[labels]) ** 2, axis=1
        )
        return float(np.mean(distances_sq))

    @property
    def is_fitted(self) -> bool:
        """Check if codebook has been fitted."""
        return self.centroids is not None

    def __repr__(self) -> str:
        return f"KMeansCodebook(n_codes={self.n_codes}, fitted={self.is_fitted})"