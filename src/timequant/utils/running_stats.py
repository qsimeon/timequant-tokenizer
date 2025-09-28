"""Online statistics using Welford's algorithm.

Provides numerically stable computation of running mean and variance
for scalar or vector inputs in a streaming fashion.
"""

from __future__ import annotations
import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, float, int]


class RunningStats:
    """Numerically stable running mean/variance using Welford's method.

    Supports scalar or vector inputs. Use `update(x)` per time step or 
    `update_batch(x, axis=0)` for batch updates.

    Parameters
    ----------
    dim : int or None
        Dimensionality of input data. If None, inferred from first update.
    eps : float
        Small epsilon to add to variance for numerical stability.

    Attributes
    ----------
    n : int
        Number of samples seen.
    mean : np.ndarray or None
        Running mean estimate.
    M2 : np.ndarray or None  
        Sum of squares of differences from the mean (for variance calculation).
    eps : float
        Numerical stability epsilon.
    """

    def __init__(self, dim: int | None = None, eps: float = 1e-8):
        self.n = 0
        self.mean = None if dim is None else np.zeros(dim, dtype=float)
        self.M2 = None if dim is None else np.zeros(dim, dtype=float)
        self.eps = eps

    def update(self, x: ArrayLike) -> None:
        """Update statistics with a single observation.

        Parameters
        ----------
        x : array_like
            Single observation (scalar or vector).
        """
        x = np.asarray(x, dtype=float)
        
        # Initialize on first update if needed
        if self.mean is None:
            self.mean = np.zeros_like(x, dtype=float)
            self.M2 = np.zeros_like(x, dtype=float)
        
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def update_batch(self, X: np.ndarray, axis: int = 0) -> None:
        """Update statistics with multiple observations.

        Parameters
        ----------
        X : np.ndarray
            Batch of observations.
        axis : int
            Axis along which to iterate (default: 0, iterate over rows).
        """
        for x in np.moveaxis(X, axis, 0):
            self.update(x)

    @property
    def var(self) -> np.ndarray:
        """Sample variance estimate.
        
        Returns
        -------
        np.ndarray
            Sample variance (unbiased, using n-1 denominator).
        """
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.M2 / (self.n - 1)

    @property  
    def std(self) -> np.ndarray:
        """Sample standard deviation estimate.
        
        Returns
        -------
        np.ndarray
            Standard deviation with eps added for stability.
        """
        return np.sqrt(self.var + self.eps)

    def normalize(self, x: ArrayLike) -> np.ndarray:
        """Standardize input using current statistics.

        Parameters
        ----------  
        x : array_like
            Input to standardize.
            
        Returns
        -------
        np.ndarray
            Standardized input: (x - mean) / std.
        """
        x = np.asarray(x, dtype=float)
        if self.mean is None:
            raise ValueError("No data seen yet. Call update() first.")
        return (x - self.mean) / self.std

    def reset(self) -> None:
        """Reset all statistics to initial state."""
        self.n = 0
        if self.mean is not None:
            self.mean.fill(0.0)
            self.M2.fill(0.0)

    def __repr__(self) -> str:
        return (f"RunningStats(n={self.n}, "
                f"mean={self.mean}, std={self.std if self.mean is not None else None})")