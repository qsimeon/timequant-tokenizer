"""Gaussian quantile binning for tokenization.

Maps standardized (z-score) values to discrete tokens using 
precomputed Gaussian quantile boundaries.
"""

import numpy as np
import bisect
from scipy.stats import norm
from typing import Optional


class QuantileBinner:
    """Maps a standardized value to a token id by N(0,1) quantile bins.

    Uses precomputed quantile boundaries from standard normal distribution
    to map z-scores to discrete token IDs via binary search.

    Parameters
    ----------
    vocab_size : int
        Number of bins V (vocabulary size).
    boundaries : np.ndarray, optional
        Optional precomputed boundaries (length V-1). If None, uses N(0,1) quantiles.
        
    Attributes
    ----------
    vocab_size : int
        Vocabulary size.
    boundaries : np.ndarray
        Quantile boundaries for binning, shape (vocab_size-1,).
    """

    def __init__(self, vocab_size: int, boundaries: Optional[np.ndarray] = None):
        self.vocab_size = int(vocab_size)
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
            
        if boundaries is None:
            # Create equally-spaced quantiles from standard normal
            qs = np.linspace(0, 1, self.vocab_size + 1)[1:-1]  # Skip 0 and 1
            self.boundaries = norm.ppf(qs).astype(np.float32)
        else:
            self.boundaries = np.asarray(boundaries, dtype=np.float32)
            
        if self.boundaries.shape[0] != self.vocab_size - 1:
            raise ValueError(
                f"Expected {self.vocab_size - 1} boundaries, "
                f"got {self.boundaries.shape[0]}"
            )

    def encode_scalar(self, z: float) -> int:
        """Encode a single z-score to token ID.
        
        Bins are defined as:
        - Token 0: (-inf, boundaries[0]]
        - Token 1: (boundaries[0], boundaries[1]]
        - ...
        - Token V-1: (boundaries[-1], inf)
        
        Parameters
        ----------
        z : float
            Standardized input value.
            
        Returns
        -------
        int
            Token ID in range [0, vocab_size).
        """
        return bisect.bisect_right(self.boundaries.tolist(), float(z))

    def encode_batch(self, z: np.ndarray) -> np.ndarray:
        """Encode batch of z-scores to token IDs.
        
        Parameters
        ----------
        z : np.ndarray
            Array of standardized values.
            
        Returns
        -------
        np.ndarray
            Array of token IDs, same shape as input.
        """
        z_flat = z.flatten()
        tokens = np.searchsorted(self.boundaries, z_flat, side='right')
        return tokens.reshape(z.shape)

    def decode_scalar(self, tok: int) -> float:
        """Decode token ID to approximate z-score (bin midpoint).
        
        Uses the quantile midpoint of the corresponding bin as the 
        decoded value.
        
        Parameters
        ----------
        tok : int
            Token ID.
            
        Returns
        -------
        float
            Approximate z-score (bin midpoint).
        """
        if not (0 <= tok < self.vocab_size):
            raise ValueError(f"Token {tok} out of range [0, {self.vocab_size})")
            
        # Get bin boundaries
        if tok == 0:
            lo, hi = -np.inf, self.boundaries[0]
        elif tok == self.vocab_size - 1:
            lo, hi = self.boundaries[-1], np.inf
        else:
            lo, hi = self.boundaries[tok - 1], self.boundaries[tok]
        
        # Use quantile midpoint
        if np.isinf(lo):
            q_lo = 0.0
        else:
            q_lo = float(norm.cdf(lo))
            
        if np.isinf(hi):
            q_hi = 1.0
        else:
            q_hi = float(norm.cdf(hi))
            
        q_mid = 0.5 * (q_lo + q_hi)
        return float(norm.ppf(q_mid))

    def decode_batch(self, tokens: np.ndarray) -> np.ndarray:
        """Decode batch of tokens to approximate z-scores.
        
        Parameters
        ----------
        tokens : np.ndarray
            Array of token IDs.
            
        Returns
        -------
        np.ndarray
            Array of approximate z-scores, same shape as input.
        """
        tokens_flat = tokens.flatten()
        z_flat = np.array([self.decode_scalar(int(tok)) for tok in tokens_flat])
        return z_flat.reshape(tokens.shape)

    def get_bin_edges(self) -> np.ndarray:
        """Get all bin edges including -inf and +inf.
        
        Returns
        -------
        np.ndarray
            Bin edges, shape (vocab_size + 1,).
        """
        return np.concatenate([[-np.inf], self.boundaries, [np.inf]])

    def __repr__(self) -> str:
        return f"QuantileBinner(vocab_size={self.vocab_size})"