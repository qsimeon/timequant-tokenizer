"""Gaussian-Quantile tokenizer for 1D streams.

Combines online statistics with Gaussian quantile binning for 
streaming tokenization of univariate time series.
"""

import numpy as np
from typing import Union
from ..utils.running_stats import RunningStats, ArrayLike
from .quantile_binner import QuantileBinner


class GQTokenizer:
    """Gaussian-Quantile tokenizer for 1D streams.

    Maintains running statistics and maps each observation to a discrete
    token using Gaussian quantile binning of the standardized values.

    Usage:
        tok = GQTokenizer(V=64)
        for x_t in stream: 
            tok.update(x_t)
            token = tok.encode(x_t)

    Parameters
    ----------
    V : int
        Vocabulary size (number of tokens).
    eps : float
        Small epsilon for numerical stability in standardization.
    warmup_samples : int
        Number of initial samples to see before full normalization kicks in.
        During warmup, uses population std = 1.0.

    Attributes
    ----------
    stats : RunningStats
        Online statistics tracker.
    binner : QuantileBinner  
        Quantile-based token mapper.
    vocab_size : int
        Vocabulary size.
    warmup_samples : int
        Warmup period length.
    eps : float
        Numerical epsilon.
    """

    def __init__(self, V: int, eps: float = 1e-8, warmup_samples: int = 10):
        self.vocab_size = V
        self.stats = RunningStats(eps=eps)
        self.binner = QuantileBinner(V)
        self.eps = eps
        self.warmup_samples = warmup_samples

    def update(self, x: ArrayLike) -> None:
        """Update running statistics with new observation.

        Parameters
        ----------
        x : array_like
            New observation (scalar).
        """
        self.stats.update(x)

    def encode(self, x: ArrayLike) -> int:
        """Encode observation to token ID.

        Parameters
        ----------
        x : array_like
            Observation to encode (scalar).

        Returns
        -------
        int
            Token ID in range [0, vocab_size).
        """
        x = np.asarray(x, dtype=float)
        
        if self.stats.mean is None:
            raise ValueError("No data seen yet. Call update() first.")
        
        # During warmup, use simple z-score with std=1
        if self.stats.n < self.warmup_samples:
            z = float(x - self.stats.mean)
        else:
            z = float((x - self.stats.mean) / self.stats.std)
        
        # Clip extreme values to avoid issues with infinite boundaries
        z = np.clip(z, -10.0, 10.0)
        
        return self.binner.encode_scalar(z)

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode batch of observations to token IDs.
        
        Note: This assumes statistics have already been updated with all data.

        Parameters
        ----------
        X : np.ndarray
            Batch of observations.

        Returns
        -------
        np.ndarray
            Array of token IDs, same shape as input.
        """
        if self.stats.mean is None:
            raise ValueError("No data seen yet. Call update() first.")
            
        X = np.asarray(X, dtype=float)
        
        if self.stats.n < self.warmup_samples:
            Z = X - self.stats.mean
        else:
            Z = (X - self.stats.mean) / self.stats.std
            
        Z = np.clip(Z, -10.0, 10.0)
        return self.binner.encode_batch(Z)

    def decode(self, tok: int) -> float:
        """Decode token ID back to approximate original value.

        Parameters
        ----------
        tok : int
            Token ID to decode.

        Returns
        -------
        float
            Approximate original value (bin midpoint mapped back to x-space).
        """
        if self.stats.mean is None:
            raise ValueError("No data seen yet. Call update() first.")
            
        z = self.binner.decode_scalar(tok)
        
        if self.stats.n < self.warmup_samples:
            return float(self.stats.mean + z)
        else:
            return float(self.stats.mean + z * self.stats.std)

    def decode_batch(self, tokens: np.ndarray) -> np.ndarray:
        """Decode batch of tokens to approximate original values.

        Parameters
        ----------
        tokens : np.ndarray
            Array of token IDs.

        Returns
        -------
        np.ndarray
            Array of approximate original values, same shape as input.
        """
        if self.stats.mean is None:
            raise ValueError("No data seen yet. Call update() first.")
            
        Z = self.binner.decode_batch(tokens)
        
        if self.stats.n < self.warmup_samples:
            return self.stats.mean + Z
        else:
            return self.stats.mean + Z * self.stats.std

    def get_vocab_utilization(self, tokens: np.ndarray) -> float:
        """Calculate vocabulary utilization from token sequence.

        Parameters
        ----------
        tokens : np.ndarray
            Sequence of token IDs.

        Returns
        -------
        float
            Fraction of vocabulary actually used (unique tokens / vocab_size).
        """
        unique_tokens = len(np.unique(tokens))
        return unique_tokens / self.vocab_size

    def reset(self) -> None:
        """Reset tokenizer to initial state."""
        self.stats.reset()

    @property 
    def is_fitted(self) -> bool:
        """Check if tokenizer has seen any data."""
        return self.stats.n > 0

    def __repr__(self) -> str:
        return (f"GQTokenizer(vocab_size={self.vocab_size}, "
                f"n_samples={self.stats.n}, "
                f"fitted={self.is_fitted})")