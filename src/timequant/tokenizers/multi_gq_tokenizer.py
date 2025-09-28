"""Per-dimension GQ + codebook reduction to vocab V.

Multivariate extension of GQ tokenization using per-dimension binning
followed by vector quantization with a learned codebook.
"""

import numpy as np
from typing import Optional, List
from .gq_tokenizer import GQTokenizer
from ..quantize.kmeans import KMeansCodebook


class MultiGQTokenizer:
    """Per-dimension GQ + codebook reduction to vocab V.

    Steps:
      1) Maintain D independent GQTokenizers with per-dim V_dim bins (e.g., 2 or 4).
      2) Concatenate per-dim bin IDs into a vector code.  
      3) Map vector code to final token via nearest centroid in a learned codebook.

    Training requires a pass to collect code vectors then fit the codebook.
    Inference uses the fixed codebook.

    Parameters
    ----------
    D : int
        Dimensionality of input data.
    V_dim : int  
        Vocabulary size per dimension (typically 2-4).
    V : int
        Final vocabulary size after codebook reduction.
    eps : float
        Numerical stability epsilon.
    warmup_samples : int
        Warmup samples for each dimension.

    Attributes
    ----------
    D : int
        Input dimensionality.
    V_dim : int
        Per-dimension vocabulary size.
    V : int  
        Final vocabulary size.
    per_dim : List[GQTokenizer]
        List of per-dimension tokenizers.
    codebook : KMeansCodebook
        Vector quantization codebook.
    """

    def __init__(
        self, 
        D: int, 
        V_dim: int = 2, 
        V: int = 256, 
        eps: float = 1e-8,
        warmup_samples: int = 10
    ):
        self.D = D
        self.V_dim = V_dim  
        self.V = V
        self.per_dim = [
            GQTokenizer(V_dim, eps=eps, warmup_samples=warmup_samples) 
            for _ in range(D)
        ]
        self.codebook = KMeansCodebook(n_codes=V)

    def update(self, x: np.ndarray) -> None:
        """Update statistics for all dimensions.

        Parameters
        ----------
        x : np.ndarray
            Input vector, shape (D,).
        """
        x = np.asarray(x)
        if x.shape != (self.D,):
            raise ValueError(f"Expected shape ({self.D},), got {x.shape}")
            
        for d in range(self.D):
            self.per_dim[d].update(x[d])

    def update_batch(self, X: np.ndarray) -> None:
        """Update statistics with batch of vectors.

        Parameters
        ----------
        X : np.ndarray
            Batch of input vectors, shape (N, D).
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"Expected shape (N, {self.D}), got {X.shape}")
            
        for x in X:
            self.update(x)

    def encode_vector_code(self, x: np.ndarray) -> np.ndarray:
        """Encode to per-dimension code vector (before codebook).

        Parameters
        ----------
        x : np.ndarray
            Input vector, shape (D,).

        Returns
        -------
        np.ndarray
            Per-dimension code vector, shape (D,), dtype int.
        """
        x = np.asarray(x)
        if x.shape != (self.D,):
            raise ValueError(f"Expected shape ({self.D},), got {x.shape}")
            
        codes = np.array([
            self.per_dim[d].encode(x[d]) for d in range(self.D)
        ], dtype=np.int32)
        
        return codes

    def encode_vector_code_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode batch to per-dimension code vectors.

        Parameters
        ----------
        X : np.ndarray
            Batch of input vectors, shape (N, D).

        Returns
        -------
        np.ndarray
            Batch of code vectors, shape (N, D), dtype int.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"Expected shape (N, {self.D}), got {X.shape}")
            
        N = X.shape[0]
        codes = np.zeros((N, self.D), dtype=np.int32)
        
        for d in range(self.D):
            codes[:, d] = self.per_dim[d].encode_batch(X[:, d])
            
        return codes

    def fit_codebook(
        self, 
        codes: np.ndarray, 
        seed: int = 0, 
        max_iter: int = 100,
        batch_size: int = 1024
    ) -> None:
        """Fit codebook on collection of vector codes.

        Parameters
        ----------
        codes : np.ndarray
            Collection of vector codes, shape (N, D).
        seed : int
            Random seed for k-means.
        max_iter : int
            Maximum k-means iterations.
        batch_size : int
            Batch size for mini-batch k-means.
        """
        if codes.ndim != 2 or codes.shape[1] != self.D:
            raise ValueError(f"Expected shape (N, {self.D}), got {codes.shape}")
            
        self.codebook.fit(
            codes, 
            seed=seed, 
            max_iter=max_iter, 
            batch_size=batch_size
        )

    def encode(self, x: np.ndarray) -> int:
        """Encode to final token (after codebook reduction).

        Parameters
        ----------
        x : np.ndarray
            Input vector, shape (D,).

        Returns  
        -------
        int
            Final token ID in range [0, V).
        """
        if not self.codebook.is_fitted:
            raise ValueError("Codebook not fitted. Call fit_codebook() first.")
            
        code = self.encode_vector_code(x)
        return int(self.codebook.predict(code))

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode batch to final tokens.

        Parameters
        ----------
        X : np.ndarray
            Batch of input vectors, shape (N, D).

        Returns
        -------
        np.ndarray
            Array of final token IDs, shape (N,).
        """
        if not self.codebook.is_fitted:
            raise ValueError("Codebook not fitted. Call fit_codebook() first.")
            
        codes = self.encode_vector_code_batch(X)
        return self.codebook.predict_batch(codes)

    def decode_vector_code(self, code: np.ndarray) -> np.ndarray:
        """Decode vector code back to approximate original vector.

        Parameters
        ----------
        code : np.ndarray
            Vector code, shape (D,).

        Returns
        -------
        np.ndarray
            Approximate original vector, shape (D,).
        """
        code = np.asarray(code)
        if code.shape != (self.D,):
            raise ValueError(f"Expected shape ({self.D},), got {code.shape}")
            
        return np.array([
            self.per_dim[d].decode(int(code[d])) for d in range(self.D)
        ])

    def decode(self, tok: int) -> np.ndarray:
        """Decode final token to approximate original vector.

        Parameters
        ----------
        tok : int
            Final token ID.

        Returns
        -------
        np.ndarray
            Approximate original vector, shape (D,).
        """
        if not self.codebook.is_fitted:
            raise ValueError("Codebook not fitted. Call fit_codebook() first.")
            
        code = self.codebook.decode(tok)  # Get nearest centroid
        return self.decode_vector_code(code)

    def get_vocab_utilization(self, tokens: np.ndarray) -> float:
        """Calculate final vocabulary utilization.

        Parameters
        ----------
        tokens : np.ndarray
            Sequence of final token IDs.

        Returns
        -------
        float
            Fraction of final vocabulary used.
        """
        unique_tokens = len(np.unique(tokens))
        return unique_tokens / self.V

    def get_per_dim_utilization(self, codes: np.ndarray) -> np.ndarray:
        """Calculate per-dimension vocabulary utilization.

        Parameters
        ----------
        codes : np.ndarray
            Collection of vector codes, shape (N, D).

        Returns
        -------
        np.ndarray
            Per-dimension utilization fractions, shape (D,).
        """
        if codes.ndim != 2 or codes.shape[1] != self.D:
            raise ValueError(f"Expected shape (N, {self.D}), got {codes.shape}")
            
        util = np.zeros(self.D)
        for d in range(self.D):
            unique_codes_d = len(np.unique(codes[:, d]))
            util[d] = unique_codes_d / self.V_dim
            
        return util

    def reset(self) -> None:
        """Reset all tokenizers to initial state."""
        for tokenizer in self.per_dim:
            tokenizer.reset()
        self.codebook = KMeansCodebook(n_codes=self.V)

    @property
    def is_fitted(self) -> bool:
        """Check if all components are fitted."""
        stats_fitted = all(tok.is_fitted for tok in self.per_dim)
        return stats_fitted and self.codebook.is_fitted

    def __repr__(self) -> str:
        n_samples = self.per_dim[0].stats.n if self.per_dim else 0
        return (f"MultiGQTokenizer(D={self.D}, V_dim={self.V_dim}, V={self.V}, "
                f"n_samples={n_samples}, fitted={self.is_fitted})")