"""TimeQuant: Streaming Gaussian-Quantile Tokenizer for Time Series.

A lightweight library for tokenizing continuous time series data using
streaming Gaussian quantile binning with optional vector quantization.
"""

__version__ = "0.1.0"

from .tokenizers.gq_tokenizer import GQTokenizer
from .tokenizers.multi_gq_tokenizer import MultiGQTokenizer

__all__ = ["GQTokenizer", "MultiGQTokenizer"]