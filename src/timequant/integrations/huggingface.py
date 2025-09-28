"""Hugging Face Datasets integration for TimeQuant tokenizers.

Provides helpers for tokenizing time series data using HF Datasets,
including Features definitions and dataset mapping functions.
"""

from typing import Dict, Any, Optional, Callable, Union
import numpy as np

try:
    from datasets import Features, Sequence, Value, Array2D, Array3D
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from ..tokenizers.gq_tokenizer import GQTokenizer
from ..tokenizers.multi_gq_tokenizer import MultiGQTokenizer


def tokens_feature(name: str = "tokens") -> Features:
    """Create HF Features schema for tokenized time series.
    
    Parameters
    ----------
    name : str
        Name of the tokens column.
        
    Returns
    -------
    Features
        HuggingFace Features object.
    """
    if not HAS_DATASETS:
        raise ImportError("datasets package required for HF integration")
    
    return Features({name: Sequence(Value("int32"))})


def timeseries_tokens_feature(
    timeseries_name: str = "timeseries",
    tokens_name: str = "tokens"
) -> Features:
    """Create Features schema for both raw timeseries and tokens.
    
    Parameters
    ----------
    timeseries_name : str
        Name of the raw timeseries column.
    tokens_name : str
        Name of the tokens column.
        
    Returns
    -------
    Features
        Combined Features object.
    """
    if not HAS_DATASETS:
        raise ImportError("datasets package required for HF integration")
        
    return Features({
        timeseries_name: Sequence(Value("float32")),
        tokens_name: Sequence(Value("int32"))
    })


def create_tokenize_map_fn(
    tokenizer: Union[GQTokenizer, MultiGQTokenizer],
    input_column: str = "timeseries",
    output_column: str = "tokens",
    update_stats: bool = True
) -> Callable:
    """Create a map function for tokenizing datasets.
    
    Parameters
    ----------
    tokenizer : GQTokenizer or MultiGQTokenizer
        Fitted tokenizer instance.
    input_column : str
        Name of input column containing time series data.
    output_column : str
        Name of output column to store tokens.
    update_stats : bool
        Whether to update tokenizer statistics during mapping.
        
    Returns
    -------
    Callable
        Map function suitable for dataset.map().
    """
    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize batch of time series."""
        timeseries = batch[input_column]
        batch_tokens = []
        
        for ts in timeseries:
            ts_array = np.array(ts, dtype=np.float32)
            
            if isinstance(tokenizer, GQTokenizer):
                # Univariate case
                if update_stats:
                    for x in ts_array:
                        tokenizer.update(x)
                tokens = [tokenizer.encode(x) for x in ts_array]
            
            elif isinstance(tokenizer, MultiGQTokenizer):
                # Multivariate case - ts_array should be shape (T, D)
                if ts_array.ndim == 1:
                    ts_array = ts_array.reshape(-1, 1)
                    
                if update_stats:
                    for x_vec in ts_array:
                        tokenizer.update(x_vec)
                tokens = [tokenizer.encode(x_vec) for x_vec in ts_array]
            
            else:
                raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")
                
            batch_tokens.append(tokens)
        
        return {**batch, output_column: batch_tokens}
    
    return tokenize_fn


def create_streaming_tokenize_fn(
    tokenizer_class: Union[type, str],
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    input_column: str = "timeseries", 
    output_column: str = "tokens"
) -> Callable:
    """Create streaming tokenize function that fits tokenizer on-the-fly.
    
    Useful for cases where you want to fit the tokenizer during the first
    pass through the dataset.
    
    Parameters
    ----------
    tokenizer_class : type or str
        Tokenizer class or string name ('GQ' or 'MultiGQ').
    tokenizer_kwargs : dict, optional
        Keyword arguments for tokenizer initialization.
    input_column : str
        Input column name.
    output_column : str
        Output column name.
        
    Returns
    -------
    Callable
        Streaming tokenize function.
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
        
    # Resolve tokenizer class
    if isinstance(tokenizer_class, str):
        if tokenizer_class.lower() in ('gq', 'gqtokenizer'):
            tokenizer_class = GQTokenizer
        elif tokenizer_class.lower() in ('multigq', 'multigqtokenizer'):
            tokenizer_class = MultiGQTokenizer
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer_class}")
    
    tokenizer = None
    
    def streaming_tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal tokenizer
        
        timeseries = batch[input_column]
        batch_tokens = []
        
        for ts in timeseries:
            ts_array = np.array(ts, dtype=np.float32)
            
            # Initialize tokenizer on first sample if needed
            if tokenizer is None:
                if tokenizer_class == MultiGQTokenizer:
                    if ts_array.ndim == 1:
                        D = 1
                        ts_array = ts_array.reshape(-1, 1)
                    else:
                        D = ts_array.shape[1]
                    tokenizer_kwargs.setdefault('D', D)
                
                tokenizer = tokenizer_class(**tokenizer_kwargs)
            
            # Update and encode
            if isinstance(tokenizer, GQTokenizer):
                tokens = []
                for x in ts_array:
                    tokenizer.update(x)
                    tokens.append(tokenizer.encode(x))
                    
            elif isinstance(tokenizer, MultiGQTokenizer):
                if ts_array.ndim == 1:
                    ts_array = ts_array.reshape(-1, 1)
                    
                tokens = []
                for x_vec in ts_array:
                    tokenizer.update(x_vec)
                    # For streaming, just use vector codes if no codebook fitted
                    if tokenizer.codebook.is_fitted:
                        tokens.append(tokenizer.encode(x_vec))
                    else:
                        # Return vector code as tuple/list for now
                        code = tokenizer.encode_vector_code(x_vec)
                        tokens.append(tuple(code))
                        
            batch_tokens.append(tokens)
            
        return {**batch, output_column: batch_tokens}
    
    return streaming_tokenize_fn


class DatasetTokenizer:
    """High-level wrapper for tokenizing HF datasets.
    
    Parameters
    ----------
    tokenizer : GQTokenizer or MultiGQTokenizer
        Tokenizer instance.
    input_column : str
        Input column name.
    output_column : str  
        Output column name.
    """
    
    def __init__(
        self,
        tokenizer: Union[GQTokenizer, MultiGQTokenizer],
        input_column: str = "timeseries",
        output_column: str = "tokens"
    ):
        if not HAS_DATASETS:
            raise ImportError("datasets package required for DatasetTokenizer")
            
        self.tokenizer = tokenizer
        self.input_column = input_column
        self.output_column = output_column
    
    def fit_and_transform(self, dataset, num_proc: int = 1, batch_size: int = 1000):
        """Fit tokenizer and transform dataset in one pass.
        
        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset.
        num_proc : int
            Number of processes for mapping.
        batch_size : int
            Batch size for mapping.
            
        Returns
        -------
        Dataset
            Transformed dataset with tokens.
        """
        map_fn = create_tokenize_map_fn(
            self.tokenizer, 
            self.input_column, 
            self.output_column,
            update_stats=True
        )
        
        return dataset.map(
            map_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Tokenizing time series"
        )
    
    def transform(self, dataset, num_proc: int = 1, batch_size: int = 1000):
        """Transform dataset using fitted tokenizer.
        
        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset.
        num_proc : int
            Number of processes.
        batch_size : int
            Batch size.
            
        Returns
        -------
        Dataset
            Transformed dataset.
        """
        if not self.tokenizer.is_fitted:
            raise ValueError("Tokenizer not fitted. Use fit_and_transform() first.")
            
        map_fn = create_tokenize_map_fn(
            self.tokenizer,
            self.input_column, 
            self.output_column,
            update_stats=False
        )
        
        return dataset.map(
            map_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Tokenizing time series"
        )