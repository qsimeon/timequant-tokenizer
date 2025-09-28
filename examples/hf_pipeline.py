#!/usr/bin/env python3
"""Example: TimeQuant tokenization with HuggingFace Datasets.

This script demonstrates how to use TimeQuant tokenizers with 
HuggingFace Datasets for streaming time series tokenization.
"""

import numpy as np
from datasets import Dataset
from timequant.tokenizers.gq_tokenizer import GQTokenizer
from timequant.tokenizers.multi_gq_tokenizer import MultiGQTokenizer
from timequant.integrations.huggingface import (
    create_tokenize_map_fn, 
    DatasetTokenizer,
    timeseries_tokens_feature
)


def create_sample_dataset(n_samples=1000, seq_length=100, n_dims=1):
    """Create a sample time series dataset."""
    np.random.seed(42)
    
    if n_dims == 1:
        # Univariate time series
        data = []
        for _ in range(n_samples):
            # Generate AR(1) process with different parameters
            phi = np.random.uniform(0.5, 0.9)
            noise_std = np.random.uniform(0.1, 0.5)
            
            ts = [0.0]
            for _ in range(seq_length - 1):
                ts.append(phi * ts[-1] + np.random.normal(0, noise_std))
                
            data.append(ts)
    else:
        # Multivariate time series
        data = []
        for _ in range(n_samples):
            ts = np.random.randn(seq_length, n_dims).tolist()
            data.append(ts)
    
    return Dataset.from_dict({
        "timeseries": data,
        "sample_id": list(range(n_samples))
    })


def example_univariate_tokenization():
    """Example of univariate tokenization."""
    print("=== Univariate Tokenization Example ===")
    
    # Create sample dataset
    dataset = create_sample_dataset(n_samples=100, seq_length=50, n_dims=1)
    print(f"Dataset: {len(dataset)} samples, sequence length: 50")
    
    # Initialize tokenizer
    tokenizer = GQTokenizer(V=16)
    
    # Create dataset tokenizer wrapper
    ds_tokenizer = DatasetTokenizer(tokenizer, input_column="timeseries", output_column="tokens")
    
    # Tokenize dataset
    tokenized_dataset = ds_tokenizer.fit_and_transform(dataset, batch_size=10)
    
    # Examine results
    sample = tokenized_dataset[0]
    print(f"Original shape: {len(sample['timeseries'])}")
    print(f"Tokens shape: {len(sample['tokens'])}")
    print(f"Sample tokens: {sample['tokens'][:10]}...")
    print(f"Vocab utilization: {tokenizer.get_vocab_utilization(np.array(sample['tokens'])):.3f}")
    
    print(f"Tokenizer stats: n={tokenizer.stats.n}, mean={tokenizer.stats.mean:.3f}, std={tokenizer.stats.std:.3f}")
    print()


def example_multivariate_tokenization():
    """Example of multivariate tokenization."""
    print("=== Multivariate Tokenization Example ===")
    
    # Create multivariate dataset
    dataset = create_sample_dataset(n_samples=50, seq_length=30, n_dims=3)
    print(f"Dataset: {len(dataset)} samples, sequence length: 30, dimensions: 3")
    
    # Initialize multivariate tokenizer
    tokenizer = MultiGQTokenizer(D=3, V_dim=4, V=64)
    
    # We need to do a two-pass approach for multivariate:
    # 1. First pass: collect vector codes
    # 2. Fit codebook
    # 3. Second pass: tokenize with fitted codebook
    
    print("Phase 1: Collecting vector codes...")
    codes_list = []
    
    def collect_codes_fn(batch):
        batch_codes = []
        for ts in batch["timeseries"]:
            ts_array = np.array(ts, dtype=np.float32)  # Shape: (T, D)
            
            seq_codes = []
            for t in range(ts_array.shape[0]):
                tokenizer.update(ts_array[t])
                code = tokenizer.encode_vector_code(ts_array[t])
                seq_codes.append(code.tolist())
                codes_list.append(code)
                
            batch_codes.append(seq_codes)
            
        return {**batch, "vector_codes": batch_codes}
    
    # First pass: collect codes
    dataset_with_codes = dataset.map(
        collect_codes_fn, 
        batched=True, 
        batch_size=5,
        desc="Collecting vector codes"
    )
    
    print(f"Collected {len(codes_list)} vector codes")
    print(f"Per-dim utilization: {tokenizer.get_per_dim_utilization(np.array(codes_list))}")
    
    # Fit codebook
    print("Phase 2: Fitting codebook...")
    codes_array = np.array(codes_list)
    tokenizer.fit_codebook(codes_array)
    print(f"Codebook fitted with {tokenizer.codebook.n_codes} centroids")
    
    # Second pass: final tokenization
    print("Phase 3: Final tokenization...")
    
    def final_tokenize_fn(batch):
        batch_tokens = []
        for ts in batch["timeseries"]:
            ts_array = np.array(ts, dtype=np.float32)
            
            tokens = []
            for t in range(ts_array.shape[0]):
                token = tokenizer.encode(ts_array[t])
                tokens.append(int(token))
                
            batch_tokens.append(tokens)
            
        return {**batch, "tokens": batch_tokens}
    
    final_dataset = dataset_with_codes.map(
        final_tokenize_fn,
        batched=True,
        batch_size=5,
        desc="Final tokenization"
    )
    
    # Examine results
    sample = final_dataset[0]
    print(f"Original shape: {np.array(sample['timeseries']).shape}")
    print(f"Tokens shape: {len(sample['tokens'])}")
    print(f"Sample tokens: {sample['tokens'][:10]}...")
    
    all_tokens = []
    for item in final_dataset:
        all_tokens.extend(item['tokens'])
    all_tokens = np.array(all_tokens)
    
    print(f"Final vocab utilization: {tokenizer.get_vocab_utilization(all_tokens):.3f}")
    print()


def example_streaming_tokenization():
    """Example of streaming tokenization without pre-fitting."""
    print("=== Streaming Tokenization Example ===")
    
    # This would be useful for very large datasets where you can't 
    # afford two passes
    
    from timequant.integrations.huggingface import create_streaming_tokenize_fn
    
    dataset = create_sample_dataset(n_samples=20, seq_length=25, n_dims=1)
    
    # Create streaming tokenizer function
    streaming_fn = create_streaming_tokenize_fn(
        tokenizer_class="GQ",
        tokenizer_kwargs={"V": 8},
        input_column="timeseries",
        output_column="tokens"
    )
    
    # Apply streaming tokenization
    tokenized_dataset = dataset.map(
        streaming_fn,
        batched=True,
        batch_size=5,
        desc="Streaming tokenization"
    )
    
    # Examine results
    sample = tokenized_dataset[0]
    print(f"Sample tokens: {sample['tokens'][:10]}...")
    print("Note: Streaming tokenization may give different results as statistics evolve")
    print()


def main():
    """Run all examples."""
    print("TimeQuant HuggingFace Integration Examples")
    print("=" * 50)
    
    try:
        example_univariate_tokenization()
        example_multivariate_tokenization() 
        example_streaming_tokenization()
        
        print("All examples completed successfully!")
        
    except ImportError as e:
        print(f"Missing optional dependency: {e}")
        print("Install with: pip install timequant-tokenizer[hf]")


if __name__ == "__main__":
    main()