# timequant-tokenizer

A small, dependency-light library for **streaming tokenization of continuous time series** via Gaussian-quantile binning + (optional) vector quantization.

> v0.1 scope: univariate tokenizer (online), per-dimension multivariate, optional codebook reduction to vocab V using k-means (offline), Hugging Face Datasets glue, and `temporaldata` interoperability.

## Quick Start

```python
from timequant.tokenizers.gq_tokenizer import GQTokenizer

# Create a tokenizer with 64 vocabulary tokens
tok = GQTokenizer(V=64)

# Stream processing
for x_t in stream:  # scalar values
    tok.update(x_t)     # Update running statistics
    token_id = tok.encode(x_t)  # Get token for current value
    print(f"Value: {x_t:.3f} -> Token: {token_id}")
```

### Multivariate Example

```python
from timequant.tokenizers.multi_gq_tokenizer import MultiGQTokenizer
import numpy as np

# 4-dimensional data with 2 bins per dimension, final vocab of 256
mtok = MultiGQTokenizer(D=4, V_dim=2, V=256)

# Collect some data to train the codebook
codes = []
for x_vec in training_data:  # shape (4,) vectors
    mtok.update(x_vec)
    code = mtok.encode_vector_code(x_vec)
    codes.append(code)

# Fit the codebook
mtok.fit_codebook(np.array(codes))

# Now use for streaming tokenization
for x_vec in stream:
    mtok.update(x_vec)
    final_token = mtok.encode(x_vec)  # Single token ID
```

## Features

- **Streaming**: Online statistics with Welford's method, O(1) memory per dimension
- **Composable**: Univariate → multivariate → codebook reduction
- **Lightweight**: Core dependencies: NumPy, SciPy, scikit-learn, h5py
- **Fast**: Binary search tokenization, vectorized operations where possible
- **Integrations**: Hugging Face Datasets, temporaldata adapters

## Installation

```bash
# Basic installation
pip install timequant-tokenizer

# With Hugging Face integration
pip install timequant-tokenizer[hf]

# Development installation
pip install -e .[dev]
```

## How it Works

TimeQuant uses **Gaussian quantile binning** for streaming tokenization:

1. **Online Statistics**: Maintains running mean/variance using Welford's algorithm
2. **Standardization**: Normalizes each sample to z-score: `z = (x - μ) / σ`
3. **Quantile Binning**: Maps z-scores to tokens using precomputed N(0,1) quantile boundaries
4. **Multivariate Extension**: Per-dimension tokenization + k-means codebook reduction

This approach is similar to SAX (Symbolic Aggregate approXimation) but:
- Fully streaming (no windowing required)
- Per-timestep tokenization
- Clean extension to multivariate data via vector quantization

## Repository Structure

```
timequant-tokenizer/
├── src/timequant/
│   ├── utils/running_stats.py          # Welford's online statistics  
│   ├── tokenizers/
│   │   ├── quantile_binner.py          # Gaussian quantile boundaries
│   │   ├── gq_tokenizer.py             # 1D streaming tokenizer
│   │   └── multi_gq_tokenizer.py       # Multivariate + codebook
│   ├── quantize/kmeans.py              # Lightweight k-means
│   ├── io/hdf5.py                      # State persistence
│   └── integrations/
│       ├── huggingface.py              # HF Datasets helpers
│       └── temporaldata_adapter.py     # temporaldata integration
├── tests/                              # Unit tests
├── examples/hf_pipeline.py             # HF Datasets example
└── notebooks/00_quickstart.ipynb       # Interactive demo
```

## Benchmarks & Baselines

We compare against:
- **SAX**: Symbolic Aggregate approXimation 
- **SFA**: Symbolic Fourier Approximation

Metrics:
- Token utilization (unique tokens / vocab size)
- Perplexity on token streams
- Downstream task performance (forecasting, anomaly detection)
- Throughput (tokens/second)

Target performance: ≥5M tokens/min (univariate, CPU)

## Roadmap

**v0.1** (current):
- [x] Univariate streaming tokenizer
- [x] Multivariate with codebook reduction  
- [x] HF Datasets integration
- [x] temporaldata adapters
- [ ] Comprehensive benchmarks

**Future versions**:
- Online whitening (full covariance)
- Gaussian copula transforms
- Residual/delta coding
- Adaptive codebooks
- Task-specific token transformers

## Contributing

This project aims to contribute a tokenization module to [temporaldata](https://github.com/neuro-galaxy/temporaldata). See our [contribution guidelines](CONTRIBUTING.md) for details on:

- Code style (ruff, mypy)
- Testing requirements (≥80% coverage)
- Benchmark criteria
- Documentation standards

## License

Apache-2.0

## Citation

If you use this work, please cite:

```bibtex
@misc{timequant2024,
  title={TimeQuant: Streaming Gaussian-Quantile Tokenization for Time Series},
  author={Simeon, Quilee},
  year={2024},
  url={https://github.com/qsimeon/timequant-tokenizer}
}
```