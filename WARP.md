# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

TimeQuant is a lightweight library for streaming tokenization of continuous time series data using Gaussian-quantile binning with optional vector quantization. The core innovation is using Welford's algorithm for online statistics combined with Gaussian quantile binning to create discrete tokens from continuous time series in a streaming fashion.

## Development Commands

### Installation and Setup
```bash
# Development installation with all dependencies
pip install -e .[dev,all]

# Basic installation
pip install -e .

# Install with Hugging Face integration only
pip install -e .[hf]
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/timequant --cov-report=term-missing

# Run specific test files
pytest tests/test_gq_tokenizer.py
pytest tests/test_multi_gq_tokenizer.py

# Run single test
pytest tests/test_gq_tokenizer.py::TestGQTokenizer::test_basic_usage
```

### Code Quality
```bash
# Lint code with ruff
ruff check src/ tests/ examples/

# Format code with ruff
ruff format src/ tests/ examples/

# Type checking with mypy
mypy src/timequant
```

### Development Workflow
```bash
# Run quick checks before committing
ruff check src/ tests/ && ruff format src/ tests/ && pytest

# Run examples to verify functionality
python examples/hf_pipeline.py
```

## Architecture and Design

### Core Components

**Streaming Statistics (`utils/running_stats.py`)**
- `RunningStats`: Implements Welford's algorithm for numerically stable online mean/variance computation
- Supports both scalar and vector inputs with O(1) memory per dimension
- Critical for streaming tokenization without requiring data storage

**Quantile Binning (`tokenizers/quantile_binner.py`)**
- Converts standardized values (z-scores) to discrete tokens using precomputed Gaussian quantiles
- Uses binary search for O(log V) token lookup where V is vocabulary size
- Forms the mathematical foundation for the tokenization approach

**Primary Tokenizers**
- `GQTokenizer`: Univariate streaming tokenizer combining online stats + quantile binning
- `MultiGQTokenizer`: Extends to multivariate data via per-dimension tokenization + k-means vector quantization

### Key Design Patterns

**Streaming-First Architecture**: All components designed for online processing where data arrives one sample at a time. Statistics are updated incrementally without storing historical data.

**Composable Design**: Univariate → Multivariate → Codebook reduction. Each layer builds on the previous while maintaining the streaming paradigm.

**Warmup Handling**: During initial samples (configurable warmup period), uses simplified normalization until sufficient statistics are available.

### Data Flow

1. **Raw Time Series** → `RunningStats.update()` → **Updated Statistics**
2. **Raw Value** → `(x - μ) / σ` → **Z-Score** 
3. **Z-Score** → `QuantileBinner.encode()` → **Token ID**
4. **Multivariate**: Vector of Token IDs → `KMeans.predict()` → **Final Token**

### Integration Points

**Hugging Face (`integrations/huggingface.py`)**
- `DatasetTokenizer`: Wrapper for batch processing HF datasets
- `create_streaming_tokenize_fn`: Factory for dataset.map() functions
- Handles both univariate and multivariate tokenization workflows

**TemporalData (`integrations/temporaldata_adapter.py`)**
- Adapters for neuroimaging time series data
- Designed to contribute tokenization capabilities to the temporaldata ecosystem

**Persistence (`io/hdf5.py`)**
- State serialization for tokenizer statistics and codebooks
- Enables saving/loading trained tokenizers for production use

## Testing Strategy

Tests are organized by component with emphasis on:
- **Streaming consistency**: Verifying streaming and batch modes produce similar results
- **Statistical properties**: Ensuring proper vocabulary utilization and token distribution
- **Edge cases**: Extreme values, empty data, warmup periods
- **Round-trip accuracy**: Encode/decode consistency within expected approximation bounds

## Key Implementation Details

**Numerical Stability**: Z-scores are clipped to [-10, 10] range to prevent issues with extreme outliers affecting quantile boundaries.

**Warmup Period**: During initial `warmup_samples` (default 10), uses unit standard deviation for normalization until sample variance stabilizes.

**Vector Quantization**: Multivariate tokenizer uses two-phase approach: collect per-dimension codes, then fit k-means codebook for final vocabulary reduction.

**Memory Efficiency**: Core tokenizers maintain O(1) memory per dimension regardless of stream length, critical for long time series.

## Development Notes

- Code follows `ruff` formatting with 88-character line length
- Type hints required for all public interfaces  
- Minimum test coverage target: 80%
- All examples in `examples/` directory should remain functional
- Performance target: ≥5M tokens/min for univariate CPU processing