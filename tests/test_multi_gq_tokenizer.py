"""Tests for multi-GQ tokenizer module."""

import numpy as np
import pytest
from timequant.tokenizers.multi_gq_tokenizer import MultiGQTokenizer


class TestMultiGQTokenizer:
    """Test cases for MultiGQTokenizer class."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tok = MultiGQTokenizer(D=4, V_dim=2, V=256)
        
        assert tok.D == 4
        assert tok.V_dim == 2 
        assert tok.V == 256
        assert len(tok.per_dim) == 4
        assert not tok.is_fitted

    def test_basic_usage(self):
        """Test basic multivariate tokenizer usage."""
        tok = MultiGQTokenizer(D=3, V_dim=2, V=8)
        
        # Generate some multivariate data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        
        # Update statistics
        for x in X:
            tok.update(x)
        
        # Collect codes for fitting codebook
        codes = []
        for x in X:
            code = tok.encode_vector_code(x)
            codes.append(code)
            
        codes = np.array(codes)
        
        # Fit codebook
        tok.fit_codebook(codes)
        
        # Now can encode to final tokens
        tokens = []
        for x in X:
            token = tok.encode(x)
            tokens.append(token)
        
        tokens = np.array(tokens)
        
        # Check properties
        assert len(tokens) == len(X)
        assert np.all(tokens >= 0)
        assert np.all(tokens < tok.V)
        assert tok.is_fitted

    def test_vector_codes(self):
        """Test vector code generation."""
        tok = MultiGQTokenizer(D=2, V_dim=4, V=16)
        
        # Generate and update with some data
        np.random.seed(42)
        X = np.random.randn(50, 2)
        
        for x in X:
            tok.update(x)
        
        # Test vector codes
        codes = []
        for x in X:
            code = tok.encode_vector_code(x)
            assert code.shape == (2,)
            assert np.all(code >= 0)
            assert np.all(code < 4)  # V_dim
            codes.append(code)
            
        codes = np.array(codes)
        assert codes.shape == (50, 2)

    def test_batch_operations(self):
        """Test batch encoding operations."""
        tok = MultiGQTokenizer(D=2, V_dim=2, V=4)
        
        np.random.seed(42)
        X = np.random.randn(20, 2)
        
        # Update statistics
        tok.update_batch(X)
        
        # Test batch vector code generation
        codes = tok.encode_vector_code_batch(X)
        assert codes.shape == (20, 2)
        assert np.all(codes >= 0)
        assert np.all(codes < 2)  # V_dim
        
        # Fit codebook and test batch encoding
        tok.fit_codebook(codes)
        tokens = tok.encode_batch(X)
        
        assert tokens.shape == (20,)
        assert np.all(tokens >= 0) 
        assert np.all(tokens < 4)  # V

    def test_decode_functionality(self):
        """Test decoding functionality."""
        tok = MultiGQTokenizer(D=2, V_dim=2, V=4)
        
        # Generate data and fit
        np.random.seed(42)
        X = np.random.randn(50, 2)
        tok.update_batch(X)
        
        codes = tok.encode_vector_code_batch(X)
        tok.fit_codebook(codes)
        
        # Test vector code decoding
        for i in range(len(X)):
            code = tok.encode_vector_code(X[i])
            decoded = tok.decode_vector_code(code)
            assert decoded.shape == (2,)
        
        # Test final token decoding
        tokens = tok.encode_batch(X)
        for token in tokens:
            decoded = tok.decode(int(token))
            assert decoded.shape == (2,)

    def test_utilization_metrics(self):
        """Test utilization calculation methods."""
        tok = MultiGQTokenizer(D=2, V_dim=4, V=8)
        
        # Generate diverse data 
        np.random.seed(42)
        X = np.random.randn(200, 2)
        tok.update_batch(X)
        
        codes = tok.encode_vector_code_batch(X)
        
        # Test per-dimension utilization
        per_dim_util = tok.get_per_dim_utilization(codes)
        assert per_dim_util.shape == (2,)
        assert np.all(per_dim_util >= 0)
        assert np.all(per_dim_util <= 1)
        
        # Fit codebook and test final utilization
        tok.fit_codebook(codes)
        tokens = tok.encode_batch(X)
        
        final_util = tok.get_vocab_utilization(tokens)
        assert 0 <= final_util <= 1

    def test_error_conditions(self):
        """Test error handling."""
        tok = MultiGQTokenizer(D=3, V_dim=2, V=8)
        
        # Wrong shape for update
        with pytest.raises(ValueError):
            tok.update(np.array([1.0, 2.0]))  # Wrong size
            
        with pytest.raises(ValueError):
            tok.update_batch(np.random.randn(10, 2))  # Wrong D
            
        # Encoding without fitted codebook
        X = np.random.randn(10, 3)
        tok.update_batch(X)
        
        with pytest.raises(ValueError):
            tok.encode(X[0])  # Codebook not fitted
            
        with pytest.raises(ValueError):
            tok.encode_batch(X)  # Codebook not fitted

    def test_reset_functionality(self):
        """Test reset functionality."""
        tok = MultiGQTokenizer(D=2, V_dim=2, V=4)
        
        # Fit the tokenizer
        X = np.random.randn(20, 2)
        tok.update_batch(X)
        codes = tok.encode_vector_code_batch(X)
        tok.fit_codebook(codes)
        
        assert tok.is_fitted
        
        # Reset
        tok.reset()
        assert not tok.is_fitted
        assert all(not t.is_fitted for t in tok.per_dim)

    def test_codebook_fitting(self):
        """Test codebook fitting with different parameters."""
        tok = MultiGQTokenizer(D=2, V_dim=2, V=4)
        
        # Generate codes
        codes = np.random.randint(0, 2, size=(100, 2))
        
        # Test fitting with different parameters
        tok.fit_codebook(codes, seed=42, max_iter=50, batch_size=32)
        
        assert tok.codebook.is_fitted
        assert tok.codebook.centroids.shape == (4, 2)

    def test_shape_validation(self):
        """Test input shape validation."""
        tok = MultiGQTokenizer(D=3, V_dim=2, V=8)
        
        # Test various invalid shapes
        with pytest.raises(ValueError):
            tok.encode_vector_code(np.array([1.0, 2.0]))  # Wrong D
            
        with pytest.raises(ValueError):
            tok.encode_vector_code_batch(np.random.randn(10, 2))  # Wrong D
            
        with pytest.raises(ValueError):
            tok.fit_codebook(np.random.randn(10, 2))  # Wrong D for codes

    def test_repr(self):
        """Test string representation.""" 
        tok = MultiGQTokenizer(D=4, V_dim=2, V=256)
        repr_str = repr(tok)
        
        assert "MultiGQTokenizer" in repr_str
        assert "D=4" in repr_str
        assert "V_dim=2" in repr_str
        assert "V=256" in repr_str
        assert "fitted=False" in repr_str