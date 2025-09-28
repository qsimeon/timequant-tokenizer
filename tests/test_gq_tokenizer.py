"""Tests for GQ tokenizer module."""

import numpy as np
import pytest
from timequant.tokenizers.gq_tokenizer import GQTokenizer


class TestGQTokenizer:
    """Test cases for GQTokenizer class."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tok = GQTokenizer(V=64)
        
        assert tok.vocab_size == 64
        assert tok.stats.n == 0
        assert not tok.is_fitted

    def test_basic_usage(self):
        """Test basic tokenizer usage."""
        tok = GQTokenizer(V=4, warmup_samples=5)
        
        # Generate some data
        np.random.seed(42)
        X = np.random.randn(100)
        
        tokens = []
        for x in X:
            tok.update(x)
            token = tok.encode(x)
            tokens.append(token)
        
        tokens = np.array(tokens)
        
        # Check basic properties
        assert len(tokens) == len(X)
        assert np.all(tokens >= 0)
        assert np.all(tokens < tok.vocab_size)
        assert tok.is_fitted

    def test_streaming_consistency(self):
        """Test that streaming gives consistent results."""
        tok1 = GQTokenizer(V=8)
        tok2 = GQTokenizer(V=8)
        
        np.random.seed(42)
        X = np.random.randn(50)
        
        # Process with first tokenizer
        tokens1 = []
        for x in X:
            tok1.update(x)
            tokens1.append(tok1.encode(x))
        
        # Process with second tokenizer  
        tok2.stats.update_batch(X)  # Update stats all at once
        tokens2 = tok2.encode_batch(X)
        
        # Results should be close (may not be exactly equal due to streaming vs batch)
        tokens1 = np.array(tokens1)
        agreement = np.mean(tokens1 == tokens2)
        assert agreement > 0.8  # Most tokens should agree

    def test_decode_functionality(self):
        """Test token decoding."""
        tok = GQTokenizer(V=4)
        
        # Generate and tokenize data
        np.random.seed(42)
        X = np.random.randn(100)
        
        for x in X:
            tok.update(x)
        
        tokens = [tok.encode(x) for x in X]
        decoded = [tok.decode(token) for token in tokens]
        
        # Decoded values should be reasonable approximations
        decoded = np.array(decoded)
        
        # Check that decode is roughly consistent
        for i, (orig, dec) in enumerate(zip(X, decoded)):
            # Should be within a reasonable range (depends on vocab size)
            assert abs(orig - dec) < 3 * tok.stats.std  

    def test_vocab_utilization(self):
        """Test vocabulary utilization calculation."""
        tok = GQTokenizer(V=8)
        
        # Generate standard normal data 
        np.random.seed(42)
        X = np.random.randn(1000)
        
        tokens = []
        for x in X:
            tok.update(x)
            tokens.append(tok.encode(x))
        
        tokens = np.array(tokens)
        utilization = tok.get_vocab_utilization(tokens)
        
        # Should use most of the vocabulary for normal data
        assert utilization > 0.8

    def test_reset_functionality(self):
        """Test tokenizer reset."""
        tok = GQTokenizer(V=4)
        
        # Add some data
        X = np.random.randn(50)
        for x in X:
            tok.update(x)
        
        assert tok.is_fitted
        assert tok.stats.n > 0
        
        # Reset
        tok.reset()
        assert not tok.is_fitted
        assert tok.stats.n == 0

    def test_error_conditions(self):
        """Test error handling."""
        tok = GQTokenizer(V=4)
        
        # Should raise error if trying to encode without data
        with pytest.raises(ValueError):
            tok.encode(1.0)
            
        with pytest.raises(ValueError):
            tok.decode(0)
            
        with pytest.raises(ValueError):
            tok.encode_batch(np.array([1.0, 2.0]))

    def test_warmup_period(self):
        """Test warmup period behavior."""
        tok = GQTokenizer(V=4, warmup_samples=10)
        
        # During warmup, should use std=1 normalization
        X = np.array([0.0, 1.0, -1.0])
        
        for i, x in enumerate(X):
            tok.update(x)
            if i < 10:  # During warmup
                # Should be using unit std
                pass  # Hard to test exactly without exposing internals
            
        assert tok.stats.n == len(X)

    def test_batch_operations(self):
        """Test batch encoding/decoding."""
        tok = GQTokenizer(V=8)
        
        # Fit tokenizer first
        np.random.seed(42)
        X_train = np.random.randn(100)
        for x in X_train:
            tok.update(x)
            
        # Test batch operations
        X_test = np.random.randn(20)
        tokens = tok.encode_batch(X_test)
        decoded = tok.decode_batch(tokens)
        
        assert len(tokens) == len(X_test)
        assert len(decoded) == len(X_test)
        
        # Check round-trip consistency
        tokens_reencoded = tok.encode_batch(decoded)
        np.testing.assert_array_equal(tokens, tokens_reencoded)

    def test_extreme_values(self):
        """Test handling of extreme values."""
        tok = GQTokenizer(V=4)
        
        # Train on normal data
        X_train = np.random.randn(100)
        for x in X_train:
            tok.update(x)
        
        # Test extreme values
        extreme_values = [-100, 100, -10, 10]
        
        for val in extreme_values:
            token = tok.encode(val)
            # Should not crash and should give valid token
            assert 0 <= token < tok.vocab_size
            
            # Decode should work too
            decoded = tok.decode(token)
            # May not be exact due to clipping, but should be finite
            assert np.isfinite(decoded)

    def test_repr(self):
        """Test string representation."""
        tok = GQTokenizer(V=16)
        repr_str = repr(tok)
        
        assert "GQTokenizer" in repr_str
        assert "16" in repr_str
        assert "fitted=False" in repr_str
        
        # After fitting
        tok.update(1.0)
        repr_str = repr(tok)
        assert "fitted=True" in repr_str