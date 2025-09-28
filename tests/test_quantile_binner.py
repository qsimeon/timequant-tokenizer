"""Tests for quantile binner module."""

import numpy as np
import pytest
from scipy.stats import norm
from timequant.tokenizers.quantile_binner import QuantileBinner


class TestQuantileBinner:
    """Test cases for QuantileBinner class."""

    def test_initialization(self):
        """Test binner initialization."""
        binner = QuantileBinner(vocab_size=4)
        
        assert binner.vocab_size == 4
        assert len(binner.boundaries) == 3  # V-1 boundaries
        
        # Boundaries should be sorted
        assert np.all(binner.boundaries[:-1] < binner.boundaries[1:])

    def test_initialization_errors(self):
        """Test initialization error cases."""
        with pytest.raises(ValueError):
            QuantileBinner(vocab_size=1)  # Must be >= 2
            
        with pytest.raises(ValueError):
            # Wrong number of boundaries
            QuantileBinner(vocab_size=4, boundaries=np.array([0.0, 1.0]))

    def test_custom_boundaries(self):
        """Test initialization with custom boundaries."""
        boundaries = np.array([-1.0, 0.0, 1.0])
        binner = QuantileBinner(vocab_size=4, boundaries=boundaries)
        
        np.testing.assert_array_equal(binner.boundaries, boundaries)

    def test_encode_scalar(self):
        """Test scalar encoding."""
        binner = QuantileBinner(vocab_size=4)
        
        # Test various z-scores
        assert binner.encode_scalar(-10.0) == 0  # Far left
        assert binner.encode_scalar(10.0) == 3   # Far right
        
        # Test around boundaries
        for i, boundary in enumerate(binner.boundaries):
            assert binner.encode_scalar(boundary - 1e-6) == i
            assert binner.encode_scalar(boundary + 1e-6) == i + 1

    def test_encode_batch(self):
        """Test batch encoding."""
        binner = QuantileBinner(vocab_size=4)
        
        z_scores = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        tokens = binner.encode_batch(z_scores)
        
        # Check consistency with scalar encoding
        expected = [binner.encode_scalar(z) for z in z_scores]
        np.testing.assert_array_equal(tokens, expected)

    def test_encode_multidimensional(self):
        """Test encoding with multidimensional input."""
        binner = QuantileBinner(vocab_size=4)
        
        Z = np.random.randn(5, 3)
        tokens = binner.encode_batch(Z)
        
        assert tokens.shape == Z.shape
        
        # Check consistency
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                assert tokens[i, j] == binner.encode_scalar(Z[i, j])

    def test_decode_scalar(self):
        """Test scalar decoding."""
        binner = QuantileBinner(vocab_size=4)
        
        for tok in range(4):
            z = binner.decode_scalar(tok)
            # Re-encoding should give same token
            assert binner.encode_scalar(z) == tok

    def test_decode_batch(self):
        """Test batch decoding."""
        binner = QuantileBinner(vocab_size=4)
        
        tokens = np.array([0, 1, 2, 3, 1, 0])
        z_scores = binner.decode_batch(tokens)
        
        # Check consistency with scalar decoding
        expected = [binner.decode_scalar(tok) for tok in tokens]
        np.testing.assert_array_equal(z_scores, expected)

    def test_decode_errors(self):
        """Test decode error cases."""
        binner = QuantileBinner(vocab_size=4)
        
        with pytest.raises(ValueError):
            binner.decode_scalar(-1)  # Token out of range
            
        with pytest.raises(ValueError):
            binner.decode_scalar(4)   # Token out of range

    def test_round_trip_consistency(self):
        """Test encode-decode round trip."""
        binner = QuantileBinner(vocab_size=8)
        
        # Test with various z-scores
        z_scores = np.linspace(-3, 3, 50)
        
        for z in z_scores:
            tok = binner.encode_scalar(z)
            z_decoded = binner.decode_scalar(tok)
            tok_re_encoded = binner.encode_scalar(z_decoded)
            
            # Token should be consistent
            assert tok == tok_re_encoded

    def test_uniform_distribution_property(self):
        """Test that standard normal samples are roughly uniformly tokenized."""
        np.random.seed(42)
        binner = QuantileBinner(vocab_size=8)
        
        # Generate standard normal samples
        z_samples = np.random.randn(10000)
        tokens = binner.encode_batch(z_samples)
        
        # Count tokens
        token_counts = np.bincount(tokens, minlength=binner.vocab_size)
        token_freqs = token_counts / len(z_samples)
        
        # Should be approximately uniform (1/V each)
        expected_freq = 1.0 / binner.vocab_size
        
        # Allow for some deviation due to finite sampling
        for freq in token_freqs:
            assert abs(freq - expected_freq) < 0.02  # Within 2%

    def test_get_bin_edges(self):
        """Test bin edges retrieval."""
        binner = QuantileBinner(vocab_size=4)
        
        edges = binner.get_bin_edges()
        
        assert len(edges) == 5  # V+1 edges
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        np.testing.assert_array_equal(edges[1:-1], binner.boundaries)

    def test_quantile_spacing(self):
        """Test that boundaries correspond to equal quantiles."""
        binner = QuantileBinner(vocab_size=4)
        
        # Check that boundaries are at the correct quantiles
        expected_quantiles = np.array([0.25, 0.5, 0.75])
        actual_quantiles = norm.cdf(binner.boundaries)
        
        np.testing.assert_allclose(actual_quantiles, expected_quantiles, rtol=1e-10)

    def test_large_vocab_size(self):
        """Test with large vocabulary size."""
        binner = QuantileBinner(vocab_size=256)
        
        assert len(binner.boundaries) == 255
        
        # Test encoding/decoding still works
        z_test = 0.5
        tok = binner.encode_scalar(z_test)
        z_decoded = binner.decode_scalar(tok)
        
        # Should be reasonably close
        assert abs(z_test - z_decoded) < 0.1  # Within reasonable range

    def test_repr(self):
        """Test string representation."""
        binner = QuantileBinner(vocab_size=16)
        repr_str = repr(binner)
        
        assert "QuantileBinner" in repr_str
        assert "16" in repr_str