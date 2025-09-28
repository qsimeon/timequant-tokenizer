"""Tests for running statistics module."""

import numpy as np
import pytest
from timequant.utils.running_stats import RunningStats


class TestRunningStats:
    """Test cases for RunningStats class."""

    def test_scalar_stats(self):
        """Test running stats with scalar inputs."""
        rs = RunningStats()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        for xi in x:
            rs.update(float(xi))
        
        # Check mean and variance match NumPy
        assert abs(rs.mean - x.mean()) < 1e-9
        assert abs(rs.var - x.var(ddof=1)) < 1e-9
        assert rs.n == len(x)

    def test_vector_stats(self):
        """Test running stats with vector inputs."""
        rs = RunningStats(dim=3)
        X = np.random.randn(100, 3)
        
        for x in X:
            rs.update(x)
        
        # Check against NumPy
        np.testing.assert_allclose(rs.mean, X.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(rs.var, X.var(axis=0, ddof=1), rtol=1e-10)
        assert rs.n == len(X)

    def test_batch_update(self):
        """Test batch update functionality."""
        rs_single = RunningStats()
        rs_batch = RunningStats()
        
        X = np.random.randn(50)
        
        # Single updates
        for x in X:
            rs_single.update(x)
        
        # Batch update
        rs_batch.update_batch(X)
        
        # Should be equivalent
        assert abs(rs_single.mean - rs_batch.mean) < 1e-10
        assert abs(rs_single.var - rs_batch.var) < 1e-10
        assert rs_single.n == rs_batch.n

    def test_normalize(self):
        """Test normalization functionality."""
        rs = RunningStats()
        X = np.random.randn(100)
        
        for x in X:
            rs.update(x)
        
        # Normalize the data
        X_norm = np.array([rs.normalize(x) for x in X])
        
        # Should have approximately zero mean and unit variance
        assert abs(X_norm.mean()) < 1e-10
        assert abs(X_norm.var() - 1.0) < 1e-2  # Some tolerance due to finite sample

    def test_reset(self):
        """Test reset functionality."""
        rs = RunningStats(dim=2)
        X = np.random.randn(10, 2)
        
        for x in X:
            rs.update(x)
        
        assert rs.n > 0
        rs.reset()
        assert rs.n == 0

    def test_edge_cases(self):
        """Test edge cases."""
        rs = RunningStats()
        
        # Single sample
        rs.update(5.0)
        assert rs.mean == 5.0
        assert rs.var == 0.0  # Only one sample
        
        # Two samples
        rs.update(7.0)
        assert rs.mean == 6.0
        assert rs.var == 2.0

    def test_error_handling(self):
        """Test error conditions."""
        rs = RunningStats()
        
        # Should raise error if trying to normalize without data
        with pytest.raises(ValueError):
            rs.normalize(1.0)

    def test_auto_initialization(self):
        """Test automatic initialization from first sample."""
        rs = RunningStats()  # No dim specified
        
        # First update with vector should auto-initialize
        x = np.array([1.0, 2.0, 3.0])
        rs.update(x)
        
        assert rs.mean.shape == (3,)
        assert rs.M2.shape == (3,)
        assert rs.n == 1

    def test_numerical_stability(self):
        """Test numerical stability with large numbers."""
        rs = RunningStats()
        
        # Large offset to test numerical stability
        offset = 1e12
        X = np.random.randn(100) + offset
        
        for x in X:
            rs.update(x)
        
        # Should still compute accurate statistics
        expected_mean = X.mean()
        expected_var = X.var(ddof=1)
        
        assert abs(rs.mean - expected_mean) / abs(expected_mean) < 1e-10
        assert abs(rs.var - expected_var) / expected_var < 1e-10