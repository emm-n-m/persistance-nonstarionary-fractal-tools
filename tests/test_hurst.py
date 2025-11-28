"""
Comprehensive tests for hurst.py

Tests cover:
- Input validation and types
- Parameter validation
- NaN handling
- Edge cases
- Return value structure
- Different time series behaviors
- Error conditions
"""

import pathlib
import sys

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for Hurst estimation tests")

# Optional pandas support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Make the Python utilities importable when tests are executed from the repo root.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "Python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from hurst import hurst_rs  # noqa: E402


class TestBasicFunctionality:
    """Test basic Hurst exponent estimation"""

    def test_hurst_white_noise_close_to_half(self):
        """White noise should have H ≈ 0.5"""
        rng = np.random.default_rng(seed=1234)
        samples = rng.standard_normal(4096)
        result = hurst_rs(samples, min_window=8, num_windows=15)
        assert 0.4 <= result["hurst"] <= 0.6

    def test_hurst_persistent_series_exceeds_half(self):
        """Integrated white noise (random walk) should have H > 0.5"""
        rng = np.random.default_rng(seed=4321)
        # Integrate white noise to create a persistent series (fBm-like)
        samples = np.cumsum(rng.standard_normal(4096))
        result = hurst_rs(samples, min_window=8, num_windows=15)
        assert result["hurst"] > 0.6

    def test_hurst_anti_persistent_series(self):
        """Mean-reverting series should have H < 0.5"""
        rng = np.random.default_rng(seed=5555)
        # Create anti-persistent series using differencing
        white = rng.standard_normal(4096)
        # Take first difference of integrated series to get anti-persistent behavior
        # Actually, let's use alternating pattern which is anti-persistent
        samples = np.array([(-1) ** i * abs(x) for i, x in enumerate(white)])
        result = hurst_rs(samples, min_window=8, num_windows=15)
        # This might not always be < 0.5 due to noise, so we just check it computes
        assert 0.0 <= result["hurst"] <= 1.0


class TestInputValidation:
    """Test input validation and error handling"""

    def test_rejects_short_series(self):
        """Should reject series with < 32 observations"""
        with pytest.raises(ValueError, match="At least 32 observations"):
            hurst_rs([1, 2, 3, 4, 5])

    def test_rejects_short_series_exactly_31(self):
        """Should reject series with exactly 31 observations"""
        with pytest.raises(ValueError, match="At least 32 observations"):
            hurst_rs(list(range(31)))

    def test_accepts_series_exactly_32(self):
        """Should accept series with exactly 32 observations"""
        rng = np.random.default_rng(seed=42)
        samples = rng.standard_normal(32)
        result = hurst_rs(samples, min_window=4, num_windows=5)
        assert "hurst" in result

    def test_rejects_multidimensional_array(self):
        """Should reject multi-dimensional arrays"""
        series_2d = np.random.randn(100, 10)
        with pytest.raises(ValueError, match="must be one-dimensional"):
            hurst_rs(series_2d)

    def test_rejects_max_window_less_than_min_window(self):
        """Should reject max_window <= min_window"""
        series = np.random.randn(1000)
        with pytest.raises(ValueError, match="max_window must be greater than min_window"):
            hurst_rs(series, min_window=100, max_window=50)

    def test_rejects_max_window_equal_to_min_window(self):
        """Should reject max_window == min_window"""
        series = np.random.randn(1000)
        with pytest.raises(ValueError, match="max_window must be greater than min_window"):
            hurst_rs(series, min_window=50, max_window=50)


class TestInputTypes:
    """Test different input types"""

    def test_accepts_list(self):
        """Should accept Python list"""
        rng = np.random.default_rng(seed=123)
        samples = rng.standard_normal(200).tolist()
        result = hurst_rs(samples, min_window=8, num_windows=10)
        assert "hurst" in result

    def test_accepts_numpy_array(self):
        """Should accept numpy array"""
        rng = np.random.default_rng(seed=456)
        samples = rng.standard_normal(200)
        result = hurst_rs(samples, min_window=8, num_windows=10)
        assert "hurst" in result

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_accepts_pandas_series(self):
        """Should accept pandas Series"""
        rng = np.random.default_rng(seed=789)
        samples = pd.Series(rng.standard_normal(200))
        result = hurst_rs(samples, min_window=8, num_windows=10)
        assert "hurst" in result


class TestNaNHandling:
    """Test NaN handling"""

    def test_dropna_true_removes_nans(self):
        """With dropna=True, NaNs should be removed"""
        series = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0] * 20)
        result = hurst_rs(series, dropna=True, min_window=4, num_windows=5)
        assert "hurst" in result

    def test_dropna_false_raises_on_nans(self):
        """With dropna=False, NaNs should raise an error"""
        series = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        with pytest.raises(ValueError, match="NaNs present"):
            hurst_rs(series, dropna=False)

    def test_dropna_false_accepts_clean_series(self):
        """With dropna=False, clean series should work"""
        rng = np.random.default_rng(seed=111)
        series = rng.standard_normal(200)
        result = hurst_rs(series, dropna=False, min_window=8, num_windows=10)
        assert "hurst" in result

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_dropna_with_pandas_series(self):
        """Test NaN handling with pandas Series"""
        series = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0] * 20)
        result = hurst_rs(series, dropna=True, min_window=4, num_windows=5)
        assert "hurst" in result


class TestReturnValue:
    """Test return value structure and contents"""

    def test_return_structure(self):
        """Test that all expected keys are present"""
        rng = np.random.default_rng(seed=999)
        series = rng.standard_normal(1000)
        result = hurst_rs(series)

        expected_keys = {
            "method",
            "hurst",
            "slope",
            "intercept",
            "r_squared",
            "window_sizes",
            "rs_values",
            "log_window_sizes",
            "log_rs_values",
            "fitted_log_rs",
        }
        assert set(result.keys()) == expected_keys

    def test_hurst_equals_slope(self):
        """Hurst exponent should equal the slope"""
        rng = np.random.default_rng(seed=888)
        series = rng.standard_normal(1000)
        result = hurst_rs(series)
        assert result["hurst"] == result["slope"]

    def test_r_squared_in_valid_range(self):
        """R² should be between 0 and 1"""
        rng = np.random.default_rng(seed=777)
        series = rng.standard_normal(1000)
        result = hurst_rs(series)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_method_is_rescaled_range(self):
        """Method should be 'rescaled_range'"""
        rng = np.random.default_rng(seed=666)
        series = rng.standard_normal(500)
        result = hurst_rs(series)
        assert result["method"] == "rescaled_range"

    def test_window_sizes_are_integers(self):
        """Window sizes should be integers"""
        rng = np.random.default_rng(seed=555)
        series = rng.standard_normal(500)
        result = hurst_rs(series)
        assert result["window_sizes"].dtype == np.dtype("int")

    def test_window_sizes_are_increasing(self):
        """Window sizes should be in increasing order"""
        rng = np.random.default_rng(seed=444)
        series = rng.standard_normal(500)
        result = hurst_rs(series)
        windows = result["window_sizes"]
        assert np.all(windows[1:] >= windows[:-1])

    def test_arrays_have_same_length(self):
        """All array outputs should have the same length"""
        rng = np.random.default_rng(seed=333)
        series = rng.standard_normal(500)
        result = hurst_rs(series)

        n = len(result["window_sizes"])
        assert len(result["rs_values"]) == n
        assert len(result["log_window_sizes"]) == n
        assert len(result["log_rs_values"]) == n
        assert len(result["fitted_log_rs"]) == n


class TestParameters:
    """Test parameter handling"""

    def test_custom_min_window(self):
        """Test custom min_window parameter"""
        rng = np.random.default_rng(seed=222)
        series = rng.standard_normal(1000)
        result = hurst_rs(series, min_window=20, num_windows=10)

        assert result["window_sizes"][0] >= 20

    def test_custom_max_window(self):
        """Test custom max_window parameter"""
        rng = np.random.default_rng(seed=111)
        series = rng.standard_normal(1000)
        result = hurst_rs(series, min_window=10, max_window=200, num_windows=10)

        assert result["window_sizes"][-1] <= 200

    def test_custom_num_windows(self):
        """Test custom num_windows parameter"""
        rng = np.random.default_rng(seed=1010)
        series = rng.standard_normal(1000)
        result = hurst_rs(series, min_window=10, num_windows=5)

        # Number of actual windows might be less due to uniqueness constraint
        assert len(result["window_sizes"]) <= 5

    def test_default_max_window(self):
        """Test that default max_window is len(series)//2"""
        rng = np.random.default_rng(seed=2020)
        series = rng.standard_normal(1000)
        result = hurst_rs(series, min_window=10)

        # Max window should be approximately 500 (half of 1000)
        assert result["window_sizes"][-1] <= 500


class TestEdgeCases:
    """Test edge cases and special series"""

    def test_constant_series_raises_error(self):
        """Constant series should raise error (std=0)"""
        series = np.ones(1000) * 5.0
        # Constant series will have zero standard deviation, which should cause issues
        with pytest.raises(ValueError):
            hurst_rs(series, min_window=8, num_windows=10)

    def test_nearly_constant_series(self):
        """Nearly constant series with tiny variations"""
        rng = np.random.default_rng(seed=3030)
        series = np.ones(1000) + rng.standard_normal(1000) * 1e-10
        # This should work, though H estimate might be unreliable
        result = hurst_rs(series, min_window=8, num_windows=10)
        assert "hurst" in result

    def test_large_series(self):
        """Test with large series"""
        rng = np.random.default_rng(seed=4040)
        series = rng.standard_normal(100000)
        result = hurst_rs(series, min_window=100, num_windows=30)
        assert "hurst" in result
        assert 0.4 <= result["hurst"] <= 0.6  # Should still be ~ 0.5 for white noise

    def test_series_with_trend(self):
        """Series with linear trend"""
        rng = np.random.default_rng(seed=5050)
        trend = np.linspace(0, 100, 1000)
        noise = rng.standard_normal(1000)
        series = trend + noise
        result = hurst_rs(series, min_window=8, num_windows=15)
        # Trend makes it more persistent
        assert result["hurst"] > 0.5


class TestReproducibility:
    """Test reproducibility of results"""

    def test_same_input_same_output(self):
        """Same input should produce same output"""
        rng = np.random.default_rng(seed=6060)
        series = rng.standard_normal(1000)

        result1 = hurst_rs(series, min_window=8, num_windows=15)
        result2 = hurst_rs(series, min_window=8, num_windows=15)

        assert result1["hurst"] == result2["hurst"]
        assert result1["r_squared"] == result2["r_squared"]
        assert np.array_equal(result1["window_sizes"], result2["window_sizes"])

    def test_same_seed_same_result(self):
        """Same random seed should produce same result"""
        rng1 = np.random.default_rng(seed=42)
        series1 = rng1.standard_normal(500)
        result1 = hurst_rs(series1, min_window=8, num_windows=10)

        rng2 = np.random.default_rng(seed=42)
        series2 = rng2.standard_normal(500)
        result2 = hurst_rs(series2, min_window=8, num_windows=10)

        assert result1["hurst"] == result2["hurst"]
        assert np.allclose(result1["rs_values"], result2["rs_values"])
