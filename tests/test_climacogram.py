"""
Comprehensive tests for climacogram.py

Tests cover:
- Input validation
- Computation correctness
- Edge cases
- Error handling
- File I/O operations
"""

import pathlib
import sys
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for climacogram tests")
pd = pytest.importorskip("pandas", reason="pandas is required for climacogram tests")
plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib is required for climacogram tests")

# Make the Python utilities importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "Python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from climacogram import (  # noqa: E402
    compute_climacogram,
    plot_climacogram,
    analyze_reservoir_climacogram,
)


class TestComputeClimacogram:
    """Tests for compute_climacogram function"""

    def test_basic_computation(self):
        """Test basic climacogram computation"""
        rng = np.random.default_rng(seed=42)
        series = rng.standard_normal(1000)
        max_scale = 50

        scales, variances = compute_climacogram(series, max_scale=max_scale)

        assert len(scales) == max_scale
        assert len(variances) == max_scale
        assert isinstance(scales, np.ndarray)
        assert isinstance(variances, np.ndarray)

    def test_scales_are_sequential(self):
        """Test that scales are sequential integers"""
        series = np.random.randn(1000)
        scales, _ = compute_climacogram(series, max_scale=20, min_scale=1)

        assert np.array_equal(scales, np.arange(1, 21))

    def test_min_scale_parameter(self):
        """Test that min_scale parameter works correctly"""
        series = np.random.randn(1000)
        min_scale = 5
        max_scale = 15

        scales, variances = compute_climacogram(series, max_scale=max_scale, min_scale=min_scale)

        assert len(scales) == max_scale - min_scale + 1
        assert scales[0] == min_scale
        assert scales[-1] == max_scale

    def test_variance_decreases_with_scale_white_noise(self):
        """For white noise, variance should decrease roughly as 1/scale"""
        rng = np.random.default_rng(seed=123)
        series = rng.standard_normal(10000)  # Large series for stable estimates
        max_scale = 100

        scales, variances = compute_climacogram(series, max_scale=max_scale)

        # Remove NaN values
        valid_mask = ~np.isnan(variances)
        scales_valid = scales[valid_mask]
        variances_valid = variances[valid_mask]

        # For white noise, log(variance) vs log(scale) should have slope ≈ -1
        # Use only the first half of scales for more stable fit
        n_points = len(scales_valid) // 2
        log_scales = np.log(scales_valid[:n_points])
        log_variances = np.log(variances_valid[:n_points])

        # Linear regression
        slope = np.polyfit(log_scales, log_variances, 1)[0]

        # Slope should be close to -1 for white noise (within tolerance)
        assert -1.3 < slope < -0.7, f"Expected slope ≈ -1 for white noise, got {slope}"

    def test_constant_series_zero_variance(self):
        """Constant series should produce zero or near-zero variance"""
        series = np.ones(1000) * 5.0
        scales, variances = compute_climacogram(series, max_scale=50)

        # Remove NaN values
        valid_variances = variances[~np.isnan(variances)]

        # All valid variances should be very close to zero
        assert np.all(valid_variances < 1e-10)

    def test_rejects_multidimensional_input(self):
        """Should raise ValueError for multi-dimensional input"""
        series_2d = np.random.randn(100, 10)

        with pytest.raises(ValueError, match="must be one-dimensional"):
            compute_climacogram(series_2d, max_scale=20)

    def test_rejects_max_scale_too_large(self):
        """Should raise ValueError if max_scale > len(series)//2"""
        series = np.random.randn(100)

        with pytest.raises(ValueError, match="max_scale.*too large"):
            compute_climacogram(series, max_scale=60)

    def test_handles_large_scales(self):
        """Test that function handles scales approaching max series length"""
        series = np.random.randn(100)
        max_scale = 48  # Close to half the series length

        scales, variances = compute_climacogram(series, max_scale=max_scale)

        # Should complete without errors
        assert len(scales) == max_scale
        assert len(variances) == max_scale
        # All variances should be non-negative or NaN
        valid_variances = variances[~np.isnan(variances)]
        assert np.all(valid_variances >= 0)

    def test_deterministic_output(self):
        """Same input should produce same output"""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 100)
        max_scale = 30

        scales1, variances1 = compute_climacogram(series, max_scale=max_scale)
        scales2, variances2 = compute_climacogram(series, max_scale=max_scale)

        assert np.array_equal(scales1, scales2)
        assert np.allclose(variances1, variances2, equal_nan=True)

    def test_small_series(self):
        """Test with minimum viable series length"""
        series = np.random.randn(20)
        max_scale = 5

        scales, variances = compute_climacogram(series, max_scale=max_scale)

        assert len(scales) == max_scale
        assert len(variances) == max_scale
        # At least the first few scales should have valid variances
        assert not np.isnan(variances[0])
        assert not np.isnan(variances[1])


class TestPlotClimacogram:
    """Tests for plot_climacogram function"""

    def test_plot_without_saving(self):
        """Test plotting without saving to file"""
        scales = np.arange(1, 50)
        variances = 1.0 / scales  # Simple power law

        # Should not raise any errors
        plot_climacogram(scales, variances, title="Test Plot", show=False)

    def test_plot_saves_to_file(self):
        """Test that plot is saved to specified path"""
        scales = np.arange(1, 50)
        variances = 1.0 / scales

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_climacogram.png"

            plot_climacogram(scales, variances, output_path=output_path, show=False)

            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File is not empty

    def test_plot_creates_parent_directories(self):
        """Test that plot_climacogram creates parent directories if needed"""
        scales = np.arange(1, 50)
        variances = 1.0 / scales

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir1" / "subdir2" / "test_climacogram.png"

            plot_climacogram(scales, variances, output_path=output_path, show=False)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_plot_custom_title(self):
        """Test that custom title is used"""
        scales = np.arange(1, 50)
        variances = 1.0 / scales
        custom_title = "My Custom Climacogram Title"

        # This should not raise errors
        plot_climacogram(scales, variances, title=custom_title, show=False)


class TestAnalyzeReservoirClimacogram:
    """Tests for analyze_reservoir_climacogram function"""

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent file"""
        fake_path = Path("/nonexistent/path/to/data.csv")

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            analyze_reservoir_climacogram(
                fake_path, reservoir_name="Test", output_path=None
            )

    def test_missing_reservoir_column_error(self):
        """Test error when reservoir column doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV without expected column
            test_data = pd.DataFrame({"WrongColumn": [1, 2, 3], "Value": [10, 20, 30]})
            data_path = Path(tmpdir) / "test_data.csv"
            test_data.to_csv(data_path, sep="\t", index=False)

            with pytest.raises(ValueError, match="Column 'Reservoir' not found"):
                analyze_reservoir_climacogram(
                    data_path, reservoir_name="Test", separator="\t"
                )

    def test_reservoir_not_found_error(self):
        """Test error when requested reservoir doesn't exist in data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV with wrong reservoir names
            test_data = pd.DataFrame({
                "Reservoir": ["ResA", "ResA", "ResB", "ResB"],
                "Value": [10, 20, 30, 40],
            })
            data_path = Path(tmpdir) / "test_data.csv"
            test_data.to_csv(data_path, sep="\t", index=False)

            with pytest.raises(ValueError, match="Reservoir 'NonExistent' not found"):
                analyze_reservoir_climacogram(
                    data_path, reservoir_name="NonExistent", separator="\t"
                )

    def test_successful_analysis(self):
        """Test successful climacogram analysis from file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create realistic test data
            rng = np.random.default_rng(seed=42)
            n_points = 200
            test_data = pd.DataFrame({
                "Reservoir": ["TestRes"] * n_points,
                "Value": rng.standard_normal(n_points),
                "Date": pd.date_range("2020-01-01", periods=n_points),
            })
            data_path = Path(tmpdir) / "test_data.csv"
            test_data.to_csv(data_path, sep="\t", index=False)

            output_path = Path(tmpdir) / "output.png"

            scales, variances = analyze_reservoir_climacogram(
                data_path,
                reservoir_name="TestRes",
                max_scale=30,
                separator="\t",
                output_path=output_path,
            )

            # Check outputs
            assert len(scales) == 30
            assert len(variances) == 30
            assert isinstance(scales, np.ndarray)
            assert isinstance(variances, np.ndarray)

            # Check that plot was saved
            assert output_path.exists()

    def test_custom_column_names(self):
        """Test with custom column names"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data with custom column names
            rng = np.random.default_rng(seed=123)
            test_data = pd.DataFrame({
                "Site": ["SiteA"] * 100,
                "Measurement": rng.standard_normal(100),
            })
            data_path = Path(tmpdir) / "test_data.csv"
            test_data.to_csv(data_path, sep=",", index=False)

            scales, variances = analyze_reservoir_climacogram(
                data_path,
                reservoir_name="SiteA",
                value_column="Measurement",
                reservoir_column="Site",
                max_scale=20,
                separator=",",
                output_path=None,
            )

            assert len(scales) == 20
            assert len(variances) == 20

    def test_filters_correct_reservoir(self):
        """Test that only the specified reservoir's data is used"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with multiple reservoirs, each with distinct values
            test_data = pd.DataFrame({
                "Reservoir": (["ResA"] * 100) + (["ResB"] * 100),
                "Value": [1.0] * 100 + [10.0] * 100,  # ResA: 1.0, ResB: 10.0
            })
            data_path = Path(tmpdir) / "test_data.csv"
            test_data.to_csv(data_path, sep="\t", index=False)

            scales_a, variances_a = analyze_reservoir_climacogram(
                data_path,
                reservoir_name="ResA",
                max_scale=20,
                separator="\t",
                output_path=None,
            )

            # ResA has constant values, so variance should be ~0
            valid_variances_a = variances_a[~np.isnan(variances_a)]
            assert np.all(valid_variances_a < 1e-10)


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_end_to_end_workflow(self):
        """Test complete workflow: generate data -> compute -> plot"""
        rng = np.random.default_rng(seed=999)
        series = rng.standard_normal(1000)

        # Compute climacogram
        scales, variances = compute_climacogram(series, max_scale=50)

        # Verify computation
        assert len(scales) == 50
        assert len(variances) == 50

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "workflow_test.png"

            # Plot
            plot_climacogram(
                scales, variances, title="Integration Test", output_path=output_path, show=False
            )

            # Verify plot was saved
            assert output_path.exists()
            assert output_path.stat().st_size > 10000  # Reasonable size for PNG

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        rng1 = np.random.default_rng(seed=42)
        series1 = rng1.standard_normal(500)
        scales1, variances1 = compute_climacogram(series1, max_scale=30)

        rng2 = np.random.default_rng(seed=42)
        series2 = rng2.standard_normal(500)
        scales2, variances2 = compute_climacogram(series2, max_scale=30)

        assert np.array_equal(scales1, scales2)
        assert np.allclose(variances1, variances2, equal_nan=True)
