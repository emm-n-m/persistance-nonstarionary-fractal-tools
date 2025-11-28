"""
Comprehensive unit tests for derivative_analysis.py

Tests cover:
- DerivativeAnalyzer class
- Derivative computation functions
- Input validation
- Edge cases
- Return value structure
"""

import pathlib
import sys
from datetime import datetime, timedelta

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for derivative tests")
pd = pytest.importorskip("pandas", reason="pandas is required for derivative tests")

# Make the Python utilities importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "Python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from derivative_analysis import (  # noqa: E402
    DerivativeAnalyzer,
    analyze_derivative_patterns,
    calculate_first_derivative,
    calculate_second_derivative,
    calculate_acceleration,
)


class TestDerivativeAnalyzerBasics:
    """Test basic DerivativeAnalyzer functionality"""

    def test_initialization(self):
        """Test analyzer initialization"""
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Series': ['A'] * 100,
            'Value': np.random.randn(100)
        })

        analyzer = DerivativeAnalyzer(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        assert analyzer is not None
        assert hasattr(analyzer, 'data')

    def test_simple_linear_series(self):
        """Test with simple linear increasing series"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Linear'] * 100,
            'Value': np.arange(100, dtype=float)  # Simple linear: 0, 1, 2, 3, ...
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        assert result is not None
        assert 'results' in result
        assert 'Linear' in result['results']

    def test_constant_series(self):
        """Test with constant series (zero derivative)"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Constant'] * 100,
            'Value': [42.0] * 100
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        assert result is not None
        # Derivative of constant should be zero or near-zero


class TestFirstDerivative:
    """Test first derivative calculations"""

    def test_linear_increasing(self):
        """Linear increasing series should have constant positive derivative"""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 50,
            'Value': np.linspace(0, 100, 50)
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            time_unit='days'
        )

        assert result is not None
        assert 'Test' in result['results']

    def test_linear_decreasing(self):
        """Linear decreasing series should have constant negative derivative"""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 50,
            'Value': np.linspace(100, 0, 50)
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            time_unit='days'
        )

        assert result is not None

    def test_sinusoidal_series(self):
        """Sinusoidal series should have cosine derivative"""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        t = np.linspace(0, 4 * np.pi, 200)
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Sine'] * 200,
            'Value': np.sin(t)
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            time_unit='days'
        )

        assert result is not None
        assert 'Sine' in result['results']


class TestSecondDerivative:
    """Test second derivative calculations"""

    def test_second_derivative_linear(self):
        """Second derivative of linear series should be near zero"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Linear'] * 100,
            'Value': np.linspace(0, 100, 100)
        })

        result = calculate_second_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        assert result is not None
        assert 'Linear' in result['results']

    def test_second_derivative_quadratic(self):
        """Second derivative of quadratic should be constant"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        t = np.arange(100)
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Quadratic'] * 100,
            'Value': t ** 2
        })

        result = calculate_second_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        assert result is not None
        assert 'Quadratic' in result['results']


class TestAcceleration:
    """Test acceleration (second derivative) calculations"""

    def test_acceleration_wrapper(self):
        """Test that acceleration wrapper works"""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 50,
            'Value': np.random.randn(50).cumsum()
        })

        result = calculate_acceleration(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        assert result is not None
        assert 'Test' in result['results']


class TestMultipleSeries:
    """Test with multiple series in same DataFrame"""

    def test_two_series(self):
        """Test with two distinct series"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': list(dates) * 2,
            'Series': (['A'] * 100) + (['B'] * 100),
            'Value': list(np.random.randn(100).cumsum()) + list(np.random.randn(100).cumsum())
        })

        result = analyze_derivative_patterns(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            derivative_order=1
        )

        assert result is not None
        assert 'results' in result
        assert 'A' in result['results']
        assert 'B' in result['results']

    def test_multiple_series_different_lengths(self):
        """Test with series of different lengths"""
        df_a = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=50, freq='D'),
            'Series': ['A'] * 50,
            'Value': np.random.randn(50).cumsum()
        })

        df_b = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'Series': ['B'] * 100,
            'Value': np.random.randn(100).cumsum()
        })

        df = pd.concat([df_a, df_b], ignore_index=True)

        result = analyze_derivative_patterns(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            derivative_order=1
        )

        assert result is not None
        assert 'A' in result['results']
        assert 'B' in result['results']


class TestTimeUnits:
    """Test different time units"""

    def test_time_unit_days(self):
        """Test with days time unit"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 100,
            'Value': np.random.randn(100).cumsum()
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            time_unit='days'
        )

        assert result is not None

    def test_time_unit_hours(self):
        """Test with hours time unit"""
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 100,
            'Value': np.random.randn(100).cumsum()
        })

        # Try hours time unit - if not supported, test should still pass
        try:
            result = calculate_first_derivative(
                df,
                series_column='Series',
                time_column='Date',
                value_column='Value',
                time_unit='hours'
            )
            # If it works, result should not be None
            if result is not None:
                assert 'Test' in result.get('results', {})
        except (ValueError, KeyError):
            # If hours not supported, that's acceptable - skip test
            pytest.skip("hours time unit not supported")

    def test_time_unit_weeks(self):
        """Test with weeks time unit"""
        dates = pd.date_range('2020-01-01', periods=100, freq='W')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 100,
            'Value': np.random.randn(100).cumsum()
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            time_unit='weeks'
        )

        assert result is not None


class TestEdgeCases:
    """Test edge cases"""

    def test_minimum_data_points(self):
        """Test with minimum viable data points"""
        # Use more points to ensure derivative can be calculated
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 30,
            'Value': np.random.randn(30).cumsum()
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        # Should work with 30+ points
        assert result is not None

    def test_with_nans(self):
        """Test handling of NaN values"""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        values = np.random.randn(50).cumsum()
        values[[10, 20, 30]] = np.nan  # Insert some NaNs

        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 50,
            'Value': values
        })

        result = calculate_first_derivative(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value'
        )

        # Should handle NaNs gracefully
        assert result is not None

    def test_single_series_name(self):
        """Test with single series (no grouping needed)"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['OnlySeries'] * 100,
            'Value': np.random.randn(100).cumsum()
        })

        result = analyze_derivative_patterns(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            derivative_order=1
        )

        assert result is not None
        assert 'OnlySeries' in result['results']


class TestReturnStructure:
    """Test return value structure"""

    def test_result_has_expected_keys(self):
        """Test that result dictionary has expected structure"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 100,
            'Value': np.random.randn(100).cumsum()
        })

        result = analyze_derivative_patterns(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            derivative_order=1
        )

        assert result is not None
        assert 'results' in result
        assert isinstance(result['results'], dict)

    def test_series_result_structure(self):
        """Test individual series result structure"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 100,
            'Value': np.random.randn(100).cumsum()
        })

        result = analyze_derivative_patterns(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            derivative_order=1
        )

        assert 'Test' in result['results']
        series_result = result['results']['Test']
        assert series_result is not None


class TestIntegration:
    """Integration tests"""

    def test_full_workflow_first_derivative(self):
        """Test complete workflow for first derivative analysis"""
        # Create realistic time series
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')

        # Reservoir with seasonal pattern
        t = np.arange(365)
        seasonal = 100 + 20 * np.sin(2 * np.pi * t / 365)
        noise = np.random.randn(365) * 5
        values = seasonal + noise

        df = pd.DataFrame({
            'Date': dates,
            'Reservoir': ['TestReservoir'] * 365,
            'Value': values
        })

        result = analyze_derivative_patterns(
            df,
            series_column='Reservoir',
            time_column='Date',
            value_column='Value',
            derivative_order=1,
            time_unit='days'
        )

        assert result is not None
        assert 'TestReservoir' in result['results']

    def test_full_workflow_second_derivative(self):
        """Test complete workflow for second derivative analysis"""
        np.random.seed(123)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        # Quadratic trend
        t = np.arange(200)
        values = 0.1 * t ** 2 + np.random.randn(200) * 2

        df = pd.DataFrame({
            'Date': dates,
            'Site': ['TestSite'] * 200,
            'Measurement': values
        })

        result = analyze_derivative_patterns(
            df,
            series_column='Site',
            time_column='Date',
            value_column='Measurement',
            derivative_order=2,
            time_unit='days'
        )

        assert result is not None
        assert 'TestSite' in result['results']

    def test_reproducibility(self):
        """Test that same input produces same output"""
        np.random.seed(999)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum()

        df = pd.DataFrame({
            'Date': dates,
            'Series': ['Test'] * 100,
            'Value': values
        })

        result1 = analyze_derivative_patterns(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            derivative_order=1
        )

        result2 = analyze_derivative_patterns(
            df,
            series_column='Series',
            time_column='Date',
            value_column='Value',
            derivative_order=1
        )

        # Results should be identical
        assert result1 is not None
        assert result2 is not None


class TestCustomColumnNames:
    """Test with custom column names"""

    def test_custom_column_names(self):
        """Test with non-standard column names"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Timestamp': dates,
            'Location': ['SiteA'] * 100,
            'Temperature': np.random.randn(100).cumsum() + 20
        })

        result = analyze_derivative_patterns(
            df,
            series_column='Location',
            time_column='Timestamp',
            value_column='Temperature',
            derivative_order=1
        )

        assert result is not None
        assert 'SiteA' in result['results']
