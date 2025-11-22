# Python Package Installation Guide

## Quick Install

```bash
# Install from source (in project directory)
pip install -e .

# Or install from GitHub
pip install git+https://github.com/emm-n-m/persistance-nonstarionary-fractal-tools.git
```

## Installation Options

### Option 1: Editable Install (For Development)

```bash
# Clone or navigate to the repository
cd persistance-nonstarionary-fractal-tools

# Install in editable mode with core dependencies
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Option 2: Standard Install

```bash
# Install as regular package
pip install .

# Or with specific extras
pip install ".[timeseries]"  # Time series analysis tools
pip install ".[viz]"          # Visualization tools
pip install ".[dev]"          # Development tools
```

### Option 3: Direct from GitHub

```bash
# Latest version from main branch
pip install git+https://github.com/emm-n-m/persistance-nonstarionary-fractal-tools.git

# Specific branch
pip install git+https://github.com/emm-n-m/persistance-nonstarionary-fractal-tools.git@branch-name
```

## Available Extras

- **`timeseries`**: Advanced time series analysis packages (statsmodels, arch, ruptures, etc.)
- **`viz`**: Visualization tools (plotly, jupyter, ipywidgets)
- **`dev`**: Development tools (pytest, black, flake8, mypy)
- **`all`**: Everything above

```bash
# Install with multiple extras
pip install -e ".[dev,timeseries]"
```

## Verify Installation

```python
import timeseries_tools

# Check version
print(timeseries_tools.__version__)

# Show package info
timeseries_tools.show_info()

# Test Hurst estimation
from timeseries_tools import hurst_rs
import numpy as np

series = np.random.randn(1000)
result = hurst_rs(series)
print(f"Hurst exponent: {result['hurst']:.3f}")
```

## Usage

### Hurst Exponent Estimation

```python
from timeseries_tools import hurst_rs
import pandas as pd

# From pandas Series
series = pd.Series([...])
result = hurst_rs(series)

print(f"Hurst: {result['hurst']:.3f}")
print(f"R²: {result['r_squared']:.3f}")
```

### Derivative Analysis

```python
from timeseries_tools import analyze_derivative_patterns
import pandas as pd

# Load your data
data = pd.DataFrame({
    'Date': [...],
    'Series': [...],
    'Value': [...]
})

# Analyze first derivative
results = analyze_derivative_patterns(
    data,
    derivative_order=1,
    time_unit='days',
    direction_labels=['rising', 'falling']
)

# View summary
print(results['summary'])
```

### Climacogram Analysis

```python
from timeseries_tools import compute_climacogram, plot_climacogram
import numpy as np

series = np.random.randn(1000)
scales, variances = compute_climacogram(series, max_scale=50)

# Plot
plot_climacogram(scales, variances, title='My Series Climacogram')
```

## Development Setup

For contributing to the package:

```bash
# Clone the repository
git clone https://github.com/emm-n-m/persistance-nonstarionary-fractal-tools.git
cd persistance-nonstarionary-fractal-tools

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dev dependencies
pip install -e ".[all]"

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=timeseries_tools tests/

# Format code
black Python/

# Type check
mypy Python/
```

## Troubleshooting

### Import Error

If you get import errors after installation:

```python
# Check if package is installed
pip list | grep timeseries

# Try uninstalling and reinstalling
pip uninstall timeseries-tools
pip install -e .
```

### Missing Dependencies

If you get missing module errors:

```bash
# Install all optional dependencies
pip install -e ".[all]"

# Or install specific extras
pip install -e ".[timeseries]"
```

### Windows Issues

On Windows, you might need:

```bash
# Use pip with --user if you don't have admin rights
pip install --user -e .

# Or create a virtual environment
python -m venv venv
venv\Scripts\activate
pip install -e .
```

## Uninstall

```bash
pip uninstall timeseries-tools
```

## Package Structure

After installation, you can import:

```python
# Main package
import timeseries_tools

# Individual modules
from timeseries_tools import hurst_rs
from timeseries_tools import DerivativeAnalyzer
from timeseries_tools import compute_climacogram

# Check what's available
print(timeseries_tools.__all__)
```

## Next Steps

- Read the full documentation in `Python/PYTHON_SETUP_GUIDE.md`
- Check out `Python/example_usage.py` for examples
- Run `pytest tests/` to ensure everything works
