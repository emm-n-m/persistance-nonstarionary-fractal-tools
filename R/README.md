# R Package: timeseries.tools

## Installation

```r
# Install from GitHub
# install.packages("devtools")
devtools::install_github("emm-n-m/persistance-nonstarionary-fractal-tools")

# Or install locally
devtools::install()
```

## Quick Start

```r
library(timeseries.tools)

# Load data
data <- read_csv_dynamic("your_data.csv", date_column = "Date")

# Analyze first derivative (velocity)
velocity_results <- analyze_derivative_patterns(
  data$data,
  derivative_order = 1,
  direction_labels = c("rising", "falling")
)

# Detect regime changes
regime_results <- analyze_regime_changes(
  data$data,
  methods = c("variance", "mean", "trend")
)

# Simulate HK process
sim <- simulate_hk_spectral(n = 1000, H = 0.7)
plot(sim$timeseries, type = "l")
```

## Main Functions

### Derivative Analysis
- `analyze_derivative_patterns()` - Comprehensive nth-order derivative analysis
- `calculate_derivative_metrics()` - Core derivative computation
- `detect_derivative_regimes()` - Regime change detection in derivatives

### Regime Change Detection
- `analyze_regime_changes()` - Multi-method regime detection
- `detect_variance_change()` - Variance shift detection
- `detect_mean_change()` - Mean level shift detection
- `detect_trend_change()` - Trend reversal detection

### HK Process Simulation
- `simulate_hk_spectral()` - Spectral synthesis method
- `simulate_hk_arfima()` - ARFIMA-based simulation
- `simulate_hk_wavelet()` - Wavelet synthesis
- `simulate_hk_circulant()` - Exact circulant embedding

### Utilities
- `read_csv_dynamic()` - Flexible CSV reader with date parsing
- `explore_data()` - Quick data exploration

## Testing

```r
# Run all tests
devtools::test()

# Check package
devtools::check()
```

## Documentation Status

⚠️ **Note:** Full roxygen2 documentation is planned. Currently using manual NAMESPACE.

To generate documentation:
```r
# TODO: Add roxygen2 comments to functions
# roxygen2::roxygenise()
```
