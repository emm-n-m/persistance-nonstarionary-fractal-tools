"""
Timeseries Tools - Analysis of Non-Stationary, Fractal, and Long Memory Time Series

This package provides tools for:
- Hurst exponent estimation (rescaled range method)
- Derivative pattern analysis (nth-order derivatives with regime detection)
- Climacogram analysis (variance scaling properties)
- EEG data loading and analysis (long-range correlations)
- Example workflows and utilities

Modules:
    hurst: Hurst exponent estimation using rescaled range (R/S) method
    derivative_analysis: Comprehensive nth-order derivative analysis
    climacogram: Climacogram computation and visualization
    eeg_loader: Generic EEG data loader and analyzer (CSV, EDF formats)
    example_usage: Example workflows and demonstrations

Author: Emmanouil Mavrogiorgis
License: MIT
"""

__version__ = "0.0.1"
__author__ = "Emmanouil Mavrogiorgis"
__email__ = "emm.n.black@gmail.com"

# Import main functions for convenient access
try:
    from .hurst import hurst_rs
except ImportError:
    hurst_rs = None

try:
    from .derivative_analysis import (
        DerivativeAnalyzer,
        analyze_derivative_patterns,
        calculate_first_derivative,
        calculate_second_derivative,
        calculate_acceleration,
    )
except ImportError:
    DerivativeAnalyzer = None
    analyze_derivative_patterns = None
    calculate_first_derivative = None
    calculate_second_derivative = None
    calculate_acceleration = None

try:
    from .climacogram import (
        compute_climacogram,
        plot_climacogram,
        analyze_reservoir_climacogram,
    )
except ImportError:
    compute_climacogram = None
    plot_climacogram = None
    analyze_reservoir_climacogram = None

try:
    from .eeg_loader import (
        EEGLoader,
        EEGAnalyzer,
        quick_analysis,
    )
except ImportError:
    EEGLoader = None
    EEGAnalyzer = None
    quick_analysis = None

# Define what's available when using "from timeseries_tools import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Hurst analysis
    "hurst_rs",
    # Derivative analysis
    "DerivativeAnalyzer",
    "analyze_derivative_patterns",
    "calculate_first_derivative",
    "calculate_second_derivative",
    "calculate_acceleration",
    # Climacogram
    "compute_climacogram",
    "plot_climacogram",
    "analyze_reservoir_climacogram",
    # EEG analysis
    "EEGLoader",
    "EEGAnalyzer",
    "quick_analysis",
]


def get_version():
    """Return the version string."""
    return __version__


def show_info():
    """Display package information."""
    print(f"Timeseries Tools v{__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print("\nAvailable modules:")
    print("  - hurst: Hurst exponent estimation")
    print("  - derivative_analysis: Derivative pattern analysis")
    print("  - climacogram: Variance scaling analysis")
    print("  - eeg_loader: EEG data loading and analysis (CSV, EDF)")
    print("\nFor help on a specific module:")
    print("  import timeseries_tools")
    print("  help(timeseries_tools.hurst_rs)")
    print("  help(timeseries_tools.EEGLoader)")
