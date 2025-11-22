# ============================================================================
# Time Series Tools - Complete Loader
# ============================================================================
# This script loads all functions from the persistance-nonstarionary-fractal-tools
# project. Run this at the start of each R session.
#
# Usage:
#   source("load_tools.R")
#
# Author: Emmanouil Mavrogiorgis
# License: MIT
# ============================================================================

cat("Loading Time Series Analysis Tools...\n")
cat("=====================================\n\n")

# Load required packages
required_packages <- c("dplyr", "tidyr", "lubridate", "fracdiff", "ggplot2")
missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

if(length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages)
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(fracdiff)
  library(ggplot2)
})

cat("✓ Required packages loaded\n\n")

# Source all R files
cat("Loading R modules:\n")

source("R/Tools/csv_reader_tool.R")
cat("  ✓ CSV Reader Tool\n")

source("R/derivative.R")
cat("  ✓ Derivative Analysis\n")

source("R/regime_change.R")
cat("  ✓ Regime Change Detection (Generalized)\n")

source("R/regime_change_focused.R")
cat("  ✓ Regime Change Detection (Focused)\n")

source("R/hk_simulator.R")
cat("  ✓ Hurst-Kolmogorov Simulator\n")

source("R/monthly_velocity_anomalies.R")
cat("  ✓ Monthly Velocity Anomalies\n")

tryCatch({
  source("R/hktest.R")
  cat("  ✓ HK Test Suite\n")
}, error = function(e) {
  cat("  ⚠ HK Test Suite (optional - some dependencies may be missing)\n")
})

tryCatch({
  source("R/high_level_analysis.R")
  cat("  ✓ High Level Analysis\n")
}, error = function(e) {
  cat("  ⚠ High Level Analysis (optional)\n")
})

cat("\n=====================================\n")
cat("✓ All tools loaded successfully!\n")
cat("=====================================\n\n")

cat("QUICK START GUIDE\n")
cat("-----------------\n\n")

cat("1. Load Data:\n")
cat("   data <- read_csv_dynamic('your_data.csv', date_column = 'Date')\n\n")

cat("2. Derivative Analysis:\n")
cat("   velocity <- analyze_derivative_patterns(data$data, derivative_order = 1)\n\n")

cat("3. Regime Change Detection:\n")
cat("   regimes <- analyze_regime_changes(data$data, methods = c('mean', 'trend'))\n\n")

cat("4. HK Process Simulation:\n")
cat("   sim <- simulate_hk_spectral(n = 1000, H = 0.7)\n")
cat("   plot(sim$timeseries, type = 'l')\n\n")

cat("MAIN FUNCTIONS AVAILABLE\n")
cat("------------------------\n")
cat("Data Loading:\n")
cat("  • read_csv_dynamic()       - Flexible CSV reader with date parsing\n")
cat("  • explore_data()           - Quick data exploration\n\n")

cat("Derivative Analysis:\n")
cat("  • analyze_derivative_patterns()  - Complete nth-order analysis\n")
cat("  • calculate_derivative_metrics() - Core derivative computation\n")
cat("  • detect_derivative_regimes()    - Regime change in derivatives\n\n")

cat("Regime Change Detection:\n")
cat("  • analyze_regime_changes()    - Multi-method detection\n")
cat("  • detect_variance_change()    - Variance shift detection\n")
cat("  • detect_mean_change()        - Mean level shifts\n")
cat("  • detect_trend_change()       - Trend reversals\n\n")

cat("HK Process Simulation:\n")
cat("  • simulate_hk_spectral()      - Spectral synthesis method\n")
cat("  • simulate_hk_arfima()        - ARFIMA-based simulation\n")
cat("  • simulate_hk_wavelet()       - Wavelet synthesis\n")
cat("  • simulate_hk_circulant()     - Exact circulant embedding\n")
cat("  • compare_hk_methods()        - Compare all methods\n\n")

cat("Monthly Anomalies:\n")
cat("  • analyze_monthly_anomalies() - Detect monthly change anomalies\n\n")

cat("=====================================\n")
cat("Ready to analyze! Type a function name followed by ? for help.\n")
cat("Example: ?analyze_derivative_patterns\n")
cat("=====================================\n")
