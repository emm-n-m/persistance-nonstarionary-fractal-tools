# ============================================================================
# Example Workflow - Complete Analysis Pipeline
# ============================================================================
# This script demonstrates a complete analysis workflow using the
# persistance-nonstarionary-fractal-tools package.
#
# Author: Emmanouil Mavrogiorgis
# ============================================================================

# Load all tools
source("load_tools.R")

# ============================================================================
# EXAMPLE 1: HK Process Simulation and Validation
# ============================================================================

cat("\n\n=== EXAMPLE 1: HK PROCESS SIMULATION ===\n")

# Simulate HK process with Hurst exponent 0.7 (persistent)
set.seed(42)
n <- 1000
H <- 0.7

cat("\nSimulating HK process (n =", n, ", H =", H, ")\n")

# Try spectral method
sim_spectral <- simulate_hk_spectral(n = n, H = H)

# Try ARFIMA method
sim_arfima <- simulate_hk_arfima(n = n, H = H)

# Try circulant method
sim_circulant <- simulate_hk_circulant(n = n, H = H)

# Plot comparison
par(mfrow = c(3, 1), mar = c(4, 4, 2, 1))
plot(sim_spectral$timeseries, type = "l", col = "blue",
     main = "HK Process - Spectral Method", ylab = "Value")
plot(sim_arfima$timeseries, type = "l", col = "red",
     main = "HK Process - ARFIMA Method", ylab = "Value")
plot(sim_circulant$timeseries, type = "l", col = "green",
     main = "HK Process - Circulant Method", ylab = "Value")
par(mfrow = c(1, 1))

cat("\n✓ HK simulation complete. Three methods compared.\n")

# ============================================================================
# EXAMPLE 2: Synthetic Data with Known Regime Change
# ============================================================================

cat("\n\n=== EXAMPLE 2: REGIME CHANGE DETECTION ===\n")

# Create synthetic data with regime change
n1 <- 150
n2 <- 150
dates1 <- seq(as.Date("2020-01-01"), by = "day", length.out = n1)
dates2 <- seq(dates1[n1] + 1, by = "day", length.out = n2)

# Before regime change: mean = 100, sd = 10
# After regime change: mean = 200, sd = 30 (mean shift + variance increase)
set.seed(123)
values1 <- rnorm(n1, mean = 100, sd = 10)
values2 <- rnorm(n2, mean = 200, sd = 30)

synthetic_data <- data.frame(
  Date = c(dates1, dates2),
  Series = "Synthetic",
  Value = c(values1, values2)
)

cat("\nCreated synthetic data with regime change at day", n1, "\n")

# Detect regime changes
regime_results <- analyze_regime_changes(
  synthetic_data,
  methods = c("variance", "mean", "trend"),
  transform = "none"
)

# Plot the data with detected changepoints
plot(synthetic_data$Date, synthetic_data$Value, type = "l",
     main = "Synthetic Data with Regime Change",
     xlab = "Date", ylab = "Value", col = "darkblue")
abline(v = dates1[n1], col = "red", lwd = 2, lty = 2)
text(dates1[n1], max(synthetic_data$Value) * 0.9,
     "True Change", col = "red", pos = 4)

cat("\n✓ Regime change detection complete. Check results$summary\n")

# ============================================================================
# EXAMPLE 3: Derivative Analysis on Trending Data
# ============================================================================

cat("\n\n=== EXAMPLE 3: DERIVATIVE ANALYSIS ===\n")

# Create data with trend reversal
n <- 200
dates <- seq(as.Date("2020-01-01"), by = "day", length.out = n)

# Increasing trend for first half, then decreasing
trend_values <- c(
  100 + (1:100) * 0.5,  # Increasing
  100 + 50 - (1:100) * 0.5  # Decreasing
)

derivative_data <- data.frame(
  Date = dates,
  Series = "TrendReversal",
  Value = trend_values + rnorm(n, sd = 2)
)

cat("\nCreated data with trend reversal at day 100\n")

# Analyze first derivative (velocity)
velocity_results <- analyze_derivative_patterns(
  derivative_data,
  derivative_order = 1,
  time_unit = "days",
  direction_labels = c("rising", "falling")
)

# Plot original data and derivative
par(mfrow = c(2, 1), mar = c(4, 4, 2, 1))
plot(derivative_data$Date, derivative_data$Value, type = "l",
     main = "Original Data - Trend Reversal",
     xlab = "Date", ylab = "Value", col = "darkblue")
abline(v = dates[100], col = "red", lty = 2)

# Extract and plot derivative
deriv_ts <- extract_derivative_timeseries(
  velocity_results$results$TrendReversal$derivative_metrics
)
plot(deriv_ts$Date, deriv_ts$Derivative, type = "l",
     main = "First Derivative (Velocity)",
     xlab = "Date", ylab = "Rate of Change", col = "darkgreen")
abline(h = 0, col = "gray", lty = 2)
abline(v = dates[100], col = "red", lty = 2)
par(mfrow = c(1, 1))

cat("\n✓ Derivative analysis complete. Check velocity_results$summary\n")

# ============================================================================
# EXAMPLE 4: Real-World Usage Pattern (If you have data)
# ============================================================================

cat("\n\n=== EXAMPLE 4: REAL-WORLD WORKFLOW ===\n")
cat("\nTo analyze your own data, use this pattern:\n\n")

cat("# 1. Load your data\n")
cat("my_data <- read_csv_dynamic(\n")
cat("  'path/to/your/data.csv',\n")
cat("  date_column = 'Date',\n")
cat("  clean_names = TRUE\n")
cat(")\n\n")

cat("# 2. Explore the data\n")
cat("explore_data(my_data)\n\n")

cat("# 3. Run derivative analysis\n")
cat("velocity <- analyze_derivative_patterns(\n")
cat("  my_data$data,\n")
cat("  derivative_order = 1,\n")
cat("  time_unit = 'days',\n")
cat("  direction_labels = c('increasing', 'decreasing')\n")
cat(")\n\n")

cat("# 4. Detect regime changes\n")
cat("regimes <- analyze_regime_changes(\n")
cat("  my_data$data,\n")
cat("  methods = c('variance', 'mean', 'trend'),\n")
cat("  transform = 'none'  # or 'log', 'sqrt', etc.\n")
cat(")\n\n")

cat("# 5. Extract results\n")
cat("View(velocity$summary)\n")
cat("View(regimes$summary)\n\n")

# ============================================================================
# Summary
# ============================================================================

cat("\n\n=== WORKFLOW COMPLETE ===\n")
cat("\nAll examples ran successfully!\n")
cat("\nGenerated objects:\n")
cat("  • sim_spectral, sim_arfima, sim_circulant - HK simulations\n")
cat("  • synthetic_data - Synthetic data with regime change\n")
cat("  • regime_results - Regime change detection results\n")
cat("  • derivative_data - Data with trend reversal\n")
cat("  • velocity_results - Derivative analysis results\n")
cat("  • deriv_ts - Extracted derivative time series\n\n")

cat("Next steps:\n")
cat("  1. Explore the results objects\n")
cat("  2. Try with your own data\n")
cat("  3. Adjust parameters as needed\n")
cat("  4. Create custom visualizations\n\n")

cat("For help on any function, type: ?function_name\n")
cat("Example: ?analyze_derivative_patterns\n\n")
