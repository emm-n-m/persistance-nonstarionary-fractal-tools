# Memory Structure Regime Change Detection for H-K Processes
# Detects changes in persistence, memory, and correlation structure

library(tidyverse)
library(lubridate)

# Simple data reader
read_reservoir_data <- function(file_path) {
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  data$Date <- as.Date(data$Date, format = "%m/%d/%Y")
  data <- data %>%
    filter(!is.na(Date), !is.na(Value), Value > 0) %>%
    arrange(Date, Reservoir)
  return(data)
}

# Calculate DFA Hurst exponent (simplified)
calculate_dfa_hurst <- function(ts_data) {
  n <- length(ts_data)

  if (n < 100) return(NA)

  # Clean data
  finite_idx <- is.finite(ts_data)
  if (sum(finite_idx) < 100) return(NA)

  clean_data <- ts_data[finite_idx]
  n <- length(clean_data)

  # Remove mean and integrate
  y <- clean_data - mean(clean_data)
  y_int <- cumsum(y)

  # Window sizes
  min_win <- 10
  max_win <- floor(n / 8)

  if (max_win < min_win) return(NA)

  windows <- unique(round(seq(min_win, max_win, length.out = 6)))
  if (length(windows) < 3) return(NA)

  fluctuations <- numeric(length(windows))

  for (i in seq_along(windows)) {
    win_size <- windows[i]
    n_segments <- floor(n / win_size)

    if (n_segments < 2) {
      fluctuations[i] <- NA
      next
    }

    segment_flucts <- numeric(n_segments)

    for (j in 1:n_segments) {
      start_idx <- (j - 1) * win_size + 1
      end_idx <- j * win_size

      segment_data <- y_int[start_idx:end_idx]
      x_vals <- 1:length(segment_data)

      if (length(segment_data) > 2) {
        x_mean <- mean(x_vals)
        y_mean <- mean(segment_data)

        numerator <- sum((x_vals - x_mean) * (segment_data - y_mean))
        denominator <- sum((x_vals - x_mean)^2)

        if (denominator > 0) {
          slope <- numerator / denominator
          intercept <- y_mean - slope * x_mean
          trend <- intercept + slope * x_vals
          residuals <- segment_data - trend
          segment_flucts[j] <- sqrt(mean(residuals^2))
        } else {
          segment_flucts[j] <- sd(segment_data)
        }
      } else {
        segment_flucts[j] <- NA
      }
    }

    valid_flucts <- segment_flucts[is.finite(segment_flucts) & segment_flucts > 0]
    if (length(valid_flucts) > 0) {
      fluctuations[i] <- mean(valid_flucts)
    } else {
      fluctuations[i] <- NA
    }
  }

  # Fit power law
  valid_idx <- is.finite(fluctuations) & fluctuations > 0
  if (sum(valid_idx) < 3) return(NA)

  log_windows <- log(windows[valid_idx])
  log_fluct <- log(fluctuations[valid_idx])

  x_mean <- mean(log_windows)
  y_mean <- mean(log_fluct)

  numerator <- sum((log_windows - x_mean) * (log_fluct - y_mean))
  denominator <- sum((log_windows - x_mean)^2)

  if (denominator > 0) {
    hurst <- numerator / denominator
    if (is.finite(hurst) && hurst > 0.2 && hurst < 2.0) {
      return(hurst)
    }
  }

  return(NA)
}

# Calculate autocorrelation decay characteristics
calculate_acf_memory <- function(ts_data, max_lag = 100) {
  if (length(ts_data) < max_lag * 3) return(NA)

  acf_result <- acf(ts_data, lag.max = max_lag, plot = FALSE)
  acf_values <- acf_result$acf[-1]  # Remove lag 0

  # Find first lag where ACF drops below 0.1 (memory cutoff)
  memory_cutoff <- which(acf_values < 0.1)[1]
  if (is.na(memory_cutoff)) memory_cutoff <- max_lag

  # Calculate integral of ACF (total memory)
  acf_integral <- sum(acf_values[acf_values > 0])

  # Fit power law to ACF decay
  lags <- 1:max_lag
  positive_acf <- acf_values > 0

  if (sum(positive_acf) > 10) {
    log_lags <- log(lags[positive_acf])
    log_acf <- log(acf_values[positive_acf])

    lm_fit <- lm(log_acf ~ log_lags)
    acf_slope <- coef(lm_fit)[2]
  } else {
    acf_slope <- NA
  }

  return(list(
    memory_cutoff = memory_cutoff,
    acf_integral = acf_integral,
    acf_slope = acf_slope
  ))
}

# Detect changes in Hurst exponent using PROPER rolling windows (recent to past)
detect_hurst_regime_change <- function(values, dates, reservoir_name,
                                       window_years = 5, step_years = 1) {

  cat("\n=== MEMORY REGIME ANALYSIS:", reservoir_name, "===\n")

  n <- length(values)
  window_size <- floor(window_years * 365.25)
  step_size <- floor(step_years * 365.25)

  if (n < window_size) {
    cat("Insufficient data for", window_years, "year windows\n")
    return(NULL)
  }

  # Calculate rolling windows FROM RECENT TO PAST (correct approach)
  cat("Calculating rolling Hurst exponents (recent to past)...\n")
  cat("Total data points:", n, "\n")
  cat("Window size (days):", window_size, "\n")
  cat("Step size (days):", step_size, "\n")
  cat("Data range:", as.character(dates[1]), "to", as.character(dates[n]), "\n")

  # Maximum number of windows possible
  max_windows <- floor((n - window_size) / step_size) + 1
  cat("Maximum possible windows:", max_windows, "\n")

  rolling_hurst <- numeric(max_windows)
  rolling_dates <- as.Date(numeric(max_windows), origin = "1970-01-01")
  rolling_acf_slope <- numeric(max_windows)
  rolling_memory_cutoff <- numeric(max_windows)

  valid_windows <- 0

  for (i in 1:max_windows) {
    # CORRECTED: Start from the END and work backwards
    end_idx <- n - (i - 1) * step_size
    start_idx <- end_idx - window_size + 1

    # Debug output for first few windows
    if (i <= 5) {
      cat("Window", i, "calculation: start_idx =", start_idx, "end_idx =", end_idx, "\n")
    }

    # Check if we have valid indices
    if (start_idx < 1) {
      cat("Window", i, "invalid: start_idx =", start_idx, "< 1, stopping\n")
      break
    }

    valid_windows <- valid_windows + 1

    window_data <- values[start_idx:end_idx]
    window_dates <- dates[start_idx:end_idx]

    # More detailed debug output
    if (i <= 5) {
      cat("Window", i, "dates:", as.character(window_dates[1]), "to",
          as.character(window_dates[length(window_dates)]), "\n")
      cat("Window", i, "data points:", length(window_data), "\n")
    }

    # Calculate memory properties
    rolling_hurst[valid_windows] <- calculate_dfa_hurst(window_data)
    rolling_dates[valid_windows] <- window_dates[floor(length(window_dates)/2)]  # Mid-point date

    # ACF properties
    acf_props <- calculate_acf_memory(window_data)
    rolling_acf_slope[valid_windows] <- acf_props$acf_slope
    rolling_memory_cutoff[valid_windows] <- acf_props$memory_cutoff

    # Debug Hurst calculation
    if (i <= 5) {
      cat("Window", i, "Hurst:", round(rolling_hurst[valid_windows], 3), "\n\n")
    }
  }

  # Trim to valid windows
  rolling_hurst <- rolling_hurst[1:valid_windows]
  rolling_dates <- rolling_dates[1:valid_windows]
  rolling_acf_slope <- rolling_acf_slope[1:valid_windows]
  rolling_memory_cutoff <- rolling_memory_cutoff[1:valid_windows]

  # Remove invalid calculations
  valid_idx <- !is.na(rolling_hurst) & rolling_hurst > 0.3 & rolling_hurst < 2.0
  rolling_hurst <- rolling_hurst[valid_idx]
  rolling_dates <- rolling_dates[valid_idx]
  rolling_acf_slope <- rolling_acf_slope[valid_idx]
  rolling_memory_cutoff <- rolling_memory_cutoff[valid_idx]

  if (length(rolling_hurst) < 3) {
    cat("Insufficient valid rolling windows\n")
    return(NULL)
  }

  cat("Valid rolling windows:", length(rolling_hurst), "\n")
  cat("Analysis period:", as.character(min(rolling_dates)), "to", as.character(max(rolling_dates)), "\n")
  cat("Hurst range:", round(range(rolling_hurst), 3), "\n")

  # IMPORTANT: Reverse the order so chronological time goes forward
  # (since we calculated from recent to past)
  reverse_idx <- length(rolling_hurst):1
  rolling_hurst <- rolling_hurst[reverse_idx]
  rolling_dates <- rolling_dates[reverse_idx]
  rolling_acf_slope <- rolling_acf_slope[reverse_idx]
  rolling_memory_cutoff <- rolling_memory_cutoff[reverse_idx]

  # Detect changepoints in Hurst evolution
  hurst_changepoints <- detect_memory_changepoints(
    rolling_hurst, rolling_dates, "Hurst Exponent"
  )

  # Detect changepoints in ACF slope
  acf_changepoints <- detect_memory_changepoints(
    rolling_acf_slope, rolling_dates, "ACF Slope"
  )

  # Detect changepoints in memory cutoff
  memory_changepoints <- detect_memory_changepoints(
    rolling_memory_cutoff, rolling_dates, "Memory Cutoff"
  )

  # Compile results
  results <- list(
    reservoir = reservoir_name,
    rolling_dates = rolling_dates,
    rolling_hurst = rolling_hurst,
    rolling_acf_slope = rolling_acf_slope,
    rolling_memory_cutoff = rolling_memory_cutoff,
    hurst_changepoints = hurst_changepoints,
    acf_changepoints = acf_changepoints,
    memory_changepoints = memory_changepoints
  )

  # Print summary
  print_memory_summary(results)

  return(results)
}

# Detect changepoints in memory time series
detect_memory_changepoints <- function(values, dates, property_name) {
  n <- length(values)

  if (n < 6) return(NULL)

  cat("\nDetecting changepoints in", property_name, "...\n")

  # Test potential changepoints (avoid first and last 2 points)
  test_indices <- 3:(n-2)

  if (length(test_indices) < 2) return(NULL)

  scores <- numeric(length(test_indices))

  for (i in seq_along(test_indices)) {
    cp_idx <- test_indices[i]

    before_values <- values[1:cp_idx]
    after_values <- values[(cp_idx+1):n]

    if (length(before_values) >= 3 && length(after_values) >= 3) {
      before_mean <- mean(before_values, na.rm = TRUE)
      after_mean <- mean(after_values, na.rm = TRUE)

      before_var <- var(before_values, na.rm = TRUE)
      after_var <- var(after_values, na.rm = TRUE)

      # Score based on both mean and variance changes
      if (is.finite(before_mean) && is.finite(after_mean) &&
          is.finite(before_var) && is.finite(after_var) &&
          before_var > 0 && after_var > 0) {

        # Normalized mean change
        pooled_sd <- sqrt((before_var + after_var) / 2)
        mean_change_score <- abs(after_mean - before_mean) / pooled_sd

        # Variance ratio
        var_change_score <- abs(log(after_var / before_var))

        # Combined score
        scores[i] <- mean_change_score + 0.5 * var_change_score
      } else {
        scores[i] <- 0
      }
    } else {
      scores[i] <- 0
    }
  }

  # Find most significant changepoint
  if (length(scores) > 0 && max(scores, na.rm = TRUE) > 0.5) {
    max_idx <- which.max(scores)
    best_cp_idx <- test_indices[max_idx]
    best_score <- scores[max_idx]

    before_values <- values[1:best_cp_idx]
    after_values <- values[(best_cp_idx+1):n]

    result <- list(
      property = property_name,
      changepoint_date = dates[best_cp_idx],
      changepoint_index = best_cp_idx,
      score = best_score,
      before_mean = mean(before_values, na.rm = TRUE),
      after_mean = mean(after_values, na.rm = TRUE),
      before_var = var(before_values, na.rm = TRUE),
      after_var = var(after_values, na.rm = TRUE)
    )

    cat(property_name, "changepoint:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 3), "\n")
    cat("Before mean:", round(result$before_mean, 3),
        "After mean:", round(result$after_mean, 3), "\n")

    return(result)
  } else {
    cat("No significant", property_name, "changepoint detected\n")
    return(NULL)
  }
}

# Print comprehensive memory analysis summary
print_memory_summary <- function(results) {
  cat("\n=== MEMORY STRUCTURE SUMMARY ===\n")
  cat("Reservoir:", results$reservoir, "\n")
  cat("Analysis period:", as.character(min(results$rolling_dates)), "to",
      as.character(max(results$rolling_dates)), "\n")
  cat("Rolling windows:", length(results$rolling_hurst), "\n\n")

  # Memory property ranges
  cat("Memory Property Ranges:\n")
  cat("Hurst Exponent   :", round(range(results$rolling_hurst, na.rm = TRUE), 3), "\n")
  cat("ACF Slope        :", round(range(results$rolling_acf_slope, na.rm = TRUE), 3), "\n")
  cat("Memory Cutoff    :", round(range(results$rolling_memory_cutoff, na.rm = TRUE), 1), "days\n\n")

  # Changepoint summary
  cat("Detected Memory Regime Changes:\n")
  cat(paste(rep("=", 35), collapse = ""), "\n")

  changepoint_types <- c("hurst_changepoints", "acf_changepoints", "memory_changepoints")
  type_names <- c("Hurst Exponent", "ACF Slope", "Memory Cutoff")

  for (i in seq_along(changepoint_types)) {
    cp_type <- changepoint_types[i]
    type_name <- type_names[i]

    if (!is.null(results[[cp_type]])) {
      cp <- results[[cp_type]]
      cat(sprintf("%-15s: %s (%.3f → %.3f)\n",
                  type_name,
                  as.character(cp$changepoint_date),
                  cp$before_mean,
                  cp$after_mean))
    } else {
      cat(sprintf("%-15s: No change detected\n", type_name))
    }
  }
  cat("\n")

  # Memory evolution trends
  if (length(results$rolling_hurst) > 3) {
    hurst_trend <- cor(as.numeric(results$rolling_dates), results$rolling_hurst, use = "complete.obs")
    cat("Hurst Evolution Trend:", round(hurst_trend, 3))
    if (abs(hurst_trend) > 0.3) {
      direction <- ifelse(hurst_trend > 0, "INCREASING", "DECREASING")
      cat(" (", direction, " persistence)\n")
    } else {
      cat(" (STABLE persistence)\n")
    }
  }
}

# Main analysis function
analyze_memory_regimes <- function(csv_file, reservoirs = NULL,
                                   window_years = 5, step_years = 1) {
  cat("=== MEMORY STRUCTURE REGIME DETECTION ===\n")
  cat("Window size:", window_years, "years\n")
  cat("Step size:", step_years, "year(s)\n\n")

  # Read data
  data <- read_reservoir_data(csv_file)

  if (is.null(reservoirs)) {
    reservoirs <- unique(data$Reservoir)
  }

  results_list <- list()

  for (reservoir in reservoirs) {
    reservoir_data <- data %>%
      filter(Reservoir == reservoir) %>%
      arrange(Date)

    if (nrow(reservoir_data) > window_years * 365 * 2) {
      # Log transform for better H-K analysis
      log_values <- log(reservoir_data$Value + 1e-6)

      results <- detect_hurst_regime_change(
        values = log_values,
        dates = reservoir_data$Date,
        reservoir_name = reservoir,
        window_years = window_years,
        step_years = step_years
      )

      if (!is.null(results)) {
        results_list[[reservoir]] <- results
      }
    } else {
      cat("Skipping", reservoir, "- insufficient data for", window_years, "year windows\n")
    }
  }

  # Create summary table
  cat("\n=== MEMORY REGIME SUMMARY TABLE ===\n")
  create_memory_summary_table(results_list)

  return(results_list)
}

# Create summary table of all memory regime changes
create_memory_summary_table <- function(results_list) {
  if (length(results_list) == 0) {
    cat("No memory regime changes detected\n")
    return()
  }

  summary_data <- data.frame(
    Reservoir = character(),
    Hurst_Change = character(),
    ACF_Change = character(),
    Memory_Change = character(),
    Current_Hurst = numeric(),
    Hurst_Trend = character(),
    stringsAsFactors = FALSE
  )

  for (name in names(results_list)) {
    result <- results_list[[name]]

    hurst_date <- if (!is.null(result$hurst_changepoints)) as.character(result$hurst_changepoints$changepoint_date) else "None"
    acf_date <- if (!is.null(result$acf_changepoints)) as.character(result$acf_changepoints$changepoint_date) else "None"
    memory_date <- if (!is.null(result$memory_changepoints)) as.character(result$memory_changepoints$changepoint_date) else "None"

    current_hurst <- tail(result$rolling_hurst, 1)
    hurst_trend_val <- cor(as.numeric(result$rolling_dates), result$rolling_hurst, use = "complete.obs")
    hurst_trend <- if (abs(hurst_trend_val) > 0.3) {
      ifelse(hurst_trend_val > 0, "Increasing", "Decreasing")
    } else {
      "Stable"
    }

    summary_data <- rbind(summary_data, data.frame(
      Reservoir = name,
      Hurst_Change = hurst_date,
      ACF_Change = acf_date,
      Memory_Change = memory_date,
      Current_Hurst = round(current_hurst, 3),
      Hurst_Trend = hurst_trend,
      stringsAsFactors = FALSE
    ))
  }

  print(summary_data)

  # Check for recent memory regime changes
  cat("\n=== RECENT MEMORY REGIME CHECK ===\n")
  for (i in 1:nrow(summary_data)) {
    reservoir <- summary_data$Reservoir[i]
    dates <- c(summary_data$Hurst_Change[i], summary_data$ACF_Change[i],
               summary_data$Memory_Change[i])

    # Check for recent changes (2020-2025)
    recent_detected <- any(grepl("202[0-5]", dates))

    if (recent_detected) {
      cat(reservoir, ": ✓ DETECTED recent memory regime change\n")
    } else {
      cat(reservoir, ": ✗ No recent memory regime change detected\n")
    }

    # Check Hurst trend
    trend <- summary_data$Hurst_Trend[i]
    current_h <- summary_data$Current_Hurst[i]
    cat("  Current H =", current_h, "- Trend:", trend, "\n")
  }
}

# Example usage:
 memory_results <- analyze_memory_regimes("data/water_reserves.csv")
