# Copyright (c) 2025 [Emmanouil Mavrogiorgis, emm.n.black@gmail.com]
# All rights reserved.
#
# This project was developed with assistance from Claude Code by Anthropic.
# Claude Code is an AI-powered coding assistant (https://www.anthropic.com).
#
# Licensed under MIT License
# See LICENSE file for details

# Reservoir Data Long Memory and Hurst-Kolmogorov Analysis
# Enhanced version with temporal analysis

# Load required libraries
library(tidyverse)
library(fracdiff)
library(longmemo)
library(wavelets)
library(forecast)
library(lubridate)
library(tseries)  # For stationarity tests

# Function to read and prepare the data
prepare_reservoir_data <- function(file_path) {
  # Read the CSV file
  data <- read.csv(file_path, stringsAsFactors = FALSE)

  cat("Raw data structure:\n")
  cat("Columns:", names(data), "\n")
  cat("First few rows:\n")
  print(head(data))

  # Check the date format in the data
  cat("Sample date values:", head(data$Date, 10), "\n")

  # Try multiple date formats
  date_formats <- c("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d")

  parsed_dates <- NULL
  for (fmt in date_formats) {
    cat("Trying date format:", fmt, "\n")
    tryCatch({
      test_dates <- as.Date(data$Date, format = fmt)
      n_valid <- sum(!is.na(test_dates))
      cat("Valid dates with format", fmt, ":", n_valid, "out of", length(test_dates), "\n")

      if (n_valid > length(test_dates) * 0.9) {  # If >90% parse successfully
        parsed_dates <- test_dates
        cat("Using date format:", fmt, "\n")
        break
      }
    }, error = function(e) {
      cat("Format", fmt, "failed:", e$message, "\n")
    })
  }

  if (is.null(parsed_dates)) {
    cat("ERROR: Could not parse dates with any format!\n")
    cat("Date column sample:", head(data$Date, 20), "\n")
    return(NULL)
  }

  # Convert Date column to Date type
  data$Date <- parsed_dates

  # Sort by date and reservoir
  data <- data %>%
    arrange(Date, Reservoir) %>%
    filter(!is.na(Value), Value > 0, !is.na(Date))  # Remove missing values AND invalid dates

  # Add data quality summary
  data_summary <- data %>%
    group_by(Reservoir) %>%
    summarise(
      first_date = min(Date, na.rm = TRUE),
      last_date = max(Date, na.rm = TRUE),
      n_observations = n(),
      mean_value = mean(Value, na.rm = TRUE),
      years_of_data = as.numeric(difftime(max(Date, na.rm = TRUE), min(Date, na.rm = TRUE), units = "days")) / 365.25,
      .groups = "drop"
    )

  cat("=== DATA SUMMARY (FIXED) ===\n")
  print(data_summary)
  cat("\n")

  return(data)
}

# Simple diagnostic function to check what's wrong with seasonality
diagnose_seasonality <- function(ts_data, dates, reservoir_name) {
  cat("\n=== SEASONALITY DIAGNOSIS for", reservoir_name, "===\n")

  # Check data basics
  cat("Data length:", length(ts_data), "\n")
  cat("Date length:", length(dates), "\n")
  cat("Data range:", round(range(ts_data, na.rm = TRUE), 2), "\n")

  # Check dates
  cat("Date class:", class(dates), "\n")
  cat("Date range:", range(dates, na.rm = TRUE), "\n")

  # Extract months
  month <- as.numeric(format(as.Date(dates), "%m"))
  cat("Month range:", range(month, na.rm = TRUE), "\n")
  cat("Unique months:", sort(unique(month)), "\n")

  # Calculate monthly means manually
  if (length(unique(month)) >= 12) {
    monthly_means <- tapply(ts_data, month, mean, na.rm = TRUE)
    cat("Monthly means:\n")
    print(round(monthly_means, 0))

    cat("Min monthly mean:", round(min(monthly_means, na.rm = TRUE), 0), "\n")
    cat("Max monthly mean:", round(max(monthly_means, na.rm = TRUE), 0), "\n")
    cat("Ratio (max/min):", round(max(monthly_means, na.rm = TRUE) / min(monthly_means, na.rm = TRUE), 4), "\n")
  }

  cat("=== END DIAGNOSIS ===\n\n")
}

# Simple R/S Hurst calculation (fallback)
calculate_rs_hurst <- function(ts_data) {
  n <- length(ts_data)
  if (n < 10) return(NA)

  # Remove trend
  y <- ts_data - mean(ts_data)

  # Calculate cumulative sum
  z <- cumsum(y)

  # Calculate range
  R <- max(z) - min(z)

  # Calculate standard deviation
  S <- sd(ts_data)

  if (S == 0) return(NA)

  # R/S statistic
  rs <- R / S

  # Hurst exponent approximation
  hurst <- log(rs) / log(n)

  return(hurst)
}

# Manual DFA implementation - robust version
calculate_dfa_hurst <- function(ts_data) {
  n <- length(ts_data)

  if (n < 50) {
    return(NA)
  }

  # Check for valid data
  finite_idx <- is.finite(ts_data)
  if (sum(finite_idx) < 50) {
    return(NA)
  }

  # Use only finite data
  clean_data <- ts_data[finite_idx]
  n <- length(clean_data)

  # Remove mean and integrate
  y <- clean_data - mean(clean_data)
  y_int <- cumsum(y)

  # Conservative window sizes
  min_win <- 10
  max_win <- floor(n / 8)  # More conservative

  if (max_win < min_win) {
    return(NA)
  }

  # Linear spacing of window sizes
  n_windows <- min(8, floor((max_win - min_win) / 5) + 1)
  windows <- unique(round(seq(min_win, max_win, length.out = n_windows)))

  if (length(windows) < 3) {
    return(NA)
  }

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

      # Simple linear detrending
      if (length(segment_data) > 2) {
        # Use simple linear fit
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

  # Check results
  valid_idx <- is.finite(fluctuations) & fluctuations > 0
  n_valid <- sum(valid_idx)

  if (n_valid < 3) {
    return(NA)
  }

  # Fit power law
  log_windows <- log(windows[valid_idx])
  log_fluct <- log(fluctuations[valid_idx])

  # Simple linear regression
  x_mean <- mean(log_windows)
  y_mean <- mean(log_fluct)

  numerator <- sum((log_windows - x_mean) * (log_fluct - y_mean))
  denominator <- sum((log_windows - x_mean)^2)

  if (denominator > 0) {
    hurst <- numerator / denominator

    # Sanity check
    if (is.finite(hurst) && hurst > 0.2 && hurst < 2.0) {
      return(hurst)
    } else {
      return(NA)
    }
  } else {
    return(NA)
  }
}

# Function to calculate Hurst exponent using multiple methods
calculate_hurst_exponent <- function(ts_data, methods = c("rs", "dfa", "per")) {
  results <- list()

  # R/S Statistic (Rescaled Range) using longmemo package
  if ("rs" %in% methods) {
    tryCatch({
      # Use longmemo package for R/S analysis
      rs_result <- hurstSpec(ts_data, method = "standard", freq = frequency(ts_data))
      results$rs_hurst <- rs_result$exponent
    }, error = function(e) {
      # Fallback: simple R/S calculation
      results$rs_hurst <- calculate_rs_hurst(ts_data)
    })
  }

  # Detrended Fluctuation Analysis using manual implementation
  if ("dfa" %in% methods) {
    tryCatch({
      results$dfa_hurst <- calculate_dfa_hurst(ts_data)
    }, error = function(e) {
      results$dfa_hurst <- NA
    })
  }

  # Periodogram method using longmemo
  if ("per" %in% methods) {
    tryCatch({
      per_result <- hurstSpec(ts_data, method = "per", freq = frequency(ts_data))
      results$per_hurst <- per_result$exponent
    }, error = function(e) {
      results$per_hurst <- NA
    })
  }

  return(results)
}

# Function to fit ARFIMA model
fit_arfima_model <- function(ts_data) {
  tryCatch({
    # Fit ARFIMA model
    arfima_fit <- fracdiff(ts_data, nar = 1, nma = 1)

    # Extract fractional differencing parameter
    d_param <- arfima_fit$d
    hurst_est <- d_param + 0.5  # H = d + 0.5 for ARFIMA

    return(list(
      d_parameter = d_param,
      hurst_arfima = hurst_est,
      aic = arfima_fit$log.likelihood,
      model = arfima_fit
    ))
  }, error = function(e) {
    return(list(
      d_parameter = NA,
      hurst_arfima = NA,
      aic = NA,
      model = NULL
    ))
  })
}

# Function to analyze temporal correlations
analyze_correlations <- function(ts_data, max_lag = 50) {
  # Calculate autocorrelation function
  acf_result <- acf(ts_data, lag.max = max_lag, plot = FALSE)

  # Fit power law to ACF decay
  lags <- 1:max_lag
  acf_values <- acf_result$acf[2:(max_lag + 1)]

  # Remove negative or zero correlations for log-log fit
  valid_idx <- acf_values > 0
  if (sum(valid_idx) > 5) {
    log_lags <- log(lags[valid_idx])
    log_acf <- log(acf_values[valid_idx])

    # Linear fit in log-log space
    lm_fit <- lm(log_acf ~ log_lags)
    power_law_slope <- coef(lm_fit)[2]
  } else {
    power_law_slope <- NA
  }

  return(list(
    acf_result = acf_result,
    power_law_slope = power_law_slope
  ))
}

# Function to perform wavelet analysis
wavelet_analysis <- function(ts_data) {
  tryCatch({
    # Ensure data length is power of 2
    n <- length(ts_data)
    n_pow2 <- 2^floor(log2(n))
    ts_subset <- ts_data[1:n_pow2]

    # Perform discrete wavelet transform
    wt <- dwt(ts_subset, filter = "haar")

    # Calculate energy at each scale
    energies <- sapply(wt@W, function(x) sum(x^2))
    scales <- 1:length(energies)

    # Fit power law to energy vs scale
    if (length(energies) > 3) {
      log_scales <- log(scales)
      log_energies <- log(energies)
      lm_fit <- lm(log_energies ~ log_scales)
      scaling_exponent <- coef(lm_fit)[2]
    } else {
      scaling_exponent <- NA
    }

    return(list(
      wavelet_transform = wt,
      scaling_exponent = scaling_exponent,
      energies = energies
    ))
  }, error = function(e) {
    return(list(
      wavelet_transform = NULL,
      scaling_exponent = NA,
      energies = NA
    ))
  })
}

# Function to test for structural breaks and non-stationarity
test_stationarity <- function(ts_data) {
  results <- list()

  # Check for valid data
  if (length(ts_data) < 10 || all(is.na(ts_data))) {
    results$adf_pvalue <- NA
    results$adf_statistic <- NA
    results$kpss_pvalue <- NA
    results$kpss_statistic <- NA
    return(results)
  }

  # Clean data
  clean_data <- ts_data[is.finite(ts_data)]
  if (length(clean_data) < 10) {
    results$adf_pvalue <- NA
    results$adf_statistic <- NA
    results$kpss_pvalue <- NA
    results$kpss_statistic <- NA
    return(results)
  }

  # Augmented Dickey-Fuller test
  tryCatch({
    adf_test <- adf.test(clean_data, alternative = "stationary")
    results$adf_pvalue <- adf_test$p.value
    results$adf_statistic <- as.numeric(adf_test$statistic)
  }, error = function(e) {
    results$adf_pvalue <- NA
    results$adf_statistic <- NA
  })

  # KPSS test
  tryCatch({
    kpss_test <- kpss.test(clean_data)
    results$kpss_pvalue <- kpss_test$p.value
    results$kpss_statistic <- as.numeric(kpss_test$statistic)
  }, error = function(e) {
    results$kpss_pvalue <- NA
    results$kpss_statistic <- NA
  })

  return(results)
}

# Function to analyze seasonality using proper tools
analyze_seasonality_advanced <- function(ts_data, dates, frequency = 365.25) {
  results <- list()

  tryCatch({
    n <- length(ts_data)

    # Create proper time series object
    ts_obj <- ts(ts_data, frequency = frequency)

    # Method 1: STL decomposition (most robust)
    stl_result <- tryCatch({
      stl(ts_obj, s.window = "periodic", robust = TRUE)
    }, error = function(e) {
      tryCatch({
        stl(ts_obj, s.window = 13, robust = TRUE)  # Fixed window
      }, error = function(e2) {
        NULL
      })
    })

    if (!is.null(stl_result)) {
      seasonal_comp <- as.numeric(stl_result$time.series[,"seasonal"])
      trend_comp <- as.numeric(stl_result$time.series[,"trend"])
      remainder_comp <- as.numeric(stl_result$time.series[,"remainder"])

      # Calculate seasonal strength (variance decomposition)
      total_var <- var(ts_data, na.rm = TRUE)
      seasonal_var <- var(seasonal_comp, na.rm = TRUE)
      trend_var <- var(trend_comp, na.rm = TRUE)

      results$seasonal_strength_stl <- seasonal_var / total_var
      results$trend_strength <- trend_var / total_var

      # Seasonal range from STL
      seasonal_range_stl <- (max(seasonal_comp, na.rm = TRUE) - min(seasonal_comp, na.rm = TRUE))
      results$seasonal_range_stl <- seasonal_range_stl / mean(ts_data, na.rm = TRUE)

      # Deseasonalized series
      results$deseasonalized_stl <- ts_data - seasonal_comp
    }

    # Method 2: Simple monthly aggregation for validation
    month <- as.numeric(format(as.Date(dates), "%m"))
    month <- pmax(1, pmin(12, month))  # Ensure 1-12 range

    if (length(unique(month)) >= 12) {
      monthly_stats <- data.frame(
        value = ts_data,
        month = month
      ) %>%
        group_by(month) %>%
        summarise(
          mean_val = mean(value, na.rm = TRUE),
          sd_val = sd(value, na.rm = TRUE),
          n_obs = n(),
          .groups = "drop"
        )

      if (nrow(monthly_stats) >= 12) {
        monthly_means <- monthly_stats$mean_val

        # Coefficient of variation across months
        mean_of_means <- mean(monthly_means, na.rm = TRUE)
        if (mean_of_means > 0) {
          seasonal_cv <- sd(monthly_means, na.rm = TRUE) / mean_of_means
        } else {
          seasonal_cv <- 0
        }

        # Seasonal range (max/min months)
        min_mean <- min(monthly_means, na.rm = TRUE)
        max_mean <- max(monthly_means, na.rm = TRUE)
        if (min_mean > 0) {
          seasonal_range <- max_mean / min_mean
        } else {
          seasonal_range <- 1
        }

        # Peak and low months
        peak_month <- monthly_stats$month[which.max(monthly_means)]
        low_month <- monthly_stats$month[which.min(monthly_means)]

        results$seasonal_cv <- seasonal_cv
        results$seasonal_range <- seasonal_range
        results$peak_month <- peak_month
        results$low_month <- low_month
        results$monthly_stats <- monthly_stats
      }
    }

    # Use the best available deseasonalized series
    if (!is.null(results$deseasonalized_stl)) {
      results$deseasonalized <- results$deseasonalized_stl
      results$method_used <- "STL"
    } else if (!is.null(results$monthly_stats)) {
      # Fallback: simple monthly deseasonalization
      monthly_effects <- rep(0, length(ts_data))
      for (m in 1:12) {
        month_idx <- which(month == m)
        if (length(month_idx) > 0) {
          month_mean <- mean(ts_data[month_idx], na.rm = TRUE)
          overall_mean <- mean(ts_data, na.rm = TRUE)
          monthly_effects[month_idx] <- month_mean - overall_mean
        }
      }
      results$deseasonalized <- ts_data - monthly_effects
      results$method_used <- "Monthly"
    } else {
      results$deseasonalized <- ts_data  # No deseasonalization possible
      results$method_used <- "None"
    }

  }, error = function(e) {
    # Set all results to NA
    results$seasonal_strength_stl <- NA
    results$seasonal_range <- 1.00
    results$seasonal_cv <- 0
    results$peak_month <- NA
    results$low_month <- NA
    results$deseasonalized <- ts_data
    results$method_used <- "Failed"
  })

  return(results)
}

# Function to calculate Hurst on deseasonalized data
calculate_deseasonalized_hurst <- function(deseasonalized_data) {
  if (is.null(deseasonalized_data) || length(deseasonalized_data) < 100) {
    return(NA)
  }

  tryCatch({
    # Use DFA on deseasonalized data
    hurst_deseas <- calculate_dfa_hurst(deseasonalized_data)
    return(hurst_deseas)
  }, error = function(e) {
    return(NA)
  })
}

# Function to detect regime changes and time-varying Hurst
analyze_temporal_evolution <- function(ts_data, dates, reservoir_name, window_years = 5) {
  results <- list()

  cat("=== TEMPORAL ANALYSIS for", reservoir_name, "===\n")

  n <- length(ts_data)
  window_size <- floor(window_years * 365.25)  # Convert years to days

  if (n < 2 * window_size) {
    cat("Insufficient data for temporal analysis\n")
    return(results)
  }

  # 1. ROLLING HURST EXPONENT
  cat("Calculating rolling Hurst exponents...\n")
  step_size <- floor(window_size / 2)  # 50% overlap
  n_windows <- floor((n - window_size) / step_size) + 1

  rolling_hurst <- numeric(n_windows)
  rolling_dates <- as.Date(numeric(n_windows), origin = "1970-01-01")
  rolling_mean <- numeric(n_windows)
  rolling_var <- numeric(n_windows)

  for (i in 1:n_windows) {
    start_idx <- (i - 1) * step_size + 1
    end_idx <- start_idx + window_size - 1

    if (end_idx > n) break

    window_data <- ts_data[start_idx:end_idx]
    window_dates <- dates[start_idx:end_idx]

    # Calculate Hurst for this window
    hurst_val <- calculate_dfa_hurst(window_data)
    rolling_hurst[i] <- hurst_val
    rolling_dates[i] <- window_dates[floor(length(window_dates)/2)]  # Mid-point date
    rolling_mean[i] <- mean(window_data, na.rm = TRUE)
    rolling_var[i] <- var(window_data, na.rm = TRUE)
  }

  # Remove incomplete windows
  valid_idx <- !is.na(rolling_hurst) & rolling_hurst > 0.3 & rolling_hurst < 2.0
  rolling_hurst <- rolling_hurst[valid_idx]
  rolling_dates <- rolling_dates[valid_idx]
  rolling_mean <- rolling_mean[valid_idx]
  rolling_var <- rolling_var[valid_idx]

  if (length(rolling_hurst) > 3) {
    results$rolling_hurst <- rolling_hurst
    results$rolling_dates <- rolling_dates
    results$rolling_mean <- rolling_mean
    results$rolling_var <- rolling_var

    cat("Rolling Hurst range:", round(range(rolling_hurst, na.rm = TRUE), 3), "\n")
    cat("Rolling Hurst trend:", round(cor(as.numeric(rolling_dates), rolling_hurst, use = "complete.obs"), 3), "\n")
  }

  # 2. DECADE ANALYSIS
  cat("Analyzing by decades...\n")

  # Group by decades
  year <- as.numeric(format(dates, "%Y"))
  decade <- floor(year / 10) * 10

  decade_stats <- data.frame(
    year = year,
    decade = decade,
    value = ts_data
  ) %>%
    group_by(decade) %>%
    summarise(
      n_obs = n(),
      mean_val = mean(value, na.rm = TRUE),
      var_val = var(value, na.rm = TRUE),
      min_year = min(year),
      max_year = max(year),
      .groups = "drop"
    ) %>%
    filter(n_obs > 365)  # At least 1 year of data

  if (nrow(decade_stats) >= 2) {
    results$decade_stats <- decade_stats

    cat("Decade analysis:\n")
    print(decade_stats)

    # Calculate Hurst for each decade with sufficient data
    decade_hurst <- numeric(nrow(decade_stats))
    for (i in 1:nrow(decade_stats)) {
      dec <- decade_stats$decade[i]
      decade_data <- ts_data[decade == dec]

      if (length(decade_data) > 500) {
        decade_hurst[i] <- calculate_dfa_hurst(decade_data)
      } else {
        decade_hurst[i] <- NA
      }
    }

    decade_stats$hurst <- decade_hurst
    results$decade_hurst <- decade_hurst

    cat("Hurst by decade:\n")
    valid_decades <- !is.na(decade_hurst)
    if (sum(valid_decades) > 0) {
      print(data.frame(
        decade = decade_stats$decade[valid_decades],
        hurst = round(decade_hurst[valid_decades], 3)
      ))
    }
  }

  # 3. SIMPLE CHANGEPOINT DETECTION
  cat("Detecting potential regime changes...\n")

  # Simple changepoint detection using variance changes
  changepoints <- tryCatch({
    # Use cumulative sum of squares for changepoint detection
    cumsum_sq <- cumsum((ts_data - mean(ts_data, na.rm = TRUE))^2)
    n_data <- length(ts_data)

    # Look for significant changes in slope
    test_points <- seq(floor(n_data * 0.2), floor(n_data * 0.8), by = floor(n_data * 0.1))
    changepoint_scores <- numeric(length(test_points))

    for (i in seq_along(test_points)) {
      cp <- test_points[i]

      # Test if there's a significant change at this point
      before_slope <- (cumsum_sq[cp] - cumsum_sq[1]) / (cp - 1)
      after_slope <- (cumsum_sq[n_data] - cumsum_sq[cp]) / (n_data - cp)

      changepoint_scores[i] <- abs(log(after_slope / before_slope))
    }

    # Find the most significant changepoint
    max_score_idx <- which.max(changepoint_scores)
    best_changepoint <- test_points[max_score_idx]

    if (changepoint_scores[max_score_idx] > 0.5) {  # Threshold for significance
      list(
        changepoint_index = best_changepoint,
        changepoint_date = dates[best_changepoint],
        changepoint_score = changepoint_scores[max_score_idx]
      )
    } else {
      NULL
    }
  }, error = function(e) {
    NULL
  })

  if (!is.null(changepoints)) {
    results$changepoint_date <- changepoints$changepoint_date
    results$changepoint_score <- changepoints$changepoint_score

    cat("Potential regime change detected at:", as.character(changepoints$changepoint_date), "\n")
    cat("Change score:", round(changepoints$changepoint_score, 3), "\n")

    # Compare Hurst before and after changepoint
    cp_idx <- changepoints$changepoint_index
    before_data <- ts_data[1:cp_idx]
    after_data <- ts_data[(cp_idx+1):length(ts_data)]

    if (length(before_data) > 100 && length(after_data) > 100) {
      hurst_before <- calculate_dfa_hurst(before_data)
      hurst_after <- calculate_dfa_hurst(after_data)

      results$hurst_before_change <- hurst_before
      results$hurst_after_change <- hurst_after

      cat("Hurst before change:", round(hurst_before, 3), "\n")
      cat("Hurst after change:", round(hurst_after, 3), "\n")
      cat("Hurst change:", round(hurst_after - hurst_before, 3), "\n")
    }
  }

  # 4. TIME TREND ANALYSIS
  cat("Checking for trends and correlations...\n")

  # Time trend in the data
  time_numeric <- as.numeric(dates)
  trend_correlation <- cor(time_numeric, ts_data, use = "complete.obs")

  results$time_trend <- trend_correlation
  cat("Overall time trend correlation:", round(trend_correlation, 3), "\n")

  cat("=== END TEMPORAL ANALYSIS ===\n\n")

  return(results)
}

# Main analysis function for each reservoir
analyze_reservoir <- function(data, reservoir_name) {
  cat("\n=== Analyzing", reservoir_name, "===\n")

  # Filter data for specific reservoir
  reservoir_data <- data %>%
    filter(Reservoir == reservoir_name) %>%
    arrange(Date)

  # Check data quality
  n_obs <- nrow(reservoir_data)
  if (n_obs == 0) {
    cat("No data available for", reservoir_name, "\n")
    return(NULL)
  }

  # Calculate years of data with error checking
  years_span <- NA
  tryCatch({
    date_range <- range(reservoir_data$Date, na.rm = TRUE)
    years_span <- as.numeric(difftime(date_range[2], date_range[1], units = "days")) / 365.25

    # Check for valid date calculation
    if (is.na(years_span) || is.infinite(years_span)) {
      years_span <- n_obs / 365.25  # Fallback estimate
      cat("Date calculation issue - using observation count estimate\n")
    }

    cat("Data span:", as.character(date_range[1]), "to", as.character(date_range[2]), "\n")
    cat("Years of data:", round(years_span, 1), "\n")
    cat("Number of observations:", n_obs, "\n")
  }, error = function(e) {
    years_span <- n_obs / 365.25  # Fallback estimate
    cat("Date processing error - using observation count estimate\n")
    cat("Years of data (estimated):", round(years_span, 1), "\n")
    cat("Number of observations:", n_obs, "\n")
  })

  # Minimum requirements for long memory analysis
  min_obs_longmem <- 500  # At least ~1.5 years of daily data
  min_years <- 3          # At least 3 years for meaningful long memory analysis

  if (n_obs < 50) {
    cat("INSUFFICIENT DATA: Less than 50 observations for", reservoir_name, "\n")
    return(NULL)
  }

  if (n_obs < min_obs_longmem) {
    cat("WARNING: Only", n_obs, "observations. Long memory estimates may be unreliable.\n")
  }

  if (!is.na(years_span) && years_span < min_years) {
    cat("WARNING: Only", round(years_span, 1), "years of data. Long memory analysis needs longer series.\n")
  }

  # Extract time series
  ts_values <- reservoir_data$Value
  dates <- reservoir_data$Date

  # Check for zero values (commissioning issues)
  zero_count <- sum(ts_values == 0)
  if (zero_count > 0) {
    cat("Found", zero_count, "zero values (possibly pre-commissioning)\n")
  }

  # Create time series object
  ts_obj <- ts(ts_values, frequency = 365.25)  # Daily data

  # Log-transform to stabilize variance (common in hydrology)
  # Add small constant to handle any remaining zeros
  log_ts <- log(ts_values + 1e-6)

  # DIAGNOSTIC: Check what's happening with seasonality data
  diagnose_seasonality(ts_values, dates, reservoir_name)  # Use raw data first
  diagnose_seasonality(log_ts, dates, paste(reservoir_name, "(log)"))  # Then log data

  # Calculate first differences (for non-stationary data)
  diff_ts <- diff(log_ts)

  # Initialize all result lists
  hurst_results <- list(rs_hurst = NA, dfa_hurst = NA, per_hurst = NA)
  arfima_results <- list(hurst_arfima = NA, d_parameter = NA, aic = NA)
  corr_results <- list(power_law_slope = NA, acf_result = NULL)
  wavelet_results <- list(scaling_exponent = NA)
  stationarity_results <- list(adf_pvalue = NA, kpss_pvalue = NA)
  seasonality_results <- list(seasonal_cv = NA, seasonal_range = NA,
                              peak_month = NA, low_month = NA,
                              seasonal_strength = NA, hurst_deseasonalized = NA)
  temporal_results <- list(changepoint_date = NA, time_trend = NA)

  # Only run detailed analysis if we have sufficient data
  if (n_obs >= min_obs_longmem && (!is.na(years_span) && years_span >= min_years)) {
    cat("Running full long memory analysis...\n")

    tryCatch({
      cat("Calculating Hurst exponents...\n")
      hurst_results <- calculate_hurst_exponent(log_ts)
    }, error = function(e) {
      cat("Hurst calculation failed:", e$message, "\n")
    })

    tryCatch({
      cat("Fitting ARFIMA model...\n")
      arfima_results <- fit_arfima_model(diff_ts)
    }, error = function(e) {
      cat("ARFIMA fitting failed:", e$message, "\n")
    })

    tryCatch({
      cat("Analyzing correlations...\n")
      corr_results <- analyze_correlations(log_ts, max_lag = min(50, floor(n_obs/10)))
    }, error = function(e) {
      cat("Correlation analysis failed:", e$message, "\n")
    })

    tryCatch({
      cat("Performing wavelet analysis...\n")
      wavelet_results <- wavelet_analysis(log_ts)
    }, error = function(e) {
      cat("Wavelet analysis failed:", e$message, "\n")
    })

    tryCatch({
      cat("Analyzing seasonality...\n")
      seasonality_results <- analyze_seasonality_advanced(log_ts, dates)

      # Calculate Hurst on deseasonalized data if available
      if (!is.null(seasonality_results$deseasonalized)) {
        cat("Calculating Hurst on deseasonalized data...\n")
        hurst_deseas <- calculate_deseasonalized_hurst(seasonality_results$deseasonalized)
        seasonality_results$hurst_deseasonalized <- hurst_deseas
      }
    }, error = function(e) {
      cat("Seasonality analysis failed:", e$message, "\n")
    })

    tryCatch({
      cat("Analyzing temporal evolution...\n")
      temporal_results <- analyze_temporal_evolution(log_ts, dates, reservoir_name)
    }, error = function(e) {
      cat("Temporal analysis failed:", e$message, "\n")
    })

    tryCatch({
      cat("Testing stationarity...\n")
      stationarity_results <- test_stationarity(log_ts)
    }, error = function(e) {
      cat("Stationarity tests failed:", e$message, "\n")
      stationarity_results <- list(adf_pvalue = NA, kpss_pvalue = NA)
    })

  } else {
    cat("Skipping detailed long memory analysis due to insufficient data.\n")
    cat("Running basic analysis only...\n")

    tryCatch({
      corr_results <- analyze_correlations(log_ts, max_lag = min(20, floor(n_obs/5)))
    }, error = function(e) {
      cat("Basic correlation analysis failed:", e$message, "\n")
    })

    tryCatch({
      seasonality_results <- analyze_seasonality_advanced(log_ts, dates)
    }, error = function(e) {
      cat("Seasonality analysis failed:", e$message, "\n")
    })
  }

  # Compile results
  results <- list(
    reservoir = reservoir_name,
    n_observations = length(ts_values),
    years_of_data = years_span,
    date_range = c(min(dates), max(dates)),
    sufficient_data = (n_obs >= min_obs_longmem && (!is.na(years_span) && years_span >= min_years)),
    hurst_rs = hurst_results$rs_hurst,
    hurst_dfa = hurst_results$dfa_hurst,
    hurst_per = hurst_results$per_hurst,
    hurst_arfima = arfima_results$hurst_arfima,
    d_parameter = arfima_results$d_parameter,
    acf_power_slope = corr_results$power_law_slope,
    wavelet_scaling = wavelet_results$scaling_exponent,
    adf_pvalue = stationarity_results$adf_pvalue,
    kpss_pvalue = stationarity_results$kpss_pvalue,
    seasonal_cv = seasonality_results$seasonal_cv,
    seasonal_range = seasonality_results$seasonal_range,
    peak_month = seasonality_results$peak_month,
    low_month = seasonality_results$low_month,
    seasonal_strength = seasonality_results$seasonal_strength,
    hurst_deseasonalized = seasonality_results$hurst_deseasonalized,
    changepoint_date = temporal_results$changepoint_date,
    hurst_trend = if(length(temporal_results$rolling_hurst) > 0) cor(as.numeric(temporal_results$rolling_dates), temporal_results$rolling_hurst, use = "complete.obs") else NA,
    time_trend = temporal_results$time_trend,
    raw_data = reservoir_data,
    log_ts = log_ts,
    acf_result = corr_results$acf_result,
    temporal_analysis = temporal_results
  )

  return(results)
}

# Function to create diagnostic plots
create_diagnostic_plots <- function(analysis_results) {
  reservoir_name <- analysis_results$reservoir

  # Set up plotting area
  par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))

  # 1. Time series plot
  plot(analysis_results$raw_data$Date, analysis_results$raw_data$Value,
       type = "l", main = paste(reservoir_name, "- Raw Data"),
       xlab = "Date", ylab = "Reservoir Level")

  # 2. Log-transformed series
  plot(analysis_results$raw_data$Date, analysis_results$log_ts,
       type = "l", main = "Log-transformed Series",
       xlab = "Date", ylab = "log(Level)")

  # 3. Rolling Hurst if available
  if (!is.null(analysis_results$temporal_analysis$rolling_hurst)) {
    plot(analysis_results$temporal_analysis$rolling_dates,
         analysis_results$temporal_analysis$rolling_hurst,
         type = "l", main = "Rolling Hurst Exponent",
         xlab = "Date", ylab = "Hurst Exponent")
    abline(h = 0.5, col = "red", lty = 2)
  } else {
    plot.new()
    text(0.5, 0.5, "Rolling Hurst\nNot Available", cex = 1.2)
  }

  # 4. Autocorrelation function
  if (!is.null(analysis_results$acf_result)) {
    plot(analysis_results$acf_result, main = "Autocorrelation Function")
  } else {
    plot.new()
    text(0.5, 0.5, "ACF not available", cex = 1.2)
  }

  # 5. Distribution of values
  hist(analysis_results$log_ts, main = "Distribution (Log Scale)",
       xlab = "log(Level)", prob = TRUE)
  lines(density(analysis_results$log_ts), col = "red")

  # 6. Q-Q plot for normality
  qqnorm(analysis_results$log_ts, main = "Q-Q Plot")
  qqline(analysis_results$log_ts, col = "red")

  par(mfrow = c(1, 1))
}

# Function to summarize results
summarize_results <- function(all_results) {
  # Helper function to safely extract and round numeric values
  safe_round <- function(values, digits = 3) {
    numeric_values <- as.numeric(values)
    numeric_values[is.na(numeric_values)] <- NA
    return(round(numeric_values, digits))
  }

  # Create summary table
  summary_df <- data.frame(
    Reservoir = sapply(all_results, function(x) ifelse(is.null(x$reservoir), "Unknown", x$reservoir)),
    N_Obs = sapply(all_results, function(x) ifelse(is.null(x$n_observations), NA, x$n_observations)),
    Years_Data = safe_round(sapply(all_results, function(x) ifelse(is.null(x$years_of_data), NA, x$years_of_data)), 1),
    Sufficient_Data = sapply(all_results, function(x) ifelse(is.null(x$sufficient_data), FALSE, x$sufficient_data)),
    Hurst_RS = safe_round(sapply(all_results, function(x) ifelse(is.null(x$hurst_rs), NA, x$hurst_rs))),
    Hurst_DFA = safe_round(sapply(all_results, function(x) ifelse(is.null(x$hurst_dfa), NA, x$hurst_dfa))),
    Hurst_PER = safe_round(sapply(all_results, function(x) ifelse(is.null(x$hurst_per), NA, x$hurst_per))),
    Hurst_ARFIMA = safe_round(sapply(all_results, function(x) ifelse(is.null(x$hurst_arfima), NA, x$hurst_arfima))),
    d_param = safe_round(sapply(all_results, function(x) ifelse(is.null(x$d_parameter), NA, x$d_parameter))),
    ACF_Slope = safe_round(sapply(all_results, function(x) ifelse(is.null(x$acf_power_slope), NA, x$acf_power_slope))),
    ADF_pval = safe_round(sapply(all_results, function(x) ifelse(is.null(x$adf_pvalue), NA, x$adf_pvalue))),
    Seasonal_CV = safe_round(sapply(all_results, function(x) ifelse(is.null(x$seasonal_cv), NA, x$seasonal_cv))),
    Seasonal_Range = safe_round(sapply(all_results, function(x) ifelse(is.null(x$seasonal_range), NA, x$seasonal_range)), 2),
    Peak_Month = sapply(all_results, function(x) ifelse(is.null(x$peak_month), NA, x$peak_month)),
    Low_Month = sapply(all_results, function(x) ifelse(is.null(x$low_month), NA, x$low_month)),
    Seasonal_Strength = safe_round(sapply(all_results, function(x) ifelse(is.null(x$seasonal_strength), NA, x$seasonal_strength))),
    Hurst_Deseasonalized = safe_round(sapply(all_results, function(x) ifelse(is.null(x$hurst_deseasonalized), NA, x$hurst_deseasonalized))),
    Changepoint_Date = sapply(all_results, function(x) ifelse(is.null(x$changepoint_date) || is.na(x$changepoint_date), "None", as.character(x$changepoint_date))),
    Hurst_Trend = safe_round(sapply(all_results, function(x) ifelse(is.null(x$hurst_trend), NA, x$hurst_trend))),
    Time_Trend = safe_round(sapply(all_results, function(x) ifelse(is.null(x$time_trend), NA, x$time_trend))),
    stringsAsFactors = FALSE
  )

  return(summary_df)
}

# Main execution script
main_analysis <- function(csv_file = "water_reserves.csv") {
  cat("Starting Enhanced Long Memory Analysis of Reservoir Data\n")
  cat("=", rep("=", 50), "\n")

  # Read and prepare data
  cat("Reading data...\n")
  data <- prepare_reservoir_data(csv_file)

  if (is.null(data)) {
    cat("ERROR: Could not read data file\n")
    return(NULL)
  }

  # Get unique reservoirs
  reservoirs <- unique(data$Reservoir)
  cat("Found reservoirs:", paste(reservoirs, collapse = ", "), "\n")

  # Create total system reserves time series
  cat("\nCreating total system reserves...\n")
  total_reserves <- data %>%
    group_by(Date) %>%
    summarise(Total_Value = sum(Value, na.rm = TRUE), .groups = "drop") %>%
    arrange(Date)

  cat("Total system data span:", nrow(total_reserves), "observations\n")
  cat("Date range:", as.character(min(total_reserves$Date)), "to", as.character(max(total_reserves$Date)), "\n")

  # Analyze individual reservoirs
  all_results <- list()
  for (reservoir in reservoirs) {
    results <- analyze_reservoir(data, reservoir)
    if (!is.null(results)) {
      all_results[[reservoir]] <- results

      # Create diagnostic plots
      cat("Creating diagnostic plots for", reservoir, "\n")
      create_diagnostic_plots(results)

      # Wait for user to view plots
      if (interactive()) {
        readline(prompt = "Press [enter] to continue to next reservoir...")
      }
    }
  }

  # Analyze total system reserves
  cat("\n=== Analyzing Total System Reserves ===\n")

  # Create a pseudo reservoir data structure for total reserves
  total_data_structure <- data.frame(
    Date = total_reserves$Date,
    Reservoir = "TOTAL_SYSTEM",
    Value = total_reserves$Total_Value
  )

  # Analyze total system using the same framework
  total_results <- analyze_reservoir(total_data_structure, "TOTAL_SYSTEM")

  if (!is.null(total_results)) {
    all_results[["TOTAL_SYSTEM"]] <- total_results

    # Create diagnostic plots for total system
    cat("Creating diagnostic plots for Total System\n")
    create_diagnostic_plots(total_results)

    if (interactive()) {
      readline(prompt = "Press [enter] to continue to summary...")
    }
  }

  # Summarize results
  cat("\n=== ENHANCED SUMMARY RESULTS ===\n")
  summary_table <- summarize_results(all_results)
  print(summary_table)

  # Enhanced interpretation for temporal analysis
  cat("\n=== TEMPORAL INSIGHTS ===\n")
  for (name in names(all_results)) {
    result <- all_results[[name]]
    temporal <- result$temporal_analysis

    if (!is.null(temporal) && length(temporal) > 0) {
      cat("\n", name, ":\n")

      if (!is.null(temporal$decade_stats)) {
        cat("- Decades analyzed:", paste(temporal$decade_stats$decade, collapse = ", "), "\n")
      }

      if (!is.na(result$changepoint_date)) {
        cat("- Regime change detected:", as.character(result$changepoint_date), "\n")
      }

      if (!is.na(result$time_trend)) {
        trend_direction <- ifelse(result$time_trend > 0.1, "increasing",
                                  ifelse(result$time_trend < -0.1, "decreasing", "stable"))
        cat("- Long-term trend:", trend_direction, "(", round(result$time_trend, 3), ")\n")
      }

      if (!is.na(result$hurst_trend)) {
        hurst_direction <- ifelse(result$hurst_trend > 0.1, "increasing persistence",
                                  ifelse(result$hurst_trend < -0.1, "decreasing persistence", "stable persistence"))
        cat("- Hurst evolution:", hurst_direction, "(", round(result$hurst_trend, 3), ")\n")
      }
    }
  }

  cat("\n=== ACTIONABLE INSIGHTS ===\n")
  cat("1. Check Changepoint_Date column for regime changes\n")
  cat("2. Hurst_Trend shows if persistence is increasing/decreasing over time\n")
  cat("3. Time_Trend indicates long-term water level changes\n")
  cat("4. Compare Hurst_DFA vs Hurst_Deseasonalized for seasonal effects\n")

  return(list(results = all_results, summary = summary_table, total_reserves = total_reserves))
}

# Run the analysis
# Uncomment the line below to execute:
results <- main_analysis("data/water_reserves.csv")
