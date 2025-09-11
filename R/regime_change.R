# Copyright (c) 2025 [Emmanouil Mavrogiorgis, emm.n.black@gmail.com]
# All rights reserved.
#
# This project was developed with assistance from Claude Code by Anthropic.
# Claude Code is an AI-powered coding assistant (https://www.anthropic.com).
#
# Licensed under MIT License
# See LICENSE file for details

# Generalized Time Series Regime Change Detection - Pure Analytics
# Detects changes in statistical properties: mean, variance, trend, correlation structure
# Operates on pre-loaded data from csv_reader_tool.R

library(tidyverse)
library(lubridate)

# Main regime change detection function
detect_regime_changes <- function(data,
                                  series_column = "Series",
                                  time_column = "Date",
                                  value_column = "Value",
                                  series_name = NULL,
                                  methods = c("variance", "mean", "trend", "correlation"),
                                  min_segment_fraction = 0.2,
                                  recent_focus_fraction = 0.5,
                                  transform = "none",
                                  significance_thresholds = list(
                                    variance = 0.3,
                                    mean = 0.1,
                                    trend = 0.001,
                                    correlation = 0.2
                                  )) {

  # Extract series data
  if (!is.null(series_name)) {
    series_data <- data %>%
      filter(!!sym(series_column) == series_name) %>%
      arrange(!!sym(time_column))

    analysis_name <- series_name
    cat("\n=== REGIME CHANGE ANALYSIS:", series_name, "===\n")
  } else {
    # Assume single time series
    series_data <- data %>%
      arrange(!!sym(time_column))

    analysis_name <- "Single Series"
    cat("\n=== REGIME CHANGE ANALYSIS:", analysis_name, "===\n")
  }

  n <- nrow(series_data)
  if (n < 100) {
    cat("Insufficient data for regime change analysis\n")
    return(NULL)
  }

  values <- series_data[[value_column]]
  dates <- series_data[[time_column]]

  # Ensure dates are Date objects
  if (!inherits(dates, "Date")) {
    dates <- as.Date(dates)
  }

  # Apply transformation
  transformed_values <- apply_transformation(values, transform)

  cat("Total data points:", n, "\n")
  cat("Date range:", as.character(range(dates)), "\n")
  cat("Transform applied:", transform, "\n")

  # Calculate minimum segment size
  min_segment_size <- floor(n * min_segment_fraction)
  recent_cutoff_size <- floor(n * recent_focus_fraction)

  cat("Minimum segment size:", min_segment_size, "observations\n")
  cat("Recent focus window:", recent_cutoff_size, "observations\n")

  # Store all results
  results <- list(
    series_name = analysis_name,
    n_total = n,
    date_range = c(min(dates), max(dates)),
    transform = transform,
    methods_used = methods
  )

  # Apply each requested method
  for (method in methods) {
    cat("\n--- Applying", toupper(method), "method ---\n")

    method_result <- switch(method,
                            "variance" = detect_variance_change(transformed_values, dates, min_segment_size,
                                                                significance_thresholds$variance),
                            "mean" = detect_mean_change(transformed_values, dates, min_segment_size,
                                                        significance_thresholds$mean),
                            "trend" = detect_trend_change(transformed_values, dates, min_segment_size,
                                                          significance_thresholds$trend),
                            "correlation" = detect_correlation_change(transformed_values, dates, min_segment_size,
                                                                      significance_thresholds$correlation),
                            NULL
    )

    results[[paste0(method, "_change")]] <- method_result
  }

  # Focus on recent period if enough data
  if (n > recent_cutoff_size * 2) {
    cat("\n--- Recent period analysis ---\n")
    recent_start_idx <- n - recent_cutoff_size + 1
    recent_values <- transformed_values[recent_start_idx:n]
    recent_dates <- dates[recent_start_idx:n]

    # Apply trend detection to recent period (most sensitive to recent changes)
    recent_result <- detect_trend_change(recent_values, recent_dates,
                                         floor(length(recent_values) * 0.2),
                                         significance_thresholds$trend * 0.5)  # Lower threshold for recent

    results$recent_trend_change <- recent_result
  }

  # Summary
  cat("\n=== REGIME CHANGE SUMMARY ===\n")
  print_regime_summary(results)

  return(results)
}

# Apply various transformations to stabilize variance or enhance signals
apply_transformation <- function(values, transform) {
  switch(transform,
         "none" = values,
         "log" = {
           if (any(values <= 0)) {
             log(values - min(values, na.rm = TRUE) + 1e-6)
           } else {
             log(values)
           }
         },
         "sqrt" = sqrt(pmax(values - min(values, na.rm = TRUE), 0)),
         "diff" = c(NA, diff(values)),
         "log_diff" = c(NA, diff(log(pmax(values, 1e-6)))),
         "standardize" = scale(values)[,1],
         "detrend" = {
           time_numeric <- 1:length(values)
           trend_model <- lm(values ~ time_numeric)
           residuals(trend_model)
         },
         values  # default: no transformation
  )
}

# Detect changes in variance (volatility regime changes)
detect_variance_change <- function(values, dates, min_segment_size, threshold = 0.3) {
  n <- length(values)

  # Test potential changepoints
  test_points <- seq(min_segment_size, n - min_segment_size, by = max(1, floor(n/100)))

  if (length(test_points) < 2) {
    cat("Insufficient data points for variance change detection\n")
    return(NULL)
  }

  scores <- numeric(length(test_points))

  for (i in seq_along(test_points)) {
    cp <- test_points[i]

    before_var <- var(values[1:cp], na.rm = TRUE)
    after_var <- var(values[(cp+1):n], na.rm = TRUE)

    # Score based on log ratio of variances
    if (before_var > 0 && after_var > 0) {
      scores[i] <- abs(log(after_var / before_var))
    } else {
      scores[i] <- 0
    }
  }

  # Find most significant change
  max_idx <- which.max(scores)
  best_cp <- test_points[max_idx]
  best_score <- scores[max_idx]

  if (best_score > threshold) {
    before_var <- var(values[1:best_cp], na.rm = TRUE)
    after_var <- var(values[(best_cp+1):n], na.rm = TRUE)

    result <- list(
      changepoint_date = dates[best_cp],
      changepoint_index = best_cp,
      score = best_score,
      before_variance = before_var,
      after_variance = after_var,
      variance_ratio = after_var / before_var,
      change_type = ifelse(after_var > before_var, "increased", "decreased")
    )

    cat("Variance change detected:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 3), "\n")
    cat("Variance", result$change_type, "by factor of", round(result$variance_ratio, 2), "\n")

    return(result)
  } else {
    cat("No significant variance change detected (max score:", round(best_score, 3), ")\n")
    return(NULL)
  }
}

# Detect changes in mean level
detect_mean_change <- function(values, dates, min_segment_size, threshold = 0.1) {
  n <- length(values)

  test_points <- seq(min_segment_size, n - min_segment_size, by = max(1, floor(n/100)))

  if (length(test_points) < 2) {
    cat("Insufficient data points for mean change detection\n")
    return(NULL)
  }

  scores <- numeric(length(test_points))

  overall_sd <- sd(values, na.rm = TRUE)

  for (i in seq_along(test_points)) {
    cp <- test_points[i]

    before_mean <- mean(values[1:cp], na.rm = TRUE)
    after_mean <- mean(values[(cp+1):n], na.rm = TRUE)

    # Score based on standardized mean difference
    if (overall_sd > 0) {
      scores[i] <- abs(after_mean - before_mean) / overall_sd
    } else {
      scores[i] <- 0
    }
  }

  max_idx <- which.max(scores)
  best_cp <- test_points[max_idx]
  best_score <- scores[max_idx]

  if (best_score > threshold) {
    before_mean <- mean(values[1:best_cp], na.rm = TRUE)
    after_mean <- mean(values[(best_cp+1):n], na.rm = TRUE)

    result <- list(
      changepoint_date = dates[best_cp],
      changepoint_index = best_cp,
      score = best_score,
      before_mean = before_mean,
      after_mean = after_mean,
      mean_change = after_mean - before_mean,
      change_type = ifelse(after_mean > before_mean, "increased", "decreased")
    )

    cat("Mean change detected:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 3), "\n")
    cat("Mean", result$change_type, "from", round(before_mean, 3), "to", round(after_mean, 3), "\n")

    return(result)
  } else {
    cat("No significant mean change detected (max score:", round(best_score, 3), ")\n")
    return(NULL)
  }
}

# Detect changes in trend direction or slope
detect_trend_change <- function(values, dates, min_segment_size, threshold = 0.001) {
  n <- length(values)

  # Convert dates to numeric for trend calculation
  dates_numeric <- as.numeric(dates)

  test_points <- seq(min_segment_size, n - min_segment_size, by = max(1, floor(n/100)))

  if (length(test_points) < 2) {
    cat("Insufficient data points for trend change detection\n")
    return(NULL)
  }

  scores <- numeric(length(test_points))
  trends_before <- numeric(length(test_points))
  trends_after <- numeric(length(test_points))

  overall_sd <- sd(values, na.rm = TRUE)

  for (i in seq_along(test_points)) {
    cp <- test_points[i]

    # Calculate trends before and after
    before_idx <- 1:cp
    after_idx <- (cp+1):n

    if (length(before_idx) > 3 && length(after_idx) > 3) {
      # Linear regression for trends
      before_model <- lm(values[before_idx] ~ dates_numeric[before_idx])
      after_model <- lm(values[after_idx] ~ dates_numeric[after_idx])

      trend_before <- coef(before_model)[2]
      trend_after <- coef(after_model)[2]

      trends_before[i] <- trend_before
      trends_after[i] <- trend_after

      # Score based on trend difference normalized by data scale
      if (overall_sd > 0) {
        # Convert daily trends to more interpretable units
        trend_diff <- abs(trend_after - trend_before) * 365.25  # Annualized difference
        scores[i] <- trend_diff / overall_sd
      } else {
        scores[i] <- 0
      }
    } else {
      scores[i] <- 0
    }
  }

  max_idx <- which.max(scores)
  best_cp <- test_points[max_idx]
  best_score <- scores[max_idx]

  if (best_score > threshold) {
    trend_before <- trends_before[max_idx]
    trend_after <- trends_after[max_idx]

    # Determine change type
    change_type <- "modified"
    if (sign(trend_before) != sign(trend_after)) {
      change_type <- "reversed"
    } else if (abs(trend_after) > abs(trend_before)) {
      change_type <- "accelerated"
    } else {
      change_type <- "decelerated"
    }

    result <- list(
      changepoint_date = dates[best_cp],
      changepoint_index = best_cp,
      score = best_score,
      trend_before = trend_before,
      trend_after = trend_after,
      trend_change = trend_after - trend_before,
      annualized_trend_before = trend_before * 365.25,
      annualized_trend_after = trend_after * 365.25,
      change_type = change_type
    )

    cat("Trend change detected:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 6), "\n")
    cat("Trend", change_type, ": from", round(result$annualized_trend_before, 3),
        "to", round(result$annualized_trend_after, 3), "per year\n")

    return(result)
  } else {
    cat("No significant trend change detected (max score:", round(best_score, 6), ")\n")
    return(NULL)
  }
}

# Detect changes in autocorrelation structure
detect_correlation_change <- function(values, dates, min_segment_size, threshold = 0.2) {
  n <- length(values)

  test_points <- seq(min_segment_size, n - min_segment_size, by = max(1, floor(n/50)))

  if (length(test_points) < 2) {
    cat("Insufficient data points for correlation change detection\n")
    return(NULL)
  }

  scores <- numeric(length(test_points))

  for (i in seq_along(test_points)) {
    cp <- test_points[i]

    before_values <- values[1:cp]
    after_values <- values[(cp+1):n]

    # Calculate lag-1 autocorrelation for each segment
    if (length(before_values) > 10 && length(after_values) > 10) {
      before_acf <- tryCatch({
        acf(before_values, lag.max = 1, plot = FALSE)$acf[2]
      }, error = function(e) NA)

      after_acf <- tryCatch({
        acf(after_values, lag.max = 1, plot = FALSE)$acf[2]
      }, error = function(e) NA)

      if (!is.na(before_acf) && !is.na(after_acf)) {
        scores[i] <- abs(after_acf - before_acf)
      } else {
        scores[i] <- 0
      }
    } else {
      scores[i] <- 0
    }
  }

  max_idx <- which.max(scores)
  best_cp <- test_points[max_idx]
  best_score <- scores[max_idx]

  if (best_score > threshold) {
    before_values <- values[1:best_cp]
    after_values <- values[(best_cp+1):n]

    before_acf <- acf(before_values, lag.max = 1, plot = FALSE)$acf[2]
    after_acf <- acf(after_values, lag.max = 1, plot = FALSE)$acf[2]

    result <- list(
      changepoint_date = dates[best_cp],
      changepoint_index = best_cp,
      score = best_score,
      before_autocorr = before_acf,
      after_autocorr = after_acf,
      correlation_change = after_acf - before_acf,
      change_type = ifelse(abs(after_acf) > abs(before_acf), "more correlated", "less correlated")
    )

    cat("Correlation structure change detected:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 3), "\n")
    cat("Lag-1 autocorrelation changed from", round(before_acf, 3), "to", round(after_acf, 3), "\n")

    return(result)
  } else {
    cat("No significant correlation change detected (max score:", round(best_score, 3), ")\n")
    return(NULL)
  }
}

# Print comprehensive regime change summary
print_regime_summary <- function(results) {
  cat("Series:", results$series_name, "\n")
  cat("Data period:", as.character(results$date_range[1]), "to",
      as.character(results$date_range[2]), "\n")
  cat("Transform applied:", results$transform, "\n")
  cat("Total observations:", results$n_total, "\n\n")

  # Check each method for detected changes
  methods <- c("variance_change", "mean_change", "trend_change", "correlation_change", "recent_trend_change")
  method_names <- c("Variance", "Mean", "Trend", "Correlation", "Recent Trend")

  cat("Detected Regime Changes:\n")
  cat(paste(rep("=", 40), collapse = ""), "\n")

  changes_detected <- FALSE

  for (i in seq_along(methods)) {
    method <- methods[i]
    name <- method_names[i]

    if (!is.null(results[[method]])) {
      changes_detected <- TRUE
      change <- results[[method]]

      cat(sprintf("%-15s: %s (score: %.3f, %s)\n",
                  name,
                  as.character(change$changepoint_date),
                  change$score,
                  change$change_type))
    } else {
      cat(sprintf("%-15s: No change detected\n", name))
    }
  }

  if (!changes_detected) {
    cat("No significant regime changes detected in any method.\n")
  }

  cat("\n")
}

# Create summary table for multiple series
create_regime_summary_table <- function(results_list) {
  if (length(results_list) == 0) {
    cat("No regime change results to summarize\n")
    return(NULL)
  }

  summary_data <- data.frame(
    Series = character(),
    Variance_Change = character(),
    Mean_Change = character(),
    Trend_Change = character(),
    Correlation_Change = character(),
    Recent_Change = character(),
    Total_Changes = integer(),
    stringsAsFactors = FALSE
  )

  for (name in names(results_list)) {
    result <- results_list[[name]]

    # Extract change dates
    variance_date <- if (!is.null(result$variance_change)) as.character(result$variance_change$changepoint_date) else "None"
    mean_date <- if (!is.null(result$mean_change)) as.character(result$mean_change$changepoint_date) else "None"
    trend_date <- if (!is.null(result$trend_change)) as.character(result$trend_change$changepoint_date) else "None"
    corr_date <- if (!is.null(result$correlation_change)) as.character(result$correlation_change$changepoint_date) else "None"
    recent_date <- if (!is.null(result$recent_trend_change)) as.character(result$recent_trend_change$changepoint_date) else "None"

    # Count total changes
    total_changes <- sum(c(
      !is.null(result$variance_change),
      !is.null(result$mean_change),
      !is.null(result$trend_change),
      !is.null(result$correlation_change),
      !is.null(result$recent_trend_change)
    ))

    summary_data <- rbind(summary_data, data.frame(
      Series = name,
      Variance_Change = variance_date,
      Mean_Change = mean_date,
      Trend_Change = trend_date,
      Correlation_Change = corr_date,
      Recent_Change = recent_date,
      Total_Changes = total_changes,
      stringsAsFactors = FALSE
    ))
  }

  print(summary_data)
  return(summary_data)
}

# Main analysis function for multiple series
analyze_regime_changes <- function(data,
                                   series_column = "Series",
                                   time_column = "Date",
                                   value_column = "Value",
                                   series_names = NULL,
                                   methods = c("variance", "mean", "trend"),
                                   transform = "none",
                                   recent_focus_years = 5,
                                   ...) {

  cat("=== GENERALIZED REGIME CHANGE DETECTION ===\n")
  cat("Methods:", paste(methods, collapse = ", "), "\n")
  cat("Transform:", transform, "\n")
  cat("Recent focus:", recent_focus_years, "years\n\n")

  # Validate columns
  required_cols <- c(series_column, time_column, value_column)
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  # Get series names
  if (is.null(series_names)) {
    if (series_column %in% names(data)) {
      series_names <- unique(data[[series_column]])
    } else {
      series_names <- "Single_Series"
    }
  }

  results_list <- list()

  for (series_name in series_names) {
    results <- detect_regime_changes(
      data = data,
      series_column = series_column,
      time_column = time_column,
      value_column = value_column,
      series_name = if(series_name == "Single_Series") NULL else series_name,
      methods = methods,
      transform = transform,
      recent_focus_fraction = recent_focus_years / 20,  # Assume ~20 year max span
      ...
    )

    if (!is.null(results)) {
      results_list[[series_name]] <- results
    }
  }

  # Create summary table
  if (length(results_list) > 0) {
    cat("\n=== OVERALL REGIME CHANGE SUMMARY ===\n")
    summary_table <- create_regime_summary_table(results_list)

    return(list(
      results = results_list,
      summary = summary_table,
      analysis_params = list(
        methods = methods,
        transform = transform,
        recent_focus_years = recent_focus_years
      )
    ))
  } else {
    cat("No series had sufficient data for analysis\n")
    return(NULL)
  }
}

# Convenience wrapper for single time series
analyze_single_regime <- function(data,
                                  time_column = "Date",
                                  value_column = "Value",
                                  methods = c("variance", "mean", "trend"),
                                  transform = "none",
                                  ...) {

  # Add dummy series column
  data$Series <- "Single_Series"

  return(analyze_regime_changes(
    data = data,
    series_column = "Series",
    time_column = time_column,
    value_column = value_column,
    series_names = "Single_Series",
    methods = methods,
    transform = transform,
    ...
  ))
}

cat("Generalized Regime Change Detection loaded successfully!\n")
cat("Main functions:\n")
cat("  - analyze_regime_changes(): Multi-series regime change detection\n")
cat("  - analyze_single_regime(): Single time series regime detection\n")
cat("  - detect_regime_changes(): Core detection for one series\n")
cat("Available methods: variance, mean, trend, correlation\n")
cat("Available transforms: none, log, sqrt, diff, log_diff, standardize, detrend\n")

# Example workflow:
# 1. data_obj <- read_csv_dynamic("your_data.csv")
# 2. results <- analyze_regime_changes(data_obj$data,
#                                     methods = c("variance", "mean", "trend"),
#                                     transform = "log")
# 3. View(results$summary)  # See all detected regime changes
