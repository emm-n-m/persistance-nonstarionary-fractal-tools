# Copyright (c) 2025 [Emmanouil Mavrogiorgis, emm.n.black@gmail.com]
# All rights reserved.
#
# This project was developed with assistance from Claude Code by Anthropic.
# Claude Code is an AI-powered coding assistant (https://www.anthropic.com).
#
# Licensed under MIT License
# See LICENSE file for details

# Generalized Time Series Derivative Analysis - Pure Analytics
# Calculates nth-order derivatives and detects regime changes in derivative patterns
# Operates on pre-loaded data from csv_reader_tool.R

library(tidyverse)
library(lubridate)

# Calculate nth-order derivative metrics - generalized
calculate_derivative_metrics <- function(data,
                                         series_column = "Series",
                                         time_column = "Date",
                                         value_column = "Value",
                                         series_name = NULL,
                                         derivative_order = 1,
                                         time_unit = "days",
                                         min_change_threshold = 0,
                                         direction_labels = c("increasing", "decreasing")) {

  # If series_name is provided, filter to that series
  if (!is.null(series_name)) {
    series_data <- data %>%
      filter(!!sym(series_column) == series_name) %>%
      arrange(!!sym(time_column))

    analysis_name <- series_name
    cat("\n=== ", derivative_order,
        ifelse(derivative_order == 1, "st", ifelse(derivative_order == 2, "nd", ifelse(derivative_order == 3, "rd", "th"))),
        " DERIVATIVE ANALYSIS:", series_name, "===\n")
  } else {
    # Assume single time series data
    series_data <- data %>%
      arrange(!!sym(time_column))

    analysis_name <- "Single Series"
    cat("\n=== ", derivative_order,
        ifelse(derivative_order == 1, "st", ifelse(derivative_order == 2, "nd", ifelse(derivative_order == 3, "rd", "th"))),
        " DERIVATIVE ANALYSIS:", analysis_name, "===\n")
  }

  n_obs <- nrow(series_data)
  if (n_obs < (derivative_order + 10)) {
    cat("Insufficient data for", derivative_order, "order derivative analysis\n")
    return(NULL)
  }

  values <- series_data[[value_column]]
  times <- series_data[[time_column]]

  # Ensure times are Date objects
  if (!inherits(times, "Date")) {
    times <- as.Date(times)
  }

  # Calculate time differences in specified units
  time_multiplier <- switch(time_unit,
                            "seconds" = 1 / (24 * 3600),
                            "minutes" = 1 / (24 * 60),
                            "hours" = 1 / 24,
                            "days" = 1,
                            "weeks" = 7,
                            "months" = 30.44,
                            "years" = 365.25,
                            1  # default to days
  )

  # Calculate derivatives iteratively
  current_values <- values
  current_times <- times
  derivative_times_list <- list()

  for (order in 1:derivative_order) {
    time_diffs <- as.numeric(diff(current_times)) / time_multiplier
    value_diffs <- diff(current_values)

    # Calculate derivative
    derivatives <- value_diffs / time_diffs

    # Store the times for this derivative order (corresponds to intervals)
    derivative_times_list[[order]] <- current_times[-1]

    # Update for next iteration
    current_values <- derivatives
    current_times <- current_times[-1]

    # Remove invalid derivatives
    valid_idx <- is.finite(derivatives) & abs(value_diffs) >= min_change_threshold
    current_values <- current_values[valid_idx]
    current_times <- current_times[valid_idx]

    if (length(current_values) < 5) {
      cat("Insufficient valid derivatives at order", order, "\n")
      return(NULL)
    }
  }

  # Final derivatives and times
  final_derivatives <- current_values
  final_times <- current_times

  cat("Valid", derivative_order, "order derivative measurements:", length(final_derivatives), "\n")
  cat("Time range:", as.character(range(final_times)), "\n")
  cat("Derivative unit:", paste0("per_", time_unit, ifelse(derivative_order > 1, paste0("^", derivative_order), "")), "\n")

  # Classify derivatives by direction (sign)
  pos_derivatives <- final_derivatives[final_derivatives > 0]  # Positive direction
  neg_derivatives <- final_derivatives[final_derivatives < 0]  # Negative direction

  # Create derivative labels based on order
  derivative_names <- c("1st derivative (rate)", "2nd derivative (acceleration)",
                        "3rd derivative (jerk)", "4th derivative", "5th derivative")
  derivative_name <- if(derivative_order <= 5) derivative_names[derivative_order] else paste0(derivative_order, "th derivative")

  results <- list(
    series_name = analysis_name,
    derivative_order = derivative_order,
    derivative_name = derivative_name,
    time_unit = time_unit,
    times = final_times,
    derivatives = final_derivatives,
    derivative_times_list = derivative_times_list,  # All intermediate times

    # Basic statistics
    mean_derivative = mean(final_derivatives, na.rm = TRUE),
    median_derivative = median(final_derivatives, na.rm = TRUE),
    derivative_range = range(final_derivatives, na.rm = TRUE),
    derivative_sd = sd(final_derivatives, na.rm = TRUE),

    # Direction-based statistics
    n_positive = length(pos_derivatives),
    n_negative = length(neg_derivatives),
    mean_positive_derivative = if(length(pos_derivatives) > 0) mean(pos_derivatives) else NA,
    mean_negative_derivative = if(length(neg_derivatives) > 0) mean(neg_derivatives) else NA,
    max_positive_derivative = if(length(pos_derivatives) > 0) max(pos_derivatives) else NA,
    max_negative_derivative = if(length(neg_derivatives) > 0) min(neg_derivatives) else NA,

    # Proportions
    positive_fraction = length(pos_derivatives) / length(final_derivatives),
    negative_fraction = length(neg_derivatives) / length(final_derivatives),

    # Labels for interpretation
    direction_labels = direction_labels
  )

  # Print summary
  cat("Mean", derivative_name, ":", round(results$mean_derivative, 6),
      "per", time_unit, ifelse(derivative_order > 1, paste0("^", derivative_order), ""), "\n")
  cat("Range:", round(results$derivative_range, 6), "\n")
  cat("Positive", direction_labels[1], "periods:", results$n_positive,
      "(", round(results$positive_fraction * 100, 1), "%)\n")
  cat("Negative", direction_labels[2], "periods:", results$n_negative,
      "(", round(results$negative_fraction * 100, 1), "%)\n")

  if (!is.na(results$mean_positive_derivative)) {
    cat("Mean positive", derivative_name, ":", round(results$mean_positive_derivative, 6), "\n")
    cat("Max positive", derivative_name, ":", round(results$max_positive_derivative, 6), "\n")
  }

  if (!is.na(results$mean_negative_derivative)) {
    cat("Mean negative", derivative_name, ":", round(results$mean_negative_derivative, 6), "\n")
    cat("Max negative", derivative_name, ":", round(results$max_negative_derivative, 6), "\n")
  }

  return(results)
}

# Convenience functions for common derivative orders
calculate_first_derivative <- function(data, ...) {
  calculate_derivative_metrics(data, derivative_order = 1, ...)
}

calculate_second_derivative <- function(data, ...) {
  calculate_derivative_metrics(data, derivative_order = 2, ...)
}

calculate_acceleration <- function(data, ...) {
  calculate_derivative_metrics(data, derivative_order = 2,
                               direction_labels = c("accelerating", "decelerating"), ...)
}

# Detect derivative regime changes - generalized
detect_derivative_regimes <- function(derivative_results,
                                      window_size = 365,
                                      step_size = 30,
                                      window_unit = "days") {

  if (is.null(derivative_results)) return(NULL)

  derivatives <- derivative_results$derivatives
  times <- derivative_results$times
  n <- length(derivatives)

  cat("\nDetecting", derivative_results$derivative_name, "regime changes...\n")

  # Convert window and step sizes based on unit
  if (window_unit == "observations") {
    window_obs <- window_size
    step_obs <- step_size
  } else {
    # Convert time-based windows to observations
    time_multiplier <- switch(window_unit,
                              "days" = 1,
                              "weeks" = 7,
                              "months" = 30.44,
                              "years" = 365.25,
                              1
    )

    # Estimate observations per time unit based on data density
    total_days <- as.numeric(diff(range(times)))
    obs_per_day <- n / total_days

    window_obs <- floor(window_size * time_multiplier * obs_per_day)
    step_obs <- floor(step_size * time_multiplier * obs_per_day)
  }

  if (n < window_obs * 2) {
    cat("Insufficient data for derivative regime analysis\n")
    cat("Need at least", window_obs * 2, "observations, have", n, "\n")
    return(NULL)
  }

  # Calculate rolling derivative metrics
  n_windows <- floor((n - window_obs) / step_obs) + 1

  rolling_mean_deriv <- numeric(n_windows)
  rolling_positive_frac <- numeric(n_windows)
  rolling_max_positive <- numeric(n_windows)
  rolling_max_negative <- numeric(n_windows)
  rolling_volatility <- numeric(n_windows)
  rolling_times <- as.Date(numeric(n_windows), origin = "1970-01-01")

  for (i in 1:n_windows) {
    start_idx <- (i - 1) * step_obs + 1
    end_idx <- start_idx + window_obs - 1

    if (end_idx > n) break

    window_derivatives <- derivatives[start_idx:end_idx]
    window_times <- times[start_idx:end_idx]

    # Calculate window metrics
    rolling_mean_deriv[i] <- mean(window_derivatives, na.rm = TRUE)
    rolling_positive_frac[i] <- sum(window_derivatives > 0) / length(window_derivatives)
    rolling_volatility[i] <- sd(window_derivatives, na.rm = TRUE)

    pos_deriv <- window_derivatives[window_derivatives > 0]
    neg_deriv <- window_derivatives[window_derivatives < 0]

    rolling_max_positive[i] <- if(length(pos_deriv) > 0) max(pos_deriv) else 0
    rolling_max_negative[i] <- if(length(neg_deriv) > 0) min(neg_deriv) else 0

    rolling_times[i] <- window_times[floor(length(window_times)/2)]
  }

  # Remove incomplete windows
  valid_windows <- !is.na(rolling_mean_deriv)
  rolling_mean_deriv <- rolling_mean_deriv[valid_windows]
  rolling_positive_frac <- rolling_positive_frac[valid_windows]
  rolling_max_positive <- rolling_max_positive[valid_windows]
  rolling_max_negative <- rolling_max_negative[valid_windows]
  rolling_volatility <- rolling_volatility[valid_windows]
  rolling_times <- rolling_times[valid_windows]

  cat("Rolling derivative windows:", length(rolling_mean_deriv), "\n")
  cat("Analysis period:", as.character(range(rolling_times)), "\n")

  # Detect regime changes in derivative patterns
  regime_changes <- detect_derivative_changepoints(
    rolling_mean_deriv, rolling_positive_frac, rolling_volatility, rolling_times,
    derivative_results$derivative_name
  )

  return(list(
    rolling_times = rolling_times,
    rolling_mean_deriv = rolling_mean_deriv,
    rolling_positive_frac = rolling_positive_frac,
    rolling_max_positive = rolling_max_positive,
    rolling_max_negative = rolling_max_negative,
    rolling_volatility = rolling_volatility,
    regime_changes = regime_changes,
    window_obs = window_obs,
    step_obs = step_obs,
    derivative_order = derivative_results$derivative_order
  ))
}

# Detect changepoints in derivative patterns - enhanced
detect_derivative_changepoints <- function(mean_derivatives, positive_fractions,
                                           volatilities, times, derivative_name,
                                           significance_threshold = 1.5) {
  n <- length(mean_derivatives)

  if (n < 6) return(NULL)

  cat("Detecting", derivative_name, "changepoints...\n")

  # Test potential changepoints (avoid edges)
  test_indices <- 3:(n-2)

  if (length(test_indices) < 2) return(NULL)

  mean_deriv_scores <- numeric(length(test_indices))
  positive_frac_scores <- numeric(length(test_indices))
  volatility_scores <- numeric(length(test_indices))

  for (i in seq_along(test_indices)) {
    cp_idx <- test_indices[i]

    # Mean derivative change score
    before_mean_deriv <- mean(mean_derivatives[1:cp_idx], na.rm = TRUE)
    after_mean_deriv <- mean(mean_derivatives[(cp_idx+1):n], na.rm = TRUE)

    deriv_change <- abs(after_mean_deriv - before_mean_deriv)
    pooled_sd_deriv <- sqrt((var(mean_derivatives[1:cp_idx], na.rm = TRUE) +
                               var(mean_derivatives[(cp_idx+1):n], na.rm = TRUE)) / 2)

    mean_deriv_scores[i] <- if(pooled_sd_deriv > 0) deriv_change / pooled_sd_deriv else 0

    # Positive fraction change score
    before_pos_frac <- mean(positive_fractions[1:cp_idx], na.rm = TRUE)
    after_pos_frac <- mean(positive_fractions[(cp_idx+1):n], na.rm = TRUE)

    positive_frac_scores[i] <- abs(after_pos_frac - before_pos_frac)

    # Volatility change score
    before_vol <- mean(volatilities[1:cp_idx], na.rm = TRUE)
    after_vol <- mean(volatilities[(cp_idx+1):n], na.rm = TRUE)

    if (before_vol > 0 && after_vol > 0) {
      volatility_scores[i] <- abs(log(after_vol / before_vol))
    } else {
      volatility_scores[i] <- 0
    }
  }

  results <- list()

  # Mean derivative regime change
  if (length(mean_deriv_scores) > 0 && max(mean_deriv_scores, na.rm = TRUE) > significance_threshold) {
    best_deriv_idx <- which.max(mean_deriv_scores)
    cp_idx <- test_indices[best_deriv_idx]

    before_deriv <- mean(mean_derivatives[1:cp_idx], na.rm = TRUE)
    after_deriv <- mean(mean_derivatives[(cp_idx+1):n], na.rm = TRUE)

    results$derivative_change <- list(
      changepoint_date = times[cp_idx],
      score = max(mean_deriv_scores, na.rm = TRUE),
      before_derivative = before_deriv,
      after_derivative = after_deriv,
      derivative_change = after_deriv - before_deriv
    )

    cat(derivative_name, "regime change:", as.character(times[cp_idx]),
        "- Score:", round(max(mean_deriv_scores, na.rm = TRUE), 2), "\n")
    cat("Before", derivative_name, ":", round(before_deriv, 6),
        "After", derivative_name, ":", round(after_deriv, 6), "\n")
  }

  # Direction regime change
  if (length(positive_frac_scores) > 0 && max(positive_frac_scores, na.rm = TRUE) > 0.2) {
    best_frac_idx <- which.max(positive_frac_scores)
    cp_idx <- test_indices[best_frac_idx]

    before_frac <- mean(positive_fractions[1:cp_idx], na.rm = TRUE)
    after_frac <- mean(positive_fractions[(cp_idx+1):n], na.rm = TRUE)

    results$direction_change <- list(
      changepoint_date = times[cp_idx],
      score = max(positive_frac_scores, na.rm = TRUE),
      before_positive_frac = before_frac,
      after_positive_frac = after_frac,
      direction_change = after_frac - before_frac
    )

    cat("Direction regime change:", as.character(times[cp_idx]),
        "- Score:", round(max(positive_frac_scores, na.rm = TRUE), 3), "\n")
    cat("Before positive fraction:", round(before_frac, 3),
        "After positive fraction:", round(after_frac, 3), "\n")
  }

  # Volatility regime change
  if (length(volatility_scores) > 0 && max(volatility_scores, na.rm = TRUE) > 0.5) {
    best_vol_idx <- which.max(volatility_scores)
    cp_idx <- test_indices[best_vol_idx]

    before_vol <- mean(volatilities[1:cp_idx], na.rm = TRUE)
    after_vol <- mean(volatilities[(cp_idx+1):n], na.rm = TRUE)

    results$volatility_change <- list(
      changepoint_date = times[cp_idx],
      score = max(volatility_scores, na.rm = TRUE),
      before_volatility = before_vol,
      after_volatility = after_vol,
      volatility_ratio = after_vol / before_vol
    )

    cat("Volatility regime change:", as.character(times[cp_idx]),
        "- Score:", round(max(volatility_scores, na.rm = TRUE), 3), "\n")
    cat("Before volatility:", round(before_vol, 6),
        "After volatility:", round(after_vol, 6), "\n")
  }

  return(results)
}

# Extract derivative time series for detailed inspection
extract_derivative_timeseries <- function(derivative_results, start_date = NULL, end_date = NULL) {
  if (is.null(derivative_results)) {
    cat("No derivative results provided\n")
    return(NULL)
  }

  dates <- derivative_results$times
  derivatives <- derivative_results$derivatives
  direction_labels <- derivative_results$direction_labels
  derivative_name <- derivative_results$derivative_name

  # Create data frame
  derivative_df <- data.frame(
    Date = dates,
    Derivative = derivatives,
    Direction = ifelse(derivatives > 0, direction_labels[1], direction_labels[2]),
    Abs_Derivative = abs(derivatives),
    Derivative_Order = derivative_results$derivative_order,
    Derivative_Name = derivative_name,
    stringsAsFactors = FALSE
  )

  # Filter by date range if specified
  if (!is.null(start_date)) {
    start_date <- as.Date(start_date)
    derivative_df <- derivative_df[derivative_df$Date >= start_date, ]
  }

  if (!is.null(end_date)) {
    end_date <- as.Date(end_date)
    derivative_df <- derivative_df[derivative_df$Date <= end_date, ]
  }

  return(derivative_df)
}

# Create generalized derivative summary table
create_derivative_summary <- function(results_list) {
  if (length(results_list) == 0) {
    cat("No derivative results to summarize\n")
    return(NULL)
  }

  summary_data <- data.frame(
    Series = character(),
    Derivative_Order = integer(),
    Mean_Derivative = numeric(),
    Positive_Rate = numeric(),
    Negative_Rate = numeric(),
    Positive_Fraction = numeric(),
    Max_Positive = numeric(),
    Max_Negative = numeric(),
    Derivative_Change_Date = character(),
    Direction_Change_Date = character(),
    Volatility_Change_Date = character(),
    stringsAsFactors = FALSE
  )

  for (name in names(results_list)) {
    result <- results_list[[name]]
    if (is.null(result$derivative_metrics)) next

    dm <- result$derivative_metrics

    # Extract regime change dates
    deriv_change_date <- "None"
    direction_change_date <- "None"
    vol_change_date <- "None"

    if (!is.null(result$regime_analysis$regime_changes)) {
      rc <- result$regime_analysis$regime_changes

      if (!is.null(rc$derivative_change)) {
        deriv_change_date <- as.character(rc$derivative_change$changepoint_date)
      }
      if (!is.null(rc$direction_change)) {
        direction_change_date <- as.character(rc$direction_change$changepoint_date)
      }
      if (!is.null(rc$volatility_change)) {
        vol_change_date <- as.character(rc$volatility_change$changepoint_date)
      }
    }

    summary_data <- rbind(summary_data, data.frame(
      Series = name,
      Derivative_Order = dm$derivative_order,
      Mean_Derivative = round(dm$mean_derivative, 6),
      Positive_Rate = round(ifelse(is.na(dm$mean_positive_derivative), 0, dm$mean_positive_derivative), 6),
      Negative_Rate = round(ifelse(is.na(dm$mean_negative_derivative), 0, dm$mean_negative_derivative), 6),
      Positive_Fraction = round(dm$positive_fraction, 3),
      Max_Positive = round(ifelse(is.na(dm$max_positive_derivative), 0, dm$max_positive_derivative), 6),
      Max_Negative = round(ifelse(is.na(dm$max_negative_derivative), 0, dm$max_negative_derivative), 6),
      Derivative_Change_Date = deriv_change_date,
      Direction_Change_Date = direction_change_date,
      Volatility_Change_Date = vol_change_date,
      stringsAsFactors = FALSE
    ))
  }

  print(summary_data)
  return(summary_data)
}

# Main generalized derivative analysis function - operates on pre-loaded data
analyze_derivative_patterns <- function(data,
                                        series_column = "Series",
                                        time_column = "Date",
                                        value_column = "Value",
                                        series_names = NULL,
                                        derivative_order = 1,
                                        time_unit = "days",
                                        window_size = 365,
                                        window_unit = "days",
                                        step_size = 30,
                                        direction_labels = c("increasing", "decreasing"),
                                        min_change_threshold = 0) {

  derivative_name <- c("1st derivative (rate)", "2nd derivative (acceleration)",
                       "3rd derivative (jerk)", "4th derivative", "5th derivative")[pmin(derivative_order, 5)]
  if (derivative_order > 5) derivative_name <- paste0(derivative_order, "th derivative")

  cat("=== ", toupper(derivative_name), " ANALYSIS ===\n")
  cat("Time unit:", time_unit, "\n")
  cat("Rolling window:", window_size, window_unit, "\n")
  cat("Step size:", step_size, window_unit, "\n\n")

  # Validate required columns
  required_cols <- c(series_column, time_column, value_column)
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  # Get series names if not specified
  if (is.null(series_names)) {
    if (series_column %in% names(data)) {
      series_names <- unique(data[[series_column]])
    } else {
      # Single time series - no series column
      series_names <- "Single_Series"
    }
  }

  results_list <- list()

  for (series_name in series_names) {
    # Calculate derivative metrics
    derivative_metrics <- calculate_derivative_metrics(
      data = data,
      series_column = series_column,
      time_column = time_column,
      value_column = value_column,
      series_name = if(series_name == "Single_Series") NULL else series_name,
      derivative_order = derivative_order,
      time_unit = time_unit,
      min_change_threshold = min_change_threshold,
      direction_labels = direction_labels
    )

    # Detect derivative regime changes
    if (!is.null(derivative_metrics)) {
      regime_analysis <- detect_derivative_regimes(
        derivative_results = derivative_metrics,
        window_size = window_size,
        window_unit = window_unit,
        step_size = step_size
      )

      results_list[[series_name]] <- list(
        derivative_metrics = derivative_metrics,
        regime_analysis = regime_analysis
      )
    }
  }

  # Create summary
  if (length(results_list) > 0) {
    cat("\n=== ", toupper(derivative_name), " SUMMARY ===\n")
    summary_table <- create_derivative_summary(results_list)

    return(list(
      results = results_list,
      summary = summary_table,
      analysis_params = list(
        derivative_order = derivative_order,
        time_unit = time_unit,
        window_size = window_size,
        window_unit = window_unit,
        direction_labels = direction_labels
      )
    ))
  } else {
    cat("No series had sufficient data for analysis\n")
    return(NULL)
  }
}

# Convenience wrapper for single time series (no series column)
analyze_single_derivative <- function(data,
                                      time_column = "Date",
                                      value_column = "Value",
                                      derivative_order = 1,
                                      time_unit = "days",
                                      direction_labels = c("increasing", "decreasing"),
                                      ...) {

  # Add dummy series column for compatibility
  data$Series <- "Single_Series"

  return(analyze_derivative_patterns(
    data = data,
    series_column = "Series",
    time_column = time_column,
    value_column = value_column,
    series_names = "Single_Series",
    derivative_order = derivative_order,
    time_unit = time_unit,
    direction_labels = direction_labels,
    ...
  ))
}

cat("Generalized Derivative Analysis loaded successfully!\n")
cat("Main functions:\n")
cat("  - analyze_derivative_patterns(): Multi-series nth-order derivative analysis\n")
cat("  - analyze_single_derivative(): Single time series derivative analysis\n")
cat("  - calculate_first_derivative(), calculate_second_derivative(), calculate_acceleration()\n")
cat("  - extract_derivative_timeseries(): Extract detailed derivative data\n")
cat("Usage: First load data with csv_reader_tool.R, then analyze\n")

# Example workflow:
# 1. data_obj <- read_csv_dynamic("stock_prices.csv", clean_names = TRUE)
# 2. # First derivative (price velocity)
#    velocity_results <- analyze_derivative_patterns(data_obj$data,
#                                                   derivative_order = 1,
#                                                   direction_labels = c("rising", "falling"))
# 3. # Second derivative (price acceleration)
#    accel_results <- analyze_derivative_patterns(data_obj$data,
#                                                derivative_order = 2,
#                                                direction_labels = c("accelerating", "decelerating"))
# 4. # Extract details
#    velocity_ts <- extract_derivative_timeseries(velocity_results$results$AAPL$derivative_metrics)
