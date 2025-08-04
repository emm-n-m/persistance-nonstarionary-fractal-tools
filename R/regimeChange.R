# Focused Regime Change Detection for Reservoir Data
# Specifically designed to detect the 2022 drought transition

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

# Focused changepoint detection using cumulative sum method
detect_regime_change <- function(values, dates, reservoir_name,
                                 min_segment_years = 2,
                                 recent_focus_years = 10) {

  cat("\n=== REGIME CHANGE ANALYSIS:", reservoir_name, "===\n")

  n <- length(values)
  if (n < 365 * 3) {
    cat("Insufficient data\n")
    return(NULL)
  }

  # Focus on recent years for better detection of 2022 change
  recent_cutoff <- max(dates) - years(recent_focus_years)
  recent_idx <- dates >= recent_cutoff

  cat("Total data points:", n, "\n")
  cat("Recent period:", as.character(recent_cutoff), "to", as.character(max(dates)), "\n")
  cat("Recent data points:", sum(recent_idx), "\n")

  # Method 1: Simple variance change detection
  cat("\nMethod 1: Variance-based detection\n")
  variance_changepoints <- detect_variance_change(values, dates, min_segment_years)

  # Method 2: Mean change detection
  cat("\nMethod 2: Mean-based detection\n")
  mean_changepoints <- detect_mean_change(values, dates, min_segment_years)

  # Method 3: Trend change detection (most relevant for 2022)
  cat("\nMethod 3: Trend-based detection\n")
  trend_changepoints <- detect_trend_change(values, dates, min_segment_years)

  # Method 4: Focus on recent period only
  if (sum(recent_idx) > 365 * 3) {
    cat("\nMethod 4: Recent period focus\n")
    recent_values <- values[recent_idx]
    recent_dates <- dates[recent_idx]
    recent_changepoints <- detect_trend_change(recent_values, recent_dates, 1)
  } else {
    recent_changepoints <- NULL
  }

  # Compile results
  results <- list(
    reservoir = reservoir_name,
    n_total = n,
    date_range = c(min(dates), max(dates)),
    variance_change = variance_changepoints,
    mean_change = mean_changepoints,
    trend_change = trend_changepoints,
    recent_change = recent_changepoints
  )

  # Summary
  cat("\n=== SUMMARY ===\n")
  print_changepoint_summary(results)

  return(results)
}

# Detect changes in variance (regime shifts often show in variance)
detect_variance_change <- function(values, dates, min_years) {
  n <- length(values)
  min_segment <- floor(min_years * 365.25)

  # Test potential changepoints
  test_points <- seq(min_segment, n - min_segment, by = 30)  # Test every month

  if (length(test_points) < 2) return(NULL)

  scores <- numeric(length(test_points))

  for (i in seq_along(test_points)) {
    cp <- test_points[i]

    before_var <- var(values[1:cp], na.rm = TRUE)
    after_var <- var(values[(cp+1):n], na.rm = TRUE)

    # Score based on ratio of variances
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

  if (best_score > 0.3) {  # Threshold for significance
    result <- list(
      changepoint_date = dates[best_cp],
      changepoint_index = best_cp,
      score = best_score,
      before_var = var(values[1:best_cp], na.rm = TRUE),
      after_var = var(values[(best_cp+1):n], na.rm = TRUE)
    )

    cat("Variance change detected:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 3), "\n")
    cat("Before variance:", round(result$before_var, 0),
        "After variance:", round(result$after_var, 0), "\n")

    return(result)
  } else {
    cat("No significant variance change detected\n")
    return(NULL)
  }
}

# Detect changes in mean level
detect_mean_change <- function(values, dates, min_years) {
  n <- length(values)
  min_segment <- floor(min_years * 365.25)

  test_points <- seq(min_segment, n - min_segment, by = 30)

  if (length(test_points) < 2) return(NULL)

  scores <- numeric(length(test_points))

  for (i in seq_along(test_points)) {
    cp <- test_points[i]

    before_mean <- mean(values[1:cp], na.rm = TRUE)
    after_mean <- mean(values[(cp+1):n], na.rm = TRUE)

    # Score based on relative change
    if (before_mean > 0) {
      scores[i] <- abs((after_mean - before_mean) / before_mean)
    } else {
      scores[i] <- 0
    }
  }

  max_idx <- which.max(scores)
  best_cp <- test_points[max_idx]
  best_score <- scores[max_idx]

  if (best_score > 0.1) {  # 10% change threshold
    result <- list(
      changepoint_date = dates[best_cp],
      changepoint_index = best_cp,
      score = best_score,
      before_mean = mean(values[1:best_cp], na.rm = TRUE),
      after_mean = mean(values[(best_cp+1):n], na.rm = TRUE)
    )

    cat("Mean change detected:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 3), "\n")
    cat("Before mean:", round(result$before_mean, 0),
        "After mean:", round(result$after_mean, 0), "\n")

    return(result)
  } else {
    cat("No significant mean change detected\n")
    return(NULL)
  }
}

# Detect changes in trend direction (most important for 2022)
detect_trend_change <- function(values, dates, min_years) {
  n <- length(values)
  min_segment <- floor(min_years * 365.25)

  # Convert dates to numeric for trend calculation
  dates_numeric <- as.numeric(dates)

  test_points <- seq(min_segment, n - min_segment, by = 30)

  if (length(test_points) < 2) return(NULL)

  scores <- numeric(length(test_points))
  trends_before <- numeric(length(test_points))
  trends_after <- numeric(length(test_points))

  for (i in seq_along(test_points)) {
    cp <- test_points[i]

    # Calculate trends before and after
    before_idx <- 1:cp
    after_idx <- (cp+1):n

    # Linear regression for trends
    if (length(before_idx) > 10 && length(after_idx) > 10) {
      trend_before <- coef(lm(values[before_idx] ~ dates_numeric[before_idx]))[2]
      trend_after <- coef(lm(values[after_idx] ~ dates_numeric[after_idx]))[2]

      trends_before[i] <- trend_before
      trends_after[i] <- trend_after

      # Score based on trend direction change
      if (sign(trend_before) != sign(trend_after)) {
        scores[i] <- abs(trend_after - trend_before) / sd(values, na.rm = TRUE)
      } else {
        scores[i] <- abs(trend_after - trend_before) / sd(values, na.rm = TRUE) * 0.5
      }
    } else {
      scores[i] <- 0
    }
  }

  max_idx <- which.max(scores)
  best_cp <- test_points[max_idx]
  best_score <- scores[max_idx]

  if (best_score > 0.001) {  # Lower threshold for trend changes
    result <- list(
      changepoint_date = dates[best_cp],
      changepoint_index = best_cp,
      score = best_score,
      trend_before = trends_before[max_idx],
      trend_after = trends_after[max_idx],
      trend_change = trends_after[max_idx] - trends_before[max_idx]
    )

    cat("Trend change detected:", as.character(result$changepoint_date),
        "- Score:", round(best_score, 6), "\n")
    cat("Trend before:", round(result$trend_before * 365, 0), "per year\n")
    cat("Trend after:", round(result$trend_after * 365, 0), "per year\n")
    cat("Trend change:", round(result$trend_change * 365, 0), "per year\n")

    return(result)
  } else {
    cat("No significant trend change detected\n")
    return(NULL)
  }
}

# Print summary of all changepoint methods
print_changepoint_summary <- function(results) {
  cat("Reservoir:", results$reservoir, "\n")
  cat("Data period:", as.character(results$date_range[1]), "to",
      as.character(results$date_range[2]), "\n\n")

  methods <- c("variance_change", "mean_change", "trend_change", "recent_change")
  method_names <- c("Variance Change", "Mean Change", "Trend Change", "Recent Focus")

  cat("Changepoint Detection Summary:\n")
  cat(paste(rep("=", 40), collapse = ""), "\n")

  for (i in seq_along(methods)) {
    method <- methods[i]
    name <- method_names[i]

    if (!is.null(results[[method]])) {
      cat(sprintf("%-15s: %s (score: %.3f)\n",
                  name,
                  as.character(results[[method]]$changepoint_date),
                  results[[method]]$score))
    } else {
      cat(sprintf("%-15s: No change detected\n", name))
    }
  }
  cat("\n")
}

# Main analysis function
analyze_regime_changes <- function(csv_file, reservoirs = NULL) {
  cat("=== FOCUSED REGIME CHANGE DETECTION ===\n")
  cat("Target: 2022 drought transition\n\n")

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

    if (nrow(reservoir_data) > 1000) {  # At least ~3 years of data
      results <- detect_regime_change(
        values = reservoir_data$Value,
        dates = reservoir_data$Date,
        reservoir_name = reservoir
      )

      if (!is.null(results)) {
        results_list[[reservoir]] <- results
      }
    } else {
      cat("Skipping", reservoir, "- insufficient data\n")
    }
  }

  # Create summary table
  cat("\n=== OVERALL SUMMARY ===\n")
  create_summary_table(results_list)

  return(results_list)
}

# Create summary table of all detected changes
create_summary_table <- function(results_list) {
  if (length(results_list) == 0) {
    cat("No regime changes detected\n")
    return()
  }

  summary_data <- data.frame(
    Reservoir = character(),
    Variance_Change = character(),
    Mean_Change = character(),
    Trend_Change = character(),
    Recent_Change = character(),
    stringsAsFactors = FALSE
  )

  for (name in names(results_list)) {
    result <- results_list[[name]]

    variance_date <- if (!is.null(result$variance_change)) as.character(result$variance_change$changepoint_date) else "None"
    mean_date <- if (!is.null(result$mean_change)) as.character(result$mean_change$changepoint_date) else "None"
    trend_date <- if (!is.null(result$trend_change)) as.character(result$trend_change$changepoint_date) else "None"
    recent_date <- if (!is.null(result$recent_change)) as.character(result$recent_change$changepoint_date) else "None"

    summary_data <- rbind(summary_data, data.frame(
      Reservoir = name,
      Variance_Change = variance_date,
      Mean_Change = mean_date,
      Trend_Change = trend_date,
      Recent_Change = recent_date,
      stringsAsFactors = FALSE
    ))
  }

  print(summary_data)

  # Check for 2022 detection
  cat("\n=== 2022 DETECTION CHECK ===\n")
  for (i in 1:nrow(summary_data)) {
    reservoir <- summary_data$Reservoir[i]
    dates <- c(summary_data$Variance_Change[i], summary_data$Mean_Change[i],
               summary_data$Trend_Change[i], summary_data$Recent_Change[i])

    # Check if any date is in 2022
    year_2022_detected <- any(grepl("2022", dates))
    year_2021_detected <- any(grepl("2021", dates))
    year_2023_detected <- any(grepl("2023", dates))

    if (year_2022_detected || year_2021_detected || year_2023_detected) {
      cat(reservoir, ": ✓ DETECTED recent regime change\n")
    } else {
      cat(reservoir, ": ✗ MISSED recent regime change\n")
    }
  }
}

# Example usage:
# results <- analyze_regime_changes("water_reserves.csv")
#
# # Or for specific reservoirs:
# results <- analyze_regime_changes("water_reserves.csv", c("Mornos", "Iliki"))
