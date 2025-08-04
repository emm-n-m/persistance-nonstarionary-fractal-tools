# Reservoir Velocity Analysis - Rate of Change Detection
# Analyzes filling/emptying velocities and their evolution over time

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

# Calculate velocity (rate of change) metrics
calculate_velocity_metrics <- function(values, dates, reservoir_name) {
  cat("\n=== VELOCITY ANALYSIS:", reservoir_name, "===\n")

  n <- length(values)
  if (n < 10) {
    cat("Insufficient data for velocity analysis\n")
    return(NULL)
  }

  # Calculate time differences (in days)
  time_diffs <- as.numeric(diff(dates))

  # Calculate value differences (absolute changes)
  value_diffs <- diff(values)

  # Calculate velocities (rate of change per day)
  velocities <- value_diffs / time_diffs

  # Remove infinite or NA velocities
  valid_idx <- is.finite(velocities)
  velocities <- velocities[valid_idx]
  velocity_dates <- dates[-1][valid_idx]  # Dates correspond to end of interval

  if (length(velocities) < 5) {
    cat("Insufficient valid velocities\n")
    return(NULL)
  }

  cat("Valid velocity measurements:", length(velocities), "\n")
  cat("Date range:", as.character(range(velocity_dates)), "\n")

  # Basic velocity statistics
  pos_velocities <- velocities[velocities > 0]  # Filling
  neg_velocities <- velocities[velocities < 0]  # Emptying

  results <- list(
    reservoir = reservoir_name,
    dates = velocity_dates,
    velocities = velocities,

    # Basic stats
    mean_velocity = mean(velocities, na.rm = TRUE),
    median_velocity = median(velocities, na.rm = TRUE),
    velocity_range = range(velocities, na.rm = TRUE),
    velocity_sd = sd(velocities, na.rm = TRUE),

    # Filling statistics
    n_filling_days = length(pos_velocities),
    mean_filling_velocity = if(length(pos_velocities) > 0) mean(pos_velocities) else NA,
    max_filling_velocity = if(length(pos_velocities) > 0) max(pos_velocities) else NA,

    # Emptying statistics
    n_emptying_days = length(neg_velocities),
    mean_emptying_velocity = if(length(neg_velocities) > 0) mean(neg_velocities) else NA,
    max_emptying_velocity = if(length(neg_velocities) > 0) min(neg_velocities) else NA,  # Most negative

    # Balance
    filling_fraction = length(pos_velocities) / length(velocities),
    emptying_fraction = length(neg_velocities) / length(velocities)
  )

  # Print basic summary
  cat("Mean velocity:", round(results$mean_velocity, 0), "units/day\n")
  cat("Velocity range:", round(results$velocity_range, 0), "\n")
  cat("Filling days:", results$n_filling_days, "(", round(results$filling_fraction * 100, 1), "%)\n")
  cat("Emptying days:", results$n_emptying_days, "(", round(results$emptying_fraction * 100, 1), "%)\n")

  if (!is.na(results$mean_filling_velocity)) {
    cat("Mean filling rate:", round(results$mean_filling_velocity, 0), "units/day\n")
    cat("Max filling rate:", round(results$max_filling_velocity, 0), "units/day\n")
  }

  if (!is.na(results$mean_emptying_velocity)) {
    cat("Mean emptying rate:", round(results$mean_emptying_velocity, 0), "units/day\n")
    cat("Max emptying rate:", round(results$max_emptying_velocity, 0), "units/day\n")
  }

  return(results)
}

# Detect velocity regime changes
detect_velocity_regimes <- function(velocity_results, window_days = 365, step_days = 30) {
  if (is.null(velocity_results)) return(NULL)

  velocities <- velocity_results$velocities
  dates <- velocity_results$dates
  n <- length(velocities)

  cat("\nDetecting velocity regime changes...\n")

  if (n < window_days * 2) {
    cat("Insufficient data for velocity regime analysis\n")
    return(NULL)
  }

  # Calculate rolling velocity metrics
  n_windows <- floor((n - window_days) / step_days) + 1

  rolling_mean_vel <- numeric(n_windows)
  rolling_filling_frac <- numeric(n_windows)
  rolling_max_fill <- numeric(n_windows)
  rolling_max_empty <- numeric(n_windows)
  rolling_dates <- as.Date(numeric(n_windows), origin = "1970-01-01")

  for (i in 1:n_windows) {
    start_idx <- (i - 1) * step_days + 1
    end_idx <- start_idx + window_days - 1

    if (end_idx > n) break

    window_velocities <- velocities[start_idx:end_idx]
    window_dates <- dates[start_idx:end_idx]

    # Calculate window metrics
    rolling_mean_vel[i] <- mean(window_velocities, na.rm = TRUE)
    rolling_filling_frac[i] <- sum(window_velocities > 0) / length(window_velocities)

    pos_vel <- window_velocities[window_velocities > 0]
    neg_vel <- window_velocities[window_velocities < 0]

    rolling_max_fill[i] <- if(length(pos_vel) > 0) max(pos_vel) else 0
    rolling_max_empty[i] <- if(length(neg_vel) > 0) min(neg_vel) else 0

    rolling_dates[i] <- window_dates[floor(length(window_dates)/2)]
  }

  # Remove incomplete windows
  valid_windows <- !is.na(rolling_mean_vel)
  rolling_mean_vel <- rolling_mean_vel[valid_windows]
  rolling_filling_frac <- rolling_filling_frac[valid_windows]
  rolling_max_fill <- rolling_max_fill[valid_windows]
  rolling_max_empty <- rolling_max_empty[valid_windows]
  rolling_dates <- rolling_dates[valid_windows]

  cat("Rolling velocity windows:", length(rolling_mean_vel), "\n")
  cat("Analysis period:", as.character(range(rolling_dates)), "\n")

  # Detect regime changes in velocity patterns
  regime_changes <- detect_velocity_changepoints(
    rolling_mean_vel, rolling_filling_frac, rolling_dates
  )

  return(list(
    rolling_dates = rolling_dates,
    rolling_mean_vel = rolling_mean_vel,
    rolling_filling_frac = rolling_filling_frac,
    rolling_max_fill = rolling_max_fill,
    rolling_max_empty = rolling_max_empty,
    regime_changes = regime_changes
  ))
}

# Detect changepoints in velocity patterns
detect_velocity_changepoints <- function(mean_velocities, filling_fractions, dates) {
  n <- length(mean_velocities)

  if (n < 6) return(NULL)

  cat("Detecting velocity changepoints...\n")

  # Test potential changepoints
  test_indices <- 3:(n-2)  # Avoid edges

  if (length(test_indices) < 2) return(NULL)

  mean_vel_scores <- numeric(length(test_indices))
  fill_frac_scores <- numeric(length(test_indices))

  for (i in seq_along(test_indices)) {
    cp_idx <- test_indices[i]

    # Mean velocity change
    before_mean_vel <- mean(mean_velocities[1:cp_idx], na.rm = TRUE)
    after_mean_vel <- mean(mean_velocities[(cp_idx+1):n], na.rm = TRUE)

    vel_change <- abs(after_mean_vel - before_mean_vel)
    pooled_sd_vel <- sqrt((var(mean_velocities[1:cp_idx], na.rm = TRUE) +
                             var(mean_velocities[(cp_idx+1):n], na.rm = TRUE)) / 2)

    mean_vel_scores[i] <- if(pooled_sd_vel > 0) vel_change / pooled_sd_vel else 0

    # Filling fraction change
    before_fill_frac <- mean(filling_fractions[1:cp_idx], na.rm = TRUE)
    after_fill_frac <- mean(filling_fractions[(cp_idx+1):n], na.rm = TRUE)

    fill_frac_scores[i] <- abs(after_fill_frac - before_fill_frac)
  }

  # Find most significant changes
  best_vel_idx <- which.max(mean_vel_scores)
  best_fill_idx <- which.max(fill_frac_scores)

  results <- list()

  # Velocity regime change
  if (length(mean_vel_scores) > 0 && max(mean_vel_scores, na.rm = TRUE) > 1.0) {
    cp_idx <- test_indices[best_vel_idx]

    before_vel <- mean(mean_velocities[1:cp_idx], na.rm = TRUE)
    after_vel <- mean(mean_velocities[(cp_idx+1):n], na.rm = TRUE)

    results$velocity_change <- list(
      changepoint_date = dates[cp_idx],
      score = max(mean_vel_scores, na.rm = TRUE),
      before_velocity = before_vel,
      after_velocity = after_vel,
      velocity_change = after_vel - before_vel
    )

    cat("Velocity regime change:", as.character(dates[cp_idx]), "\n")
    cat("Before velocity:", round(before_vel, 0), "After velocity:", round(after_vel, 0), "\n")
  }

  # Filling fraction regime change
  if (length(fill_frac_scores) > 0 && max(fill_frac_scores, na.rm = TRUE) > 0.2) {
    cp_idx <- test_indices[best_fill_idx]

    before_frac <- mean(filling_fractions[1:cp_idx], na.rm = TRUE)
    after_frac <- mean(filling_fractions[(cp_idx+1):n], na.rm = TRUE)

    results$filling_change <- list(
      changepoint_date = dates[cp_idx],
      score = max(fill_frac_scores, na.rm = TRUE),
      before_filling_frac = before_frac,
      after_filling_frac = after_frac,
      filling_change = after_frac - before_frac
    )

    cat("Filling fraction regime change:", as.character(dates[cp_idx]), "\n")
    cat("Before filling fraction:", round(before_frac, 3), "After filling fraction:", round(after_frac, 3), "\n")
  }

  return(results)
}

# Create velocity summary table
create_velocity_summary <- function(results_list) {
  if (length(results_list) == 0) {
    cat("No velocity results to summarize\n")
    return()
  }

  summary_data <- data.frame(
    Reservoir = character(),
    Mean_Velocity = numeric(),
    Filling_Rate = numeric(),
    Emptying_Rate = numeric(),
    Filling_Fraction = numeric(),
    Max_Fill_Rate = numeric(),
    Max_Empty_Rate = numeric(),
    Velocity_Change_Date = character(),
    Filling_Change_Date = character(),
    stringsAsFactors = FALSE
  )

  for (name in names(results_list)) {
    result <- results_list[[name]]
    if (is.null(result$velocity_metrics)) next

    vm <- result$velocity_metrics

    # Regime change dates
    vel_change_date <- "None"
    fill_change_date <- "None"

    if (!is.null(result$regime_analysis$regime_changes)) {
      rc <- result$regime_analysis$regime_changes
      if (!is.null(rc$velocity_change)) {
        vel_change_date <- as.character(rc$velocity_change$changepoint_date)
      }
      if (!is.null(rc$filling_change)) {
        fill_change_date <- as.character(rc$filling_change$changepoint_date)
      }
    }

    summary_data <- rbind(summary_data, data.frame(
      Reservoir = name,
      Mean_Velocity = round(vm$mean_velocity, 0),
      Filling_Rate = round(ifelse(is.na(vm$mean_filling_velocity), 0, vm$mean_filling_velocity), 0),
      Emptying_Rate = round(ifelse(is.na(vm$mean_emptying_velocity), 0, vm$mean_emptying_velocity), 0),
      Filling_Fraction = round(vm$filling_fraction, 3),
      Max_Fill_Rate = round(ifelse(is.na(vm$max_filling_velocity), 0, vm$max_filling_velocity), 0),
      Max_Empty_Rate = round(ifelse(is.na(vm$max_emptying_velocity), 0, vm$max_emptying_velocity), 0),
      Velocity_Change_Date = vel_change_date,
      Filling_Change_Date = fill_change_date,
      stringsAsFactors = FALSE
    ))
  }

  print(summary_data)

  # Check for recent velocity changes
  cat("\n=== RECENT VELOCITY REGIME CHECK ===\n")
  for (i in 1:nrow(summary_data)) {
    reservoir <- summary_data$Reservoir[i]
    vel_date <- summary_data$Velocity_Change_Date[i]
    fill_date <- summary_data$Filling_Change_Date[i]

    recent_vel <- grepl("202[0-5]", vel_date)
    recent_fill <- grepl("202[0-5]", fill_date)

    if (recent_vel || recent_fill) {
      cat(reservoir, ": ✓ DETECTED recent velocity regime change\n")
      if (recent_vel) cat("  Velocity change:", vel_date, "\n")
      if (recent_fill) cat("  Filling pattern change:", fill_date, "\n")
    } else {
      cat(reservoir, ": ✗ No recent velocity regime change\n")
    }
  }
}

# Extract velocity time series for inspection
extract_velocity_timeseries <- function(velocity_results, start_date = NULL, end_date = NULL) {
  if (is.null(velocity_results)) {
    cat("No velocity results provided\n")
    return(NULL)
  }

  dates <- velocity_results$dates
  velocities <- velocity_results$velocities

  # Create data frame
  velocity_df <- data.frame(
    Date = dates,
    Velocity = velocities,
    Velocity_Sign = ifelse(velocities > 0, "Filling", "Emptying"),
    Abs_Velocity = abs(velocities)
  )

  # Filter by date range if specified
  if (!is.null(start_date)) {
    start_date <- as.Date(start_date)
    velocity_df <- velocity_df[velocity_df$Date >= start_date, ]
  }

  if (!is.null(end_date)) {
    end_date <- as.Date(end_date)
    velocity_df <- velocity_df[velocity_df$Date <= end_date, ]
  }

  return(velocity_df)
}

# Show velocity time series with summary stats
show_velocity_timeseries <- function(velocity_results, reservoir_name,
                                     start_date = NULL, end_date = NULL,
                                     show_top_n = 20) {

  cat("\n=== VELOCITY TIME SERIES:", reservoir_name, "===\n")

  velocity_df <- extract_velocity_timeseries(velocity_results, start_date, end_date)

  if (is.null(velocity_df) || nrow(velocity_df) == 0) {
    cat("No velocity data available for specified period\n")
    return(NULL)
  }

  cat("Period:", as.character(range(velocity_df$Date)), "\n")
  cat("Total observations:", nrow(velocity_df), "\n\n")

  # Show top filling events
  cat("TOP", show_top_n, "FILLING EVENTS:\n")
  top_filling <- velocity_df[velocity_df$Velocity > 0, ] %>%
    arrange(desc(Velocity)) %>%
    head(show_top_n)

  if (nrow(top_filling) > 0) {
    print(top_filling[, c("Date", "Velocity")])
  } else {
    cat("No filling events in this period\n")
  }

  cat("\nTOP", show_top_n, "EMPTYING EVENTS:\n")
  top_emptying <- velocity_df[velocity_df$Velocity < 0, ] %>%
    arrange(Velocity) %>%
    head(show_top_n)

  if (nrow(top_emptying) > 0) {
    print(top_emptying[, c("Date", "Velocity")])
  } else {
    cat("No emptying events in this period\n")
  }

  # Recent period analysis
  cat("\n=== RECENT PERIOD ANALYSIS ===\n")
  recent_cutoff <- max(velocity_df$Date) - years(2)  # Last 2 years
  recent_data <- velocity_df[velocity_df$Date >= recent_cutoff, ]

  if (nrow(recent_data) > 0) {
    cat("Recent period (last 2 years):", as.character(range(recent_data$Date)), "\n")
    cat("Recent mean velocity:", round(mean(recent_data$Velocity), 0), "units/day\n")
    cat("Recent filling fraction:", round(sum(recent_data$Velocity > 0) / nrow(recent_data), 3), "\n")

    cat("\nRecent extreme events:\n")
    recent_extremes <- recent_data[abs(recent_data$Velocity) > quantile(abs(velocity_df$Velocity), 0.95, na.rm = TRUE), ]
    if (nrow(recent_extremes) > 0) {
      print(recent_extremes[order(recent_extremes$Date), c("Date", "Velocity", "Velocity_Sign")])
    } else {
      cat("No recent extreme velocity events\n")
    }
  }

  return(velocity_df)
}

# Plot velocity time series (basic text-based visualization)
plot_velocity_summary <- function(velocity_results, reservoir_name,
                                  recent_years = 5) {

  if (is.null(velocity_results)) return(NULL)

  cat("\n=== VELOCITY PLOT SUMMARY:", reservoir_name, "===\n")

  # Get recent data
  cutoff_date <- max(velocity_results$dates) - years(recent_years)
  recent_idx <- velocity_results$dates >= cutoff_date

  recent_dates <- velocity_results$dates[recent_idx]
  recent_velocities <- velocity_results$velocities[recent_idx]

  cat("Plotting last", recent_years, "years:", as.character(range(recent_dates)), "\n")

  # Create simple text-based plot data
  # Bin by months for visualization
  monthly_data <- data.frame(
    Date = recent_dates,
    Velocity = recent_velocities,
    Year = year(recent_dates),
    Month = month(recent_dates)
  )

  monthly_summary <- monthly_data %>%
    group_by(Year, Month) %>%
    summarise(
      Date = as.Date(paste(Year[1], Month[1], "01", sep = "-")),
      Mean_Velocity = mean(Velocity, na.rm = TRUE),
      Max_Velocity = max(Velocity, na.rm = TRUE),
      Min_Velocity = min(Velocity, na.rm = TRUE),
      Filling_Days = sum(Velocity > 0),
      Emptying_Days = sum(Velocity < 0),
      .groups = "drop"
    )

  cat("\nMONTHLY VELOCITY SUMMARY (Recent", recent_years, "years):\n")
  print(monthly_summary)

  return(monthly_summary)
}
analyze_reservoir_velocities <- function(csv_file, reservoirs = NULL,
                                         window_days = 365, step_days = 30) {
  cat("=== RESERVOIR VELOCITY ANALYSIS ===\n")
  cat("Rolling window:", window_days, "days\n")
  cat("Step size:", step_days, "days\n\n")

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

    if (nrow(reservoir_data) > 100) {
      # Calculate velocity metrics
      velocity_metrics <- calculate_velocity_metrics(
        values = reservoir_data$Value,
        dates = reservoir_data$Date,
        reservoir_name = reservoir
      )

      # Detect velocity regime changes
      if (!is.null(velocity_metrics)) {
        regime_analysis <- detect_velocity_regimes(
          velocity_results = velocity_metrics,
          window_days = window_days,
          step_days = step_days
        )

        results_list[[reservoir]] <- list(
          velocity_metrics = velocity_metrics,
          regime_analysis = regime_analysis
        )
      }
    } else {
      cat("Skipping", reservoir, "- insufficient data\n")
    }
  }

  # Create summary
  cat("\n=== VELOCITY SUMMARY TABLE ===\n")
  create_velocity_summary(results_list)

  return(results_list)
}

# Example usage:
# velocity_results <- analyze_reservoir_velocities("water_reserves.csv")
#
# # Extract velocity time series for a specific reservoir
# iliki_velocities <- velocity_results[["Iliki"]]$velocity_metrics
# velocity_timeseries <- show_velocity_timeseries(iliki_velocities, "Iliki",
#                                                start_date = "2020-01-01")
#
# # Plot recent velocity patterns
# plot_velocity_summary(iliki_velocities, "Iliki", recent_years = 3)
#
# # Get velocity data frame for custom analysis
# velocity_df <- extract_velocity_timeseries(iliki_velocities,
#                                           start_date = "2022-01-01",
#                                           end_date = "2025-01-01")
# head(velocity_df)
