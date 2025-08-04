# Monthly Reservoir Change Anomaly Detection
# Analyzes anomalies in Reserve(now) - Reserve(last_month)

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

# Calculate monthly changes and detect anomalies
analyze_monthly_anomalies <- function(values, dates, reservoir_name) {
  cat("\n=== MONTHLY ANOMALY ANALYSIS:", reservoir_name, "===\n")

  # Create monthly data
  monthly_data <- data.frame(
    Date = dates,
    Value = values,
    Year = year(dates),
    Month = month(dates)
  ) %>%
    group_by(Year, Month) %>%
    summarise(
      Date = as.Date(paste(Year[1], Month[1], "01", sep = "-")),
      End_Value = last(Value),  # End of month value
      .groups = "drop"
    ) %>%
    arrange(Date)

  if (nrow(monthly_data) < 12) {
    cat("Insufficient monthly data\n")
    return(NULL)
  }

  # Calculate monthly changes: Reserve(now) - Reserve(last_month)
  monthly_data$Monthly_Change <- c(NA, diff(monthly_data$End_Value))
  monthly_data$Month_Name <- month(monthly_data$Date, label = TRUE)

  # Remove first row (no previous month)
  monthly_data <- monthly_data[-1, ]

  cat("Monthly data points:", nrow(monthly_data), "\n")
  cat("Date range:", as.character(range(monthly_data$Date)), "\n")

  # Calculate seasonal baselines (expected change for each month)
  seasonal_baselines <- monthly_data %>%
    group_by(Month = month(Date)) %>%
    summarise(
      Month_Name = first(month(Date, label = TRUE)),
      Expected_Change = median(Monthly_Change, na.rm = TRUE),  # Robust to outliers
      Expected_Change_Mean = mean(Monthly_Change, na.rm = TRUE),
      Change_SD = sd(Monthly_Change, na.rm = TRUE),
      N_Observations = sum(!is.na(Monthly_Change)),
      .groups = "drop"
    )

  cat("\nSEASONAL BASELINES:\n")
  print(seasonal_baselines[, c("Month_Name", "Expected_Change", "Change_SD", "N_Observations")])

  # Calculate anomalies: Actual - Expected
  monthly_data$Expected_Change <- seasonal_baselines$Expected_Change[month(monthly_data$Date)]
  monthly_data$Expected_SD <- seasonal_baselines$Change_SD[month(monthly_data$Date)]
  monthly_data$Anomaly <- monthly_data$Monthly_Change - monthly_data$Expected_Change
  monthly_data$Standardized_Anomaly <- monthly_data$Anomaly / monthly_data$Expected_SD

  # Identify significant anomalies (>2 standard deviations)
  threshold <- 2.0
  monthly_data$Significant_Anomaly <- abs(monthly_data$Standardized_Anomaly) > threshold

  # Find extreme anomalies
  extreme_anomalies <- monthly_data[which(monthly_data$Significant_Anomaly & !is.na(monthly_data$Significant_Anomaly)), ]

  cat("\nEXTREME ANOMALIES (>", threshold, "σ):\n")
  if (nrow(extreme_anomalies) > 0) {
    extreme_summary <- extreme_anomalies %>%
      select(Date, Month_Name, Monthly_Change, Expected_Change, Anomaly, Standardized_Anomaly) %>%
      arrange(Date)
    print(extreme_summary)
  } else {
    cat("No extreme anomalies detected\n")
  }

  # Recent period analysis (last 5 years)
  recent_cutoff <- max(monthly_data$Date) - years(5)
  recent_data <- monthly_data[monthly_data$Date >= recent_cutoff, ]

  cat("\nRECENT ANOMALIES (Last 5 years):\n")
  cat("Period:", as.character(range(recent_data$Date)), "\n")

  recent_significant <- recent_data[which(recent_data$Significant_Anomaly & !is.na(recent_data$Significant_Anomaly)), ]
  if (nrow(recent_significant) > 0) {
    recent_summary <- recent_significant %>%
      select(Date, Month_Name, Monthly_Change, Expected_Change, Anomaly, Standardized_Anomaly) %>%
      arrange(desc(abs(Standardized_Anomaly)))
    print(recent_summary)
  } else {
    cat("No significant recent anomalies\n")
  }

  # Detect anomaly clusters (consecutive months of significant anomalies)
  anomaly_clusters <- detect_anomaly_clusters(monthly_data)

  return(list(
    reservoir = reservoir_name,
    monthly_data = monthly_data,
    seasonal_baselines = seasonal_baselines,
    extreme_anomalies = extreme_anomalies,
    recent_anomalies = recent_significant,
    anomaly_clusters = anomaly_clusters
  ))
}

# Detect clusters of consecutive anomalous months
detect_anomaly_clusters <- function(monthly_data) {
  if (nrow(monthly_data) < 3) return(NULL)

  # Create sequence of anomaly indicators
  anomaly_sequence <- monthly_data$Significant_Anomaly
  anomaly_sequence[is.na(anomaly_sequence)] <- FALSE

  # Find runs of consecutive anomalies
  rle_result <- rle(anomaly_sequence)
  anomaly_runs <- which(rle_result$values == TRUE & rle_result$lengths >= 2)  # At least 2 consecutive months

  if (length(anomaly_runs) == 0) {
    cat("\nNo anomaly clusters detected\n")
    return(NULL)
  }

  cat("\nANOMALY CLUSTERS (2+ consecutive months):\n")

  clusters <- list()
  for (i in seq_along(anomaly_runs)) {
    run_idx <- anomaly_runs[i]

    # Find start and end positions
    end_pos <- sum(rle_result$lengths[1:run_idx])
    start_pos <- end_pos - rle_result$lengths[run_idx] + 1

    cluster_data <- monthly_data[start_pos:end_pos, ]

    cluster_info <- list(
      start_date = cluster_data$Date[1],
      end_date = cluster_data$Date[nrow(cluster_data)],
      duration_months = nrow(cluster_data),
      mean_anomaly = mean(cluster_data$Standardized_Anomaly, na.rm = TRUE),
      total_change = sum(cluster_data$Monthly_Change, na.rm = TRUE),
      cluster_data = cluster_data
    )

    clusters[[i]] <- cluster_info

    cat("Cluster", i, ":", as.character(cluster_info$start_date), "to",
        as.character(cluster_info$end_date), "\n")
    cat("  Duration:", cluster_info$duration_months, "months\n")
    cat("  Mean anomaly:", round(cluster_info$mean_anomaly, 2), "σ\n")
    cat("  Total change:", round(cluster_info$total_change / 1e6, 1), "million units\n\n")
  }

  return(clusters)
}

# Create comprehensive anomaly summary
create_anomaly_summary <- function(results_list) {
  if (length(results_list) == 0) {
    cat("No anomaly results to summarize\n")
    return()
  }

  summary_data <- data.frame(
    Reservoir = character(),
    Total_Months = integer(),
    Extreme_Anomalies = integer(),
    Recent_Anomalies = integer(),
    Anomaly_Clusters = integer(),
    Worst_Anomaly_Date = character(),
    Worst_Anomaly_Value = numeric(),
    stringsAsFactors = FALSE
  )

  for (name in names(results_list)) {
    result <- results_list[[name]]

    # Find worst anomaly
    monthly_data <- result$monthly_data
    worst_idx <- which.max(abs(monthly_data$Standardized_Anomaly))

    worst_date <- if(length(worst_idx) > 0) as.character(monthly_data$Date[worst_idx]) else "None"
    worst_value <- if(length(worst_idx) > 0) monthly_data$Standardized_Anomaly[worst_idx] else NA

    summary_data <- rbind(summary_data, data.frame(
      Reservoir = name,
      Total_Months = nrow(monthly_data),
      Extreme_Anomalies = nrow(result$extreme_anomalies),
      Recent_Anomalies = nrow(result$recent_anomalies),
      Anomaly_Clusters = length(result$anomaly_clusters),
      Worst_Anomaly_Date = worst_date,
      Worst_Anomaly_Value = round(worst_value, 2),
      stringsAsFactors = FALSE
    ))
  }

  print(summary_data)

  # Check for 2022-2023 anomalies
  cat("\n=== 2022-2023 DROUGHT ANOMALY CHECK ===\n")
  for (i in 1:nrow(summary_data)) {
    reservoir <- summary_data$Reservoir[i]
    result <- results_list[[reservoir]]

    # Check for anomalies in 2022-2023
    drought_period <- result$monthly_data$Date >= as.Date("2022-01-01") &
      result$monthly_data$Date <= as.Date("2023-12-31")

    drought_anomalies <- result$monthly_data[drought_period &
                                               result$monthly_data$Significant_Anomaly &
                                               !is.na(result$monthly_data$Significant_Anomaly), ]

    if (nrow(drought_anomalies) > 0) {
      cat(reservoir, ": ✓ DETECTED", nrow(drought_anomalies), "anomalies in 2022-2023\n")
      worst_drought <- drought_anomalies[which.max(abs(drought_anomalies$Standardized_Anomaly)), ]
      cat("  Worst:", as.character(worst_drought$Date), "- Anomaly:",
          round(worst_drought$Standardized_Anomaly, 2), "σ\n")
    } else {
      cat(reservoir, ": ✗ No significant anomalies detected in 2022-2023\n")
    }
  }
}

# Main anomaly analysis function
analyze_reservoir_anomalies <- function(csv_file, reservoirs = NULL, anomaly_threshold = 2.0) {
  cat("=== MONTHLY RESERVOIR CHANGE ANOMALY DETECTION ===\n")
  cat("Anomaly threshold:", anomaly_threshold, "standard deviations\n\n")

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

    if (nrow(reservoir_data) > 365 * 2) {  # At least 2 years
      results <- analyze_monthly_anomalies(
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

  # Create summary
  cat("\n=== ANOMALY SUMMARY TABLE ===\n")
  create_anomaly_summary(results_list)

  return(results_list)
}

# Example usage:
# anomaly_results <- analyze_reservoir_anomalies("water_reserves.csv")
#
# # Access specific reservoir results
# mornos_anomalies <- anomaly_results[["Mornos"]]
# View(mornos_anomalies$monthly_data)  # See all monthly changes and anomalies
#
# # Focus on recent period
# recent_data <- mornos_anomalies$monthly_data[mornos_anomalies$monthly_data$Date >= as.Date("2020-01-01"), ]
# print(recent_data[, c("Date", "Monthly_Change", "Expected_Change", "Anomaly", "Standardized_Anomaly")])
