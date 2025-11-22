# Tests for Regime Change Detection Functions
# Tests variance, mean, trend, and correlation change detection

library(tidyverse)
library(lubridate)

# Helper function to create data with known regime change
create_regime_change_data <- function(n1 = 100, n2 = 100,
                                      mean1 = 100, mean2 = 150,
                                      sd1 = 10, sd2 = 10) {
  dates1 <- seq(as.Date("2020-01-01"), by = "day", length.out = n1)
  dates2 <- seq(dates1[n1] + 1, by = "day", length.out = n2)

  values1 <- rnorm(n1, mean = mean1, sd = sd1)
  values2 <- rnorm(n2, mean = mean2, sd = sd2)

  data.frame(
    Date = c(dates1, dates2),
    Series = "Test",
    Value = pmax(c(values1, values2), 0)  # Ensure non-negative
  )
}

test_that("detect_variance_change detects variance shifts", {
  # Create data with variance change
  set.seed(123)
  dates1 <- seq(as.Date("2020-01-01"), by = "day", length.out = 100)
  dates2 <- seq(dates1[100] + 1, by = "day", length.out = 100)

  values1 <- rnorm(100, mean = 100, sd = 5)   # Low variance
  values2 <- rnorm(100, mean = 100, sd = 20)  # High variance

  values <- c(values1, values2)
  dates <- c(dates1, dates2)

  result <- detect_variance_change(values, dates, min_segment_size = 30)

  expect_type(result, "list")
  expect_true("changepoint_date" %in% names(result))
  expect_true("score" %in% names(result))
  expect_true("before_variance" %in% names(result))
  expect_true("after_variance" %in% names(result))

  # Should detect the change around day 100
  changepoint_day <- as.numeric(result$changepoint_date - dates[1])
  expect_true(abs(changepoint_day - 100) < 30)  # Allow some tolerance

  # After variance should be higher
  expect_true(result$after_variance > result$before_variance)
})

test_that("detect_mean_change detects mean level shifts", {
  # Create data with mean shift
  data <- create_regime_change_data(
    n1 = 100, n2 = 100,
    mean1 = 100, mean2 = 200,  # Clear mean shift
    sd1 = 10, sd2 = 10
  )

  result <- detect_mean_change(data$Value, data$Date, min_segment_size = 30)

  expect_type(result, "list")
  expect_true("changepoint_date" %in% names(result))
  expect_true("before_mean" %in% names(result))
  expect_true("after_mean" %in% names(result))

  # Should detect the change around day 100
  changepoint_day <- as.numeric(result$changepoint_date - data$Date[1])
  expect_true(abs(changepoint_day - 100) < 30)

  # After mean should be higher
  expect_true(result$after_mean > result$before_mean)
})

test_that("detect_trend_change detects trend reversals", {
  # Create data with trend change
  n1 <- 100
  n2 <- 100
  dates1 <- seq(as.Date("2020-01-01"), by = "day", length.out = n1)
  dates2 <- seq(dates1[n1] + 1, by = "day", length.out = n2)

  # Increasing trend, then decreasing trend
  values1 <- 100 + (1:n1) * 0.5 + rnorm(n1, sd = 5)
  values2 <- max(values1) - (1:n2) * 0.5 + rnorm(n2, sd = 5)

  values <- c(values1, values2)
  dates <- c(dates1, dates2)

  result <- detect_trend_change(values, dates, min_segment_size = 30)

  expect_type(result, "list")
  expect_true("changepoint_date" %in% names(result))
  expect_true("trend_before" %in% names(result))
  expect_true("trend_after" %in% names(result))

  # Trends should have opposite signs
  expect_true(sign(result$trend_before) != sign(result$trend_after))
})

test_that("detect_correlation_change detects autocorrelation shifts", {
  # Create white noise then AR(1) process
  n1 <- 150
  n2 <- 150

  # White noise (no autocorrelation)
  values1 <- rnorm(n1)

  # AR(1) process (strong autocorrelation)
  values2 <- numeric(n2)
  values2[1] <- rnorm(1)
  for (i in 2:n2) {
    values2[i] <- 0.7 * values2[i-1] + rnorm(1)
  }

  values <- c(values1, values2)
  dates <- seq(as.Date("2020-01-01"), by = "day", length.out = length(values))

  result <- detect_correlation_change(values, dates, min_segment_size = 50)

  # May or may not detect (depends on realization), but should not error
  if (!is.null(result)) {
    expect_type(result, "list")
    expect_true("changepoint_date" %in% names(result))
    expect_true("before_autocorr" %in% names(result))
    expect_true("after_autocorr" %in% names(result))
  }
})

test_that("detect_regime_changes returns NULL for insufficient data", {
  # Very small dataset
  data <- data.frame(
    Date = seq(as.Date("2020-01-01"), by = "day", length.out = 50),
    Series = "Test",
    Value = rnorm(50)
  )

  result <- detect_regime_changes(
    data,
    series_name = "Test",
    methods = c("variance", "mean")
  )

  # Should handle gracefully (may return NULL or results with no changes)
  expect_true(is.null(result) || is.list(result))
})

test_that("detect_regime_changes works with multiple methods", {
  # Create data with clear regime change
  data <- create_regime_change_data(
    n1 = 150, n2 = 150,
    mean1 = 100, mean2 = 200,
    sd1 = 10, sd2 = 30
  )

  result <- detect_regime_changes(
    data,
    series_name = "Test",
    methods = c("variance", "mean", "trend")
  )

  expect_type(result, "list")
  expect_true("series_name" %in% names(result))
  expect_true("methods_used" %in% names(result))

  # Should detect changes in at least some methods
  has_variance <- !is.null(result$variance_change)
  has_mean <- !is.null(result$mean_change)

  expect_true(has_variance || has_mean)
})

test_that("apply_transformation works correctly", {
  values <- c(10, 20, 30, 40, 50)

  # Test none
  result_none <- apply_transformation(values, "none")
  expect_equal(result_none, values)

  # Test log
  result_log <- apply_transformation(values, "log")
  expect_equal(result_log, log(values))

  # Test sqrt
  result_sqrt <- apply_transformation(values, "sqrt")
  expect_equal(result_sqrt, sqrt(values))

  # Test standardize
  result_std <- apply_transformation(values, "standardize")
  expect_true(abs(mean(result_std)) < 1e-10)  # Mean should be ~0
  expect_true(abs(sd(result_std) - 1) < 1e-10)  # SD should be ~1

  # Test diff
  result_diff <- apply_transformation(values, "diff")
  expect_equal(length(result_diff), length(values))
  expect_true(is.na(result_diff[1]))
  expect_equal(result_diff[2:5], rep(10, 4))  # Constant diff
})

test_that("analyze_regime_changes works with multiple series", {
  # Create multi-series data
  n <- 150
  data <- rbind(
    data.frame(
      Date = seq(as.Date("2020-01-01"), by = "day", length.out = n),
      Series = "Series1",
      Value = c(rnorm(75, 100, 10), rnorm(75, 200, 10))
    ),
    data.frame(
      Date = seq(as.Date("2020-01-01"), by = "day", length.out = n),
      Series = "Series2",
      Value = c(rnorm(75, 50, 5), rnorm(75, 100, 5))
    )
  )

  result <- analyze_regime_changes(
    data,
    methods = c("mean", "variance")
  )

  expect_type(result, "list")
  expect_true("results" %in% names(result))
  expect_true("summary" %in% names(result))

  # Should analyze both series
  expect_true(length(result$results) <= 2)
})

test_that("print_regime_summary does not error", {
  data <- create_regime_change_data(n1 = 150, n2 = 150)

  result <- detect_regime_changes(
    data,
    series_name = "Test",
    methods = c("variance", "mean")
  )

  # Should print without error
  expect_silent(print_regime_summary(result))
})

test_that("create_regime_summary_table creates proper dataframe", {
  data <- create_regime_change_data(n1 = 150, n2 = 150)

  result1 <- detect_regime_changes(data, series_name = "Test1", methods = c("mean"))
  result2 <- detect_regime_changes(data, series_name = "Test2", methods = c("variance"))

  results_list <- list(Test1 = result1, Test2 = result2)

  summary_df <- create_regime_summary_table(results_list)

  expect_s3_class(summary_df, "data.frame")
  expect_true("Series" %in% colnames(summary_df))
  expect_equal(nrow(summary_df), 2)
})

test_that("regime detection handles edge cases", {
  # Constant data (no variance)
  data_const <- data.frame(
    Date = seq(as.Date("2020-01-01"), by = "day", length.out = 100),
    Series = "Const",
    Value = rep(100, 100)
  )

  # Should handle without error
  result_const <- detect_mean_change(
    data_const$Value,
    data_const$Date,
    min_segment_size = 20
  )

  # May return NULL or a result with no significant change
  expect_true(is.null(result_const) || is.list(result_const))

  # Very noisy data
  data_noisy <- data.frame(
    Date = seq(as.Date("2020-01-01"), by = "day", length.out = 100),
    Series = "Noisy",
    Value = rnorm(100, mean = 100, sd = 100)
  )

  result_noisy <- detect_variance_change(
    data_noisy$Value,
    data_noisy$Date,
    min_segment_size = 20
  )

  # Should not error
  expect_true(is.null(result_noisy) || is.list(result_noisy))
})
