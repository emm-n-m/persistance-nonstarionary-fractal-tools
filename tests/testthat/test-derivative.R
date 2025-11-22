# Tests for Derivative Analysis Functions
# Tests derivative calculation, regime detection, and time series extraction

library(tidyverse)
library(lubridate)

# Helper function to create test data
create_test_timeseries <- function(n = 100, trend = 0, noise_sd = 1) {
  dates <- seq(as.Date("2020-01-01"), by = "day", length.out = n)
  values <- 100 + trend * (1:n) + rnorm(n, sd = noise_sd)

  data.frame(
    Date = dates,
    Series = "Test",
    Value = pmax(values, 0)  # Ensure non-negative
  )
}

test_that("calculate_derivative_metrics validates input data", {
  # Create test data
  data <- create_test_timeseries(n = 100)

  # Should work with valid data
  result <- calculate_derivative_metrics(
    data,
    series_column = "Series",
    time_column = "Date",
    value_column = "Value",
    series_name = "Test",
    derivative_order = 1
  )

  expect_type(result, "list")
  expect_true("derivatives" %in% names(result))
  expect_true("times" %in% names(result))
})

test_that("calculate_derivative_metrics returns NULL for insufficient data", {
  # Create very small dataset
  data <- create_test_timeseries(n = 5)

  # Should return NULL due to insufficient data
  result <- calculate_derivative_metrics(
    data,
    series_column = "Series",
    time_column = "Date",
    value_column = "Value",
    series_name = "Test",
    derivative_order = 1
  )

  expect_null(result)
})

test_that("calculate_derivative_metrics computes first derivative correctly", {
  # Create data with known trend
  n <- 100
  data <- create_test_timeseries(n = n, trend = 2, noise_sd = 0.1)

  result <- calculate_derivative_metrics(
    data,
    series_column = "Series",
    time_column = "Date",
    value_column = "Value",
    series_name = "Test",
    derivative_order = 1,
    time_unit = "days"
  )

  expect_false(is.null(result))
  expect_equal(result$derivative_order, 1)
  expect_true(is.numeric(result$derivatives))

  # Mean derivative should be close to trend (2 per day)
  # Allow some tolerance due to noise
  expect_true(abs(mean(result$derivatives) - 2) < 1)
})

test_that("calculate_derivative_metrics handles different time units", {
  data <- create_test_timeseries(n = 100, trend = 1)

  # Test with days
  result_days <- calculate_derivative_metrics(
    data,
    series_name = "Test",
    derivative_order = 1,
    time_unit = "days"
  )

  # Test with weeks
  result_weeks <- calculate_derivative_metrics(
    data,
    series_name = "Test",
    derivative_order = 1,
    time_unit = "weeks"
  )

  expect_false(is.null(result_days))
  expect_false(is.null(result_weeks))

  # Derivatives in weeks should be ~7x larger than days
  ratio <- mean(result_weeks$derivatives) / mean(result_days$derivatives)
  expect_true(abs(ratio - 7) < 2)  # Allow some tolerance
})

test_that("calculate_derivative_metrics computes second derivative", {
  # Create data with quadratic trend (constant acceleration)
  n <- 100
  dates <- seq(as.Date("2020-01-01"), by = "day", length.out = n)
  time_numeric <- 1:n
  values <- 100 + 2 * time_numeric + 0.5 * time_numeric^2

  data <- data.frame(
    Date = dates,
    Series = "Test",
    Value = values
  )

  result <- calculate_derivative_metrics(
    data,
    series_name = "Test",
    derivative_order = 2,
    time_unit = "days"
  )

  expect_false(is.null(result))
  expect_equal(result$derivative_order, 2)

  # Second derivative should be approximately constant (= 1.0)
  expect_true(sd(result$derivatives) < 0.5)  # Low variance indicates constant
})

test_that("calculate_derivative_metrics returns proper structure", {
  data <- create_test_timeseries(n = 100)

  result <- calculate_derivative_metrics(
    data,
    series_name = "Test",
    derivative_order = 1
  )

  # Check all required fields
  expect_true("series_name" %in% names(result))
  expect_true("derivative_order" %in% names(result))
  expect_true("derivative_name" %in% names(result))
  expect_true("time_unit" %in% names(result))
  expect_true("times" %in% names(result))
  expect_true("derivatives" %in% names(result))
  expect_true("mean_derivative" %in% names(result))
  expect_true("median_derivative" %in% names(result))
  expect_true("derivative_range" %in% names(result))
  expect_true("positive_fraction" %in% names(result))
  expect_true("negative_fraction" %in% names(result))
})

test_that("calculate_derivative_metrics classifies directions correctly", {
  # Create increasing trend data
  data <- create_test_timeseries(n = 100, trend = 5, noise_sd = 0.1)

  result <- calculate_derivative_metrics(
    data,
    series_name = "Test",
    derivative_order = 1,
    time_unit = "days"
  )

  # Should have mostly positive derivatives (increasing trend)
  expect_true(result$positive_fraction > 0.8)
  expect_true(result$n_positive > result$n_negative)
})

test_that("extract_derivative_timeseries creates proper dataframe", {
  data <- create_test_timeseries(n = 100)

  derivative_results <- calculate_derivative_metrics(
    data,
    series_name = "Test",
    derivative_order = 1
  )

  # Extract timeseries
  df <- extract_derivative_timeseries(derivative_results)

  expect_s3_class(df, "data.frame")
  expect_true("Date" %in% colnames(df))
  expect_true("Derivative" %in% colnames(df))
  expect_true("Direction" %in% colnames(df))
  expect_true("Abs_Derivative" %in% colnames(df))
  expect_equal(nrow(df), length(derivative_results$derivatives))
})

test_that("extract_derivative_timeseries filters by date range", {
  data <- create_test_timeseries(n = 365)  # 1 year of data

  derivative_results <- calculate_derivative_metrics(
    data,
    series_name = "Test",
    derivative_order = 1
  )

  # Extract subset
  df <- extract_derivative_timeseries(
    derivative_results,
    start_date = "2020-06-01",
    end_date = "2020-09-01"
  )

  expect_true(min(df$Date) >= as.Date("2020-06-01"))
  expect_true(max(df$Date) <= as.Date("2020-09-01"))
  expect_true(nrow(df) < length(derivative_results$derivatives))
})

test_that("analyze_derivative_patterns works with multiple series", {
  # Create data with multiple series
  n <- 100
  data <- rbind(
    data.frame(
      Date = seq(as.Date("2020-01-01"), by = "day", length.out = n),
      Series = "Series1",
      Value = 100 + (1:n) * 0.5 + rnorm(n, sd = 5)
    ),
    data.frame(
      Date = seq(as.Date("2020-01-01"), by = "day", length.out = n),
      Series = "Series2",
      Value = 200 + (1:n) * 1.5 + rnorm(n, sd = 5)
    )
  )

  result <- analyze_derivative_patterns(
    data,
    derivative_order = 1,
    time_unit = "days"
  )

  expect_type(result, "list")
  expect_true("results" %in% names(result))
  expect_true("summary" %in% names(result))
  expect_equal(length(result$results), 2)
  expect_true("Series1" %in% names(result$results))
  expect_true("Series2" %in% names(result$results))
})

test_that("convenience functions work correctly", {
  data <- create_test_timeseries(n = 100, trend = 2)

  # Test calculate_first_derivative
  result1 <- calculate_first_derivative(data, series_name = "Test")
  expect_equal(result1$derivative_order, 1)

  # Test calculate_second_derivative
  result2 <- calculate_second_derivative(data, series_name = "Test")
  expect_equal(result2$derivative_order, 2)

  # Test calculate_acceleration
  result3 <- calculate_acceleration(data, series_name = "Test")
  expect_equal(result3$derivative_order, 2)
  expect_equal(result3$direction_labels, c("accelerating", "decelerating"))
})
