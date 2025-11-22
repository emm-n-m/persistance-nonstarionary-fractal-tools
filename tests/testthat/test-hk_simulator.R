# Tests for Hurst-Kolmogorov Process Simulator
# Tests the four simulation methods: spectral, ARFIMA, wavelet, circulant

test_that("generate_fbm_dh validates Hurst parameter", {
  # Should reject H <= 0
  expect_error(
    generate_fbm_dh(100, H = 0),
    "Hurst exponent must be between 0 and 1"
  )

  # Should reject H >= 1
  expect_error(
    generate_fbm_dh(100, H = 1),
    "Hurst exponent must be between 0 and 1"
  )

  # Should reject H < 0
  expect_error(
    generate_fbm_dh(100, H = -0.5),
    "Hurst exponent must be between 0 and 1"
  )

  # Should reject H > 1
  expect_error(
    generate_fbm_dh(100, H = 1.5),
    "Hurst exponent must be between 0 and 1"
  )
})

test_that("generate_fbm_dh produces correct length output", {
  n <- 100
  result <- generate_fbm_dh(n, H = 0.7)

  expect_equal(length(result), n)
  expect_true(is.numeric(result))
  expect_true(all(is.finite(result)))
})

test_that("generate_fbm_dh produces different results with different seeds", {
  n <- 100
  H <- 0.7

  set.seed(123)
  result1 <- generate_fbm_dh(n, H)

  set.seed(456)
  result2 <- generate_fbm_dh(n, H)

  expect_false(identical(result1, result2))
})

test_that("generate_fbm_dh produces reproducible results with same seed", {
  n <- 100
  H <- 0.7

  set.seed(123)
  result1 <- generate_fbm_dh(n, H)

  set.seed(123)
  result2 <- generate_fbm_dh(n, H)

  expect_equal(result1, result2)
})

test_that("simulate_hk_spectral returns proper structure", {
  skip_if_not_installed("fracdiff")

  n <- 100
  H <- 0.7
  result <- simulate_hk_spectral(n, H)

  expect_type(result, "list")
  expect_named(result, c("timeseries", "method", "parameters"))
  expect_equal(result$method, "spectral_synthesis")
  expect_equal(length(result$timeseries), n)
  expect_true(is.numeric(result$timeseries))
})

test_that("simulate_hk_arfima returns proper structure", {
  skip_if_not_installed("fracdiff")

  n <- 100
  H <- 0.7
  result <- simulate_hk_arfima(n, H)

  expect_type(result, "list")
  expect_named(result, c("timeseries", "method", "parameters"))
  expect_equal(result$method, "arfima")
  expect_equal(length(result$timeseries), n)
  expect_true(is.numeric(result$timeseries))

  # Check d parameter conversion
  expected_d <- H - 0.5
  expect_equal(result$parameters$d, expected_d)
})

test_that("simulate_hk_wavelet returns proper structure", {
  skip_if_not_installed("wavelets")

  n <- 128  # Power of 2 for wavelets
  H <- 0.7
  result <- simulate_hk_wavelet(n, H)

  expect_type(result, "list")
  expect_named(result, c("timeseries", "method", "parameters"))
  expect_equal(result$method, "wavelet_synthesis")
  expect_equal(length(result$timeseries), n)
  expect_true(is.numeric(result$timeseries))
})

test_that("simulate_hk_circulant returns proper structure", {
  n <- 100
  H <- 0.7
  result <- simulate_hk_circulant(n, H)

  expect_type(result, "list")
  expect_named(result, c("timeseries", "method", "parameters"))
  expect_equal(result$method, "circulant_embedding")
  expect_equal(length(result$timeseries), n)
  expect_true(is.numeric(result$timeseries))
})

test_that("HK simulations produce different results for different H values", {
  skip_if_not_installed("fracdiff")

  n <- 100
  set.seed(42)

  # Simulate with H = 0.3 (anti-persistent)
  result_low <- simulate_hk_spectral(n, H = 0.3)

  set.seed(42)
  # Simulate with H = 0.9 (highly persistent)
  result_high <- simulate_hk_spectral(n, H = 0.9)

  # The series should be different
  expect_false(identical(result_low$timeseries, result_high$timeseries))

  # Basic sanity check: different H should produce different characteristics
  # (This is a weak test, but better than nothing)
  var_low <- var(result_low$timeseries)
  var_high <- var(result_high$timeseries)

  # Just check they're both positive and finite
  expect_true(is.finite(var_low) && var_low > 0)
  expect_true(is.finite(var_high) && var_high > 0)
})

test_that("HK simulations produce finite values", {
  skip_if_not_installed("fracdiff")

  n <- 100
  H <- 0.7

  # Test all methods
  result_spectral <- simulate_hk_spectral(n, H)
  result_arfima <- simulate_hk_arfima(n, H)
  result_circulant <- simulate_hk_circulant(n, H)

  expect_true(all(is.finite(result_spectral$timeseries)))
  expect_true(all(is.finite(result_arfima$timeseries)))
  expect_true(all(is.finite(result_circulant$timeseries)))
})

test_that("Spectral method works for edge cases of H", {
  skip_if_not_installed("fracdiff")

  n <- 100

  # Near 0.5 (almost white noise)
  result_mid <- simulate_hk_spectral(n, H = 0.5)
  expect_equal(length(result_mid$timeseries), n)
  expect_true(all(is.finite(result_mid$timeseries)))

  # Near boundaries
  result_low <- simulate_hk_spectral(n, H = 0.1)
  expect_equal(length(result_low$timeseries), n)
  expect_true(all(is.finite(result_low$timeseries)))

  result_high <- simulate_hk_spectral(n, H = 0.95)
  expect_equal(length(result_high$timeseries), n)
  expect_true(all(is.finite(result_high$timeseries)))
})
