# Copyright (c) 2025 [Emmanouil Mavrogiorgis, emm.n.black@gmail.com]
# All rights reserved.
#
# This project was developed with assistance from Claude Code by Anthropic.
# Claude Code is an AI-powered coding assistant (https://www.anthropic.com).
#
# Licensed under MIT License
# See LICENSE file for details

# Transparent Hurst-Kolmogorov Process Simulation

library(tidyverse)

# Method 1: Spectral Synthesis (Koutsoyiannis approach)
# Based on the mathematical foundation in the PDF
simulate_hk_spectral <- function(n, H = 0.7, lambda = 1, alpha = 1,
                                 method = "koutsoyiannis") {

  cat("Simulating HK process using spectral method\n")
  cat("Parameters: n =", n, ", H =", H, ", lambda =", lambda, ", alpha =", alpha, "\n")

  # Frequency domain approach
  # Generate frequencies
  freq <- (1:(n/2)) * 2 * pi / n

  # Power spectral density for HK process
  # S(f) = lambda * (alpha * f)^(-2H-1) for f >> 1/alpha
  psd <- lambda * (alpha * freq)^(-(2*H + 1))

  # Generate complex Gaussian random numbers
  real_part <- rnorm(length(freq))
  imag_part <- rnorm(length(freq))

  # Scale by square root of PSD
  complex_coeff <- (real_part + 1i * imag_part) * sqrt(psd)

  # Create full frequency representation (conjugate symmetry)
  full_fft <- c(0, complex_coeff, 0, rev(Conj(complex_coeff)))

  # Inverse FFT to get time series
  time_series <- Re(fft(full_fft, inverse = TRUE))

  # Normalize and return first n points
  result <- time_series[1:n]
  result <- result - mean(result)  # Remove mean

  return(list(
    timeseries = result,
    method = "spectral_synthesis",
    parameters = list(n = n, H = H, lambda = lambda, alpha = alpha)
  ))
}

# Method 2: Fractional Differencing (ARFIMA approach)
simulate_hk_arfima <- function(n, H = 0.7, ar = 0, ma = 0, sigma = 1) {

  cat("Simulating HK process using ARFIMA method\n")

  # Convert Hurst to fractional differencing parameter
  d <- H - 0.5
  cat("Fractional differencing parameter d =", d, "\n")

  library(fracdiff)

  # Simulate ARFIMA process
  arfima_sim <- fracdiff.sim(n, ar = ar, ma = ma, d = d, sigma = sigma)

  return(list(
    timeseries = as.numeric(arfima_sim$series),
    method = "arfima",
    parameters = list(n = n, H = H, d = d, ar = ar, ma = ma, sigma = sigma)
  ))
}

# Method 3: Wavelet-based synthesis (Abry & Veitch approach)
simulate_hk_wavelet <- function(n, H = 0.7, J = NULL) {

  cat("Simulating HK process using wavelet synthesis\n")

  library(wavelets)

  # Determine number of decomposition levels
  if(is.null(J)) {
    J <- floor(log2(n)) - 2
  }

  cat("Using", J, "wavelet decomposition levels\n")

  # Initialize with white noise
  signal <- rnorm(n)

  # Apply discrete wavelet transform
  wt <- dwt(signal, filter = "haar", n.levels = J)

  # Scale wavelet coefficients according to H
  for(j in 1:J) {
    scale_factor <- 2^(j * (H - 0.5))
    wt@W[[j]] <- wt@W[[j]] * scale_factor
  }

  # Scale approximation coefficients
  wt@V[[J]] <- wt@V[[J]] * 2^(J * (H - 0.5))

  # Inverse wavelet transform
  result <- idwt(wt)

  return(list(
    timeseries = as.numeric(result),
    method = "wavelet_synthesis",
    parameters = list(n = n, H = H, J = J)
  ))
}

# Method 4: Circulant matrix embedding (exact method)
simulate_hk_circulant <- function(n, H = 0.7) {

  cat("Simulating HK process using circulant embedding (exact method)\n")

  # This is the most mathematically rigorous approach
  # Based on exact covariance structure

  # Extend to nearest power of 2 for FFT efficiency
  m <- 2^ceiling(log2(2*n))

  # Generate autocovariance sequence for fBm
  # R(k) = 0.5 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))
  k <- 0:(m-1)
  R <- 0.5 * (abs(k+1)^(2*H) - 2*abs(k)^(2*H) + abs(k-1)^(2*H))

  # Create circulant matrix first row
  circ_row <- c(R[1:(n+1)], R[(n):2])

  # Extend to full size
  if(length(circ_row) < m) {
    circ_row <- c(circ_row, rep(0, m - length(circ_row)))
  }

  # Get eigenvalues via FFT
  eigenvals <- Re(fft(circ_row))

  # Check for non-negative eigenvalues (required for valid covariance)
  if(any(eigenvals < -1e-10)) {
    warning("Some eigenvalues are negative - covariance matrix may not be positive definite")
    eigenvals[eigenvals < 0] <- 0
  }

  # Generate complex Gaussian random variables
  Z <- rnorm(m) + 1i * rnorm(m)
  Z[1] <- rnorm(1)  # First element should be real
  Z[m/2 + 1] <- rnorm(1)  # Middle element should be real

  # Make conjugate symmetric
  for(k in 2:(m/2)) {
    Z[m + 2 - k] <- Conj(Z[k])
  }

  # Scale by square root of eigenvalues
  scaled_Z <- Z * sqrt(eigenvals)

  # Inverse FFT
  result <- Re(fft(scaled_Z, inverse = TRUE)) / sqrt(m)

  # Return first n values
  result <- result[1:n]
  result <- result - mean(result)  # Remove mean

  return(list(
    timeseries = result,
    method = "circulant_embedding",
    parameters = list(n = n, H = H, m = m)
  ))
}

# Validation function to check if simulation reproduces correct H
validate_hk_simulation <- function(sim_result, expected_H) {

  cat("\n=== VALIDATION OF HK SIMULATION ===\n")
  cat("Method:", sim_result$method, "\n")
  cat("Expected H:", expected_H, "\n")

  timeseries <- sim_result$timeseries

  # Method 1: DFA estimation
  dfa_result <- calculate_dfa_hurst(timeseries)  # From your existing code
  cat("DFA estimated H:", round(dfa_result, 3), "\n")

  # Method 2: R/S statistic
  rs_hurst <- calculate_rs_hurst(timeseries)  # From your existing code
  cat("R/S estimated H:", round(rs_hurst, 3), "\n")

  # Method 3: Variance scaling (climacogram)
  climaco_result <- calculate_climacogram(timeseries)
  cat("Climacogram estimated H:", round(climaco_result$hurst, 3), "\n")

  # Check quality
  dfa_error <- abs(dfa_result - expected_H)
  cat("DFA error:", round(dfa_error, 3), "\n")

  if(dfa_error < 0.05) {
    cat("✓ EXCELLENT simulation quality\n")
  } else if(dfa_error < 0.1) {
    cat("✓ GOOD simulation quality\n")
  } else if(dfa_error < 0.15) {
    cat("⚠ FAIR simulation quality\n")
  } else {
    cat("✗ POOR simulation quality - check parameters\n")
  }

  return(list(
    dfa_hurst = dfa_result,
    rs_hurst = rs_hurst,
    climaco_hurst = climaco_result$hurst,
    dfa_error = dfa_error
  ))
}

# Comparison function - test all methods
compare_hk_methods <- function(n = 1000, H = 0.7, seed = 123) {

  set.seed(seed)
  cat("=== COMPARING HK SIMULATION METHODS ===\n")
  cat("Parameters: n =", n, ", H =", H, "\n\n")

  methods <- list(
    spectral = function() simulate_hk_spectral(n, H),
    arfima = function() simulate_hk_arfima(n, H),
    wavelet = function() simulate_hk_wavelet(n, H),
    circulant = function() simulate_hk_circulant(n, H)
  )

  results <- list()
  validations <- list()

  for(method_name in names(methods)) {
    cat("Testing", method_name, "method...\n")

    tryCatch({
      sim_result <- methods[[method_name]]()
      validation <- validate_hk_simulation(sim_result, H)

      results[[method_name]] <- sim_result
      validations[[method_name]] <- validation

    }, error = function(e) {
      cat("ERROR in", method_name, "method:", e$message, "\n")
      results[[method_name]] <- NULL
      validations[[method_name]] <- NULL
    })

    cat("\n")
  }

  # Summary comparison
  cat("=== SUMMARY COMPARISON ===\n")
  for(method_name in names(validations)) {
    if(!is.null(validations[[method_name]])) {
      val <- validations[[method_name]]
      cat(sprintf("%-12s: DFA H = %.3f (error = %.3f)\n",
                  method_name, val$dfa_hurst, val$dfa_error))
    }
  }

  return(list(simulations = results, validations = validations))
}

# Example usage and testing
cat("Transparent HK Simulation Toolkit Loaded!\n")
cat("All methods are based on published mathematical algorithms\n")
cat("No black boxes - you can inspect every line of code!\n\n")

cat("Available functions:\n")
cat("- simulate_hk_spectral(): Frequency domain synthesis\n")
cat("- simulate_hk_arfima(): ARFIMA-based simulation\n")
cat("- simulate_hk_wavelet(): Wavelet-based synthesis\n")
cat("- simulate_hk_circulant(): Exact circulant embedding\n")
cat("- validate_hk_simulation(): Check simulation quality\n")
cat("- compare_hk_methods(): Compare all methods\n\n")

cat("Example usage:\n")
cat("results <- compare_hk_methods(n = 1000, H = 0.7)\n")
