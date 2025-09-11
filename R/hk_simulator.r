# Hurst-Kolmogorov Process Simulator
# Generates synthetic time series with specified Hurst exponent
# Multiple methods: Fractional Brownian Motion, ARFIMA, and Wavelet-based

library(tidyverse)
library(lubridate)
library(fracdiff)

# ============================================================================
# CORE H-K SIMULATION FUNCTIONS
# ============================================================================

#' Generate Fractional Brownian Motion using Davies-Harte method
#' 
#' @param n Length of series
#' @param H Hurst exponent (0.5 < H < 1 for persistent, 0 < H < 0.5 for anti-persistent)
#' @param sigma Standard deviation scaling
#' @return Fractional Brownian Motion increments
generate_fbm_dh <- function(n, H, sigma = 1) {
  
  if (H <= 0 || H >= 1) {
    stop("Hurst exponent must be between 0 and 1")
  }
  
  # For fBm increments (fractional Gaussian noise)
  # Covariance function: R(k) = 0.5 * sigma^2 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))
  
  # Create covariance matrix for first row of circulant matrix
  m <- 2 * n
  covariances <- numeric(m)
  
  # First element
  covariances[1] <- sigma^2
  
  # Positive lags
  for (k in 1:(n-1)) {
    covariances[k + 1] <- 0.5 * sigma^2 * (abs(k + 1)^(2*H) - 2 * abs(k)^(2*H) + abs(k - 1)^(2*H))
  }
  
  # Negative lags (symmetric)
  for (k in 1:(n-1)) {
    covariances[m - k + 1] <- covariances[k + 1]
  }
  
  # Handle the special case at n
  covariances[n + 1] <- 0.5 * sigma^2 * (abs(n)^(2*H) - 2 * abs(n-1)^(2*H) + abs(n-2)^(2*H))
  
  # Generate using FFT method
  # Get eigenvalues via FFT
  eigenvals <- Re(fft(covariances))
  
  # Check for numerical issues
  if (any(eigenvals < -1e-10)) {
    warning("Negative eigenvalues detected, using approximate method")
    eigenvals[eigenvals < 0] <- 0
  }
  
  # Generate complex Gaussian random variables
  z1 <- rnorm(m/2 + 1)
  z2 <- c(0, rnorm(m/2 - 1), 0)
  
  z_complex <- complex(real = z1, imaginary = z2)
  
  # Scale by square root of eigenvalues
  scaled_z <- z_complex * sqrt(pmax(eigenvals[1:(m/2 + 1)], 0))
  
  # Create full vector for inverse FFT
  full_z <- c(scaled_z, Conj(scaled_z[(m/2):2]))
  
  # Inverse FFT and take real part
  result <- Re(fft(full_z, inverse = TRUE)) / m
  
  # Return first n values
  return(result[1:n])
}

#' Simple fractional differencing method (alternative to Davies-Harte)
#' 
#' @param n Length of series  
#' @param H Hurst exponent
#' @param sigma Standard deviation
#' @return fGn sequence
generate_fbm_simple <- function(n, H, sigma = 1) {
  
  # Generate white noise
  white_noise <- rnorm(n, mean = 0, sd = sigma)
  
  # For H ≠ 0.5, apply fractional differencing
  if (abs(H - 0.5) < 1e-10) {
    return(white_noise)  # Standard Brownian motion case
  }
  
  # Fractional differencing parameter
  d <- H - 0.5
  
  # Use fracdiff approach for simulation
  if (abs(d) < 0.4) {  # Stable range for fracdiff
    tryCatch({
      # Generate using fracdiff package
      ar_coef <- numeric(0)  # No AR terms
      ma_coef <- numeric(0)  # No MA terms
      
      # Use fdSim for simulation (if available) or approximate
      weights <- numeric(min(n, 100))
      weights[1] <- 1
      
      for (k in 2:length(weights)) {
        weights[k] <- weights[k-1] * (d + 1 - k) / (k - 1)
      }
      
      # Apply filter
      result <- filter(white_noise, weights, method = "convolution", sides = 1)
      result[is.na(result)] <- white_noise[is.na(result)]
      
      return(as.numeric(result))
      
    }, error = function(e) {
      warning("Fractional differencing failed, using approximation")
      return(white_noise)
    })
  } else {
    # For extreme values, use approximation
    return(white_noise * sqrt(2*H))
  }
}

#' Generate H-K process using wavelet-based method
#' 
#' @param n Length of series
#' @param H Hurst exponent  
#' @param sigma Standard deviation
#' @return Wavelet-based H-K process
generate_hk_wavelet <- function(n, H, sigma = 1) {
  
  # Ensure n is power of 2 for efficiency
  n_pow2 <- 2^ceiling(log2(n))
  
  # Generate coefficients at multiple scales
  max_scale <- floor(log2(n_pow2)) - 1
  
  # Initialize result
  result <- numeric(n_pow2)
  
  # Generate at each scale with appropriate variance scaling
  for (j in 1:max_scale) {
    n_coef <- 2^(max_scale - j + 1)
    
    # Variance scaling for Hurst process
    scale_var <- sigma^2 * (2^j)^(2*H - 1)
    
    # Generate coefficients
    coefficients <- rnorm(n_coef, mean = 0, sd = sqrt(scale_var))
    
    # Simple aggregation (this is a simplified approach)
    # In practice, you'd use proper wavelet reconstruction
    segment_length <- n_pow2 / n_coef
    
    for (k in 1:n_coef) {
      start_idx <- (k-1) * segment_length + 1
      end_idx <- min(k * segment_length, n_pow2)
      result[start_idx:end_idx] <- result[start_idx:end_idx] + coefficients[k]
    }
  }
  
  # Add fine-scale noise
  result <- result + rnorm(n_pow2, mean = 0, sd = sigma * 0.1)
  
  # Return only the requested length
  return(result[1:n])
}

# ============================================================================
# ARFIMA SIMULATION  
# ============================================================================

#' Generate ARFIMA process
#' 
#' @param n Length of series
#' @param ar AR coefficients
#' @param ma MA coefficients  
#' @param d Fractional differencing parameter (d = H - 0.5)
#' @param sigma Innovation variance
#' @return ARFIMA time series
generate_arfima <- function(n, ar = numeric(0), ma = numeric(0), d, sigma = 1) {
  
  tryCatch({
    # Use fracdiff package for ARFIMA simulation
    # Generate longer series and trim to avoid startup effects
    n_sim <- n + 500
    
    arfima_sim <- fracdiff.sim(n = n_sim, ar = ar, ma = ma, d = d, 
                               innov = rnorm(n_sim, sd = sigma))
    
    # Return the last n observations
    return(arfima_sim$series[(n_sim - n + 1):n_sim])
    
  }, error = function(e) {
    warning(paste("ARFIMA simulation failed:", e$message, "- using fGn approximation"))
    return(generate_fbm_simple(n, H = d + 0.5, sigma = sigma))
  })
}

# ============================================================================
# HIGH-LEVEL SIMULATION FUNCTIONS
# ============================================================================

#' Main H-K simulator function
#' 
#' @param n Length of time series
#' @param H Hurst exponent (0 < H < 1)
#' @param method Simulation method: "fbm", "arfima", "wavelet"
#' @param sigma Standard deviation of innovations
#' @param trend Linear trend coefficient 
#' @param seasonal_amplitude Amplitude of seasonal component
#' @param seasonal_period Period of seasonality (365.25 for annual)
#' @param noise_level Additional white noise level
#' @return List with simulated series and parameters
simulate_hk_process <- function(n, H, method = "fbm", sigma = 1, 
                               trend = 0, seasonal_amplitude = 0, 
                               seasonal_period = 365.25, noise_level = 0) {
  
  cat("Simulating H-K process with H =", H, "using method:", method, "\n")
  cat("Length:", n, "observations\n")
  
  # Generate base H-K process
  if (method == "fbm") {
    if (n > 2000) {
      hk_series <- generate_fbm_simple(n, H, sigma)  # Faster for large n
    } else {
      hk_series <- generate_fbm_dh(n, H, sigma)     # More accurate for smaller n
    }
  } else if (method == "arfima") {
    d <- H - 0.5
    hk_series <- generate_arfima(n, d = d, sigma = sigma)
  } else if (method == "wavelet") {
    hk_series <- generate_hk_wavelet(n, H, sigma)
  } else {
    stop("Method must be 'fbm', 'arfima', or 'wavelet'")
  }
  
  # Add deterministic components
  time_index <- 1:n
  
  # Linear trend
  if (trend != 0) {
    trend_component <- trend * time_index
    hk_series <- hk_series + trend_component
  }
  
  # Seasonal component
  if (seasonal_amplitude > 0) {
    seasonal_component <- seasonal_amplitude * sin(2 * pi * time_index / seasonal_period)
    hk_series <- hk_series + seasonal_component
  }
  
  # Additional white noise
  if (noise_level > 0) {
    white_noise <- rnorm(n, mean = 0, sd = noise_level)
    hk_series <- hk_series + white_noise
  }
  
  # Create result list
  result <- list(
    series = hk_series,
    parameters = list(
      n = n,
      H = H,
      method = method,
      sigma = sigma,
      trend = trend,
      seasonal_amplitude = seasonal_amplitude,
      seasonal_period = seasonal_period,
      noise_level = noise_level
    ),
    time_index = time_index
  )
  
  return(result)
}

#' Simulate multiple H-K realizations for testing
#' 
#' @param n_realizations Number of independent realizations
#' @param n Length of each series
#' @param H Hurst exponent
#' @param method Simulation method
#' @param ... Additional parameters passed to simulate_hk_process
#' @return List of realizations
simulate_hk_ensemble <- function(n_realizations, n, H, method = "fbm", ...) {
  
  cat("Generating", n_realizations, "H-K realizations with H =", H, "\n")
  
  realizations <- list()
  
  for (i in 1:n_realizations) {
    if (i %% max(1, floor(n_realizations/10)) == 0) {
      cat("Realization", i, "of", n_realizations, "\n")
    }
    
    realizations[[i]] <- simulate_hk_process(n = n, H = H, method = method, ...)
  }
  
  return(realizations)
}

# ============================================================================
# REALISTIC RESERVOIR SIMULATION
# ============================================================================

#' Simulate realistic reservoir data with H-K characteristics
#' 
#' @param n Number of days to simulate
#' @param H Hurst exponent for underlying process
#' @param base_level Average reservoir level
#' @param volatility Relative volatility (0.1 = 10% coefficient of variation)
#' @param seasonal_strength Strength of seasonal pattern (0-1)
#' @param trend_per_year Linear trend per year
#' @param min_level Minimum possible reservoir level
#' @param max_level Maximum possible reservoir level
#' @param start_date Starting date for time series
#' @param reservoir_name Name for the simulated reservoir
#' @return Data frame with Date, Reservoir, Value columns
simulate_reservoir_hk <- function(n, H, base_level = 1000, volatility = 0.15,
                                 seasonal_strength = 0.3, trend_per_year = 0,
                                 min_level = 0, max_level = 2000,
                                 start_date = as.Date("2000-01-01"),
                                 reservoir_name = "Simulated_HK_Reservoir") {
  
  cat("Simulating realistic reservoir with H-K characteristics\n")
  cat("Hurst exponent:", H, "\n")
  cat("Base level:", base_level, "units\n")
  cat("Volatility:", volatility, "\n")
  cat("Seasonal strength:", seasonal_strength, "\n")
  
  # Generate dates
  dates <- seq(from = start_date, by = "day", length.out = n)
  
  # Simulate H-K innovations on log scale for multiplicative effects
  sigma_innovation <- volatility
  
  hk_sim <- simulate_hk_process(
    n = n, 
    H = H, 
    method = "fbm",
    sigma = sigma_innovation,
    trend = trend_per_year / 365.25,  # Daily trend
    seasonal_amplitude = seasonal_strength * volatility,
    seasonal_period = 365.25,
    noise_level = volatility * 0.1  # Small amount of white noise
  )
  
  # Transform to reservoir levels
  # Use log-normal transformation to ensure positive values
  log_base <- log(base_level)
  log_levels <- log_base + hk_sim$series
  
  # Convert back to levels
  simulated_levels <- exp(log_levels)
  
  # Apply bounds
  simulated_levels <- pmax(min_level, pmin(max_level, simulated_levels))
  
  # Create data frame in same format as real data
  reservoir_data <- data.frame(
    Date = dates,
    Reservoir = reservoir_name,
    Value = simulated_levels,
    stringsAsFactors = FALSE
  )
  
  # Add metadata
  attr(reservoir_data, "simulation_params") <- hk_sim$parameters
  attr(reservoir_data, "true_H") <- H
  
  cat("Simulated reservoir levels:\n")
  cat("Range:", round(range(simulated_levels), 0), "\n")
  cat("Mean:", round(mean(simulated_levels), 0), "\n")
  cat("CV:", round(sd(simulated_levels) / mean(simulated_levels), 3), "\n")
  
  return(reservoir_data)
}

#' Simulate multiple reservoirs with different H values
#' 
#' @param H_values Vector of Hurst exponents to simulate
#' @param n Number of days for each reservoir
#' @param ... Additional parameters for simulate_reservoir_hk
#' @return Combined data frame with multiple reservoirs
simulate_reservoir_system <- function(H_values, n, ...) {
  
  cat("Simulating reservoir system with", length(H_values), "reservoirs\n")
  cat("H values:", paste(H_values, collapse = ", "), "\n")
  
  all_data <- data.frame()
  
  for (i in seq_along(H_values)) {
    H <- H_values[i]
    reservoir_name <- paste0("Reservoir_H", sprintf("%.2f", H))
    
    cat("\nSimulating", reservoir_name, "\n")
    
    reservoir_data <- simulate_reservoir_hk(
      n = n, 
      H = H, 
      reservoir_name = reservoir_name,
      ...
    )
    
    all_data <- rbind(all_data, reservoir_data)
  }
  
  return(all_data)
}

# ============================================================================
# VALIDATION AND TESTING FUNCTIONS
# ============================================================================

#' Validate H-K simulation by estimating Hurst exponent
#' 
#' @param hk_simulation Result from simulate_hk_process
#' @param methods Vector of estimation methods to use
#' @return List with estimated H values
validate_hk_simulation <- function(hk_simulation, methods = c("dfa", "rs")) {
  
  series <- hk_simulation$series
  true_H <- hk_simulation$parameters$H
  
  cat("Validating H-K simulation\n")
  cat("True H:", true_H, "\n")
  cat("Series length:", length(series), "\n")
  
  estimates <- list(true_H = true_H)
  
  # DFA method (simplified version)
  if ("dfa" %in% methods) {
    tryCatch({
      estimates$dfa_H <- estimate_hurst_dfa_simple(series)
      cat("DFA estimate:", round(estimates$dfa_H, 3), "\n")
    }, error = function(e) {
      cat("DFA estimation failed\n")
      estimates$dfa_H <- NA
    })
  }
  
  # R/S method
  if ("rs" %in% methods) {
    tryCatch({
      estimates$rs_H <- estimate_hurst_rs_simple(series)
      cat("R/S estimate:", round(estimates$rs_H, 3), "\n")
    }, error = function(e) {
      cat("R/S estimation failed\n")
      estimates$rs_H <- NA
    })
  }
  
  return(estimates)
}

# Simple DFA implementation for validation
estimate_hurst_dfa_simple <- function(x) {
  n <- length(x)
  x_centered <- x - mean(x)
  y <- cumsum(x_centered)
  
  scales <- unique(round(seq(10, n/4, length.out = 20)))
  fluctuations <- numeric(length(scales))
  
  for (i in seq_along(scales)) {
    scale <- scales[i]
    n_segments <- floor(n / scale)
    
    segment_flucts <- numeric(n_segments)
    for (j in 1:n_segments) {
      start_idx <- (j-1) * scale + 1
      end_idx <- j * scale
      segment <- y[start_idx:end_idx]
      
      # Linear detrending
      t <- 1:length(segment)
      fit <- lm(segment ~ t)
      residuals <- segment - fitted(fit)
      segment_flucts[j] <- sqrt(mean(residuals^2))
    }
    
    fluctuations[i] <- mean(segment_flucts)
  }
  
  # Fit power law
  valid_idx <- fluctuations > 0 & is.finite(fluctuations)
  if (sum(valid_idx) < 3) return(NA)
  
  log_scales <- log(scales[valid_idx])
  log_flucts <- log(fluctuations[valid_idx])
  
  fit <- lm(log_flucts ~ log_scales)
  return(as.numeric(coef(fit)[2]))
}

# Simple R/S implementation for validation  
estimate_hurst_rs_simple <- function(x) {
  n <- length(x)
  x_centered <- x - mean(x)
  
  # Calculate cumulative sum
  cum_sum <- cumsum(x_centered)
  
  # Calculate range
  R <- max(cum_sum) - min(cum_sum)
  
  # Calculate standard deviation
  S <- sd(x)
  
  if (S == 0) return(NA)
  
  # Hurst exponent
  H <- log(R/S) / log(n)
  
  return(H)
}

# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

#' Demonstrate H-K simulation capabilities
#' 
#' @param demo_type Type of demonstration: "basic", "ensemble", "reservoir", "validation"
demonstrate_hk_simulation <- function(demo_type = "basic") {
  
  cat("=== HURST-KOLMOGOROV SIMULATION DEMONSTRATION ===\n")
  cat("Demo type:", demo_type, "\n\n")
  
  if (demo_type == "basic") {
    cat("Basic H-K simulation with different H values\n")
    
    H_values <- c(0.3, 0.5, 0.7, 0.9)
    n <- 1000
    
    for (H in H_values) {
      cat("\n--- H =", H, "---\n")
      sim <- simulate_hk_process(n = n, H = H, sigma = 1)
      
      # Basic statistics
      cat("Mean:", round(mean(sim$series), 3), "\n")
      cat("SD:", round(sd(sim$series), 3), "\n")
      cat("Range:", round(range(sim$series), 3), "\n")
      
      # Validate
      validation <- validate_hk_simulation(sim)
    }
    
  } else if (demo_type == "ensemble") {
    cat("Ensemble simulation for testing estimators\n")
    
    ensemble <- simulate_hk_ensemble(
      n_realizations = 5,
      n = 500,
      H = 0.8,
      method = "fbm"
    )
    
    cat("Generated", length(ensemble), "realizations\n")
    
    # Validate each realization
    for (i in 1:length(ensemble)) {
      cat("\nRealization", i, ":\n")
      validation <- validate_hk_simulation(ensemble[[i]])
    }
    
  } else if (demo_type == "reservoir") {
    cat("Realistic reservoir simulation\n")
    
    # Simulate reservoir system
    reservoir_system <- simulate_reservoir_system(
      H_values = c(0.6, 0.8),
      n = 2000,  # ~5.5 years
      base_level = 1500,
      volatility = 0.2,
      seasonal_strength = 0.4
    )
    
    cat("\nGenerated reservoir system with", nrow(reservoir_system), "observations\n")
    cat("Reservoirs:", unique(reservoir_system$Reservoir), "\n")
    cat("Date range:", as.character(range(reservoir_system$Date)), "\n")
    
    return(reservoir_system)
    
  } else if (demo_type == "validation") {
    cat("Method validation across H values\n")
    
    H_test_values <- seq(0.3, 0.9, by = 0.1)
    validation_results <- data.frame(
      True_H = numeric(),
      DFA_H = numeric(),
      RS_H = numeric(),
      DFA_Error = numeric(),
      RS_Error = numeric()
    )
    
    for (H in H_test_values) {
      cat("Testing H =", H, "\n")
      
      sim <- simulate_hk_process(n = 1000, H = H, method = "fbm")
      validation <- validate_hk_simulation(sim)
      
      dfa_error <- validation$dfa_H - H
      rs_error <- validation$rs_H - H
      
      validation_results <- rbind(validation_results, data.frame(
        True_H = H,
        DFA_H = validation$dfa_H,
        RS_H = validation$rs_H,
        DFA_Error = dfa_error,
        RS_Error = rs_error
      ))
    }
    
    cat("\nValidation Results:\n")
    print(validation_results)
    
    return(validation_results)
  }
}

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Run demonstrations
cat("H-K Simulator loaded successfully!\n")
cat("Available functions:\n")
cat("- simulate_hk_process(): Main simulation function\n")
cat("- simulate_reservoir_hk(): Realistic reservoir simulation\n") 
cat("- simulate_reservoir_system(): Multiple reservoirs\n")
cat("- demonstrate_hk_simulation(): Run demonstrations\n")
cat("\nExample usage:\n")
cat("demo_data <- demonstrate_hk_simulation('reservoir')\n")
cat("validation <- demonstrate_hk_simulation('validation')\n")

# Uncomment to run demonstrations:
# demonstrate_hk_simulation("basic")
# reservoir_data <- demonstrate_hk_simulation("reservoir")
# validation_results <- demonstrate_hk_simulation("validation")