# Generic CSV Reader with Dynamic Variable Assignment
# This tool reads CSV files and creates variables named after column headers
# Designed for flexible data processing workflows

library(readr)
library(dplyr)
library(tidyr)
library(stringr)

#' Read CSV and create dynamic variables
#'
#' @param file_path Path to the CSV file
#' @param date_column Name of the date column (if exists)
#' @param date_format Format of the date column (auto-detected if NULL)
#' @param clean_names Should column names be cleaned? (default: TRUE)
#' @param verbose Print information about the loaded data? (default: TRUE)
#' @return List containing the data and metadata
read_csv_dynamic <- function(file_path,
                             date_column = NULL,
                             date_format = NULL,
                             clean_names = TRUE,
                             verbose = TRUE) {

  # Read the CSV file
  if (verbose) cat("Reading CSV file:", file_path, "\n")

  data <- read_csv(file_path, show_col_types = FALSE)

  # Store original column names
  original_names <- colnames(data)

  # Clean column names if requested
  if (clean_names) {
    clean_column_names <- function(names) {
      names %>%
        # Remove special characters and replace with underscores
        str_replace_all("[^A-Za-z0-9_]", "_") %>%
        # Remove multiple underscores
        str_replace_all("_{2,}", "_") %>%
        # Remove leading/trailing underscores
        str_replace_all("^_|_$", "") %>%
        # Convert to lowercase
        str_to_lower() %>%
        # Ensure names don't start with numbers
        str_replace_all("^([0-9])", "x\\1")
    }

    colnames(data) <- clean_column_names(colnames(data))

    if (verbose) {
      cat("Column names cleaned:\n")
      for (i in seq_along(original_names)) {
        if (original_names[i] != colnames(data)[i]) {
          cat("  ", original_names[i], "->", colnames(data)[i], "\n")
        }
      }
    }
  }

  # Handle date column if specified
  if (!is.null(date_column)) {
    # Check both original and cleaned column names
    original_date_col <- date_column
    cleaned_date_col <- if (clean_names) {
      date_column %>%
        str_replace_all("[^A-Za-z0-9_]", "_") %>%
        str_replace_all("_{2,}", "_") %>%
        str_replace_all("^_|_$", "") %>%
        str_to_lower() %>%
        str_replace_all("^([0-9])", "x\\1")
    } else {
      date_column
    }

    # Find the actual date column name
    actual_date_col <- NULL
    if (original_date_col %in% colnames(data)) {
      actual_date_col <- original_date_col
    } else if (cleaned_date_col %in% colnames(data)) {
      actual_date_col <- cleaned_date_col
    }

    if (!is.null(actual_date_col)) {
      if (verbose) cat("Processing date column:", actual_date_col, "\n")

      # Try to parse dates
      if (is.null(date_format)) {
        data[[actual_date_col]] <- as.POSIXct(data[[actual_date_col]], tz = "UTC")
      } else {
        data[[actual_date_col]] <- as.POSIXct(data[[actual_date_col]], format = date_format, tz = "UTC")
      }

      # Check for parsing failures
      na_dates <- sum(is.na(data[[actual_date_col]]))
      if (na_dates > 0 && verbose) {
        cat("Warning:", na_dates, "dates could not be parsed\n")
      }

      # Update date_column in metadata to reflect actual column name used
      date_column <- actual_date_col
    } else {
      warning("Date column '", original_date_col, "' not found in data (tried both original and cleaned names)")
      date_column <- NULL
    }
  }

  # Print data summary
  if (verbose) {
    cat("\nData Summary:\n")
    cat("  Rows:", nrow(data), "\n")
    cat("  Columns:", ncol(data), "\n")
    cat("  Column types:\n")

    # Show available columns first
    if (verbose) {
      cat("  Available columns:\n")
      for (col in colnames(data)) {
        cat("    ", col, "\n")
      }
    }

    col_info <- data %>%
      summarise_all(class) %>%
      pivot_longer(everything(), names_to = "column", values_to = "type")

    for (i in 1:nrow(col_info)) {
      cat("    ", col_info$column[i], ":", col_info$type[i], "\n")
    }

    # Show date range if date column exists
    if (!is.null(date_column) && date_column %in% colnames(data)) {
      date_range <- range(data[[date_column]], na.rm = TRUE)
      cat("  Date range:", as.character(date_range[1]), "to", as.character(date_range[2]), "\n")
    }
  }

  # Create the result list
  result <- list(
    data = data,
    metadata = list(
      file_path = file_path,
      original_names = original_names,
      current_names = colnames(data),
      n_rows = nrow(data),
      n_cols = ncol(data),
      date_column = date_column,
      read_time = Sys.time()
    )
  )

  return(result)
}

#' Extract variables from loaded data
#'
#' @param data_obj Object returned by read_csv_dynamic
#' @param variables Vector of variable names to extract (NULL for all)
#' @param as_list Return as named list? (default: FALSE, assigns to global env)
#' @return Named list of variables or assigns to global environment
extract_variables <- function(data_obj, variables = NULL, as_list = FALSE) {

  data <- data_obj$data

  # Select variables
  if (is.null(variables)) {
    variables <- colnames(data)
  } else {
    # Check if requested variables exist
    missing_vars <- setdiff(variables, colnames(data))
    if (length(missing_vars) > 0) {
      warning("Variables not found: ", paste(missing_vars, collapse = ", "))
      variables <- intersect(variables, colnames(data))
    }
  }

  # Extract variables
  var_list <- list()
  for (var in variables) {
    var_list[[var]] <- data[[var]]
  }

  if (as_list) {
    return(var_list)
  } else {
    # Assign to global environment
    list2env(var_list, envir = .GlobalEnv)
    cat("Variables assigned to global environment:\n")
    cat("  ", paste(variables, collapse = ", "), "\n")
    return(invisible(var_list))
  }
}

#' Quick data exploration function
#'
#' @param data_obj Object returned by read_csv_dynamic
#' @param n_sample Number of rows to sample for preview (default: 6)
explore_data <- function(data_obj, n_sample = 6) {

  data <- data_obj$data
  metadata <- data_obj$metadata

  cat("=== DATA EXPLORATION ===\n\n")

  # Basic info
  cat("File:", metadata$file_path, "\n")
  cat("Dimensions:", metadata$n_rows, "x", metadata$n_cols, "\n")
  cat("Read at:", as.character(metadata$read_time), "\n\n")

  # Sample of data
  cat("Data preview (first", min(n_sample, nrow(data)), "rows):\n")
  print(head(data, n_sample))

  cat("\nColumn summary:\n")
  print(summary(data))

  # Missing values
  missing_summary <- data %>%
    summarise_all(~sum(is.na(.))) %>%
    pivot_longer(everything(), names_to = "column", values_to = "missing") %>%
    filter(missing > 0)

  if (nrow(missing_summary) > 0) {
    cat("\nMissing values:\n")
    print(missing_summary)
  } else {
    cat("\nNo missing values detected.\n")
  }
}

# Example usage:
#
# # Load data
# drought_data <- read_csv_dynamic("weather_data.csv",
#                                date_column = "time",
#                                clean_names = TRUE)
#
# # Explore the data
# explore_data(drought_data)
#
# # Extract specific variables for analysis
# drought_vars <- c("time", "soil_moisture_0_to_7cm", "precipitation",
#                   "et0_fao_evapotranspiration", "vapour_pressure_deficit")
# extract_variables(drought_data, drought_vars)
#
# # Now you can use: time, soil_moisture_0_to_7cm, precipitation, etc.
#
# # Or get as a list for further processing
# var_list <- extract_variables(drought_data, drought_vars, as_list = TRUE)

cat("CSV Reader Tool loaded successfully!\n")
cat("Main functions:\n")
cat("  - read_csv_dynamic(): Load CSV with dynamic variable creation\n")
cat("  - extract_variables(): Extract specific variables\n")
cat("  - explore_data(): Quick data exploration\n")
