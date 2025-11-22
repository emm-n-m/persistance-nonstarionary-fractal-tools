# Copyright (c) 2025 [Emmanouil Mavrogiorgis, emm.n.black@gmail.com]
# All rights reserved.
#
# This project was developed with assistance from Claude Code by Anthropic.
# Claude Code is an AI-powered coding assistant (https://www.anthropic.com).
#
# Licensed under MIT License
# See LICENSE file for details

"""
Climacogram Analysis for Time Series
Estimates variance scaling properties across different time scales

A climacogram plots the variance of aggregated values against the aggregation scale.
It's useful for detecting long-range dependence and scaling behavior in time series.
"""

from typing import Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_climacogram(
    series: np.ndarray, max_scale: int, min_scale: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute climacogram for a time series.

    The climacogram shows how variance changes across different aggregation scales,
    revealing long-range dependence and scaling properties.

    Parameters
    ----------
    series : np.ndarray
        One-dimensional time series data
    max_scale : int
        Maximum aggregation scale (window size)
    min_scale : int, optional
        Minimum aggregation scale (default: 1)

    Returns
    -------
    scales : np.ndarray
        Array of aggregation scales
    variances : np.ndarray
        Variance at each scale

    Examples
    --------
    >>> series = np.random.randn(1000)
    >>> scales, variances = compute_climacogram(series, max_scale=50)
    >>> plt.loglog(scales, variances)
    """
    if series.ndim != 1:
        raise ValueError("Input series must be one-dimensional")

    if max_scale > len(series) // 2:
        raise ValueError(
            f"max_scale ({max_scale}) too large for series length ({len(series)}). "
            f"Should be at most {len(series) // 2}"
        )

    scales = np.arange(min_scale, max_scale + 1)
    variances = []

    for scale in scales:
        # Create non-overlapping chunks of the specified scale
        chunks = [series[i : i + scale] for i in range(0, len(series), scale)]

        # Compute means of complete chunks only
        chunk_means = [np.mean(chunk) for chunk in chunks if len(chunk) == scale]

        if len(chunk_means) < 2:
            # Need at least 2 chunks to compute variance
            variances.append(np.nan)
        else:
            var = np.var(chunk_means, ddof=1)
            variances.append(var)

    return scales, np.array(variances)


def plot_climacogram(
    scales: np.ndarray,
    variances: np.ndarray,
    title: str = "Climacogram",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot climacogram on log-log scale.

    Parameters
    ----------
    scales : np.ndarray
        Aggregation scales
    variances : np.ndarray
        Variance at each scale
    title : str, optional
        Plot title (default: "Climacogram")
    output_path : Path, optional
        If provided, save plot to this path
    show : bool, optional
        If True, display the plot (default: True)
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(scales, variances, marker="o", markersize=4, linewidth=1.5)
    plt.xlabel("Scale (window size)", fontsize=12)
    plt.ylabel("Variance of aggregated values", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, which="both")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved climacogram to '{output_path}'")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_reservoir_climacogram(
    data_path: Path,
    reservoir_name: str,
    value_column: str = "Value",
    reservoir_column: str = "Reservoir",
    max_scale: int = 50,
    separator: str = "\t",
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data and compute climacogram for a specific reservoir.

    Parameters
    ----------
    data_path : Path
        Path to CSV file containing time series data
    reservoir_name : str
        Name of the reservoir to analyze
    value_column : str, optional
        Column name containing values (default: 'Value')
    reservoir_column : str, optional
        Column name containing reservoir identifiers (default: 'Reservoir')
    max_scale : int, optional
        Maximum aggregation scale (default: 50)
    separator : str, optional
        CSV separator (default: tab)
    output_path : Path, optional
        If provided, save plot to this path

    Returns
    -------
    scales : np.ndarray
        Aggregation scales
    variances : np.ndarray
        Variance at each scale

    Raises
    ------
    FileNotFoundError
        If data file doesn't exist
    ValueError
        If reservoir not found in data
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path, sep=separator)

    # Check if reservoir exists
    if reservoir_column not in df.columns:
        raise ValueError(f"Column '{reservoir_column}' not found in data")

    if reservoir_name not in df[reservoir_column].values:
        available = df[reservoir_column].unique()
        raise ValueError(
            f"Reservoir '{reservoir_name}' not found. "
            f"Available: {', '.join(map(str, available))}"
        )

    # Filter for specific reservoir
    filtered = df[df[reservoir_column] == reservoir_name]

    # Extract value series
    series = filtered[value_column].values

    # Compute climacogram
    scales, variances = compute_climacogram(series, max_scale=max_scale)

    # Plot
    title = f"Climacogram for {reservoir_name}"
    plot_climacogram(scales, variances, title=title, output_path=output_path)

    return scales, variances


if __name__ == "__main__":
    # Example usage
    print("Climacogram Analysis Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from pathlib import Path
    from climacogram import analyze_reservoir_climacogram

    # Analyze a specific reservoir
    data_path = Path('data/water_reserves.csv')
    scales, variances = analyze_reservoir_climacogram(
        data_path,
        reservoir_name='Mornos',
        max_scale=50,
        output_path=Path('output/mornos_climacogram.png')
    )

    # Or compute directly from a series
    from climacogram import compute_climacogram, plot_climacogram
    import numpy as np

    series = np.random.randn(1000)
    scales, variances = compute_climacogram(series, max_scale=50)
    plot_climacogram(scales, variances, title='My Series')
    """)
