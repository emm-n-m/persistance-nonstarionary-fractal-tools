"""Utilities for estimating the Hurst exponent of a time series."""

from __future__ import annotations

from typing import Optional, Sequence, Union, Dict

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas optional
    pd = None  # type: ignore[assignment]


if pd is None:
    ArrayLike = Union[Sequence[float], np.ndarray]
else:  # pragma: no branch - alias depends on optional pandas
    ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _prepare_series(series: ArrayLike, dropna: bool = True) -> np.ndarray:
    """Validate and prepare the input series."""
    if pd is not None and isinstance(series, pd.Series):
        values = series.to_numpy(dtype=float)
    else:
        values = np.asarray(series, dtype=float)

    if values.ndim != 1:
        raise ValueError("Input series must be one-dimensional")

    if dropna:
        values = values[~np.isnan(values)]
    elif np.isnan(values).any():
        raise ValueError("NaNs present in series – set dropna=True to ignore them")

    if values.size < 32:
        raise ValueError("At least 32 observations are required for Hurst estimation")

    return values


def _mean_rs(segment: np.ndarray) -> Optional[float]:
    """Return the rescaled range for a segment, or None if the standard deviation is zero."""
    demeaned = segment - segment.mean()
    cumulative = np.cumsum(demeaned)
    segment_range = cumulative.max() - cumulative.min()
    std = demeaned.std(ddof=1)
    if std == 0:
        return None
    return segment_range / std


def hurst_rs(
    series: ArrayLike,
    *,
    min_window: int = 8,
    max_window: Optional[int] = None,
    num_windows: int = 20,
    dropna: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """Estimate the Hurst exponent using the classical rescaled range (R/S) method.

    Parameters
    ----------
    series:
        Sequence of observations representing the time series.
    min_window:
        Smallest window size (number of observations) used in the log–log regression.
    max_window:
        Largest window size to consider. Defaults to ``len(series) // 2``.
    num_windows:
        Number of window sizes sampled on a log scale between ``min_window`` and
        ``max_window``.
    dropna:
        If ``True`` (default) remove missing values before estimation. If ``False``
        and missing values are present an error is raised.

    Returns
    -------
    dict
        A dictionary containing the estimated Hurst exponent and diagnostic
        information for the regression fit.
    """

    values = _prepare_series(series, dropna=dropna)

    if max_window is None:
        max_window = values.size // 2
    if max_window <= min_window:
        raise ValueError("max_window must be greater than min_window")

    window_candidates = np.logspace(
        np.log10(min_window), np.log10(max_window), num=num_windows
    )
    window_sizes = np.unique(window_candidates.astype(int))
    window_sizes = window_sizes[window_sizes >= min_window]

    rs_values = []
    effective_windows = []

    for window in window_sizes:
        segment_count = values.size // window
        if segment_count < 2:
            continue
        reshaped = values[: segment_count * window].reshape(segment_count, window)
        rs_segments = [_mean_rs(segment) for segment in reshaped]
        rs_segments = [val for val in rs_segments if val is not None and np.isfinite(val)]
        if not rs_segments:
            continue
        effective_windows.append(window)
        rs_values.append(np.mean(rs_segments))

    if len(effective_windows) < 2:
        raise ValueError("Insufficient valid windows for R/S regression")

    log_windows = np.log10(effective_windows)
    log_rs = np.log10(rs_values)

    slope, intercept = np.polyfit(log_windows, log_rs, deg=1)
    fitted = slope * log_windows + intercept
    residuals = log_rs - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_rs - log_rs.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return {
        "method": "rescaled_range",
        "hurst": float(slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(max(min(r_squared, 1.0), 0.0)),
        "window_sizes": np.asarray(effective_windows, dtype=int),
        "rs_values": np.asarray(rs_values, dtype=float),
        "log_window_sizes": log_windows,
        "log_rs_values": log_rs,
        "fitted_log_rs": fitted,
    }


__all__ = ["hurst_rs"]
