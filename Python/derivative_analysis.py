# Copyright (c) 2025 [Emmanouil Mavrogiorgis, emm.n.black@gmail.com]
# All rights reserved.
#
# This project was developed with assistance from Claude Code by Anthropic.
# Claude Code is an AI-powered coding assistant (https://www.anthropic.com).
#
# Licensed under MIT License
# See LICENSE file for details

"""
Generalized Time Series Derivative Analysis
Python port of derivative.R - calculates nth-order derivatives and detects regime changes

Author: Ported from R to Python
Dependencies: pandas, numpy, scipy, matplotlib
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta


class DerivativeAnalyzer:
    """
    Comprehensive derivative analysis for time series data
    Supports nth-order derivatives and regime change detection
    """
    
    def __init__(self, data: pd.DataFrame, 
                 series_column: str = 'Series',
                 time_column: str = 'Date', 
                 value_column: str = 'Value'):
        """
        Initialize the analyzer with data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with time series
        series_column : str
            Column name for series identifier
        time_column : str  
            Column name for time/date
        value_column : str
            Column name for values
        """
        self.data = data.copy()
        self.series_column = series_column
        self.time_column = time_column
        self.value_column = value_column
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[time_column]):
            self.data[time_column] = pd.to_datetime(self.data[time_column])
            
    def calculate_derivative_metrics(self, 
                                   series_name: Optional[str] = None,
                                   derivative_order: int = 1,
                                   time_unit: str = 'days',
                                   min_change_threshold: float = 0,
                                   direction_labels: List[str] = ['increasing', 'decreasing']) -> Dict:
        """
        Calculate nth-order derivative metrics for a time series
        
        Parameters:
        -----------
        series_name : str, optional
            Name of series to analyze (if None, assumes single series)
        derivative_order : int
            Order of derivative (1=velocity, 2=acceleration, etc.)
        time_unit : str
            Time unit for derivative calculation
        min_change_threshold : float
            Minimum absolute change to consider valid
        direction_labels : list
            Labels for positive/negative directions
            
        Returns:
        --------
        dict : Derivative analysis results
        """
        
        # Filter data for specific series
        if series_name is not None:
            if series_name not in self.data[self.series_column].values:
                raise ValueError(f"Series '{series_name}' not found in data")
            series_data = self.data[self.data[self.series_column] == series_name].copy()
            analysis_name = series_name
        else:
            series_data = self.data.copy()
            analysis_name = "Single Series"
            
        # Sort by time
        series_data = series_data.sort_values(self.time_column).reset_index(drop=True)
        
        n_obs = len(series_data)
        if n_obs < (derivative_order + 10):
            print(f"Insufficient data for {derivative_order} order derivative analysis")
            return None
            
        values = series_data[self.value_column].values
        times = series_data[self.time_column].values
        
        # Time multiplier for unit conversion
        time_multipliers = {
            'seconds': 1 / (24 * 3600),
            'minutes': 1 / (24 * 60), 
            'hours': 1 / 24,
            'days': 1,
            'weeks': 7,
            'months': 30.44,
            'years': 365.25
        }
        time_multiplier = time_multipliers.get(time_unit, 1)
        
        # Calculate derivatives iteratively
        current_values = values.copy()
        current_times = times.copy()
        derivative_times_list = []
        
        print(f"\n=== {self._get_derivative_name(derivative_order).upper()} ANALYSIS: {analysis_name} ===")
        
        for order in range(1, derivative_order + 1):
            # Calculate time differences in specified units
            time_diffs = np.diff(current_times).astype('timedelta64[D]').astype(float) / time_multiplier
            value_diffs = np.diff(current_values)
            
            # Calculate derivatives
            derivatives = value_diffs / time_diffs
            
            # Store times for this derivative order
            derivative_times_list.append(current_times[1:])
            
            # Update for next iteration
            current_values = derivatives.copy()
            current_times = current_times[1:]
            
            # Remove invalid derivatives
            valid_mask = (np.isfinite(derivatives) & 
                         (np.abs(value_diffs) >= min_change_threshold))
            current_values = current_values[valid_mask]
            current_times = current_times[valid_mask]
            
            if len(current_values) < 5:
                print(f"Insufficient valid derivatives at order {order}")
                return None
                
        final_derivatives = current_values
        final_times = current_times
        
        print(f"Valid {derivative_order} order derivative measurements: {len(final_derivatives)}")
        print(f"Time range: {pd.to_datetime(final_times[0])} to {pd.to_datetime(final_times[-1])}")
        
        # Classify by direction
        pos_derivatives = final_derivatives[final_derivatives > 0]
        neg_derivatives = final_derivatives[final_derivatives < 0]
        
        derivative_name = self._get_derivative_name(derivative_order)
        
        results = {
            'series_name': analysis_name,
            'derivative_order': derivative_order,
            'derivative_name': derivative_name,
            'time_unit': time_unit,
            'times': final_times,
            'derivatives': final_derivatives,
            'derivative_times_list': derivative_times_list,
            
            # Basic statistics
            'mean_derivative': np.mean(final_derivatives),
            'median_derivative': np.median(final_derivatives),
            'derivative_range': [np.min(final_derivatives), np.max(final_derivatives)],
            'derivative_sd': np.std(final_derivatives),
            
            # Direction-based statistics  
            'n_positive': len(pos_derivatives),
            'n_negative': len(neg_derivatives),
            'mean_positive_derivative': np.mean(pos_derivatives) if len(pos_derivatives) > 0 else np.nan,
            'mean_negative_derivative': np.mean(neg_derivatives) if len(neg_derivatives) > 0 else np.nan,
            'max_positive_derivative': np.max(pos_derivatives) if len(pos_derivatives) > 0 else np.nan,
            'max_negative_derivative': np.min(neg_derivatives) if len(neg_derivatives) > 0 else np.nan,
            
            # Proportions
            'positive_fraction': len(pos_derivatives) / len(final_derivatives),
            'negative_fraction': len(neg_derivatives) / len(final_derivatives),
            
            'direction_labels': direction_labels
        }
        
        self._print_derivative_summary(results)
        return results
    
    def detect_derivative_regimes(self, 
                                derivative_results: Dict,
                                window_size: int = 365,
                                step_size: int = 30,
                                window_unit: str = 'days') -> Optional[Dict]:
        """
        Detect regime changes in derivative patterns using rolling windows
        
        Parameters:
        -----------
        derivative_results : dict
            Results from calculate_derivative_metrics
        window_size : int
            Size of rolling window
        step_size : int  
            Step size between windows
        window_unit : str
            Unit for window sizing
            
        Returns:
        --------
        dict : Regime change analysis results
        """
        
        if derivative_results is None:
            return None
            
        derivatives = derivative_results['derivatives']
        times = derivative_results['times']
        n = len(derivatives)
        
        print(f"\nDetecting {derivative_results['derivative_name']} regime changes...")
        
        # Convert window sizes
        if window_unit == 'observations':
            window_obs = window_size
            step_obs = step_size
        else:
            # Convert time-based windows to observations
            time_multipliers = {'days': 1, 'weeks': 7, 'months': 30.44, 'years': 365.25}
            time_mult = time_multipliers.get(window_unit, 1)
            
            # Estimate observations per time unit
            total_days = (pd.to_datetime(times[-1]) - pd.to_datetime(times[0])).days
            obs_per_day = n / max(total_days, 1)
            
            window_obs = int(window_size * time_mult * obs_per_day)
            step_obs = int(step_size * time_mult * obs_per_day)
            
        if n < window_obs * 2:
            print(f"Insufficient data for regime analysis. Need {window_obs * 2}, have {n}")
            return None
            
        # Calculate rolling metrics
        n_windows = (n - window_obs) // step_obs + 1
        
        rolling_metrics = {
            'times': [],
            'mean_deriv': [],
            'positive_frac': [],
            'max_positive': [],
            'max_negative': [],
            'volatility': []
        }
        
        for i in range(n_windows):
            start_idx = i * step_obs
            end_idx = start_idx + window_obs
            
            if end_idx > n:
                break
                
            window_derivatives = derivatives[start_idx:end_idx]
            window_times = times[start_idx:end_idx]
            
            # Calculate window metrics
            rolling_metrics['mean_deriv'].append(np.mean(window_derivatives))
            rolling_metrics['positive_frac'].append(np.sum(window_derivatives > 0) / len(window_derivatives))
            rolling_metrics['volatility'].append(np.std(window_derivatives))
            
            pos_deriv = window_derivatives[window_derivatives > 0]
            neg_deriv = window_derivatives[window_derivatives < 0]
            
            rolling_metrics['max_positive'].append(np.max(pos_deriv) if len(pos_deriv) > 0 else 0)
            rolling_metrics['max_negative'].append(np.min(neg_deriv) if len(neg_deriv) > 0 else 0)
            rolling_metrics['times'].append(window_times[len(window_times)//2])
            
        # Convert to arrays
        for key in rolling_metrics:
            rolling_metrics[key] = np.array(rolling_metrics[key])
            
        print(f"Rolling derivative windows: {len(rolling_metrics['times'])}")
        print(f"Analysis period: {pd.to_datetime(rolling_metrics['times'][0])} to {pd.to_datetime(rolling_metrics['times'][-1])}")
        
        # Detect changepoints
        regime_changes = self._detect_derivative_changepoints(
            rolling_metrics['mean_deriv'],
            rolling_metrics['positive_frac'],
            rolling_metrics['volatility'],
            rolling_metrics['times'],
            derivative_results['derivative_name']
        )
        
        rolling_metrics['regime_changes'] = regime_changes
        rolling_metrics['window_obs'] = window_obs
        rolling_metrics['step_obs'] = step_obs
        rolling_metrics['derivative_order'] = derivative_results['derivative_order']
        
        return rolling_metrics
    
    def _detect_derivative_changepoints(self, 
                                      mean_derivatives: np.ndarray,
                                      positive_fractions: np.ndarray,
                                      volatilities: np.ndarray,
                                      times: np.ndarray,
                                      derivative_name: str,
                                      significance_threshold: float = 1.5) -> Dict:
        """
        Detect changepoints in derivative patterns
        """
        n = len(mean_derivatives)
        
        if n < 6:
            return {}
            
        print(f"Detecting {derivative_name} changepoints...")
        
        # Test potential changepoints (avoid edges)
        test_indices = range(2, n-2)
        
        if len(test_indices) < 2:
            return {}
            
        # Calculate scores for different types of changes
        mean_deriv_scores = []
        positive_frac_scores = []
        volatility_scores = []
        
        for cp_idx in test_indices:
            # Mean derivative change
            before_mean = np.mean(mean_derivatives[:cp_idx])
            after_mean = np.mean(mean_derivatives[cp_idx:])
            
            deriv_change = abs(after_mean - before_mean)
            pooled_sd = np.sqrt((np.var(mean_derivatives[:cp_idx]) + 
                               np.var(mean_derivatives[cp_idx:])) / 2)
            
            mean_deriv_scores.append(deriv_change / pooled_sd if pooled_sd > 0 else 0)
            
            # Positive fraction change
            before_pos_frac = np.mean(positive_fractions[:cp_idx])
            after_pos_frac = np.mean(positive_fractions[cp_idx:])
            positive_frac_scores.append(abs(after_pos_frac - before_pos_frac))
            
            # Volatility change
            before_vol = np.mean(volatilities[:cp_idx])
            after_vol = np.mean(volatilities[cp_idx:])
            
            if before_vol > 0 and after_vol > 0:
                volatility_scores.append(abs(np.log(after_vol / before_vol)))
            else:
                volatility_scores.append(0)
                
        results = {}
        
        # Check for significant changes
        if len(mean_deriv_scores) > 0 and max(mean_deriv_scores) > significance_threshold:
            best_idx = np.argmax(mean_deriv_scores)
            cp_idx = test_indices[best_idx]
            
            results['derivative_change'] = {
                'changepoint_date': pd.to_datetime(times[cp_idx]),
                'score': max(mean_deriv_scores),
                'before_derivative': np.mean(mean_derivatives[:cp_idx]),
                'after_derivative': np.mean(mean_derivatives[cp_idx:]),
                'derivative_change': np.mean(mean_derivatives[cp_idx:]) - np.mean(mean_derivatives[:cp_idx])
            }
            
            print(f"{derivative_name} regime change: {results['derivative_change']['changepoint_date']}")
            print(f"Score: {results['derivative_change']['score']:.2f}")
            
        if len(positive_frac_scores) > 0 and max(positive_frac_scores) > 0.2:
            best_idx = np.argmax(positive_frac_scores)
            cp_idx = test_indices[best_idx]
            
            results['direction_change'] = {
                'changepoint_date': pd.to_datetime(times[cp_idx]),
                'score': max(positive_frac_scores),
                'before_positive_frac': np.mean(positive_fractions[:cp_idx]),
                'after_positive_frac': np.mean(positive_fractions[cp_idx:]),
            }
            
            print(f"Direction regime change: {results['direction_change']['changepoint_date']}")
            
        if len(volatility_scores) > 0 and max(volatility_scores) > 0.5:
            best_idx = np.argmax(volatility_scores)
            cp_idx = test_indices[best_idx]
            
            results['volatility_change'] = {
                'changepoint_date': pd.to_datetime(times[cp_idx]),
                'score': max(volatility_scores),
                'before_volatility': np.mean(volatilities[:cp_idx]),
                'after_volatility': np.mean(volatilities[cp_idx:]),
                'volatility_ratio': np.mean(volatilities[cp_idx:]) / np.mean(volatilities[:cp_idx])
            }
            
            print(f"Volatility regime change: {results['volatility_change']['changepoint_date']}")
            
        return results
    
    def extract_derivative_timeseries(self, 
                                    derivative_results: Dict,
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Extract derivative time series as DataFrame for further analysis
        """
        if derivative_results is None:
            print("No derivative results provided")
            return None
            
        dates = pd.to_datetime(derivative_results['times'])
        derivatives = derivative_results['derivatives']
        direction_labels = derivative_results['direction_labels']
        derivative_name = derivative_results['derivative_name']
        
        df = pd.DataFrame({
            'Date': dates,
            'Derivative': derivatives,
            'Direction': np.where(derivatives > 0, direction_labels[0], direction_labels[1]),
            'Abs_Derivative': np.abs(derivatives),
            'Derivative_Order': derivative_results['derivative_order'],
            'Derivative_Name': derivative_name
        })
        
        # Filter by date range
        if start_date is not None:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
            
        return df
    
    def _get_derivative_name(self, order: int) -> str:
        """Get human-readable name for derivative order"""
        names = {1: "1st derivative (rate)", 
                2: "2nd derivative (acceleration)",
                3: "3rd derivative (jerk)",
                4: "4th derivative",
                5: "5th derivative"}
        return names.get(order, f"{order}th derivative")
    
    def _print_derivative_summary(self, results: Dict):
        """Print summary of derivative analysis"""
        derivative_name = results['derivative_name']
        time_unit = results['time_unit']
        order = results['derivative_order']
        
        unit_suffix = f"per_{time_unit}"
        if order > 1:
            unit_suffix += f"^{order}"
            
        print(f"Mean {derivative_name}: {results['mean_derivative']:.6f} {unit_suffix}")
        print(f"Range: [{results['derivative_range'][0]:.6f}, {results['derivative_range'][1]:.6f}]")
        print(f"Positive {results['direction_labels'][0]} periods: {results['n_positive']} "
              f"({results['positive_fraction']*100:.1f}%)")
        print(f"Negative {results['direction_labels'][1]} periods: {results['n_negative']} "
              f"({results['negative_fraction']*100:.1f}%)")
        
        if not np.isnan(results['mean_positive_derivative']):
            print(f"Mean positive {derivative_name}: {results['mean_positive_derivative']:.6f}")
            print(f"Max positive {derivative_name}: {results['max_positive_derivative']:.6f}")
            
        if not np.isnan(results['mean_negative_derivative']):
            print(f"Mean negative {derivative_name}: {results['mean_negative_derivative']:.6f}")
            print(f"Max negative {derivative_name}: {results['max_negative_derivative']:.6f}")


def analyze_derivative_patterns(data: pd.DataFrame,
                              series_column: str = 'Series',
                              time_column: str = 'Date', 
                              value_column: str = 'Value',
                              series_names: Optional[List[str]] = None,
                              derivative_order: int = 1,
                              time_unit: str = 'days',
                              window_size: int = 365,
                              window_unit: str = 'days',
                              step_size: int = 30,
                              direction_labels: List[str] = ['increasing', 'decreasing'],
                              min_change_threshold: float = 0) -> Dict:
    """
    Main function for derivative pattern analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input time series data
    series_column : str
        Column containing series identifiers
    time_column : str
        Column containing time/date values
    value_column : str
        Column containing values to analyze
    series_names : list, optional
        List of series names to analyze (if None, analyzes all)
    derivative_order : int
        Order of derivative (1=velocity, 2=acceleration, etc.)
    time_unit : str
        Time unit for derivatives ('days', 'weeks', etc.)
    window_size : int
        Size of rolling windows for regime detection
    window_unit : str
        Unit for window sizing
    step_size : int
        Step size between rolling windows
    direction_labels : list
        Labels for positive/negative directions
    min_change_threshold : float
        Minimum change threshold for valid derivatives
        
    Returns:
    --------
    dict : Complete analysis results
    """
    
    analyzer = DerivativeAnalyzer(data, series_column, time_column, value_column)
    
    derivative_names = {1: "1st derivative (rate)", 2: "2nd derivative (acceleration)",
                       3: "3rd derivative (jerk)", 4: "4th derivative", 5: "5th derivative"}
    derivative_name = derivative_names.get(derivative_order, f"{derivative_order}th derivative")
    
    print(f"=== {derivative_name.upper()} ANALYSIS ===")
    print(f"Time unit: {time_unit}")
    print(f"Rolling window: {window_size} {window_unit}")
    print(f"Step size: {step_size} {window_unit}\n")
    
    # Get series names if not specified
    if series_names is None:
        if series_column in data.columns:
            series_names = data[series_column].unique().tolist()
        else:
            series_names = ["Single_Series"]
            
    results_list = {}
    
    for series_name in series_names:
        # Calculate derivative metrics
        derivative_metrics = analyzer.calculate_derivative_metrics(
            series_name=series_name if series_name != "Single_Series" else None,
            derivative_order=derivative_order,
            time_unit=time_unit,
            min_change_threshold=min_change_threshold,
            direction_labels=direction_labels
        )
        
        # Detect derivative regimes
        if derivative_metrics is not None:
            regime_analysis = analyzer.detect_derivative_regimes(
                derivative_results=derivative_metrics,
                window_size=window_size,
                window_unit=window_unit,
                step_size=step_size
            )
            
            results_list[series_name] = {
                'derivative_metrics': derivative_metrics,
                'regime_analysis': regime_analysis
            }
            
    if len(results_list) > 0:
        print(f"\n=== {derivative_name.upper()} SUMMARY ===")
        summary_table = create_derivative_summary(results_list)
        
        return {
            'results': results_list,
            'summary': summary_table,
            'analysis_params': {
                'derivative_order': derivative_order,
                'time_unit': time_unit,
                'window_size': window_size,
                'window_unit': window_unit,
                'direction_labels': direction_labels
            }
        }
    else:
        print("No series had sufficient data for analysis")
        return None


def create_derivative_summary(results_list: Dict) -> pd.DataFrame:
    """Create summary table of derivative analysis results"""
    
    if len(results_list) == 0:
        print("No derivative results to summarize")
        return None
        
    summary_data = []
    
    for name, result in results_list.items():
        if result['derivative_metrics'] is None:
            continue
            
        dm = result['derivative_metrics']
        
        # Extract regime change dates
        deriv_change_date = "None"
        direction_change_date = "None" 
        vol_change_date = "None"
        
        if result['regime_analysis'] and result['regime_analysis']['regime_changes']:
            rc = result['regime_analysis']['regime_changes']
            
            if 'derivative_change' in rc:
                deriv_change_date = rc['derivative_change']['changepoint_date'].strftime('%Y-%m-%d')
            if 'direction_change' in rc:
                direction_change_date = rc['direction_change']['changepoint_date'].strftime('%Y-%m-%d')
            if 'volatility_change' in rc:
                vol_change_date = rc['volatility_change']['changepoint_date'].strftime('%Y-%m-%d')
                
        summary_data.append({
            'Series': name,
            'Derivative_Order': dm['derivative_order'],
            'Mean_Derivative': round(dm['mean_derivative'], 6),
            'Positive_Rate': round(dm['mean_positive_derivative'] if not np.isnan(dm['mean_positive_derivative']) else 0, 6),
            'Negative_Rate': round(dm['mean_negative_derivative'] if not np.isnan(dm['mean_negative_derivative']) else 0, 6),
            'Positive_Fraction': round(dm['positive_fraction'], 3),
            'Max_Positive': round(dm['max_positive_derivative'] if not np.isnan(dm['max_positive_derivative']) else 0, 6),
            'Max_Negative': round(dm['max_negative_derivative'] if not np.isnan(dm['max_negative_derivative']) else 0, 6),
            'Derivative_Change_Date': deriv_change_date,
            'Direction_Change_Date': direction_change_date,
            'Volatility_Change_Date': vol_change_date
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    return summary_df


# Convenience functions
def calculate_first_derivative(data: pd.DataFrame, **kwargs) -> Dict:
    """Calculate first derivative (velocity) analysis"""
    return analyze_derivative_patterns(data, derivative_order=1, **kwargs)

def calculate_second_derivative(data: pd.DataFrame, **kwargs) -> Dict:
    """Calculate second derivative (acceleration) analysis"""
    return analyze_derivative_patterns(data, derivative_order=2, **kwargs)

def calculate_acceleration(data: pd.DataFrame, **kwargs) -> Dict:
    """Calculate acceleration with appropriate labels"""
    kwargs['direction_labels'] = ['accelerating', 'decelerating']
    return analyze_derivative_patterns(data, derivative_order=2, **kwargs)


if __name__ == "__main__":
    print("Derivative Analysis module loaded successfully!")
    print("Main functions:")
    print("  - analyze_derivative_patterns(): Multi-series nth-order derivative analysis")
    print("  - calculate_first_derivative(), calculate_second_derivative(), calculate_acceleration()")
    print("  - DerivativeAnalyzer class for advanced usage")
    print("\nExample usage:")
    print("  results = analyze_derivative_patterns(df, derivative_order=1)")
    print("  acceleration_results = calculate_acceleration(df)")