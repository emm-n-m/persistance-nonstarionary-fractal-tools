# Copyright (c) 2025 [Emmanouil Mavrogiorgis, emm.n.black@gmail.com]
# All rights reserved.
#
# This project was developed with assistance from Claude Code by Anthropic.
# Claude Code is an AI-powered coding assistant (https://www.anthropic.com).
#
# Licensed under MIT License
# See LICENSE file for details

"""
Example usage of the ported derivative analysis tool
Demonstrates how to use the Python version with sample data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from derivative_analysis import analyze_derivative_patterns, DerivativeAnalyzer

def create_sample_data():
    """Create sample reservoir data similar to your R scripts"""
    
    # Generate sample dates (5 years of daily data)
    start_date = datetime(2019, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(1826)]
    
    # Create sample reservoirs with different characteristics
    np.random.seed(42)
    
    data = []
    
    # Reservoir 1: Gradual decline with seasonal pattern
    base_level = 1000
    trend = -0.2  # Slow decline
    seasonal_amplitude = 200
    noise_level = 50
    
    values1 = []
    for i, date in enumerate(dates):
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * i / 365.25)
        trend_component = trend * i
        noise = np.random.normal(0, noise_level)
        value = base_level + seasonal + trend_component + noise
        values1.append(max(0, value))  # Ensure non-negative
        
        data.append({
            'Date': date,
            'Reservoir': 'Gradual_Decline',
            'Value': values1[-1]
        })
    
    # Reservoir 2: Sharp drop in 2022 (regime change)
    values2 = []
    for i, date in enumerate(dates):
        seasonal = seasonal_amplitude * 0.5 * np.sin(2 * np.pi * i / 365.25)
        
        # Sharp drop starting in 2022
        if date >= datetime(2022, 1, 1):
            trend_component = -1.5 * (i - 1095)  # Accelerated decline after 2022
        else:
            trend_component = 0.1 * i  # Slight increase before 2022
            
        noise = np.random.normal(0, noise_level)
        value = base_level + seasonal + trend_component + noise
        values2.append(max(0, value))
        
        data.append({
            'Date': date,
            'Reservoir': 'Sharp_Decline_2022',
            'Value': values2[-1]
        })
    
    # Reservoir 3: Stable with high volatility
    values3 = []
    for i, date in enumerate(dates):
        seasonal = seasonal_amplitude * 1.5 * np.sin(2 * np.pi * i / 365.25)
        noise = np.random.normal(0, noise_level * 2)  # Higher volatility
        value = base_level + seasonal + noise
        values3.append(max(0, value))
        
        data.append({
            'Date': date,
            'Reservoir': 'High_Volatility',
            'Value': values3[-1]
        })
    
    return pd.DataFrame(data)

def demonstrate_basic_usage():
    """Demonstrate basic derivative analysis usage"""
    
    print("=== CREATING SAMPLE DATA ===")
    data = create_sample_data()
    print(f"Created data with {len(data)} observations")
    print(f"Reservoirs: {data['Reservoir'].unique()}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print()
    
    print("=== FIRST DERIVATIVE ANALYSIS (VELOCITY) ===")
    # Analyze first derivative (rate of change)
    velocity_results = analyze_derivative_patterns(
        data,
        derivative_order=1,
        time_unit='days',
        direction_labels=['rising', 'falling'],
        window_size=365,  # 1 year windows
        step_size=90     # 3 month steps
    )
    
    print("\n=== SECOND DERIVATIVE ANALYSIS (ACCELERATION) ===")
    # Analyze second derivative (acceleration)
    accel_results = analyze_derivative_patterns(
        data,
        derivative_order=2,
        time_unit='days',
        direction_labels=['accelerating', 'decelerating'],
        window_size=365,
        step_size=90
    )
    
    return data, velocity_results, accel_results

def demonstrate_advanced_usage():
    """Demonstrate advanced features using the DerivativeAnalyzer class"""
    
    print("\n=== ADVANCED USAGE WITH DERIVATIVEANALYZER CLASS ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize analyzer
    analyzer = DerivativeAnalyzer(data)
    
    # Focus on one reservoir with regime change
    reservoir = 'Sharp_Decline_2022'
    
    print(f"\nAnalyzing {reservoir} in detail...")
    
    # Calculate first derivative
    first_deriv = analyzer.calculate_derivative_metrics(
        series_name=reservoir,
        derivative_order=1,
        time_unit='days',
        direction_labels=['increasing', 'decreasing']
    )
    
    # Detect regimes with smaller windows to catch 2022 change
    regime_results = analyzer.detect_derivative_regimes(
        first_deriv,
        window_size=180,  # 6 month windows
        step_size=30,     # 1 month steps
        window_unit='days'
    )
    
    # Extract detailed time series
    derivative_ts = analyzer.extract_derivative_timeseries(
        first_deriv,
        start_date='2021-01-01',
        end_date='2023-12-31'
    )
    
    print(f"\nExtracted derivative time series: {len(derivative_ts)} observations")
    print("Sample of derivative time series:")
    print(derivative_ts.head(10))
    
    return analyzer, derivative_ts, regime_results

def create_visualization_example(data, velocity_results):
    """Create some basic visualizations"""
    
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Plot original data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reservoir Analysis Example', fontsize=16)
    
    # Plot 1: Original time series
    for reservoir in data['Reservoir'].unique():
        reservoir_data = data[data['Reservoir'] == reservoir]
        axes[0, 0].plot(reservoir_data['Date'], reservoir_data['Value'], 
                       label=reservoir, alpha=0.7)
    
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].set_ylabel('Water Level')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: First derivative for one reservoir
    if velocity_results and 'Sharp_Decline_2022' in velocity_results['results']:
        deriv_data = velocity_results['results']['Sharp_Decline_2022']['derivative_metrics']
        dates = pd.to_datetime(deriv_data['times'])
        derivatives = deriv_data['derivatives']
        
        axes[0, 1].plot(dates, derivatives, color='red', alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('First Derivative (Sharp Decline Reservoir)')
        axes[0, 1].set_ylabel('Rate of Change (per day)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Distribution of derivatives
    if velocity_results:
        all_derivatives = []
        for reservoir_name, result in velocity_results['results'].items():
            if result['derivative_metrics']:
                all_derivatives.extend(result['derivative_metrics']['derivatives'])
        
        axes[1, 0].hist(all_derivatives, bins=50, alpha=0.7, color='blue')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Distribution of First Derivatives')
        axes[1, 0].set_xlabel('Derivative Value')
        axes[1, 0].set_ylabel('Frequency')
    
    # Plot 4: Summary statistics
    if velocity_results and velocity_results['summary'] is not None:
        summary = velocity_results['summary']
        
        axes[1, 1].barh(summary['Series'], summary['Positive_Fraction'])
        axes[1, 1].set_title('Fraction of Positive Derivatives by Reservoir')
        axes[1, 1].set_xlabel('Positive Fraction')
    
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'derivative_analysis_example.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to '{output_path}'")
    plt.show()

def main():
    """Main demonstration function"""
    
    print("DERIVATIVE ANALYSIS - PYTHON VERSION")
    print("=" * 50)
    
    # Basic usage demonstration
    data, velocity_results, accel_results = demonstrate_basic_usage()
    
    # Advanced usage demonstration  
    analyzer, derivative_ts, regime_results = demonstrate_advanced_usage()
    
    # Create visualizations
    create_visualization_example(data, velocity_results)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key features demonstrated:")
    print("✓ Multi-series derivative analysis")
    print("✓ Regime change detection")
    print("✓ Detailed time series extraction")
    print("✓ Summary statistics")
    print("✓ Visualization capabilities")
    
    return {
        'data': data,
        'velocity_results': velocity_results,
        'accel_results': accel_results,
        'derivative_timeseries': derivative_ts
    }

if __name__ == "__main__":
    results = main()