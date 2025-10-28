#!/usr/bin/env python3

import os
import sys
import json
from datetime import datetime

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - handled explicitly below
    try:
        import pytest
    except ModuleNotFoundError as exc:  # pragma: no cover - pytest absent
        raise ModuleNotFoundError("pandas is required to run this script") from exc

    pytest.skip(
        "pandas is required for derivative analysis smoke test", allow_module_level=True
    )

# Add Python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Python'))

from derivative_analysis import analyze_derivative_patterns, DerivativeAnalyzer

def save_results_to_files(results, output_dir='results'):
    """Save analysis results to various file formats"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving results to {output_dir}/ directory...")
    
    # 1. Save summary table as CSV
    if results['summary'] is not None:
        summary_path = os.path.join(output_dir, f'derivative_summary_{timestamp}.csv')
        results['summary'].to_csv(summary_path, index=False)
        print(f"[+] Summary table saved: {summary_path}")
    
    # 2. Save individual derivative time series for each reservoir
    # Create a dummy dataframe with the required columns for the analyzer
    dummy_df = pd.DataFrame({'Date': [], 'Reservoir': [], 'Value': []})
    analyzer = DerivativeAnalyzer(dummy_df, 'Reservoir', 'Date', 'Value')
    
    all_derivatives_df = pd.DataFrame()
    
    for reservoir_name, reservoir_results in results['results'].items():
        if reservoir_results['derivative_metrics'] is not None:
            # Extract derivative time series
            derivative_df = analyzer.extract_derivative_timeseries(reservoir_results['derivative_metrics'])
            derivative_df['Reservoir'] = reservoir_name
            
            # Save individual reservoir file
            reservoir_path = os.path.join(output_dir, f'derivatives_{reservoir_name}_{timestamp}.csv')
            derivative_df.to_csv(reservoir_path, index=False)
            print(f"[+] {reservoir_name} derivatives saved: {reservoir_path}")
            
            # Add to combined dataframe
            all_derivatives_df = pd.concat([all_derivatives_df, derivative_df], ignore_index=True)
    
    # 3. Save combined derivatives file
    if not all_derivatives_df.empty:
        combined_path = os.path.join(output_dir, f'all_derivatives_{timestamp}.csv')
        all_derivatives_df.to_csv(combined_path, index=False)
        print(f"[+] Combined derivatives saved: {combined_path}")
    
    # 4. Save regime change information as JSON
    regime_data = {}
    for reservoir_name, reservoir_results in results['results'].items():
        if reservoir_results['regime_analysis'] and reservoir_results['regime_analysis']['regime_changes']:
            regime_changes = reservoir_results['regime_analysis']['regime_changes']
            
            # Convert datetime objects to strings for JSON serialization
            regime_data[reservoir_name] = {}
            for change_type, change_info in regime_changes.items():
                if isinstance(change_info, dict):
                    regime_data[reservoir_name][change_type] = {}
                    for key, value in change_info.items():
                        if hasattr(value, 'strftime'):  # datetime object
                            regime_data[reservoir_name][change_type][key] = value.strftime('%Y-%m-%d')
                        else:
                            regime_data[reservoir_name][change_type][key] = value
    
    if regime_data:
        regime_path = os.path.join(output_dir, f'regime_changes_{timestamp}.json')
        with open(regime_path, 'w') as f:
            json.dump(regime_data, f, indent=2)
        print(f"[+] Regime changes saved: {regime_path}")
    
    # 5. Save analysis parameters
    params_path = os.path.join(output_dir, f'analysis_parameters_{timestamp}.json')
    with open(params_path, 'w') as f:
        json.dump(results['analysis_params'], f, indent=2)
    print(f"[+] Analysis parameters saved: {params_path}")
    
    return {
        'output_directory': output_dir,
        'timestamp': timestamp,
        'files_created': [f for f in os.listdir(output_dir) if timestamp in f]
    }

def test_derivative_analysis():
    print("Testing derivative_analysis.py with water_reserves.csv")
    
    # Load the data
    data_path = os.path.join('data', 'water_reserves.csv')
    print(f"Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Run the derivative analysis
        print("\n" + "="*50)
        print("RUNNING DERIVATIVE ANALYSIS")
        print("="*50)
        
        # Test with the expected column names based on the CSV structure
        results = analyze_derivative_patterns(
            data=df,
            series_column='Reservoir',  # Changed from 'Series' to 'Reservoir'
            time_column='Date',
            value_column='Value',
            derivative_order=1,
            time_unit='days'
        )
        
        if results:
            print("\nAnalysis completed successfully!")
            print(f"Number of series analyzed: {len(results['results'])}")
            
            # Save results to files
            file_info = save_results_to_files(results)
            print(f"\n[+] All results saved to '{file_info['output_directory']}' directory")
            print(f"Files created: {file_info['files_created']}")
            
            return results, file_info
        else:
            print("\nAnalysis returned no results")
            return None, None
            
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, file_info = test_derivative_analysis()
    
    if results:
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"[+] Analysis completed for {len(results['results'])} reservoirs")
        print(f"[+] Results saved in '{file_info['output_directory']}' directory")
        print("\nTo load results later:")
        print(f"  import pandas as pd")
        print(f"  summary = pd.read_csv('{file_info['output_directory']}/derivative_summary_{file_info['timestamp']}.csv')")
        print(f"  all_derivatives = pd.read_csv('{file_info['output_directory']}/all_derivatives_{file_info['timestamp']}.csv')")
        print(f"  # Individual reservoir files also available")