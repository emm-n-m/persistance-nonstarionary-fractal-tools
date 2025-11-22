#!/usr/bin/env python3
"""
Example script for analyzing EEG data

Works with:
- PhysioNet EDF files
- MILimbEEG CSV files
- Generic CSV with channel columns

Usage:
    python eeg_example.py path/to/eeg_file.csv
    python eeg_example.py path/to/eeg_file.edf --channel C3
"""

import argparse
from pathlib import Path
import sys

# Add Python directory to path if running standalone
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from eeg_loader import EEGLoader, EEGAnalyzer, quick_analysis


def analyze_single_file(file_path: Path, channel: int = 0, output_dir: Path = None):
    """
    Analyze a single EEG file.

    Parameters
    ----------
    file_path : Path
        Path to EEG file
    channel : int
        Channel index to analyze
    output_dir : Path
        Directory to save results
    """
    print("="*60)
    print("EEG Analysis Example")
    print("="*60)

    results = quick_analysis(file_path, channel=channel, output_dir=output_dir)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

    return results


def compare_channels(file_path: Path, channels: list = None, output_dir: Path = None):
    """
    Compare multiple channels from the same file.

    Parameters
    ----------
    file_path : Path
        Path to EEG file
    channels : list
        List of channel indices or names. If None, analyzes first 3.
    output_dir : Path
        Directory to save results
    """
    import matplotlib.pyplot as plt

    print("="*60)
    print("Multi-Channel Comparison")
    print("="*60)

    # Load data
    loader = EEGLoader(file_path)
    loader.load()
    print(loader.summary())

    # Select channels
    if channels is None:
        channels = list(range(min(3, len(loader.channels))))

    # Analyze each channel
    results = {}
    for ch in channels:
        if isinstance(ch, int):
            ch_name = loader.channels[ch]
        else:
            ch_name = ch

        print(f"\n--- Analyzing {ch_name} ---")

        data = loader.get_channel(ch)
        analyzer = EEGAnalyzer(data, ch_name, loader.sampling_rate)

        # Calculate Hurst
        hurst_result = analyzer.calculate_hurst()

        results[ch_name] = {
            'stats': analyzer.basic_stats(),
            'hurst': hurst_result
        }

        if 'hurst' in hurst_result:
            print(f"  Hurst: {hurst_result['hurst']:.4f} (R²: {hurst_result['r_squared']:.4f})")

    # Create comparison plot
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3*len(channels)))
    if len(channels) == 1:
        axes = [axes]

    for idx, ch in enumerate(channels):
        if isinstance(ch, int):
            ch_name = loader.channels[ch]
        else:
            ch_name = ch

        data = loader.get_channel(ch)

        # Downsample for plotting
        if len(data) > 5000:
            indices = np.linspace(0, len(data) - 1, 5000, dtype=int)
            plot_data = data[indices]
        else:
            plot_data = data

        axes[idx].plot(plot_data, linewidth=0.5, alpha=0.7)
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(f'{ch_name} - H={results[ch_name]["hurst"].get("hurst", "N/A"):.3f}')
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample')
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / f"{file_path.stem}_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")

    plt.show()

    return results


def batch_analysis(directory: Path, pattern: str = "*.csv", output_dir: Path = None):
    """
    Analyze all files matching pattern in directory.

    Parameters
    ----------
    directory : Path
        Directory containing EEG files
    pattern : str
        Glob pattern for files (default: "*.csv")
    output_dir : Path
        Directory to save results
    """
    import numpy as np

    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' found in {directory}")
        return

    print("="*60)
    print(f"Batch Analysis: {len(files)} files")
    print("="*60)

    all_results = []

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        try:
            loader = EEGLoader(file_path)
            loader.load()

            # Analyze first channel
            data = loader.get_channel(0)
            analyzer = EEGAnalyzer(data, loader.channels[0], loader.sampling_rate)

            hurst_result = analyzer.calculate_hurst()
            stats = analyzer.basic_stats()

            result = {
                'file': file_path.name,
                'channel': loader.channels[0],
                'hurst': hurst_result.get('hurst', np.nan),
                'r_squared': hurst_result.get('r_squared', np.nan),
                'mean': stats['mean'],
                'std': stats['std']
            }

            all_results.append(result)
            print(f"  Hurst: {result['hurst']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Create summary DataFrame
    import pandas as pd
    summary_df = pd.DataFrame(all_results)

    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(summary_df.describe())

    # Save summary
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        summary_path = output_dir / "batch_analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    return summary_df


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze EEG data for long-range correlations"
    )
    parser.add_argument(
        'file_path',
        type=Path,
        help='Path to EEG file or directory'
    )
    parser.add_argument(
        '--channel',
        type=int,
        default=0,
        help='Channel index to analyze (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output'),
        help='Output directory for plots (default: output/)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple channels'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all files in directory'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.csv',
        help='File pattern for batch processing (default: *.csv)'
    )

    args = parser.parse_args()

    # Import numpy here for batch analysis
    import numpy as np

    # Process based on mode
    if args.batch:
        if not args.file_path.is_dir():
            print(f"Error: {args.file_path} is not a directory")
            return 1

        batch_analysis(args.file_path, pattern=args.pattern, output_dir=args.output)

    elif args.compare:
        compare_channels(args.file_path, output_dir=args.output)

    else:
        analyze_single_file(args.file_path, channel=args.channel, output_dir=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
