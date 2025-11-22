"""
Generic EEG Data Loader and Analyzer

Supports multiple EEG dataset formats:
- PhysioNet EDF files
- CSV files (MILimbEEG, custom formats)
- Common column naming conventions

Performs exploratory analysis:
- Hurst exponent estimation
- Long-range correlation detection
- Basic visualization
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import optional dependencies
try:
    from hurst import hurst_rs
    HAS_HURST = True
except ImportError:
    HAS_HURST = False
    warnings.warn("hurst.py not found. Hurst exponent calculation disabled.")

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("MNE not installed. EDF file support disabled. Install: pip install mne")


class EEGLoader:
    """
    Generic EEG data loader supporting multiple formats.

    Automatically detects format and extracts time series from channels.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize loader with file path.

        Parameters
        ----------
        file_path : str or Path
            Path to EEG data file (CSV or EDF)
        """
        self.file_path = Path(file_path)
        self.data = None
        self.channels = None
        self.sampling_rate = None
        self.metadata = {}

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self) -> pd.DataFrame:
        """
        Load EEG data, auto-detecting format.

        Returns
        -------
        pd.DataFrame
            EEG data with channels as columns
        """
        suffix = self.file_path.suffix.lower()

        if suffix == '.edf':
            return self._load_edf()
        elif suffix == '.csv':
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_edf(self) -> pd.DataFrame:
        """Load EDF file using MNE."""
        if not HAS_MNE:
            raise ImportError(
                "MNE required for EDF files. Install: pip install mne"
            )

        raw = mne.io.read_raw_edf(self.file_path, preload=True, verbose=False)

        # Extract data
        data = raw.get_data().T  # Transpose to (samples, channels)
        self.channels = raw.ch_names
        self.sampling_rate = raw.info['sfreq']

        # Create DataFrame
        self.data = pd.DataFrame(data, columns=self.channels)

        # Add time column
        times = np.arange(len(self.data)) / self.sampling_rate
        self.data.insert(0, 'Time', times)

        self.metadata = {
            'format': 'EDF',
            'n_channels': len(self.channels),
            'n_samples': len(self.data),
            'sampling_rate': self.sampling_rate,
            'duration_seconds': len(self.data) / self.sampling_rate
        }

        return self.data

    def _load_csv(self) -> pd.DataFrame:
        """
        Load CSV file, attempting to detect format.

        Supports:
        - MILimbEEG format (electrodes as columns)
        - Time series with channel columns
        - Wide format with sample numbers
        """
        # Try reading with different separators
        for sep in [',', '\t', ';']:
            try:
                df = pd.read_csv(self.file_path, sep=sep, nrows=5)
                if len(df.columns) > 1:
                    break
            except:
                continue

        # Read full file
        df = pd.read_csv(self.file_path, sep=sep)

        # Detect format
        if self._is_milimb_format(df):
            self.data = self._parse_milimb(df)
        else:
            self.data = self._parse_generic_csv(df)

        self.channels = [col for col in self.data.columns if col not in ['Time', 'Sample']]

        # Estimate sampling rate if time column exists
        if 'Time' in self.data.columns:
            time_diffs = self.data['Time'].diff().dropna()
            if len(time_diffs) > 0:
                self.sampling_rate = 1.0 / time_diffs.median()

        self.metadata = {
            'format': 'CSV',
            'n_channels': len(self.channels),
            'n_samples': len(self.data),
            'sampling_rate': self.sampling_rate
        }

        return self.data

    def _is_milimb_format(self, df: pd.DataFrame) -> bool:
        """Detect if CSV is MILimbEEG format."""
        # MILimbEEG has first row as electrode names, first column as sample number
        first_col = df.columns[0]
        return (first_col in ['Sample', '0'] or
                str(df.iloc[0, 0]).isdigit() or
                len(df.columns) in [16, 17])  # 16 electrodes + sample column

    def _parse_milimb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse MILimbEEG format."""
        # First column is sample number, rest are electrodes
        if df.columns[0] in ['Sample', '0']:
            df = df.iloc[1:]  # Skip header row if present

        # Rename columns
        n_cols = len(df.columns)
        new_cols = ['Sample'] + [f'Ch{i}' for i in range(n_cols - 1)]
        df.columns = new_cols

        return df.astype(float)

    def _parse_generic_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse generic CSV format."""
        # Look for time column
        time_cols = ['Time', 'time', 'Timestamp', 'timestamp', 't']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break

        # If no time column, create one
        if time_col is None:
            df.insert(0, 'Time', np.arange(len(df)))
        elif time_col != 'Time':
            df = df.rename(columns={time_col: 'Time'})

        return df

    def get_channel(self, channel: Union[str, int]) -> np.ndarray:
        """
        Extract single channel as numpy array.

        Parameters
        ----------
        channel : str or int
            Channel name or index

        Returns
        -------
        np.ndarray
            Channel data
        """
        if self.data is None:
            self.load()

        if isinstance(channel, int):
            channel = self.channels[channel]

        return self.data[channel].values

    def get_channels(self, channels: Optional[List[Union[str, int]]] = None) -> Dict[str, np.ndarray]:
        """
        Extract multiple channels.

        Parameters
        ----------
        channels : list, optional
            List of channel names or indices. If None, returns all.

        Returns
        -------
        dict
            Dictionary mapping channel names to data arrays
        """
        if self.data is None:
            self.load()

        if channels is None:
            channels = self.channels

        result = {}
        for ch in channels:
            if isinstance(ch, int):
                ch = self.channels[ch]
            result[ch] = self.data[ch].values

        return result

    def summary(self) -> str:
        """Get summary of loaded data."""
        if self.data is None:
            return "No data loaded. Call load() first."

        info = [
            "="*50,
            "EEG Data Summary",
            "="*50,
            f"File: {self.file_path.name}",
            f"Format: {self.metadata.get('format', 'Unknown')}",
            f"Channels: {self.metadata.get('n_channels', 'Unknown')}",
            f"Samples: {self.metadata.get('n_samples', 'Unknown')}",
        ]

        if self.sampling_rate:
            info.append(f"Sampling Rate: {self.sampling_rate:.1f} Hz")
            if 'duration_seconds' in self.metadata:
                info.append(f"Duration: {self.metadata['duration_seconds']:.1f} seconds")

        info.append("\nAvailable Channels:")
        for i, ch in enumerate(self.channels[:10]):  # Show first 10
            info.append(f"  [{i}] {ch}")

        if len(self.channels) > 10:
            info.append(f"  ... and {len(self.channels) - 10} more")

        return "\n".join(info)


class EEGAnalyzer:
    """
    Analyze EEG time series for long-range correlations and properties.
    """

    def __init__(self, data: np.ndarray, channel_name: str = "EEG",
                 sampling_rate: Optional[float] = None):
        """
        Initialize analyzer.

        Parameters
        ----------
        data : np.ndarray
            EEG channel data
        channel_name : str
            Name of channel for labeling
        sampling_rate : float, optional
            Sampling rate in Hz
        """
        self.data = np.asarray(data, dtype=float)
        self.channel_name = channel_name
        self.sampling_rate = sampling_rate

        # Remove NaNs
        self.data = self.data[~np.isnan(self.data)]

    def calculate_hurst(self, min_window: int = 10, num_windows: int = 20) -> Dict:
        """
        Calculate Hurst exponent.

        Returns
        -------
        dict
            Hurst estimation results
        """
        if not HAS_HURST:
            return {"error": "hurst.py not available"}

        try:
            result = hurst_rs(
                self.data,
                min_window=min_window,
                num_windows=num_windows
            )
            return result
        except Exception as e:
            return {"error": str(e)}

    def basic_stats(self) -> Dict:
        """Calculate basic statistics."""
        return {
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'range': float(np.ptp(self.data)),
            'n_samples': len(self.data)
        }

    def plot_overview(self, save_path: Optional[Path] = None, max_points: int = 10000):
        """
        Create overview plot with time series and basic analysis.

        Parameters
        ----------
        save_path : Path, optional
            If provided, save plot to this path
        max_points : int
            Maximum points to plot (downsample if needed)
        """
        # Downsample if needed for plotting
        if len(self.data) > max_points:
            indices = np.linspace(0, len(self.data) - 1, max_points, dtype=int)
            plot_data = self.data[indices]
            plot_indices = indices
        else:
            plot_data = self.data
            plot_indices = np.arange(len(self.data))

        # Create time axis
        if self.sampling_rate:
            time_axis = plot_indices / self.sampling_rate
            xlabel = "Time (seconds)"
        else:
            time_axis = plot_indices
            xlabel = "Sample"

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'EEG Analysis: {self.channel_name}', fontsize=14, fontweight='bold')

        # 1. Time series
        axes[0, 0].plot(time_axis, plot_data, linewidth=0.5, alpha=0.7)
        axes[0, 0].set_xlabel(xlabel)
        axes[0, 0].set_ylabel('Amplitude (μV)')
        axes[0, 0].set_title('Raw Signal')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Histogram
        axes[0, 1].hist(self.data, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Amplitude (μV)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Power spectrum
        freqs, psd = self._compute_psd()
        axes[1, 0].semilogy(freqs, psd)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power Spectral Density')
        axes[1, 0].set_title('Power Spectrum')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Autocorrelation
        lags, acf = self._compute_acf(max_lag=min(500, len(self.data) // 4))
        axes[1, 1].plot(lags, acf)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Autocorrelation Function')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def _compute_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        from scipy import signal

        if self.sampling_rate:
            fs = self.sampling_rate
        else:
            fs = 1.0

        freqs, psd = signal.welch(self.data, fs=fs, nperseg=min(256, len(self.data) // 4))
        return freqs, psd

    def _compute_acf(self, max_lag: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Compute autocorrelation function."""
        # Demean
        data = self.data - np.mean(self.data)

        # Compute autocorrelation
        acf = np.correlate(data, data, mode='full')
        acf = acf[len(acf) // 2:]  # Take positive lags
        acf = acf / acf[0]  # Normalize

        lags = np.arange(min(max_lag, len(acf)))
        return lags, acf[:len(lags)]


def quick_analysis(file_path: Union[str, Path],
                   channel: Union[str, int] = 0,
                   output_dir: Optional[Path] = None) -> Dict:
    """
    Quick analysis of EEG file.

    Parameters
    ----------
    file_path : str or Path
        Path to EEG file
    channel : str or int
        Channel to analyze (name or index)
    output_dir : Path, optional
        Directory to save plots

    Returns
    -------
    dict
        Analysis results
    """
    print(f"Loading: {file_path}")

    # Load data
    loader = EEGLoader(file_path)
    loader.load()
    print(loader.summary())

    # Get channel
    data = loader.get_channel(channel)
    if isinstance(channel, int):
        channel_name = loader.channels[channel]
    else:
        channel_name = channel

    print(f"\nAnalyzing channel: {channel_name}")

    # Analyze
    analyzer = EEGAnalyzer(data, channel_name, loader.sampling_rate)

    # Basic stats
    stats = analyzer.basic_stats()
    print("\nBasic Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Hurst exponent
    if HAS_HURST:
        print("\nCalculating Hurst exponent...")
        hurst_result = analyzer.calculate_hurst()
        if 'error' not in hurst_result:
            print(f"  Hurst exponent: {hurst_result['hurst']:.4f}")
            print(f"  R²: {hurst_result['r_squared']:.4f}")

            # Interpretation
            h = hurst_result['hurst']
            if h < 0.5:
                interp = "Anti-persistent (mean-reverting)"
            elif h > 0.5:
                interp = "Persistent (long memory)"
            else:
                interp = "Random walk (no memory)"
            print(f"  Interpretation: {interp}")
        else:
            print(f"  Error: {hurst_result['error']}")

    # Plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plot_path = output_dir / f"{Path(file_path).stem}_{channel_name}.png"
    else:
        plot_path = None

    analyzer.plot_overview(save_path=plot_path)

    return {
        'stats': stats,
        'hurst': hurst_result if HAS_HURST else None,
        'channel': channel_name,
        'sampling_rate': loader.sampling_rate
    }


if __name__ == "__main__":
    print("EEG Loader Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from eeg_loader import EEGLoader, EEGAnalyzer, quick_analysis

    # Quick analysis of a file
    results = quick_analysis('data/eeg_file.csv', channel=0, output_dir='output')

    # Or manual workflow:
    loader = EEGLoader('data/eeg_file.csv')
    loader.load()
    print(loader.summary())

    # Get channel data
    data = loader.get_channel('C3')  # By name
    data = loader.get_channel(0)     # Or by index

    # Analyze
    analyzer = EEGAnalyzer(data, 'C3', loader.sampling_rate)
    hurst_result = analyzer.calculate_hurst()
    analyzer.plot_overview(save_path='output/analysis.png')
    """)
