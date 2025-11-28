# EEG Data Analysis - Quick Start Guide

Generic loader for EEG datasets to analyze long-range correlations and non-stationary properties.

## Supported Formats

- **EDF files** (PhysioNet format) - requires `mne` package
- **CSV files** (MILimbEEG, custom formats)
- Auto-detects format and channel structure

## Installation

```bash
# Install the package in editable mode
pip install -e .

# Optional: For EDF file support
pip install mne

# Optional: For better visualization
pip install seaborn
```

## Quick Start

### 1. Download Sample EEG Data

Choose one dataset to start:

**Option A: MILimbEEG (CSV, easiest)**
- Download: https://www.sciencedirect.com/science/article/pii/S2352340923006406
- Files: Already in CSV format
- Start with: 1-2 subjects (~100 files)

**Option B: PhysioNet on Kaggle (convenient)**
- Download: https://www.kaggle.com/datasets/brianleung2020/eeg-motor-movementimagery-dataset
- Browser download, no special tools
- Start with: 5-10 subjects

**Option C: PhysioNet CSV version**
- Download: https://data.mendeley.com/datasets/dpmtgrn8d8/4
- Pre-cleaned, CSV format
- Start with: 5 subjects

### 2. Analyze a Single File

```bash
# Analyze first channel, save plots to output/
python Python/eeg_example.py data/eeg_file.csv --output output/

# Analyze specific channel
python Python/eeg_example.py data/eeg_file.csv --channel 3

# For EDF files
python Python/eeg_example.py data/eeg_file.edf
```

### 3. Compare Multiple Channels

```bash
python Python/eeg_example.py data/eeg_file.csv --compare --output output/
```

### 4. Batch Process Multiple Files

```bash
# Process all CSV files in directory
python Python/eeg_example.py data/eeg_folder/ --batch --output results/

# Process specific pattern
python Python/eeg_example.py data/ --batch --pattern "subject_*.csv"
```

## Python API Usage

### Basic Analysis

```python
from eeg_loader import quick_analysis

# One-liner analysis
results = quick_analysis('data/eeg_file.csv', channel=0, output_dir='output')
```

### Manual Workflow

```python
from eeg_loader import EEGLoader, EEGAnalyzer

# Load data
loader = EEGLoader('data/eeg_file.csv')
loader.load()
print(loader.summary())

# Get specific channel
data = loader.get_channel('C3')  # By name
data = loader.get_channel(0)     # By index

# Analyze
analyzer = EEGAnalyzer(data, 'C3', loader.sampling_rate)

# Calculate Hurst exponent
hurst_result = analyzer.calculate_hurst()
print(f"Hurst exponent: {hurst_result['hurst']:.4f}")

# Basic statistics
stats = analyzer.basic_stats()
print(stats)

# Create plots
analyzer.plot_overview(save_path='output/analysis.png')
```

### Compare Multiple Channels

```python
from eeg_loader import EEGLoader, EEGAnalyzer

loader = EEGLoader('data/eeg_file.csv')
loader.load()

# Analyze all channels
results = {}
for channel_name in loader.channels[:5]:  # First 5 channels
    data = loader.get_channel(channel_name)
    analyzer = EEGAnalyzer(data, channel_name, loader.sampling_rate)

    hurst_result = analyzer.calculate_hurst()
    results[channel_name] = hurst_result['hurst']

# Compare
for ch, h in results.items():
    print(f"{ch}: H = {h:.4f}")
```

## What Gets Calculated

### Hurst Exponent
- **H < 0.5**: Anti-persistent (mean-reverting)
- **H ≈ 0.5**: Random walk (no memory)
- **H > 0.5**: Persistent (long memory) ← **Expected for EEG!**

EEG typically shows **H ≈ 0.6-0.9** due to long-range temporal correlations.

### Visualizations

The tool generates 4-panel plots:
1. **Raw signal** - Time series
2. **Distribution** - Histogram of amplitudes
3. **Power spectrum** - Frequency content
4. **Autocorrelation** - Temporal dependencies

## Expected Output

```
=================================================
EEG Data Summary
=================================================
File: eeg_recording.csv
Format: CSV
Channels: 16
Samples: 500
Sampling Rate: 250.0 Hz
Duration: 2.0 seconds

Available Channels:
  [0] Ch0
  [1] Ch1
  ...

Analyzing channel: Ch0

Basic Statistics:
  mean: 12.3456
  std: 45.6789
  min: -123.45
  max: 234.56
  range: 357.91
  n_samples: 500

Calculating Hurst exponent...
  Hurst exponent: 0.7234
  R²: 0.9876
  Interpretation: Persistent (long memory)

Plot saved to output/eeg_recording_Ch0.png
```

## Troubleshooting

### MNE not installed (for EDF files)
```bash
pip install mne
```

### Plot window doesn't show
The plots are saved to the output directory even if the window doesn't display.

### "hurst.py not found"
Make sure you're running from the project root or the package is installed:
```bash
pip install -e .
```

### Memory issues with large files
The loader automatically downsamples for plotting (max 10,000 points).
For analysis, it uses the full data.

## File Organization

Suggested structure:
```
persistance-nonstarionary-fractal-tools/
├── data/
│   └── eeg/
│       ├── subject_01/
│       │   ├── trial_001.csv
│       │   ├── trial_002.csv
│       │   └── ...
│       └── subject_02/
│           └── ...
├── output/
│   └── eeg_analysis/
│       ├── subject_01_trial_001_Ch0.png
│       └── ...
└── results/
    └── batch_analysis_summary.csv
```

## Next Steps

1. **Download 1-2 sample files** from one of the sources above
2. **Run quick analysis** on a single file to verify it works
3. **If interesting**, download more subjects
4. **Compare** different channels/subjects/tasks
5. **Explore** which brain regions show strongest long memory (motor cortex channels C3, C4, Cz are good candidates)

## References

**Long-range correlations in EEG:**
- Hurst exponent typically 0.6-0.9 in EEG
- Motor cortex shows strong persistence during movement
- Differences between rest/task states

**Common EEG channels:**
- **C3, C4, Cz**: Motor cortex (good for motor imagery tasks)
- **Fp1, Fp2**: Frontal (attention, planning)
- **O1, O2**: Occipital (visual processing)
