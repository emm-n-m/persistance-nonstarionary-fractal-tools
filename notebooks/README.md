# Jupyter Notebooks for Exploration

Interactive notebooks for exploring time series with long memory and non-stationary properties.

**Philosophy:** These are for **exploration**, not production code. Change parameters, try stuff, see what happens!

## Notebooks

### 01_exploration.ipynb
**General time series exploration**

- Load and visualize different types of series
- Calculate Hurst exponent with different parameters
- Compare multiple datasets side-by-side
- Quick experiments with synthetic data

**Use when:** Starting fresh, testing ideas, comparing datasets

### 02_eeg_exploration.ipynb
**EEG-specific analysis**

- Load EEG files (CSV, EDF formats)
- Compare multiple channels
- Test parameter sensitivity
- Expected: H ≈ 0.6-0.9 for EEG

**Use when:** Analyzing EEG datasets, comparing brain regions

### 03_method_comparison.ipynb
**Compare different analysis methods**

- R/S analysis vs. Climacogram
- Parameter sensitivity tests
- Detrending effects
- Method agreement checks

**Use when:** Not sure which method to trust, testing robustness

## Getting Started

### 1. Install Dependencies

```bash
# Basic (required)
pip install jupyter numpy pandas matplotlib seaborn scipy

# Or install everything:
pip install -e ".[all]"
```

### 2. Launch Jupyter

```bash
jupyter notebook notebooks/
```

### 3. Open a Notebook

- **New to the project?** Start with `01_exploration.ipynb`
- **Have EEG data?** Try `02_eeg_exploration.ipynb`
- **Testing methods?** Use `03_method_comparison.ipynb`

## Tips

### Modify Freely!

These notebooks are meant to be **changed**:
- Duplicate cells and try variations
- Change parameters and re-run
- Add your own experiments
- Break things and learn!

### Save Your Work

```bash
# Save notebook with results
File → Save and Checkpoint

# Or export to HTML to share
File → Download as → HTML
```

### Common Workflow

1. **Generate test data** OR **load your data**
2. **Run analysis** with default parameters
3. **Look at plots** - do they make sense?
4. **Try different parameters** - does H change?
5. **Compare methods** - do they agree?
6. **If confused** → try simpler/synthetic data first

## Useful Code Snippets

### Load Your Own Data

```python
# CSV file
df = pd.read_csv('../data/your_file.csv')
series = df['column_name'].values

# Or use loader
from eeg_loader import EEGLoader
loader = EEGLoader('../data/eeg_file.csv')
loader.load()
data = loader.get_channel(0)
```

### Quick Hurst Calculation

```python
from hurst import hurst_rs

result = hurst_rs(your_data, min_window=10, num_windows=20)
print(f"H = {result['hurst']:.4f}")
```

### Plot Time Series

```python
plt.figure(figsize=(12, 4))
plt.plot(your_data)
plt.title('My Data')
plt.show()
```

## Troubleshooting

**"ModuleNotFoundError"**
```bash
pip install <missing_module>
```

**"Kernel died"**
- Restart kernel: Kernel → Restart
- Or restart Jupyter

**Plots don't show**
- Make sure you have `%matplotlib inline` at the top
- Try `plt.show()` after plots

**Results seem wrong**
- Check data quality (NaNs, outliers)
- Try synthetic data first
- Compare multiple methods
- Test parameter sensitivity

## What to Look For

### Good Signs ✓
- R² > 0.95 in R/S analysis
- Straight line in log-log plots
- Stable H across parameter changes
- Methods agree (R/S, Climacogram give similar results)

### Warning Signs ⚠
- R² < 0.9
- Curved lines in log-log plots
- H changes dramatically with parameters
- Methods disagree wildly

## Next Steps

After exploring in notebooks:

1. **Found interesting patterns?** Document them!
2. **Stable method?** Move to Python scripts for batch processing
3. **Need more data?** See EEG_QUICKSTART.md for dataset sources
4. **Ready to automate?** Check Python/eeg_example.py for CLI tools

## Remember

> "All models are wrong, but some are useful." - George Box

**Explore boldly. Question results. Try different approaches. See what sticks!**
