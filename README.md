A collection of things I make while exploring processes that exhibit long-term memory, non-stationarity, and other nasty stuff like self-similarity. I am surprised that I could not find tools for these because everything falls under these categories..

## Python utilities

The `Python/` directory currently includes:

* `derivative_analysis.py` – routines for nth-order differencing, rolling windows, and regime-change inspection.
* `hurst.py` – a pure NumPy/Pandas implementation of a rescaled-range (R/S) Hurst exponent estimator that returns both the exponent and the underlying log–log regression diagnostics.
* `example_usage.py` – synthetic data generators and walkthroughs for the derivative workflow.

Install the Python requirements with:

```bash
pip install -r Python/requirements.txt
```

To estimate a Hurst exponent from the command line or a notebook:

```python
import pandas as pd
from hurst import hurst_rs

series = pd.Series(...)  # your time series data
result = hurst_rs(series)
print(f"Hurst exponent: {result['hurst']:.3f}")
```

The returned dictionary contains window sizes, mean rescaled ranges, and the fitted regression so you can visualise or validate the scaling relationship.
