import pathlib
import sys

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for Hurst estimation tests")

# Make the Python utilities importable when tests are executed from the repo root.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "Python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from hurst import hurst_rs  # noqa: E402


def test_hurst_white_noise_close_to_half():
    rng = np.random.default_rng(seed=1234)
    samples = rng.standard_normal(4096)
    result = hurst_rs(samples, min_window=8, num_windows=15)
    assert 0.4 <= result["hurst"] <= 0.6


def test_hurst_persistent_series_exceeds_half():
    rng = np.random.default_rng(seed=4321)
    # Integrate white noise to create a persistent series (fBm-like)
    samples = np.cumsum(rng.standard_normal(4096))
    result = hurst_rs(samples, min_window=8, num_windows=15)
    assert result["hurst"] > 0.6


def test_hurst_rejects_short_series():
    with pytest.raises(ValueError):
        hurst_rs([1, 2, 3, 4, 5])
