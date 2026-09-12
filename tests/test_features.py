"""
Sanity tests for src/features.py (Tasks 1.5-1.6).
Run:  pytest tests/  (or: python tests/test_features.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import add_rolling_volatility, build_observation_matrix, drop_warmup


def _returns_frame(n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"log_return": rng.normal(0, 0.01, size=n)}, index=idx)


def test_rolling_volatility_warmup_is_nan():
    frame = add_rolling_volatility(_returns_frame())
    window = config.VOLATILITY_WINDOW
    assert frame["volatility"].iloc[: window - 1].isna().all()
    assert frame["volatility"].iloc[window - 1:].notna().all()


def test_rolling_volatility_matches_manual_std():
    frame = _returns_frame(n=15)
    out = add_rolling_volatility(frame)
    w = config.VOLATILITY_WINDOW
    expected = frame["log_return"].iloc[:w].std()
    got = out["volatility"].iloc[w - 1]
    scale = np.sqrt(config.TRADING_DAYS_PER_YEAR) if config.ANNUALISE_VOLATILITY else 1.0
    assert np.isclose(got, expected * scale)


def test_drop_warmup_drops_exactly_window_minus_one():
    frame = add_rolling_volatility(_returns_frame(n=25))
    out = drop_warmup(frame)
    assert len(out) == 25 - (config.VOLATILITY_WINDOW - 1)
    assert out[config.FEATURE_COLUMNS].notna().all().all()


def test_build_observation_matrix_shape_and_dtype():
    frame = drop_warmup(add_rolling_volatility(_returns_frame(n=25)))
    matrix = build_observation_matrix(frame)
    assert matrix.shape == (len(frame), len(config.FEATURE_COLUMNS))
    assert matrix.dtype == np.float64
    assert matrix.flags["C_CONTIGUOUS"]


def test_build_observation_matrix_rejects_nan():
    frame = pd.DataFrame({
        "log_return": [0.01, np.nan],
        "volatility": [0.02, 0.02],
    }, index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="NaN"):
        build_observation_matrix(frame)


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except Exception as exc:  # noqa: BLE001 — smoke-test runner
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    sys.exit(1 if failures else 0)
