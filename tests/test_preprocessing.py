"""
Sanity tests for src/preprocessing.py (Tasks 1.3-1.4).

Run:  pytest tests/  (or: python -m pytest tests/)
These also run stand-alone with `python tests/test_preprocessing.py`, since
every test_* function is a plain function with plain asserts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataValidationError, add_log_returns, clean, validate


def _business_day_frame(prices, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=len(prices), name="Date")
    return pd.DataFrame({"Close": prices}, index=idx)


def test_add_log_returns_matches_manual_calculation():
    frame = pd.DataFrame({"price": [100.0, 110.0, 99.0]},
                          index=pd.bdate_range("2024-01-01", periods=3))
    out = add_log_returns(frame)

    expected = [np.log(110.0 / 100.0), np.log(99.0 / 110.0)]
    assert len(out) == 2  # first row (undefined return) is dropped
    assert np.allclose(out["log_return"].to_numpy(), expected)


def test_add_log_returns_drops_first_row_not_fills_zero():
    frame = pd.DataFrame({"price": [50.0, 51.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    out = add_log_returns(frame)
    # The first day has no previous price: it must be ABSENT, not a fabricated 0.0.
    assert len(out) == 1
    assert not (out["log_return"] == 0.0).any()


def test_validate_rejects_weekend_rows():
    prices = [100, 101, 102, 103, 104]
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                           "2024-01-06", "2024-01-07"])  # includes a Sat/Sun
    frame = pd.DataFrame({"Close": prices}, index=idx)
    with pytest.raises(DataValidationError, match="weekend"):
        validate(frame, verbose=False)


def test_validate_rejects_duplicate_dates():
    frame = _business_day_frame([100, 101, 102])
    frame = pd.concat([frame, frame.iloc[[0]]]).sort_index()
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(frame, verbose=False)


def test_validate_rejects_non_positive_price():
    frame = _business_day_frame([100.0, -5.0, 101.0])
    with pytest.raises(DataValidationError, match="non-positive"):
        validate(frame, verbose=False)


def test_validate_passes_clean_business_day_data():
    frame = _business_day_frame([100.0, 101.0, 99.5, 102.0])
    report = validate(frame, verbose=False)
    assert report["fatal"] == []


def test_clean_deduplicates_and_sorts():
    frame = _business_day_frame([100.0, 101.0, 102.0])
    shuffled = pd.concat([frame.iloc[[2]], frame, frame.iloc[[0]]])
    out = clean(shuffled)
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    assert list(out["price"]) == [100.0, 101.0, 102.0]


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
