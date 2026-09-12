"""
Sanity tests for src/model.py (Tasks 3.1, 3.5, and regime summary statistics).

These deliberately avoid re-fitting a real GaussianHMM (slow, and fitting
quality isn't what's under test here). label_states() and summarise_regimes()
only read a model's already-fitted attributes (means_, transmat_), so a
lightweight fake with just those attributes exercises the same code paths.

Run:  pytest tests/  (or: python tests/test_model.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import chronological_split, label_states, summarise_regimes


class _FakeModel:
    """Stand-in for a fitted GaussianHMM: only the attributes label_states()
    and summarise_regimes() actually read."""

    def __init__(self, means, transmat):
        self.means_ = np.asarray(means)
        self.transmat_ = np.asarray(transmat)


def test_label_states_sorts_by_mean_return_regardless_of_index_order():
    # State 0 happens to be the best-performing one here -- the labelling must
    # go by the VALUE of the mean return, not by state index.
    model = _FakeModel(
        means=[[+0.002, 0.01], [-0.003, 0.03], [+0.0001, 0.015]],
        transmat=np.eye(3),
    )
    labels = label_states(model)
    assert labels[0] == "Bull"      # highest mean return
    assert labels[1] == "Bear"      # lowest mean return
    assert labels[2] == "Sideways"  # in between


def test_label_states_is_invariant_to_relabelling():
    # Same three regimes, states permuted (the "label switching" problem) --
    # the NAMES assigned should be identical, only which index gets which
    # name changes.
    means_a = [[+0.002, 0.01], [-0.003, 0.03], [+0.0001, 0.015]]
    means_b = [means_a[2], means_a[0], means_a[1]]  # shuffled
    labels_a = label_states(_FakeModel(means_a, np.eye(3)))
    labels_b = label_states(_FakeModel(means_b, np.eye(3)))
    assert set(labels_a.values()) == set(labels_b.values()) == {"Bull", "Bear", "Sideways"}


def test_summarise_regimes_expected_duration_formula():
    # self-transition prob 0.9 -> expected duration 1/(1-0.9) = 10 days
    transmat = np.array([
        [0.90, 0.05, 0.05],
        [0.03, 0.95, 0.02],
        [0.10, 0.10, 0.80],
    ])
    means = [[+0.001, 0.01], [-0.002, 0.02], [0.0, 0.015]]
    model = _FakeModel(means, transmat)
    labels = label_states(model)
    summary = summarise_regimes(model, labels)

    assert np.isclose(summary.loc[0, "expected_duration_days"], 1 / (1 - 0.90))
    assert np.isclose(summary.loc[1, "expected_duration_days"], 1 / (1 - 0.95))
    assert np.isclose(summary.loc[2, "expected_duration_days"], 1 / (1 - 0.80))
    # Every state's self-transition probability is carried through unchanged
    # (summary is sorted by mean return, so reindex back to state order first).
    assert np.allclose(summary.sort_index()["self_transition_prob"].to_numpy(), np.diag(transmat))


def test_chronological_split_respects_train_end_date(monkeypatch):
    from src import config
    idx = pd.bdate_range("2024-01-01", periods=20)
    frame = pd.DataFrame({"log_return": np.zeros(20)}, index=idx)
    monkeypatch.setattr(config, "TRAIN_END_DATE", str(idx[9].date()))

    train, test = chronological_split(frame)
    assert train.index.max() <= idx[9]
    assert test.index.min() > idx[9]
    assert len(train) + len(test) == len(frame)


def test_chronological_split_raises_on_out_of_range_date(monkeypatch):
    from src import config
    idx = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"log_return": np.zeros(5)}, index=idx)
    monkeypatch.setattr(config, "TRAIN_END_DATE", "2019-01-01")  # before all data

    with pytest.raises(ValueError, match="empty"):
        chronological_split(frame)


if __name__ == "__main__":
    import sys

    class _FakeMonkeypatch:
        """Minimal monkeypatch stand-in so these tests also run without pytest."""

        def __init__(self):
            self._restore = []

        def setattr(self, obj, name, value):
            self._restore.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._restore):
                setattr(obj, name, old)

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            mp = _FakeMonkeypatch()
            try:
                if "monkeypatch" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                    fn(mp)
                else:
                    fn()
                print(f"  ok    {name}")
            except Exception as exc:  # noqa: BLE001 -- smoke-test runner
                failures += 1
                print(f"  FAIL  {name}: {exc}")
            finally:
                mp.undo()
    sys.exit(1 if failures else 0)
