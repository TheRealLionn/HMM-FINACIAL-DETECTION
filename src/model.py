"""
Sprint Tasks 3.1-3.5 — fit the Gaussian HMM, decode regimes, label them.

Pipeline:
    1. Split chronologically (Task 3.1): train = first ~4 years, test = last
       ~1 year. Chronological, not random — a random split would let the
       model "see the future" during training (look-ahead bias), which
       defeats the point of evaluating it on unseen time.
    2. Fit GaussianHMM(n_components=3) on the TRAINING matrix only
       (Task 3.2), with multiple random restarts (Task config.N_RESTARTS)
       because Baum-Welch/EM is only guaranteed to find a LOCAL optimum —
       the restart with the highest training log-likelihood is kept.
    3. Decode with Viterbi (Task 3.3) over the FULL five-year history using
       the parameters learned from the training set only. This mirrors the
       project's stated design (README section 1): Viterbi decodes the
       single most likely regime path over the whole sample, but the
       parameters that define what a "regime" looks like are never fit on
       the test period.
    4. Map states 0/1/2 to Bull/Bear/Sideways (Task 3.5) by sorting states on
       mean log return — this is the standard fix for HMM "label switching"
       (EM has no notion that state 0 should mean anything in particular;
       only the RELATIVE ordering of the fitted means is meaningful).

Run:  python -m src.model   (from the project root)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src import config
from src.features import build_observation_matrix, load_dataset

REGIME_ORDER = ("Bear", "Sideways", "Bull")  # low mean return -> high mean return
REGIMES_CSV = config.PROCESSED_DIR / "regimes.csv"


# ===========================================================================
# Task 3.1 — chronological split
# ===========================================================================
def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on config.TRAIN_END_DATE. No shuffling: order is the whole point."""
    train = frame.loc[: config.TRAIN_END_DATE]
    test = frame.loc[config.TRAIN_END_DATE:].iloc[1:]  # avoid double-counting the boundary date
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set — "
                          "check config.TRAIN_END_DATE against the data's date range.")
    print(f"  train : {len(train):4d} rows  {train.index.min().date()} to {train.index.max().date()}")
    print(f"  test  : {len(test):4d} rows  {test.index.min().date()} to {test.index.max().date()}")
    return train, test


# ===========================================================================
# Task 3.2 — fit with random restarts
# ===========================================================================
def fit_best_model(train_matrix: np.ndarray, verbose: bool = True) -> GaussianHMM:
    """Fit GaussianHMM N_RESTARTS times from different random inits, keep the best.

    Baum-Welch is EM: it climbs to the nearest local optimum of the
    likelihood surface, and where it lands depends on the (random) initial
    means/covariances. A single fit can land on a mediocre solution — e.g.
    two states that both describe "calm" and no state that describes the
    tails. Restarting from several seeds and keeping the highest TRAINING
    log-likelihood is the standard mitigation.

    A minority of seeds can also fail outright: EM occasionally drives one
    state's responsibility weights to ~0 (nothing is confidently assigned to
    it), which makes that state's covariance singular and raises a
    LinAlgError inside hmmlearn. This is a known, data-independent quirk of
    random-initialised full-covariance HMMs, not a bug in the pipeline — the
    fix is to skip that seed and keep going, not to let one bad restart
    crash the whole fit.
    """
    best_model: GaussianHMM | None = None
    best_score = -np.inf
    scores = []
    failed_seeds = []

    for i in range(config.N_RESTARTS):
        seed = config.RANDOM_SEED + i
        model = GaussianHMM(
            n_components=config.N_REGIMES,
            covariance_type=config.COVARIANCE_TYPE,
            n_iter=config.N_ITER,
            tol=config.TOL,
            random_state=seed,
            min_covar=config.MIN_COVAR,
        )
        try:
            model.fit(train_matrix)
            score = model.score(train_matrix)
        except (ValueError, np.linalg.LinAlgError) as exc:
            failed_seeds.append(seed)
            if verbose:
                print(f"  seed {seed}: fit failed ({exc.__class__.__name__}) — skipped")
            continue

        scores.append(score)
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError(
            f"All {config.N_RESTARTS} restarts failed to fit (last error on seeds "
            f"{failed_seeds}). Try increasing config.MIN_COVAR (e.g. to 1e-3 or "
            "1e-2), or set config.COVARIANCE_TYPE = \"diag\" as a more stable "
            "fallback."
        )

    scores = np.array(scores)
    if verbose:
        print(f"\n--- Task 3.2 model fitting ({config.N_RESTARTS} restarts, "
              f"{len(failed_seeds)} failed) ---")
        if failed_seeds:
            print(f"  failed seeds (skipped, not fatal): {failed_seeds}")
        print(f"  training log-likelihood: best {scores.max():.2f}, "
              f"worst {scores.min():.2f}, spread {scores.max() - scores.min():.2f}")
        print(f"  converged               : {best_model.monitor_.converged} "
              f"in {best_model.monitor_.iter} iterations")
        if scores.max() - scores.min() > 1.0:
            print("  NOTE: restarts disagree by a non-trivial margin -> EM has "
                  "multiple local optima here; keeping only the best-scoring fit "
                  "is doing real work, not a formality.")

    return best_model


# ===========================================================================
# Task 3.3 — Viterbi decoding over the full history
# ===========================================================================
def decode_full_history(model: GaussianHMM, full_matrix: np.ndarray) -> np.ndarray:
    """Most likely state path over the WHOLE sample, using train-fit parameters."""
    return model.predict(full_matrix)


# ===========================================================================
# Task 3.5 — label states by mean return (fixes label switching)
# ===========================================================================
def label_states(model: GaussianHMM) -> dict[int, str]:
    """Map raw state indices (0..K-1, meaningless order from EM) to regime names.

    Sorted on mean log return (column 0 of the observation vector): the
    lowest-return state is Bear, the highest is Bull, and whatever is left
    in between is Sideways. This is the standard fix for HMM "label
    switching" — EM never assigns a fixed meaning to state index 0, so the
    ONLY reliable way to name a state is by what it learned, not by its
    index.
    """
    mean_returns = model.means_[:, 0]
    order = np.argsort(mean_returns)  # ascending: worst return first

    if config.N_REGIMES != len(REGIME_ORDER):
        # Falls back to generic names if the model isn't 3 states — keeps this
        # function from silently mislabeling an unexpected configuration.
        names = [f"State {i}" for i in range(config.N_REGIMES)]
    else:
        names = list(REGIME_ORDER)

    label_map = {int(state_idx): names[rank] for rank, state_idx in enumerate(order)}

    # Sanity check: warn if two states' mean returns are too close to be
    # confidently distinguishable — this is exactly the failure mode where
    # sorting can flip which state is "really" Bull vs Sideways from one
    # random restart to the next.
    sorted_means = mean_returns[order]
    gaps = np.diff(sorted_means)
    if (gaps < 1e-5).any():
        print("  WARNING: two or more states have near-identical mean returns "
              "— Bull/Sideways/Bear labels may not be robust; inspect "
              "model.means_ before trusting the labels.")

    return label_map


# ===========================================================================
# Regime statistics
# ===========================================================================
def summarise_regimes(model: GaussianHMM, label_map: dict[int, str]) -> pd.DataFrame:
    """One row per state: mean return, mean volatility, expected duration, name."""
    diag = np.diag(model.transmat_)
    expected_duration = 1.0 / np.clip(1.0 - diag, 1e-6, None)  # 1/(1-A[i,i])

    rows = []
    for state_idx in range(config.N_REGIMES):
        rows.append({
            "state": state_idx,
            "regime": label_map[state_idx],
            "mean_log_return": model.means_[state_idx, 0],
            "mean_volatility": model.means_[state_idx, 1],
            "self_transition_prob": diag[state_idx],
            "expected_duration_days": expected_duration[state_idx],
        })
    summary = pd.DataFrame(rows).set_index("state").sort_values("mean_log_return")
    return summary


# ===========================================================================
# Pipeline entry point
# ===========================================================================
def run_model_pipeline(verbose: bool = True) -> dict:
    frame = load_dataset()

    print("--- Task 3.1 chronological split ---")
    train_frame, test_frame = chronological_split(frame)

    train_matrix = build_observation_matrix(train_frame)
    test_matrix = build_observation_matrix(test_frame)
    full_matrix = build_observation_matrix(frame)

    model = fit_best_model(train_matrix, verbose=verbose)

    # Held-out check: average per-observation log-likelihood should be in the
    # same ballpark on train and test. A test score far worse than train
    # signals overfitting to the training window's specific regimes.
    train_ll_avg = model.score(train_matrix) / len(train_matrix)
    test_ll_avg = model.score(test_matrix) / len(test_matrix)
    if verbose:
        print("\n--- held-out check ---")
        print(f"  avg log-likelihood / obs, train: {train_ll_avg:+.4f}")
        print(f"  avg log-likelihood / obs, test : {test_ll_avg:+.4f}")
        print(f"  gap                            : {train_ll_avg - test_ll_avg:+.4f}"
              "   (near 0 is healthy; a large positive gap suggests overfitting)")

    print("\n--- Task 3.3 Viterbi decoding (full history) ---")
    states = decode_full_history(model, full_matrix)
    switches = int((np.diff(states) != 0).sum())
    print(f"  decoded {len(states)} days, {switches} regime switches")

    print("\n--- Task 3.5 regime labelling ---")
    label_map = label_states(model)
    for state_idx, name in sorted(label_map.items()):
        print(f"  state {state_idx} -> {name}")

    summary = summarise_regimes(model, label_map)
    if verbose:
        print("\n--- regime summary ---")
        print(summary.to_string(float_format=lambda v: f"{v:+.5f}" if abs(v) < 1 else f"{v:.1f}"))

    regimes = frame.copy()
    regimes["state"] = states
    regimes["regime"] = [label_map[s] for s in states]
    regimes["split"] = np.where(regimes.index <= config.TRAIN_END_DATE, "train", "test")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    regimes.to_csv(REGIMES_CSV)
    if verbose:
        print(f"\n  saved -> {REGIMES_CSV}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_map": label_map}, config.MODEL_FILE)
    if verbose:
        print(f"  saved -> {config.MODEL_FILE}")

    if verbose:
        print("\n--- current regime (most recent trading day) ---")
        last = regimes.iloc[-1]
        print(f"  {regimes.index[-1].date()}: {last['regime']} "
              f"(log_return={last['log_return']:+.5f}, volatility={last['volatility']:.5f})")

    return {
        "model": model,
        "label_map": label_map,
        "regimes": regimes,
        "summary": summary,
        "train_ll_avg": train_ll_avg,
        "test_ll_avg": test_ll_avg,
    }


if __name__ == "__main__":
    run_model_pipeline()
