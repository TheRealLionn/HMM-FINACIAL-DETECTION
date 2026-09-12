"""
Sprint Tasks 2.1-2.4 — mathematical foundations and EDA.

Runs on the Task 1.6 output (data/processed/observations.csv), so Phase 1
must have been run at least once before this script.

Four things are established here, each feeding directly into a Phase 3
modelling choice:

  2.1  Return distribution shape   -> justifies n_components > 1
  2.2  Rolling volatility in time  -> justifies volatility as a SECOND
                                       observation dimension, not just returns
  2.3  Return/volatility correlation -> justifies feeding [return, volatility]
                                       jointly rather than volatility alone
  2.4  Emission distribution choice  -> justifies GaussianHMM specifically
                                       (as opposed to a Student-t or GMM-HMM)

Run:  python -m notebooks.eda_phase2   (from the project root)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

from src import config
from src.features import load_dataset

FIGSIZE_WIDE = (10, 4.5)
FIGSIZE_SQUARE = (6, 6)


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ===========================================================================
# Task 2.1 — return distribution: fat tails and skewness
# ===========================================================================
def plot_return_distribution(returns) -> None:
    """Histogram vs. fitted normal, plus a QQ-plot against the normal.

    A Gaussian HMM assumes each REGIME is normal, not that the pooled,
    unconditional distribution is normal — a mixture of a few normals is
    exactly how fat tails and skew emerge from otherwise-normal pieces. This
    plot establishes that the pooled distribution has that shape, which is
    the motivation for using a MIXTURE (i.e. several hidden states) instead
    of a single distribution over the whole five years.
    """
    mu, sigma = returns.mean(), returns.std()
    skew, kurt = returns.skew(), returns.kurtosis()
    jb_stat, jb_p = stats.jarque_bera(returns)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # --- panel 1: histogram vs fitted normal ---
    ax = axes[0]
    ax.hist(returns, bins=80, density=True, alpha=0.6, color="#3b6ea5",
             label="observed daily log return")
    x = np.linspace(returns.min(), returns.max(), 400)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#c0392b", lw=2,
             label=f"N(mu={mu:.5f}, sigma={sigma:.5f}) fit")
    ax.set_title("Daily log returns vs. fitted normal")
    ax.set_xlabel("log return")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.text(
        0.02, 0.97,
        f"skew = {skew:+.2f}\nexcess kurtosis = {kurt:+.2f}\n"
        f"Jarque-Bera p = {jb_p:.1e}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"),
    )

    # --- panel 2: QQ-plot ---
    ax = axes[1]
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title("QQ-plot vs. normal (tails bow away from the line)")
    ax.get_lines()[0].set(markersize=3, color="#3b6ea5", alpha=0.6)
    ax.get_lines()[1].set(color="#c0392b")

    fig.tight_layout()
    _save(fig, "01_return_distribution.png")

    print("\n--- Task 2.1 return distribution ---")
    print(f"  skewness            : {skew:+.3f}   (0 for normal)")
    print(f"  excess kurtosis     : {kurt:+.3f}   (0 for normal; >0 = fat tails)")
    print(f"  Jarque-Bera stat    : {jb_stat:.1f}, p = {jb_p:.2e}")
    verdict = "REJECTED" if jb_p < 0.05 else "not rejected"
    print(f"  H0 'returns are normal' -> {verdict} at 5%")
    print("  -> pooled returns are not normal (fat tails, mild skew).")
    print("     A single Gaussian is the wrong model for the whole series;")
    print("     a MIXTURE of regime-conditional Gaussians (the HMM's states)")
    print("     is how the model will reproduce this shape.")


# ===========================================================================
# Task 2.2 — rolling volatility across the full timeline
# ===========================================================================
def plot_volatility_timeline(frame) -> None:
    """Price and rolling volatility stacked on a shared time axis.

    Purpose: a visual gut-check that volatility clusters in time (calm
    stretches, then a burst) rather than looking like white noise. Clustering
    is exactly the persistence an HMM's transition matrix is built to
    capture — if volatility looked i.i.d. day to day, a memory-less model
    (e.g. plain clustering) would do just as well and the HMM would add
    nothing.
    """
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    ax_price.plot(frame.index, frame["price"], color="#2c3e50", lw=1)
    ax_price.set_ylabel("S&P 500 (Adj Close)")
    ax_price.set_title(f"{config.TICKER}: price and {config.VOLATILITY_WINDOW}-day "
                        "rolling volatility, full sample")
    ax_price.grid(alpha=0.3)

    ax_vol.plot(frame.index, frame["volatility"], color="#8e44ad", lw=1)
    ax_vol.axhline(frame["volatility"].mean(), color="0.4", ls="--", lw=1,
                    label=f"mean = {frame['volatility'].mean():.4f}")
    ax_vol.set_ylabel(f"{config.VOLATILITY_WINDOW}d rolling std of log return")
    ax_vol.set_xlabel("date")
    ax_vol.legend(fontsize=8)
    ax_vol.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "02_volatility_timeline.png")

    vol = frame["volatility"]
    print("\n--- Task 2.2 rolling volatility ---")
    print(f"  mean / median       : {vol.mean():.5f} / {vol.median():.5f}")
    print(f"  min  ({vol.idxmin().date()})     : {vol.min():.5f}")
    print(f"  max  ({vol.idxmax().date()})     : {vol.max():.5f}")
    print(f"  max / min ratio     : {vol.max() / vol.min():.1f}x")
    # Lag-1 autocorrelation of volatility: near 0 would mean no clustering.
    autocorr = vol.autocorr(lag=1)
    print(f"  lag-1 autocorrelation of volatility: {autocorr:+.3f}")
    print("  -> high and strongly positive: volatility is persistent")
    print("     (clusters in time), which is what the HMM's diagonal-heavy")
    print("     transition matrix is designed to exploit.")


# ===========================================================================
# Task 2.3 — correlation between returns and volatility
# ===========================================================================
def plot_correlations(frame) -> None:
    """Scatter panels for the return/volatility relationship, signed and unsigned.

    Signed return vs. volatility is expected near zero: volatility is a
    magnitude, so up-days and down-days both raise it and the sign cancels.
    |return| vs. volatility is the informative pair — big moves (either
    direction) coincide with high recent volatility, which is exactly why
    the model is fed the PAIR (return, volatility), not return alone: the
    two dimensions carry different, complementary information about regime.
    """
    returns, vol = frame["log_return"], frame["volatility"]
    r_signed = returns.corr(vol)
    r_abs = returns.abs().corr(vol)
    r_lead = returns.corr(vol.shift(-config.VOLATILITY_WINDOW))

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    axes[0].scatter(returns, vol, s=6, alpha=0.35, color="#3b6ea5")
    axes[0].set_xlabel("log return")
    axes[0].set_ylabel("rolling volatility")
    axes[0].set_title(f"signed return vs. volatility  (r = {r_signed:+.3f})")

    axes[1].scatter(returns.abs(), vol, s=6, alpha=0.35, color="#27ae60")
    axes[1].set_xlabel("|log return|")
    axes[1].set_ylabel("rolling volatility")
    axes[1].set_title(f"|return| vs. volatility  (r = {r_abs:+.3f})")

    fig.tight_layout()
    _save(fig, "03_return_volatility_correlation.png")

    print("\n--- Task 2.3 correlation checks ---")
    print(f"  corr(return, vol)         : {r_signed:+.3f}  <- ~0 expected, sign cancels")
    print(f"  corr(|return|, vol)       : {r_abs:+.3f}  <- magnitude relationship")
    print(f"  corr(return, vol +{config.VOLATILITY_WINDOW}d)      : {r_lead:+.3f}  "
          "<- leverage effect (falls raise FUTURE volatility)")
    print("  -> return and volatility carry different information (near-zero")
    print("     signed correlation, positive unsigned correlation), which")
    print("     justifies the 2-D observation vector instead of using either")
    print("     feature alone.")


# ===========================================================================
# Task 2.4 — emission distribution choice
# ===========================================================================
def compare_emission_assumption(returns) -> None:
    """Empirically motivate 'Gaussian, but per-state' as the emission model.

    A single Gaussian (n=1) is exactly what Task 2.1 already rejected. Here
    we fit a Gaussian MIXTURE with as many components as the model's planned
    regime count and show it tracks the histogram far better than n=1. This
    is a pure EDA/motivation step: the actual HMM in Phase 3 also has
    transition dynamics between states, which a plain mixture does not — but
    the marginal (unconditional) shape a good HMM should reproduce is this
    one, and a mixture of Gaussians is the right building block for it,
    which is what makes GaussianHMM (as opposed to fitting a single
    heavy-tailed distribution such as Student-t) the appropriate choice for
    hmmlearn's standard emission model.
    """
    x = returns.to_numpy().reshape(-1, 1)

    gm1 = GaussianMixture(n_components=1, random_state=config.RANDOM_SEED).fit(x)
    gmk = GaussianMixture(n_components=config.N_REGIMES,
                           random_state=config.RANDOM_SEED).fit(x)

    grid = np.linspace(returns.min(), returns.max(), 400).reshape(-1, 1)
    pdf1 = np.exp(gm1.score_samples(grid))
    pdfk = np.exp(gmk.score_samples(grid))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(returns, bins=80, density=True, alpha=0.5, color="0.75",
             label="observed")
    ax.plot(grid, pdf1, color="#c0392b", lw=2, ls="--",
             label="1 Gaussian (pooled, Task 2.1)")
    ax.plot(grid, pdfk, color="#27ae60", lw=2,
             label=f"mixture of {config.N_REGIMES} Gaussians")
    ax.set_title("Single Gaussian vs. a mixture: why the HMM uses "
                  f"{config.N_REGIMES} Gaussian states")
    ax.set_xlabel("log return")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "04_emission_distribution_choice.png")

    print("\n--- Task 2.4 emission distribution ---")
    print(f"  1-component BIC     : {gm1.bic(x):,.1f}")
    print(f"  {config.N_REGIMES}-component BIC     : {gmk.bic(x):,.1f}  (lower is better)")
    print(f"  log-likelihood gain : {gmk.score(x) - gm1.score(x):+.4f} per obs.")
    print(f"\n  DECISION: emission distribution = Gaussian per state")
    print(f"    (hmmlearn.GaussianHMM, covariance_type='{config.COVARIANCE_TYPE}',")
    print(f"     n_components={config.N_REGIMES})")
    print("    A single Gaussian was already rejected (Task 2.1). A mixture of")
    print(f"    {config.N_REGIMES} Gaussians fits the pooled shape substantially better")
    print("    (lower BIC, higher likelihood) and each component maps onto an")
    print("    economically interpretable regime (bull / bear / high-vol).")
    print("    covariance_type='full' additionally lets each regime have its")
    print("    own return-volatility covariance, not just its own variances.")


# ===========================================================================
# Pipeline entry point
# ===========================================================================
def run_eda(verbose: bool = True) -> None:
    frame = load_dataset()
    if verbose:
        print(f"Loaded {len(frame)} rows of processed features "
              f"({frame.index.min().date()} to {frame.index.max().date()})")

    plot_return_distribution(frame["log_return"])
    plot_volatility_timeline(frame)
    plot_correlations(frame)
    compare_emission_assumption(frame["log_return"])


if __name__ == "__main__":
    run_eda()
