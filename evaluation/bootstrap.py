"""Paired block bootstrap over test days (stage 4.6).

Test days are grouped into consecutive, non-overlapping blocks of `block_days`
calendar days (7 by default, 30 as a robustness check), counted from the first
test day; whole blocks are resampled with replacement, which keeps the
day-to-day autocorrelation of the ocean fields inside each block. Comparisons
are paired: both models are evaluated on the same resampled days.
"""

from datetime import date

import numpy as np

N_RESAMPLES = 2000
CI = 0.95


def block_ids(dates, block_days: int):
    """Block index of each test day: (day - first day) // block_days."""
    days = np.array([date.fromisoformat(d).toordinal() for d in dates])
    return (days - days.min()) // block_days


def resample_indices(blocks, n_resamples: int = N_RESAMPLES, seed: int = 0):
    """Yields, n_resamples times, the sample indices of a block bootstrap draw:
    as many blocks as there are, drawn with replacement, all their days kept."""
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in unique]
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        yield np.concatenate([members[c] for c in chosen])


def mean_ci(values, dates, block_days: int = 7, n_resamples: int = N_RESAMPLES, ci: float = CI, seed: int = 0):
    """Mean of a per-day series and its block-bootstrap percentile interval.
    For a paired comparison pass the per-day differences."""
    values = np.asarray(values, dtype=np.float64)
    blocks = block_ids(dates, block_days)
    draws = np.array([values[idx].mean() for idx in resample_indices(blocks, n_resamples, seed)])
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


def pooled_ci(statistic, arrays, dates, block_days: int = 7, n_resamples: int = N_RESAMPLES,
              ci: float = CI, seed: int = 0):
    """Interval for a statistic of per-day arrays pooled over the test set
    (e.g. k_c from the shell sums), recomputed on every block resample.
    `statistic(*[a[idx] for a in arrays])` must return a float."""
    blocks = block_ids(dates, block_days)
    full = statistic(*arrays)
    draws = np.array([statistic(*[a[idx] for a in arrays]) for idx in resample_indices(blocks, n_resamples, seed)])
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(full), float(lo), float(hi)
