"""The statistics the manuscript reports, defined once.

Two rules govern everything here. A difference test and an equivalence test are
always reported together, because a non-significant difference is not evidence
of sameness and a significant difference smaller than the clinical margin is
not a meaningful one. And every equivalence claim carries the minimum
detectable effect beside it, because an equivalence test on ten participants
can fail for lack of power rather than for lack of equivalence.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata, wilcoxon
from scipy.stats import t as student_t

from .config import EQUIV_ALPHA, MARGIN_MGDL, N_CONTRASTS, N_PERMUTATIONS, SEED


# ── Effect size ──────────────────────────────────────────────────────────
def rank_biserial(delta: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-).

    This is not |z|/sqrt(n); the two differ by up to 0.1 on these data. It is
    signed by the rank sums, so it can disagree with the sign of the median
    difference when a few large differences outweigh many small ones in the
    other direction.
    """
    d = np.asarray(delta, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_pos, w_neg = float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    total = w_pos + w_neg
    return 0.0 if total == 0 else (w_pos - w_neg) / total


# ── Difference ───────────────────────────────────────────────────────────
def paired_wilcoxon(delta: np.ndarray) -> dict:
    """Exact two-sided paired Wilcoxon signed-rank test.

    Exact when there are at most 25 non-zero differences, which covers every
    group here; the normal approximation would be inaccurate at n = 10.
    """
    nz = int(np.count_nonzero(delta))
    if nz == 0:
        return dict(W=np.nan, p=1.0, effect_r=0.0)
    res = wilcoxon(delta, alternative='two-sided', zero_method='wilcox',
                   method='exact' if nz <= 25 else 'approx')
    return dict(W=float(res.statistic), p=float(res.pvalue),
                effect_r=rank_biserial(delta))


def bonferroni(pvals, k: int = N_CONTRASTS) -> np.ndarray:
    """Bonferroni across the k pre-specified within-group contrasts."""
    return np.minimum(np.asarray(pvals, dtype=float) * k, 1.0)


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg false-discovery rate across the same contrasts."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


# ── Equivalence ──────────────────────────────────────────────────────────
def tost(delta: np.ndarray, margin: float = MARGIN_MGDL,
         alpha: float = EQUIV_ALPHA) -> dict:
    """Two one-sided t-tests for equivalence of the paired difference.

    The verdict is read off the (1 - 2*alpha) = 90 % interval, because that is
    the interval the two one-sided tests actually read: each spends alpha at
    one end, so "the 90 % interval lies inside the margin" and "p_TOST < 0.05"
    are the same statement. The 95 % interval is returned as a conservative
    cross-check.
    """
    d = np.asarray(delta, dtype=float)
    n, mean = d.size, float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    t90, t95 = student_t.ppf(1 - alpha, df), student_t.ppf(1 - alpha / 2, df)
    p_lo = student_t.sf((mean + margin) / se, df)     # H0: difference <= -margin
    p_hi = student_t.cdf((mean - margin) / se, df)    # H0: difference >= +margin
    ci90 = (mean - t90 * se, mean + t90 * se)
    ci95 = (mean - t95 * se, mean + t95 * se)
    # Minimum detectable effect at 80 % power, two-sided alpha = 0.05.
    mde = float((student_t.ppf(0.975, df) + student_t.ppf(0.80, df)) * se)
    return dict(n=n, mean_diff=mean, sd_diff=sd,
                ci90_lo=ci90[0], ci90_hi=ci90[1],
                ci95_lo=ci95[0], ci95_hi=ci95[1],
                p_tost=float(max(p_lo, p_hi)),
                equivalent_90=bool(ci90[0] > -margin and ci90[1] < margin),
                equivalent_95=bool(ci95[0] > -margin and ci95[1] < margin),
                mde_80=mde)


# ── Interval estimation ──────────────────────────────────────────────────
def bootstrap_ci(values, statistic=np.median, n_boot: int = 10_000,
                 seed: int = SEED, lo: float = 2.5, hi: float = 97.5):
    """Percentile bootstrap interval, resampling participants."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    draws = np.array([statistic(v[i]) for i in idx])
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))


# ── Trend across groups ──────────────────────────────────────────────────
def _concordance(v: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Sum over ordered group pairs of sign(v_higher - v_lower).

    This is the numerator of Kendall's tau-b against the severity rank. Its
    denominator depends only on the tie structure, which permutation leaves
    unchanged, so permuting this numerator gives exactly the tau-b null.
    Accepting a 2-D array lets the whole permutation null be built at once.
    """
    v = np.atleast_2d(v)
    total = np.zeros(v.shape[0])
    for a in range(len(groups) - 1):
        for b in range(a + 1, len(groups)):
            lower = v[:, groups[a]][:, :, None]
            higher = v[:, groups[b]][:, None, :]
            total += np.sign(higher - lower).sum(axis=(1, 2))
    return total


def jonckheere_terpstra(values: np.ndarray, severity: np.ndarray, rng,
                        n_perm: int = N_PERMUTATIONS,
                        chunk: int = 5_000) -> tuple[float, float]:
    """Trend test against the ordered alternative, one-sided.

    Implemented as Kendall's tau-b against the severity rank with an exact
    permutation null, which is the null the Jonckheere-Terpstra test assumes.
    A permutation null is used rather than the normal approximation because the
    groups are small and the advantage variance is strongly heteroscedastic.
    """
    tau = float(kendalltau(severity, values).statistic)
    groups = [np.flatnonzero(severity == r) for r in sorted(np.unique(severity))]
    observed = float(_concordance(values, groups)[0])
    n_ge, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        block = rng.permuted(np.tile(values, (m, 1)), axis=1)
        n_ge += int(np.count_nonzero(_concordance(block, groups) >= observed))
        done += m
    return tau, float((n_ge + 1) / (n_perm + 1))


# ── Reading a pair of tests together ─────────────────────────────────────
def verdict(significant: bool, equivalent: bool) -> str:
    """The conclusion the difference and equivalence tests support jointly.

    All four combinations occur in these data and each means something
    different, including a difference that is statistically detectable but
    smaller than the clinical margin.
    """
    if significant and not equivalent:
        return 'different beyond the margin'
    if significant and equivalent:
        return 'different but within the margin'
    if equivalent:
        return 'equivalent'
    return 'neither (inconclusive)'
