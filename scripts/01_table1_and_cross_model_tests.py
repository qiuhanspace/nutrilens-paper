"""Table 1, Supplementary Table 6 and Supplementary Table 8.

Table 1                 per-participant RMSE of the predicted 2-hour trajectory
Supplementary Table 6   the three pre-specified contrasts, difference AND
                        equivalence, with effect size and multiplicity control
Supplementary Table 8   equivalence at four candidate margins, with the minimum
                        detectable effect beside every claim

The primary comparison is NutriLens against Baseline (Demographics). Both arms
carry the same per-participant calibration offset and the baseline receives no
meal information, so the contrast isolates the photograph.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import (ARMS, COMPARATORS, GROUP_LABEL, MARGIN_MGDL,
                              REFERENCE, TABLES)
from nutrilens.curves import arm_rmse, rmse_table
from nutrilens.stats import (bh_fdr, bonferroni, bootstrap_ci, paired_wilcoxon,
                             tost, verdict)


def paired_difference(comparator: str, group: str) -> tuple[np.ndarray, pd.Series]:
    """NutriLens minus comparator, aligned on participant.

    Negative favours NutriLens.
    """
    a = arm_rmse(REFERENCE, group)
    b = arm_rmse(comparator, group)
    idx = a.index.intersection(b.index).sort_values()
    return (a.loc[idx] - b.loc[idx]).to_numpy(float), a.loc[idx]


def main() -> None:
    # ── Table 1 ──────────────────────────────────────────────────────────
    table1 = rmse_table()
    table1.to_csv(TABLES / 'table1_trajectory_rmse.csv', index=False)
    print('Table 1 — per-participant RMSE (mg/dL), mean +/- SD\n')
    wide = table1.pivot_table(index='model', columns='group',
                              values=['rmse_mean', 'rmse_sd'])
    for model in ARMS.values():
        cells = ' | '.join(
            f'{wide.loc[model, ("rmse_mean", g)]:6.2f} +/- '
            f'{wide.loc[model, ("rmse_sd", g)]:5.2f}'
            for g in GROUP_LABEL.values())
        print(f'  {model:<42s} {cells}')

    # ── Supplementary Table 6 ────────────────────────────────────────────
    rows = []
    for group, glabel in GROUP_LABEL.items():
        block, praw = [], []
        for comparator in COMPARATORS:
            delta, _ = paired_difference(comparator, group)
            w = paired_wilcoxon(delta)
            eq = tost(delta, MARGIN_MGDL)
            lo, hi = bootstrap_ci(delta, np.median)
            block.append(dict(
                group=glabel, comparator=ARMS[comparator],
                n_subjects=eq['n'], median_diff_mgdl=float(np.median(delta)),
                median_ci95_lo=lo, median_ci95_hi=hi,
                mean_diff_mgdl=eq['mean_diff'],
                ci90_lo=eq['ci90_lo'], ci90_hi=eq['ci90_hi'],
                ci95_lo=eq['ci95_lo'], ci95_hi=eq['ci95_hi'],
                W=w['W'], effect_r=w['effect_r'], p_raw=w['p'],
                margin_mgdl=MARGIN_MGDL, p_tost=eq['p_tost'],
                equivalent_tost90=eq['equivalent_90'],
                equivalent_ci95=eq['equivalent_95'],
                mde_80pct_mgdl=eq['mde_80']))
            praw.append(w['p'])
        for row, pb, qb in zip(block, bonferroni(praw), bh_fdr(praw)):
            row['p_bonferroni3'] = float(pb)
            row['q_bh'] = float(qb)
            row['verdict'] = verdict(pb < 0.05, row['equivalent_tost90'])
        rows += block
    st6 = pd.DataFrame(rows)
    st6.to_csv(TABLES / 'supptable6_cross_model_tests.csv', index=False)

    print('\nSupplementary Table 6 — difference and equivalence, margin '
          f'+/- {MARGIN_MGDL:g} mg/dL\n')
    print(st6[['group', 'comparator', 'mean_diff_mgdl', 'ci90_lo', 'ci90_hi',
               'W', 'effect_r', 'p_bonferroni3', 'equivalent_tost90',
               'mde_80pct_mgdl', 'verdict']].round(3).to_string(index=False))

    # ── Supplementary Table 8 ────────────────────────────────────────────
    rows = []
    for group, glabel in GROUP_LABEL.items():
        for comparator in COMPARATORS:
            delta, reference = paired_difference(comparator, group)
            ref = float(reference.mean())
            margins = [('+/- 3 mg/dL (primary)', 3.0),
                       ('+/- 5 mg/dL', 5.0),
                       ('+/- 5 % of reference RMSE', 0.05 * ref),
                       ('+/- 10 % of reference RMSE', 0.10 * ref)]
            for label, m in margins:
                eq = tost(delta, m)
                rows.append(dict(
                    group=glabel, comparator=ARMS[comparator],
                    n_subjects=eq['n'], reference_rmse_mgdl=ref,
                    margin_label=label, margin_mgdl=m,
                    mean_diff_mgdl=eq['mean_diff'], sd_diff_mgdl=eq['sd_diff'],
                    ci90_lo=eq['ci90_lo'], ci90_hi=eq['ci90_hi'],
                    ci95_lo=eq['ci95_lo'], ci95_hi=eq['ci95_hi'],
                    p_tost=eq['p_tost'],
                    equivalent_tost90=eq['equivalent_90'],
                    equivalent_ci95=eq['equivalent_95'],
                    mde_80pct_mgdl=eq['mde_80']))
    pd.DataFrame(rows).to_csv(TABLES / 'supptable8_equivalence_mde.csv',
                              index=False)
    print(f'\nwrote 3 tables to {TABLES}')


if __name__ == '__main__':
    main()
