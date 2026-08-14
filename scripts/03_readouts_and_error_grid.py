"""Supplementary Table 9 and the Clarke error grid (Fig. 3c).

Every clinical readout is computed from the SAME reconstructed trajectory that
Table 1 scores, rather than from a separately trained head. That is the point of
predicting a trajectory: one prediction serves every decision.

Both incremental-area definitions are reported. They differ materially in type 2
diabetes, where the trajectory spends time below its own starting value.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import ARMS, GROUP_LABEL, REFERENCE, TABLES
from nutrilens.curves import (READOUT_UNITS, clarke_summary, load_predictions,
                              readouts, reconstruct)


def main() -> None:
    # ── Supplementary Table 9 ────────────────────────────────────────────
    rows = []
    for group, glabel in GROUP_LABEL.items():
        measured = None
        for arm, label in ARMS.items():
            z = load_predictions(arm, group)
            sids = np.asarray(z['subject_ids'])
            if measured is None:
                measured = readouts(reconstruct(z['y_true']))
            predicted = readouts(reconstruct(z['y_pred_adjusted']))
            for readout in measured.columns:
                t = measured[readout].to_numpy()
                p = predicted[readout].to_numpy()
                per_subject = [np.sqrt(np.mean((p[sids == s] - t[sids == s]) ** 2))
                               for s in pd.unique(sids)]
                r = pearsonr(t, p).statistic if np.std(p) > 0 else np.nan
                rows.append(dict(
                    readout=readout, unit=READOUT_UNITS[readout],
                    group=glabel, model=label,
                    n_subjects=len(per_subject), n_meals=len(t),
                    rmse_mean=float(np.mean(per_subject)),
                    rmse_sd=float(np.std(per_subject, ddof=1)),
                    pearson_r=float(r)))
    st9 = pd.DataFrame(rows)
    st9.to_csv(TABLES / 'supptable9_curve_readouts.csv', index=False)

    print('Supplementary Table 9 — readouts derived from the predicted '
          'trajectory (NutriLens, Pearson r)\n')
    nl = st9[st9.model == ARMS[REFERENCE]]
    print(nl.pivot_table(index='readout', columns='group', values='pearson_r')
          .round(3).to_string())

    # ── Clarke error grid, Fig. 3c ───────────────────────────────────────
    rows = []
    for group, glabel in GROUP_LABEL.items():
        z = load_predictions(REFERENCE, group)
        pct = clarke_summary(z['y_true'], z['y_pred_adjusted'])
        rows.append(dict(group=glabel, n_pairs=len(z['y_true']) * 121,
                         **{f'zone_{k}_pct': round(v, 2) for k, v in pct.items()},
                         zone_AB_pct=round(pct['A'] + pct['B'], 2)))
    grid = pd.DataFrame(rows)
    grid.to_csv(TABLES / 'fig3c_clarke_error_grid.csv', index=False)
    print('\nClarke error grid, NutriLens\n')
    print(grid.to_string(index=False))
    print(f'\nwrote 2 tables to {TABLES}')


if __name__ == '__main__':
    main()
