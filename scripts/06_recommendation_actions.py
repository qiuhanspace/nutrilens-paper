"""Supplementary Table 17 and the ranked half of Fig. 4d.

For every meal whose MEASURED trajectory reached 180 mg/dL, seven
guideline-anchored modifications are applied and the meal is re-predicted. Each
action is scored as the percentage of the gap to the threshold that it closes,
for the peak and for the area above 180 mg/dL.

The ranking is the deliverable. The near-null actions are what make it
credible: a recommender that endorsed every action would be worthless.

This is the only analysis that needs the saved per-fold models, because a
counterfactual has to be re-predicted by the model that produced the original
prediction for that participant.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import (ACTION_THRESHOLD_MGDL, CATEGORICAL_FEATURES,
                              FEATURES, GROUP_SHORT, TABLES)
from nutrilens.curves import reconstruct
from nutrilens.recourse import (ACTIONS, COMPOSITION, SWAP_WINDOW_H,
                                apply_action,
                                counterfactual_on_measured, feasibility_caps,
                                iter_folds, _dmatrix)

CEILING = ACTION_THRESHOLD_MGDL


def excursion(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak, and area above the threshold, for each trajectory."""
    c = np.atleast_2d(np.asarray(curves, float))
    return c.max(axis=1), np.maximum(c - CEILING, 0.0).sum(axis=1)


def pct_gap_closed(before: np.ndarray, after: np.ndarray,
                   target: float) -> np.ndarray:
    """How much of the distance to safety the modification covers.

    Glucose quantities are rounded to 0.1 mg/dL before the ratio is taken. That
    is an order of magnitude below the resolution of the sensor these
    trajectories come from, so it cannot change a conclusion; it is done so that
    the reported percentage is a function of quantities at their reporting
    precision rather than of floating-point tails.

    Not truncated at zero: an action that makes a meal worse scores negative,
    and hiding that would flatter the ranking. The median across meals is the
    reported statistic, which is why a few extreme values do not distort it.
    """
    gap = np.round(before, 1) - target
    delta = np.round(before, 1) - np.round(after, 1)
    return np.where(gap > 0, 100.0 * delta / gap, np.nan)


def main() -> None:
    df = pd.read_csv(FEATURES)
    if 'uncertain' not in df.columns:
        df['uncertain'] = (df['meal_photo_estimation_difficulty'] > 1).astype(int)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category')

    caps = feasibility_caps(df)
    mass_cap = float(np.percentile(df['mass'].to_numpy(float), 99))
    print(f'feasibility caps (cohort P95): '
          + ', '.join(f'{k} {v:.1f} g' for k, v in caps.items())
          + f'; mass cap (P99) {mass_cap:.0f} g\n')

    rows = []
    for group, short in GROUP_SHORT.items():
        for held_out, booster, block, offset, measured9 in iter_folds(df, group):
            if len(block) == 0:
                continue
            base9 = booster.predict(_dmatrix(block)) + offset
            measured = reconstruct(measured9)
            base_peak, base_area = excursion(measured)
            unsafe = base_peak >= CEILING

            # Clock time of each meal, recovered from the cyclic encoding, so
            # that "a meal at a similar time of day" can be evaluated.
            hours = (np.arctan2(block['meal_onset_time_sin'].to_numpy(float),
                                block['meal_onset_time_cos'].to_numpy(float))
                     / (2 * np.pi) * 24.0) % 24.0

            for key, code, family, label, anchor in ACTIONS:
                if key == 'X1_meal_swap':
                    # Replacing the meal is not a composition change, so it is
                    # evaluated by swapping in another meal the participant
                    # actually ate near the same clock time, choosing the one
                    # the model predicts most favourably.
                    predicted_peak = reconstruct(base9).max(axis=1)
                    cf_block = block.copy().reset_index(drop=True)
                    available = np.zeros(len(block), bool)
                    displaced = np.zeros(len(block))
                    for i in range(len(block)):
                        dh = np.abs(hours - hours[i])
                        dh = np.minimum(dh, 24 - dh)
                        cand = np.flatnonzero((dh <= SWAP_WINDOW_H)
                                              & (np.arange(len(block)) != i))
                        if not len(cand):
                            continue
                        best = cand[np.argmin(predicted_peak[cand])]
                        cf_block.loc[i, COMPOSITION] = (
                            block.iloc[best][COMPOSITION].to_numpy())
                        available[i] = True
                        displaced[i] = abs(float(block['carbs'].iloc[i]
                                                 - block['carbs'].iloc[best]))
                else:
                    cf_block, available, displaced = apply_action(
                        block, key, caps, mass_cap)

                cf9 = booster.predict(_dmatrix(cf_block)) + offset
                cf = counterfactual_on_measured(measured, reconstruct(base9),
                                                reconstruct(cf9))
                cf_peak, cf_area = excursion(cf)

                for i in np.flatnonzero(unsafe):
                    rows.append(dict(
                        group=short, subject_id=held_out,
                        meal_epoch_id=block['meal_epoch_id'].iloc[i],
                        action=code, family=family, action_label=label,
                        available=bool(available[i]),
                        carb_displaced_g=float(displaced[i]),
                        measured_peak=float(base_peak[i]),
                        counterfactual_peak=float(cf_peak[i]),
                        pct_gap_closed_peak=float(
                            pct_gap_closed(base_peak[i:i + 1], cf_peak[i:i + 1],
                                           CEILING)[0]),
                        pct_gap_closed_area=float(
                            pct_gap_closed(base_area[i:i + 1], cf_area[i:i + 1],
                                           0.0)[0])))

    per_meal = pd.DataFrame(rows)
    per_meal.to_csv(TABLES / 'supptable17b_action_per_meal.csv', index=False)

    n_unsafe = per_meal.meal_epoch_id.nunique()
    summary = []
    anchors = {code: (family, label, anchor)
               for _, code, family, label, anchor in ACTIONS}
    for code, g in per_meal.groupby('action'):
        av = g[g.available]
        family, label, anchor = anchors[code]
        summary.append(dict(
            code=code, family=family, action=label, anchor=anchor,
            n_flagged_meals=g.meal_epoch_id.nunique(),
            available_pct=round(100 * g.available.mean(), 1),
            median_carb_displaced_g=round(av.carb_displaced_g.median(), 1),
            pct_gap_closed_peak=round(av.pct_gap_closed_peak.median(), 1),
            pct_gap_closed_area=round(av.pct_gap_closed_area.median(), 1)))
    st17 = pd.DataFrame(summary).sort_values('pct_gap_closed_peak',
                                             ascending=False)
    st17.to_csv(TABLES / 'supptable17a_action_effects.csv', index=False)

    print(f'Supplementary Table 17 — {n_unsafe} meals whose measured peak '
          f'reached {CEILING:g} mg/dL\n')
    print(st17[['code', 'family', 'action', 'available_pct',
                'median_carb_displaced_g', 'pct_gap_closed_peak',
                'pct_gap_closed_area']].to_string(index=False))
    print(f'\nwrote 2 tables to {TABLES}')


if __name__ == '__main__':
    main()
