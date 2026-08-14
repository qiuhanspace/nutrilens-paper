"""Supplementary Table 13: what each ingredient of the prediction contributes.

The obvious objection to the headline is that NutriLens beats the baselines only
because of the per-participant calibration offset. This answers it directly, and
without retraining anything: crossing the presence of image-derived nutrition,
demographics and the calibration offset gives six rungs, all of which can be
read off prediction fields already stored in every cross-validation fold.

The uncalibrated rungs are DIAGNOSTIC, not deployable. Nobody would ship an
uncalibrated model, and error without calibration is dominated by
per-participant level rather than by meal information.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import GROUP_SHORT, TABLES
from nutrilens.curves import arm_rmse
from nutrilens.stats import paired_wilcoxon

#         rung label                             arm            prediction field           image  demo  calib
LADDER = [
    ('group median curve', 'baseline-demo', 'y_pred_naive_group', False, False, False),
    ('calibration only', 'baseline-demo', 'y_pred_naive_individual', False, False, True),
    ('demographics only, no calibration', 'baseline-demo', 'y_pred_raw', False, True, False),
    ('demographics + calibration', 'baseline-demo', 'y_pred_adjusted', False, True, True),
    ('image + demographics, no calibration', 'image_logs-demo_nutrition', 'y_pred_raw', True, True, False),
    ('image + demographics + calibration', 'image_logs-demo_nutrition', 'y_pred_adjusted', True, True, True),
]
ALSO_KNOWN = {'calibration only': 'Baseline (Personal)',
              'demographics + calibration': 'Baseline (Demographics)',
              'image + demographics + calibration': 'NutriLens'}

CONTRASTS = [
    ('calibration, given demographics',
     'demographics only, no calibration', 'demographics + calibration'),
    ('calibration, given image + demographics',
     'image + demographics, no calibration', 'image + demographics + calibration'),
    ('demographics, given calibration',
     'calibration only', 'demographics + calibration'),
    ('demographics, given nothing else',
     'group median curve', 'demographics only, no calibration'),
    ('image, given demographics + calibration',
     'demographics + calibration', 'image + demographics + calibration'),
    ('image, with calibration removed',
     'demographics only, no calibration', 'image + demographics, no calibration'),
]


def main() -> None:
    rmse, rows = {}, []
    for label, arm, field, image, demo, calib in LADDER:
        row = dict(rung=label, also_known_as=ALSO_KNOWN.get(label, ''),
                   image=image, demographics=demo, calibration=calib,
                   arm=arm, prediction_field=field)
        for group, short in GROUP_SHORT.items():
            s = arm_rmse(arm, group, field=field)
            rmse[(label, short)] = s
            row[f'rmse_{short}'] = float(s.mean())
            row[f'sd_{short}'] = float(s.std(ddof=1))
        rows.append(row)
    ladder = pd.DataFrame(rows)
    ladder.to_csv(TABLES / 'supptable13a_ingredient_ladder.csv', index=False)

    rows = []
    for name, frm, to in CONTRASTS:
        row = dict(ingredient=name, from_rung=frm, to_rung=to)
        for short in GROUP_SHORT.values():
            a, b = rmse[(frm, short)], rmse[(to, short)]
            idx = a.index.intersection(b.index)
            gain = (a.loc[idx] - b.loc[idx]).to_numpy(float)   # + means it helps
            w = paired_wilcoxon(gain)
            row |= {f'gain_{short}': float(gain.mean()),
                    f'median_{short}': float(np.median(gain)),
                    f'p_{short}': w['p'], f'effect_r_{short}': w['effect_r']}
        rows.append(row)
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(TABLES / 'supptable13b_ingredient_contrasts.csv', index=False)

    print('Supplementary Table 13 — the ladder (per-participant RMSE, mg/dL)\n')
    print(ladder[['rung', 'rmse_Normo', 'rmse_Pre', 'rmse_T2D']]
          .round(2).to_string(index=False))
    print('\nMarginal gain of each ingredient (mg/dL, positive = it helps)\n')
    print(contrasts[['ingredient', 'gain_Normo', 'p_Normo', 'gain_Pre', 'p_Pre',
                     'gain_T2D', 'p_T2D']].round(3).to_string(index=False))
    print(f'\nwrote 2 tables to {TABLES}')


if __name__ == '__main__':
    main()
