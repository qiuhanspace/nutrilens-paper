"""The values behind Supplementary Figs 3, 4 and 5.

Supp Fig 3   is the model ranking an artefact of the pipeline's discretionary
             choices? Four axes: the day of prediction, the size of the
             calibration window, the calibration strategy, and the
             reconstruction scheme.
Supp Fig 4   per-participant measured against predicted trajectories.
Supp Fig 5   the two information ceilings. Where the prediction looks modest,
             the measurement is modest too.

The calibration-window sweep re-predicts with the saved per-fold models, so it
covers the NutriLens arm only; every other panel covers all four arms.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import (ARMS, CATEGORICAL_FEATURES, FEATURES, GROUPS,
                              GROUP_LABEL, GROUP_SHORT, OFFSET_CLIP_MGDL,
                              REFERENCE, SEED, TABLES, TARGETS, T_DENSE)
from nutrilens.curves import (arm_rmse, load_predictions, per_subject_rmse,
                              readouts, reconstruct)
from nutrilens.recourse import _dmatrix, iter_folds

RECONSTRUCTIONS = ['linear', 'pchip', 'cubic', 'akima']
CALIBRATION_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
STRATEGIES = {
    'none': 'y_pred_raw',
    'training-group median curve': 'y_pred_naive_group',
    'own calibration-window median curve': 'y_pred_naive_individual',
    'per-participant offset (used throughout)': 'y_pred_adjusted',
}


def load_table() -> pd.DataFrame:
    df = pd.read_csv(FEATURES)
    if 'uncertain' not in df.columns:
        df['uncertain'] = (df['meal_photo_estimation_difficulty'] > 1).astype(int)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def suppfig3a(df: pd.DataFrame) -> pd.DataFrame:
    """Per-meal error against the day since a participant's first test meal.

    Durability is measured rather than assumed. Days beyond the first week are
    too sparse to carry a claim and are reported with their counts so that is
    visible.
    """
    rows = []
    for group in GROUPS:
        for held_out, booster, block, offset, measured in iter_folds(df, group):
            if len(block) == 0:
                continue
            predicted = booster.predict(_dmatrix(block)) + offset
            error = np.sqrt(np.mean(
                (reconstruct(predicted) - reconstruct(measured)) ** 2, axis=1))
            dates = pd.to_datetime([re.search(r'\d{4}-\d{2}-\d{2}', e).group()
                                    for e in block['meal_epoch_id']])
            day = (dates - dates.min()).days
            rows += [dict(group=GROUP_SHORT[group], subject_id=held_out,
                          day=int(d), rmse=float(r))
                     for d, r in zip(day, error)]
    per_meal = pd.DataFrame(rows)
    summary = (per_meal.groupby(['group', 'day'])
               .agg(n_meals=('rmse', 'size'), median_rmse=('rmse', 'median'),
                    q25=('rmse', lambda v: v.quantile(0.25)),
                    q75=('rmse', lambda v: v.quantile(0.75)))
               .reset_index())
    print('Supp Fig. 3a — day of prediction (Spearman over days 0-6)')
    for group, g in per_meal.groupby('group'):
        early = g[g.day <= 6]
        rho, p = spearmanr(early.day, early.rmse)
        print(f'    {group:<6s} rho {rho:+.3f}  p {p:.2f}   '
              f'(days 7-8: {int((g.day >= 7).sum())} meals)')
    return summary


def suppfig3b(df: pd.DataFrame) -> pd.DataFrame:
    """Error against the size of the calibration window, NutriLens arm.

    The window is re-cut at each fraction and the offset refitted; the model
    itself is unchanged, because the calibration window is never used to fit it.
    """
    rows = []
    for group in GROUPS:
        status = df.groupby('subject_id')['t2d_status'].first()
        subjects = sorted(status[status == group].index.tolist())
        block = df[df.subject_id.isin(subjects)].reset_index(drop=True)
        for held_out, booster, _, _, _ in iter_folds(df, group):
            rows_s = block[block.subject_id == held_out].reset_index(drop=True)
            pred = booster.predict(_dmatrix(rows_s))
            y = rows_s[TARGETS].to_numpy(float)
            epochs = pd.unique(rows_s['meal_epoch_id'])
            for fraction in CALIBRATION_FRACTIONS:
                n_cal = int(fraction * len(epochs))
                if n_cal < 1 or n_cal >= len(epochs):
                    continue
                cal = rows_s['meal_epoch_id'].isin(set(epochs[:n_cal])).to_numpy()
                offset = np.clip(np.median(y[cal] - pred[cal], axis=0),
                                 -OFFSET_CLIP_MGDL, OFFSET_CLIP_MGDL)
                test = ~cal
                # Table 1's aggregation, so that the 30 % column of this sweep
                # equals the headline number rather than merely resembling it.
                rmse = per_subject_rmse(y[test], pred[test] + offset,
                                        np.full(test.sum(), held_out))
                rows.append(dict(group=GROUP_SHORT[group], subject_id=held_out,
                                 calibration_fraction=fraction,
                                 n_calibration_meals=n_cal,
                                 rmse=float(rmse.iloc[0])))
    sweep = pd.DataFrame(rows)
    summary = (sweep.groupby(['group', 'calibration_fraction'])
               .agg(n_subjects=('rmse', 'size'), mean_rmse=('rmse', 'mean'),
                    median_calibration_meals=('n_calibration_meals', 'median'))
               .reset_index())
    print('\nSupp Fig. 3b — calibration-window size, NutriLens arm')
    pivot = summary.pivot_table(index='calibration_fraction', columns='group',
                                values='mean_rmse')
    print(pivot.round(2).to_string())
    t2d = pivot['T2D']
    print(f'    T2D gains {t2d.loc[0.05] - t2d.loc[0.30]:.1f} mg/dL from 5 % to 30 %, '
          f'{t2d.loc[0.30] - t2d.loc[0.45]:.1f} more out to 45 %')
    return summary


def suppfig3cd() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration strategy, and reconstruction scheme, for all four arms."""
    rows = []
    for arm, label in ARMS.items():
        for group, glabel in GROUP_LABEL.items():
            for strategy, field in STRATEGIES.items():
                s = arm_rmse(arm, group, field=field)
                rows.append(dict(model=label, group=glabel, strategy=strategy,
                                 rmse_mean=float(s.mean()),
                                 rmse_sd=float(s.std(ddof=1))))
    strategies = pd.DataFrame(rows)

    rows = []
    for arm, label in ARMS.items():
        for group, glabel in GROUP_LABEL.items():
            for scheme in RECONSTRUCTIONS:
                s = arm_rmse(arm, group, method=scheme)
                rows.append(dict(model=label, group=glabel, reconstruction=scheme,
                                 rmse_mean=float(s.mean()),
                                 rmse_sd=float(s.std(ddof=1))))
    schemes = pd.DataFrame(rows)

    print('\nSupp Fig. 3c — calibration strategy (NutriLens, mean RMSE)')
    print(strategies[strategies.model == ARMS[REFERENCE]]
          .pivot_table(index='strategy', columns='group', values='rmse_mean')
          .round(2).to_string())
    print('\nSupp Fig. 3d — reconstruction scheme (NutriLens, mean RMSE)')
    wide = (schemes[schemes.model == ARMS[REFERENCE]]
            .pivot_table(index='reconstruction', columns='group', values='rmse_mean'))
    print(wide.round(3).to_string())
    print(f'    schemes agree to within '
          f'{(wide.max() - wide.min()).max():.3f} mg/dL')
    return strategies, schemes


def suppfig4() -> pd.DataFrame:
    """Per-participant mean measured and predicted trajectories."""
    rows = []
    for group, glabel in GROUP_LABEL.items():
        z = load_predictions(REFERENCE, group)
        measured, predicted = reconstruct(z['y_true']), reconstruct(z['y_pred_adjusted'])
        sids = np.asarray(z['subject_ids'])
        for subject in pd.unique(sids):
            m = sids == subject
            n = int(m.sum())
            for i, t in enumerate(T_DENSE):
                rows.append(dict(
                    group=glabel, subject_id=subject, n_meals=n, minute=int(t),
                    measured_mean=float(measured[m, i].mean()),
                    measured_ci=float(1.96 * measured[m, i].std(ddof=1) / np.sqrt(n)),
                    predicted_mean=float(predicted[m, i].mean()),
                    predicted_ci=float(1.96 * predicted[m, i].std(ddof=1) / np.sqrt(n))))
    return pd.DataFrame(rows)


def suppfig5a(df: pd.DataFrame) -> pd.DataFrame:
    """The in-cohort reproducibility ceiling for incremental area.

    Each meal is paired with the most nutritionally similar OTHER meal the same
    participant ate, and the two MEASURED values are correlated. The ceiling is
    a property of the measurement, so it is computed on every meal event from
    the dense measured trace and never involves the model.

    A nearest-neighbour meal is only approximately a repeat meal, so this is a
    conservative proxy and the model can exceed it. The correct statement is
    that the model is comparable to the reproducibility of the measurement,
    never that it is at the ceiling.
    """
    rng = np.random.default_rng(0)
    macros = ['carbs', 'fat', 'protein', 'fiber']
    dense = [f'postprandial_glucose_t{m:03d}' for m in range(121)]
    table = df.dropna(subset=macros + dense).reset_index(drop=True)
    curves = table[dense].to_numpy(float)
    iauc = np.trapezoid(curves - curves[:, [0]], dx=1.0, axis=1)
    status = table.groupby('subject_id')['t2d_status'].first()

    rows = []
    for group, glabel in GROUP_LABEL.items():
        block = table[table.subject_id.isin(status[status == group].index)]
        a, b = [], []
        for _, meals in block.groupby('subject_id'):
            if len(meals) < 4:
                continue
            x = meals[macros].to_numpy(float)
            x = (x - x.mean(0)) / (x.std(0) + 1e-9)
            values = iauc[meals.index.to_numpy()]
            distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            np.fill_diagonal(distance, np.inf)
            for i, j in enumerate(distance.argmin(1)):
                a.append(values[i])
                b.append(values[j])
        a, b = np.array(a), np.array(b)
        ceiling = pearsonr(a, b).statistic
        boot = [pearsonr(a[s], b[s]).statistic
                for s in rng.integers(0, len(a), size=(2000, len(a)))]

        z = load_predictions(REFERENCE, group)
        model_r = pearsonr(
            readouts(reconstruct(z['y_true']))['iAUC, net signed'],
            readouts(reconstruct(z['y_pred_adjusted']))['iAUC, net signed']).statistic
        rows.append(dict(group=glabel, n_pairs=len(a),
                         ceiling_r=float(ceiling),
                         ceiling_ci_lo=float(np.percentile(boot, 2.5)),
                         ceiling_ci_hi=float(np.percentile(boot, 97.5)),
                         model_r=float(model_r)))
    out = pd.DataFrame(rows)
    print('\nSupp Fig. 5a — iAUC against its in-cohort reproducibility ceiling')
    print(out.round(3).to_string(index=False))
    return out


def suppfig5b() -> pd.DataFrame:
    """How much of the within-participant trajectory shape is recoverable.

    Within-participant variation is decomposed into orthogonal modes, and each
    mode's recoverability is the squared correlation between its measured and
    predicted score. Only amplitude is partly recoverable, which is why a
    participant's predicted trajectories resemble one another.
    """
    rows = []
    for arm in (REFERENCE, 'image_logs-demo_nutrition_wearable_metabolic'):
        for group, glabel in GROUP_LABEL.items():
            z = load_predictions(arm, group)
            measured, predicted = reconstruct(z['y_true']), reconstruct(z['y_pred_adjusted'])
            sids = np.asarray(z['subject_ids'])
            # Centre within participant, so the modes describe how a person's
            # meals differ from each other rather than how people differ.
            centred_m, centred_p = measured.copy(), predicted.copy()
            for subject in pd.unique(sids):
                m = sids == subject
                centred_m[m] -= measured[m].mean(0)
                centred_p[m] -= predicted[m].mean(0)
            u, s, vt = np.linalg.svd(centred_m, full_matrices=False)
            variance = s ** 2 / (s ** 2).sum()
            for k, name in enumerate(['peak amplitude', 'peak timing', 'tail shape']):
                score_m = centred_m @ vt[k]
                score_p = centred_p @ vt[k]
                r2 = pearsonr(score_m, score_p).statistic ** 2
                rows.append(dict(model=ARMS[arm], group=glabel, mode=name,
                                 variance_share=float(variance[k]),
                                 recoverable_r2=float(r2)))
    out = pd.DataFrame(rows)
    print('\nSupp Fig. 5b — within-participant shape modes (NutriLens)')
    nl = out[out.model == ARMS[REFERENCE]]
    print(nl.pivot_table(index='mode', columns='group',
                         values=['variance_share', 'recoverable_r2'])
          .round(3).to_string())
    return out


def main() -> None:
    df = load_table()
    suppfig3a(df).to_csv(TABLES / 'suppfig3a_day_drift.csv', index=False)
    suppfig3b(df).to_csv(TABLES / 'suppfig3b_calibration_window.csv', index=False)
    strategies, schemes = suppfig3cd()
    strategies.to_csv(TABLES / 'suppfig3c_calibration_strategy.csv', index=False)
    schemes.to_csv(TABLES / 'suppfig3d_reconstruction.csv', index=False)
    suppfig4().to_csv(TABLES / 'suppfig4_per_participant_curves.csv', index=False)
    suppfig5a(df).to_csv(TABLES / 'suppfig5a_iauc_ceiling.csv', index=False)
    suppfig5b().to_csv(TABLES / 'suppfig5b_shape_modes.csv', index=False)
    print(f'\nwrote 7 tables to {TABLES}')


if __name__ == '__main__':
    main()
