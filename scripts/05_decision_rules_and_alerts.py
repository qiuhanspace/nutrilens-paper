"""Supplementary Tables 15, 16 and 18, and the ROC half of Fig. 4d.

Three questions, in the order the argument needs them.

Supplementary Table 16   Do the clinical decision rules agree with one another?
                         If they did, the choice of readout would not matter and
                         a scalar endpoint would be sufficient. They do not.
Supplementary Table 15   Can a photograph taken BEFORE the meal flag a breach?
                         Evaluated by discrimination, because an alert system
                         chooses its own operating point; accuracy at one
                         arbitrary cutoff would test calibration, not signal.
Supplementary Table 18   Why is the model not trained to regress incremental
                         area under the curve directly? Because iAUC is blind to
                         the excursion that the decision rules are written on.

Every rule is evaluated at the measurement window its own source specifies.
Scoring a rule at a window its source did not specify changes which meals it
selects: using the maximum over the whole 0-120 minute window for the ADA rule
over-counts breaches by about 14 %, because in this cohort the peak often
arrives before 60 minutes.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import (DECISION_RULES, GROUP_SHORT, REFERENCE, TABLES)
from nutrilens.curves import (load_predictions, readouts, reconstruct,
                              rule_breached, rule_value)


def gather() -> pd.DataFrame:
    """Measured and predicted rule values for every held-out meal."""
    rows = []
    for group, short in GROUP_SHORT.items():
        z = load_predictions(REFERENCE, group)
        measured = reconstruct(z['y_true'])
        predicted = reconstruct(z['y_pred_adjusted'])
        frame = pd.DataFrame(dict(group=short,
                                  subject=np.asarray(z['subject_ids'])))
        for name, spec in DECISION_RULES.items():
            frame[f'measured::{name}'] = rule_value(measured, spec['window'],
                                                    spec['stat'])
            frame[f'predicted::{name}'] = rule_value(predicted, spec['window'],
                                                     spec['stat'])
            frame[f'breached::{name}'] = rule_breached(measured, spec)
        for col, values in readouts(measured).items():
            frame[f'readout::{col}'] = values.to_numpy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = gather()
    n = len(df)
    flags = {name: df[f'breached::{name}'].to_numpy()
             for name in DECISION_RULES}

    # ── Supplementary Table 16: do the rules agree? ──────────────────────
    rows = []
    for name, flag in flags.items():
        row = dict(rule=name, meals_flagged=int(flag.sum()),
                   pct_of_meals=round(100 * flag.mean(), 1))
        for short in GROUP_SHORT.values():
            m = (df.group == short).to_numpy()
            row[f'flagged_{short}'] = int(flag[m].sum())
            row[f'n_{short}'] = int(m.sum())
        rows.append(row)
    counts = pd.DataFrame(rows)

    matrix = np.column_stack(list(flags.values()))
    counts.attrs['flagged_by_any'] = int((matrix.sum(1) > 0).sum())
    jac = pd.DataFrame(index=list(flags), columns=list(flags), dtype=float)
    for a, b in itertools.product(flags, repeat=2):
        union = (flags[a] | flags[b]).sum()
        jac.loc[a, b] = round((flags[a] & flags[b]).sum() / max(union, 1), 3)

    counts.to_csv(TABLES / 'supptable16a_rule_counts.csv', index=False)
    jac.to_csv(TABLES / 'supptable16b_rule_jaccard.csv')
    any_rule = int((matrix.sum(1) > 0).sum())
    all_rules = int((matrix.sum(1) == matrix.shape[1]).sum())
    print(f'Supplementary Table 16 — {n} held-out meals\n')
    print(counts.to_string(index=False))
    print(f'\n  flagged by at least one rule: {any_rule}'
          f'   by all four: {all_rules} ({100 * all_rules / any_rule:.0f} % of those flagged)')

    # ── Supplementary Table 15: the alert ────────────────────────────────
    rows = []
    for name, spec in DECISION_RULES.items():
        for short in list(GROUP_SHORT.values()) + ['All']:
            m = np.ones(n, bool) if short == 'All' else (df.group == short).to_numpy()
            y = flags[name][m].astype(int)
            score = df.loc[m, f'predicted::{name}'].to_numpy()
            prevalence = 100 * y.mean()
            if y.sum() == 0 or y.sum() == len(y):
                auroc = ap = lift = np.nan
            else:
                auroc = roc_auc_score(y, score)
                ap = average_precision_score(y, score)
                lift = ap / y.mean()
            rows.append(dict(rule=name, group=short, n_meals=int(m.sum()),
                             n_events=int(y.sum()),
                             prevalence_pct=round(prevalence, 1),
                             AUROC=auroc, average_precision=ap,
                             precision_lift=lift))
    st15 = pd.DataFrame(rows)
    st15.to_csv(TABLES / 'supptable15_alert_discrimination.csv', index=False)
    print('\nSupplementary Table 15 — flagging a breach before the meal\n')
    print(st15.pivot_table(index='rule', columns='group', values='AUROC')
          .round(3).to_string())

    # ── Supplementary Table 18: iAUC is blind to the excursion ───────────
    keys = ['Peak glucose', 'Glucose at 2 h', 'Time above 180',
            'iAUC, positive only']
    r2 = pd.DataFrame(index=keys, columns=keys, dtype=float)
    jaccard = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a, b in itertools.product(keys, repeat=2):
        x = df[f'readout::{a}'].to_numpy()
        y = df[f'readout::{b}'].to_numpy()
        r2.loc[a, b] = round(pearsonr(x, y).statistic ** 2, 3)
        # Do the two readouts select the same meals as the worst third?
        ta, tb = np.percentile(x, 200 / 3), np.percentile(y, 200 / 3)
        top_a, top_b = x >= ta, y >= tb
        jaccard.loc[a, b] = round((top_a & top_b).sum() / max((top_a | top_b).sum(), 1), 2)
    r2.to_csv(TABLES / 'supptable18a_readout_r2.csv')
    jaccard.to_csv(TABLES / 'supptable18b_readout_jaccard.csv')

    iauc = df['readout::iAUC, positive only'].to_numpy()
    decile = pd.qcut(iauc, 10, labels=False)
    middle = decile == 4
    peak = df['readout::Peak glucose'].to_numpy()[middle]
    above = df['readout::Time above 180'].to_numpy()[middle]
    print('\nSupplementary Table 18 — iAUC against the other readouts (R^2)\n')
    print(r2.to_string())
    print(f'\n  within the middle iAUC decile, peak spans '
          f'{peak.min():.1f}-{peak.max():.1f} mg/dL and time above 180 spans '
          f'{above.min():.0f}-{above.max():.0f} min')
    print(f'\nwrote 5 tables to {TABLES}')


if __name__ == '__main__':
    main()
