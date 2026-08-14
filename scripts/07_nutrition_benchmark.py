"""Fig. 2a, Supplementary Table 1 and Supplementary Table 11.

Fig. 2a / Supp Table 1   NutriLens against four published image-to-nutrition
                         methods on the identical 3,897-dish pool
Supplementary Table 11   the same metrics with the 95 prompt-optimisation dishes
                         that sit inside the evaluation pool removed

Each comparator is scored on the vision backbone its own authors published.
Those papers select their backbone deliberately, usually after comparing
several, so the published configuration is the method; re-running one on a
backbone its authors did not choose would reproduce something nobody published.

Supplementary Table 11 exists because the prompt-optimisation set overlaps the
evaluation pool while the comparators had no equivalent tuning set. That is a
leakage asymmetry favouring NutriLens, so it is quantified rather than argued
away.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import STAGE1, TABLES

METHODS = {
    'nutrilens_e2e': 'NutriLens',
    'fridolfsson2025': 'Fridolfsson 2025',
    'ohara2025': "O'Hara 2025",
    'dietai24': 'Yan 2025',
    'khlaisamniang2025': 'Khlaisamniang 2025',
}
SPLITS = ['val_n5k', 'test_n5k', 'val_january', 'test_january']
DATASET = {'n5k': 'Nutrition5k', 'january': 'January Food Benchmark'}
QUANTITIES = {
    'total_mass_g': 'mass (g)',
    'total_calories_kcal': 'calories (kcal)',
    'total_carbs_g': 'total carbs (g)',
    'total_fat_g': 'fat (g)',
    'total_protein_g': 'protein (g)',
}


def ground_truth() -> pd.DataFrame:
    """The cleaned evaluation cohort, pooled across splits."""
    frames = []
    for split in SPLITS:
        f = pd.read_csv(STAGE1 / 'cleaned_v4' / split / 'metadata.csv')
        f['split'] = split
        frames.append(f)
    gt = pd.concat(frames, ignore_index=True)
    return gt[['dish_id', 'source', 'split'] + list(QUANTITIES)]


def estimates(method: str) -> pd.DataFrame:
    """Per-dish estimates for one method, pooled across splits."""
    frames = []
    for suffix in ('_val', '_test'):
        p = STAGE1 / 'parsed' / f'{method}{suffix}_per_dish.csv'
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    est = pd.concat(frames, ignore_index=True)
    # Some parsed files contain a dish in both the validation and the test
    # export. Keeping both would weight those dishes twice.
    est = est.drop_duplicates('dish_id')
    return est[['dish_id'] + [c for c in QUANTITIES if c in est.columns]]


def score(gt: pd.DataFrame, est: pd.DataFrame, column: str) -> dict:
    """The metric set the comparator studies themselves report."""
    m = gt[['dish_id', column]].merge(est[['dish_id', column]], on='dish_id',
                                      suffixes=('_true', '_pred')).dropna()
    truth = m[f'{column}_true'].to_numpy(float)
    pred = m[f'{column}_pred'].to_numpy(float)
    error = pred - truth
    return dict(
        n_dishes=len(m),
        MAE=float(np.mean(np.abs(error))),
        # Weighted MAPE: total absolute error over total reference. A per-dish
        # percentage error is unstable when the reference value is small.
        wMAPE_pct=float(100 * np.sum(np.abs(error)) / np.sum(np.abs(truth))),
        bias=float(np.mean(error)),
        pearson_r=float(np.corrcoef(truth, pred)[0, 1]),
    )


def evaluate(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, label in METHODS.items():
        est = estimates(method)
        if est.empty:
            print(f'  no estimates found for {method}')
            continue
        for subset, name in [(gt, 'Combined')] + [
                (gt[gt.source == s], DATASET[s]) for s in DATASET]:
            for column, quantity in QUANTITIES.items():
                rows.append(dict(method=label, dataset=name, nutrient=quantity,
                                 **score(subset, est, column)))
    return pd.DataFrame(rows)


def main() -> None:
    gt = ground_truth()
    print(f'evaluation pool: {len(gt):,} dishes '
          + ', '.join(f'{DATASET[s]} {int((gt.source == s).sum()):,}'
                      for s in DATASET) + '\n')

    st1 = evaluate(gt)
    st1.to_csv(TABLES / 'supptable1_nutrition_by_dataset.csv', index=False)
    print('Supplementary Table 1 / Fig. 2a — mean absolute error, pooled\n')
    combined = st1[st1.dataset == 'Combined']
    print(combined.pivot_table(index='nutrient', columns='method', values='MAE')
          [list(METHODS.values())].round(1).to_string())

    energy = combined[combined.nutrient == 'calories (kcal)'].set_index('method')
    print('\n  NutriLens energy: MAE {MAE:.2f} kcal, weighted MAPE {wMAPE_pct:.2f} %, '
          'bias {bias:.2f} kcal, r {pearson_r:.2f}'.format(**energy.loc['NutriLens']))

    # ── Supplementary Table 11: prompt-tuning leakage ────────────────────
    tuning = pd.read_csv(STAGE1 / 'train_metadata.csv')
    tuned_ids = set(tuning['dish_id'])
    overlap = gt[gt.dish_id.isin(tuned_ids)]
    print(f'\n  {len(overlap)} of the {len(tuned_ids)} prompt-optimisation dishes '
          'fall inside the evaluation pool')

    reduced = evaluate(gt[~gt.dish_id.isin(tuned_ids)])
    st11 = pd.concat([st1.assign(pool='all val + test (as published)'),
                      reduced.assign(pool='minus the prompt-tuning dishes')],
                     ignore_index=True)
    st11.to_csv(TABLES / 'supptable11_prompt_tuning_leakage.csv', index=False)

    a = st1[(st1.dataset == 'Combined') & (st1.nutrient == 'calories (kcal)')
            & (st1.method == 'NutriLens')].iloc[0]
    b = reduced[(reduced.dataset == 'Combined')
                & (reduced.nutrient == 'calories (kcal)')
                & (reduced.method == 'NutriLens')].iloc[0]
    print(f'  NutriLens energy MAE moves {a.MAE:.2f} -> {b.MAE:.2f} kcal '
          'when they are excluded')
    print(f'\nwrote 2 tables to {TABLES}')


if __name__ == '__main__':
    main()
