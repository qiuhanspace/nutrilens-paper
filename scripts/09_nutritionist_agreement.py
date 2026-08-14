"""Supplementary Table 2 and the values behind Fig. 2b.

Agreement between NutriLens and two certified nutritionists on the 80-dish
panel: 40 Nutrition5k dishes with laboratory ground truth and 40 everyday meals
from the GLOW UP study.

Three rater pairs are reported, and the third is the one that makes the other
two readable. Agreement between the two nutritionists bounds what expert
agreement looks like on this task, so the correct claim is that the model sits
within the range of expert disagreement on this panel, not that it outperforms
a nutritionist.

Lin's concordance correlation coefficient is reported alongside Pearson r
because correlation is blind to systematic offset and scale: two raters can
correlate perfectly while one consistently estimates half the mass.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import SEED, STAGE1, TABLES

QUANTITIES = {
    'total_mass_g': 'mass (g)',
    'total_calories_kcal': 'calories (kcal)',
    'total_net_carbs_g': 'carbs (g)',
    'total_dietary_fiber_g': 'fiber (g)',
    'total_fat_g': 'fat (g)',
    'total_protein_g': 'protein (g)',
}
PAIRS = [('NutriLens', 'Annotator 1'), ('NutriLens', 'Annotator 2'),
         ('Annotator 1', 'Annotator 2')]
N_BOOTSTRAP = 1000


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient.

    Pearson r rescaled by how far the pair sits from the identity line, so a
    constant offset or a scale error is penalised.
    """
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (vx + vy + (mx - my) ** 2))


def agreement(x: np.ndarray, y: np.ndarray, rng) -> dict:
    """Both statistics, with a two-sided bootstrap p-value over dishes."""
    r = pearsonr(x, y).statistic
    ccc = lins_ccc(x, y)
    # Null: the two raters are unrelated. Permuting one against the other gives
    # the distribution of both statistics under that null.
    null_r, null_ccc = [], []
    for _ in range(N_BOOTSTRAP):
        shuffled = rng.permutation(y)
        null_r.append(pearsonr(x, shuffled).statistic)
        null_ccc.append(lins_ccc(x, shuffled))
    return dict(pearson_r=float(r),
                pearson_p=float((np.abs(null_r) >= abs(r)).mean()),
                lins_ccc=ccc,
                ccc_p=float((np.abs(null_ccc) >= abs(ccc)).mean()))


def load_raters() -> dict[str, pd.DataFrame]:
    """The three raters, on a common dish index."""
    model = pd.read_csv(STAGE1 / 'parsed' / 'nutrilens_e2e_humanai_per_dish.csv')
    a1 = pd.read_csv(STAGE1 / 'human' / 'annotator1_per_dish_totals.csv')
    a2 = pd.read_csv(STAGE1 / 'human' / 'annotator2_per_dish_totals.csv')
    mapping = pd.read_csv(STAGE1 / 'human' / 'n5k40_name_mapping.csv')

    # The annotators worked from image names; the model output is keyed by dish
    # identifier. The mapping file joins the two for the Nutrition5k half.
    key = dict(zip(mapping.iloc[:, 0], mapping.iloc[:, 1]))
    for frame in (a1, a2):
        frame['dish_id'] = frame['image'].map(lambda v: key.get(v, v))
    return {'NutriLens': model.set_index('dish_id'),
            'Annotator 1': a1.set_index('dish_id'),
            'Annotator 2': a2.set_index('dish_id')}


def main() -> None:
    raters = load_raters()
    rng = np.random.default_rng(SEED)

    # Panel membership: a dish counts if at least one human rated it.
    panel = raters['Annotator 1'].index.union(raters['Annotator 2'].index)
    source = {}
    for name, frame in raters.items():
        if 'source' in frame.columns:
            source.update(frame['source'].to_dict())
    cohorts = [('Nutrition5k', lambda s: s == 'n5k'),
               ('GLOW UP', lambda s: s == 'glowup'),
               ('Combined', lambda s: True)]

    rows = []
    for a, b in PAIRS:
        fa, fb = raters[a], raters[b]
        shared = fa.index.intersection(fb.index).intersection(panel)
        for cohort, keep in cohorts:
            dishes = [d for d in shared if keep(source.get(d, ''))]
            for column, label in QUANTITIES.items():
                if column not in fa.columns or column not in fb.columns:
                    continue
                pair = pd.DataFrame({'x': fa.loc[dishes, column],
                                     'y': fb.loc[dishes, column]}).dropna()
                if len(pair) < 10:
                    continue
                rows.append(dict(
                    rater_pair=f'{a} – {b}', panel=cohort, nutrient=label,
                    n_dishes=len(pair),
                    **agreement(pair.x.to_numpy(float), pair.y.to_numpy(float),
                                rng)))

    st2 = pd.DataFrame(rows)
    st2.to_csv(TABLES / 'supptable2_nutritionist_agreement.csv', index=False)

    print('Supplementary Table 2 / Fig. 2b — rater agreement, combined panel\n')
    combined = st2[st2.panel == 'Combined']
    print(combined.pivot_table(index='nutrient', columns='rater_pair',
                               values='lins_ccc').round(3).to_string())
    print("\n  Lin's CCC ranges, combined panel")
    for pair, g in combined.groupby('rater_pair'):
        print(f'    {pair:<28s} r {g.pearson_r.min():.2f}-{g.pearson_r.max():.2f}'
              f'   CCC {g.lins_ccc.min():.2f}-{g.lins_ccc.max():.2f}')
    print(f'\nwrote {TABLES / "supptable2_nutritionist_agreement.csv"}')


if __name__ == '__main__':
    main()
