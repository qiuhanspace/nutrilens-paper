"""The values behind Fig. 4a, 4b and 4c.

4a   what the model USES against what it NEEDS
4b   inside the image arm, which KIND of information the photograph supplies
4c   individual nutrition features, ranked

Attribution and necessity are different questions and on these data they
disagree, which is why 4a reports both. A model can lean on a feature that buys
it nothing: demographics carry roughly a quarter of the attribution in every
group, yet removing them costs nothing outside type 2 diabetes. Reporting
attribution alone would tell a reader they are a quarter of the model's value.

The calibration offset is additively commensurable with the SHAP values, being
a constant added to the same prediction, so it is included as a term. The
reported shares are therefore shares of everything driving the prediction
rather than of the feature budget alone.

Attribution comes from a separate single-output training run, because exact
TreeSHAP is intractable for a nine-dimensional output.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import (ARM_NO_DEMOGRAPHICS, DEMOGRAPHIC_FEATURES,
                              GROUP_SHORT, IMAGE_SUBGROUPS, NUTRITION_DISPLAY,
                              NUTRITION_KIND, REFERENCE, SHAP, TABLES)
from nutrilens.curves import arm_rmse, load_predictions
from nutrilens.stats import paired_wilcoxon


def mean_abs_shap(group: str) -> pd.Series:
    """Mean absolute TreeSHAP value per feature, over meals and targets."""
    stored = pd.read_csv(SHAP / group / 'global_shap.csv')
    raw = np.load(SHAP / group / 'all_shap_values.npy')      # (meals, feats, 9)
    recomputed = np.abs(raw).mean(axis=(0, 2))
    assert np.allclose(recomputed, stored.global_mean_abs_shap.to_numpy(float),
                       rtol=1e-4), f'{group}: SHAP values do not match the stored summary'
    return pd.Series(recomputed, index=stored.feature)


def ablation_cost(arm_without: str, group: str) -> tuple[float, float]:
    """RMSE cost of removing an arm, and the paired Wilcoxon p.

    Positive means the arm is worth keeping.
    """
    full = arm_rmse(REFERENCE, group)
    reduced = arm_rmse(arm_without, group)
    idx = full.index.intersection(reduced.index)
    delta = (reduced.loc[idx] - full.loc[idx]).to_numpy(float)
    return float(delta.mean()), paired_wilcoxon(delta)['p']


def main() -> None:
    shap_by_group = {g: mean_abs_shap(g) for g in GROUP_SHORT}

    # ── Fig. 4a: attribution share against ablation cost ─────────────────
    rows = []
    for group, short in GROUP_SHORT.items():
        shap = shap_by_group[group]
        image_features = [f for f in shap.index if f not in DEMOGRAPHIC_FEATURES]
        z = load_predictions(REFERENCE, group)
        offset = np.abs(np.asarray(z['y_pred_adjusted'], float)
                        - np.asarray(z['y_pred_raw'], float))
        assert len(offset) == np.load(SHAP / group / 'all_shap_values.npy').shape[0], \
            f'{group}: attribution and prediction meal counts differ'

        contribution = {
            'image-derived features': shap[image_features].sum(),
            'demographic features': shap[DEMOGRAPHIC_FEATURES].sum(),
            'calibration offset': float(offset.mean()),
        }
        total = sum(contribution.values())

        # Removing calibration means falling back to the uncalibrated model.
        full = arm_rmse(REFERENCE, group)
        raw = arm_rmse(REFERENCE, group, field='y_pred_raw')
        calib_delta = (raw - full).to_numpy(float)
        costs = {
            'image-derived features': ablation_cost('baseline-demo', group),
            'demographic features': ablation_cost(ARM_NO_DEMOGRAPHICS, group),
            'calibration offset': (float(calib_delta.mean()),
                                   paired_wilcoxon(calib_delta)['p']),
        }
        for arm, value in contribution.items():
            cost, p = costs[arm]
            rows.append(dict(group=short, arm=arm,
                             n_features=(len(image_features) if 'image' in arm
                                         else len(DEMOGRAPHIC_FEATURES)
                                         if 'demographic' in arm else 1),
                             mean_abs_shap_mgdl=round(float(value), 3),
                             share_pct=round(100 * value / total, 1),
                             ablation_cost_mgdl=round(cost, 3),
                             ablation_p=p))
    fig4a = pd.DataFrame(rows)
    fig4a.to_csv(TABLES / 'fig4a_attribution_and_ablation.csv', index=False)
    print('Fig. 4a — what the model uses against what it needs\n')
    print(fig4a.pivot_table(index='arm', columns='group',
                            values=['share_pct', 'ablation_cost_mgdl'])
          .round(2).to_string())

    # ── Fig. 4b: which kind of information the photograph supplies ───────
    rows = []
    for group, short in GROUP_SHORT.items():
        shap = shap_by_group[group]
        arm_total = sum(shap[f].sum() for f in IMAGE_SUBGROUPS.values())
        for subgroup, features in IMAGE_SUBGROUPS.items():
            value = shap[features].sum()
            rows.append(dict(group=short, subgroup=subgroup,
                             n_features=len(features),
                             mean_abs_shap_mgdl=round(float(value), 3),
                             share_of_image_arm_pct=round(100 * value / arm_total, 1)))
    fig4b = pd.DataFrame(rows)
    fig4b.to_csv(TABLES / 'fig4b_image_subgroups.csv', index=False)
    print('\nFig. 4b — inside the image arm (% of that arm)\n')
    print(fig4b.pivot_table(index='subgroup', columns='group',
                            values='share_of_image_arm_pct').round(1).to_string())

    # ── Fig. 4c: individual nutrition features, ranked ──────────────────
    frame = pd.DataFrame({short: shap_by_group[g][list(NUTRITION_DISPLAY)]
                          for g, short in GROUP_SHORT.items()})
    ranks = frame.rank(ascending=False)
    fig4c = pd.DataFrame({
        'feature': frame.index,
        'display': [NUTRITION_DISPLAY[f] for f in frame.index],
        'kind': [NUTRITION_KIND[f] for f in frame.index],
        'mean_rank': ranks.mean(axis=1).to_numpy(),
        **{f'shap_{c}': frame[c].to_numpy() for c in frame.columns},
        **{f'rank_{c}': ranks[c].astype(int).to_numpy() for c in ranks.columns},
    }).sort_values('mean_rank')
    fig4c.to_csv(TABLES / 'fig4c_nutrition_feature_ranking.csv', index=False)
    print('\nFig. 4c — individual nutrition features, by mean rank\n')
    print(fig4c[['display', 'kind', 'mean_rank', 'rank_Normo', 'rank_Pre',
                 'rank_T2D']].round(2).to_string(index=False))

    quality = fig4c[fig4c.feature == 'fiber_buffering'].iloc[0]
    quantity = fig4c[fig4c.feature == 'carbs'].iloc[0]
    print(f'\n  {quality.display}: mean rank {quality.mean_rank:.1f} '
          f'({quality.rank_Normo}/{quality.rank_Pre}/{quality.rank_T2D})')
    print(f'  {quantity.display}: mean rank {quantity.mean_rank:.1f} '
          f'({quantity.rank_Normo}/{quantity.rank_Pre}/{quantity.rank_T2D})'
          '  — quality outranks quantity')
    print(f'\nwrote 3 tables to {TABLES}')


if __name__ == '__main__':
    main()
