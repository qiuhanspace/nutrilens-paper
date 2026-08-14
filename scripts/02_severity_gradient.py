"""Supplementary Table 12: does the advantage grow with metabolic severity?

Three within-group results say nothing about whether the groups differ from one
another, so the gradient is tested directly and once. The pre-specified test is
a Jonckheere-Terpstra trend of the per-participant advantage over the
demographic baseline against the ordered alternative normoglycaemia <
prediabetes < type 2 diabetes. It is directional because the claim is
directional, and uncorrected because it is a single test.

The other two comparators are exploratory, and they function as a control: a
blanket "everything is larger in type 2 diabetes" confound would move them
together with the primary row, and it does not.

The advantage is also expressed as a percentage of the comparator's own error,
which removes the difference in overall error scale between groups.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import (ARMS, COMPARATORS, GROUPS, N_PERMUTATIONS,
                              REFERENCE, SEED_PERMUTATION, TABLES)
from nutrilens.curves import arm_rmse
from nutrilens.stats import jonckheere_terpstra

SCALES = ['absolute (mg/dL)', 'relative (% of comparator RMSE)']


def advantage(comparator: str, group: str, relative: bool) -> np.ndarray:
    """Comparator error minus NutriLens error; positive favours NutriLens."""
    a = arm_rmse(REFERENCE, group)
    b = arm_rmse(comparator, group)
    idx = a.index.intersection(b.index).sort_values()
    adv = (b.loc[idx] - a.loc[idx]).to_numpy(float)
    return 100.0 * adv / b.loc[idx].to_numpy(float) if relative else adv


def main() -> None:
    rng = np.random.default_rng(SEED_PERMUTATION)
    severity = np.concatenate([np.full(len(arm_rmse(REFERENCE, g)), i)
                               for i, g in enumerate(GROUPS)])
    rows = []
    for comparator in COMPARATORS:
        for scale in SCALES:
            per_group, stats = [], {}
            for g in GROUPS:
                adv = advantage(comparator, g, scale.startswith('relative'))
                per_group.append(adv)
                stats |= {f'n_{g}': adv.size, f'mean_{g}': float(adv.mean()),
                          f'median_{g}': float(np.median(adv)),
                          f'sd_{g}': float(adv.std(ddof=1))}
            kw = kruskal(*per_group)
            tau, p = jonckheere_terpstra(np.concatenate(per_group), severity, rng)
            rows.append(dict(
                comparator=ARMS[comparator],
                role=('primary (pre-specified)' if comparator == 'baseline-demo'
                      else 'exploratory'),
                scale=scale, **stats,
                kruskal_H=float(kw.statistic), kruskal_p=float(kw.pvalue),
                jt_tau_b=tau, jt_p_one_sided=p, n_permutations=N_PERMUTATIONS))

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / 'supptable12_severity_gradient.csv', index=False)
    print('Supplementary Table 12 — severity gradient\n')
    print(out[['comparator', 'scale', 'mean_normoglycemic', 'mean_prediabetic',
               'mean_diabetic', 'kruskal_p', 'jt_tau_b', 'jt_p_one_sided']]
          .round(4).to_string(index=False))
    print(f'\nwrote {TABLES / "supptable12_severity_gradient.csv"}')


if __name__ == '__main__':
    main()
