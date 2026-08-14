"""Run every analysis script in order and report what was produced.

    python scripts/run_all.py

Training (script 00) is excluded: it regenerates inputs that already ship with
this repository and takes hours. Pass --with-training to include it.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from nutrilens.config import FIGURES, TABLES

STEPS = [
    ('01_table1_and_cross_model_tests.py', 'Table 1, Supp Tables 6 and 8'),
    ('02_severity_gradient.py', 'Supp Table 12'),
    ('03_readouts_and_error_grid.py', 'Supp Table 9, Fig. 3c values'),
    ('04_ingredient_ladder.py', 'Supp Table 13'),
    ('05_decision_rules_and_alerts.py', 'Supp Tables 15, 16 and 18'),
    ('06_recommendation_actions.py', 'Supp Table 17'),
    ('07_nutrition_benchmark.py', 'Supp Tables 1 and 11, Fig. 2a values'),
    ('09_nutritionist_agreement.py', 'Supp Table 2, Fig. 2b values'),
    ('10_feature_attribution.py', 'Fig. 4a, 4b and 4c values'),
    ('11_sensitivity_and_ceilings.py', 'Supp Figs 3, 4 and 5 values'),
    ('12_figures.py', 'every data panel, main and supplementary'),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-training', action='store_true',
                        help='also retrain the glucose models (hours)')
    args = parser.parse_args()

    steps = ([('00_train_glucose_models.py', 'per-fold predictions')] if
             args.with_training else []) + STEPS

    failures = []
    started = time.time()
    for script, produces in steps:
        print(f'\n{"=" * 78}\n{script}  —  {produces}\n{"=" * 78}')
        result = subprocess.run([sys.executable, str(HERE / script)])
        if result.returncode != 0:
            failures.append(script)

    print(f'\n{"=" * 78}')
    print(f'{len(steps) - len(failures)}/{len(steps)} scripts completed in '
          f'{time.time() - started:.0f} s')
    print(f'  {len(list(TABLES.glob("*.csv")))} tables in {TABLES}')
    print(f'  {len(list(FIGURES.glob("*.pdf")))} figure panels in {FIGURES}')
    if failures:
        print('  FAILED: ' + ', '.join(failures))
        sys.exit(1)


if __name__ == '__main__':
    main()
