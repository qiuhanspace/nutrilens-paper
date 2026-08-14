"""Retrain the glucose models from the preprocessed meal-event table.

This regenerates the per-fold predictions that ship in
``data/stage2/predictions/``. It is not needed to reproduce any table or
figure: scripts 01 onwards read the shipped predictions. It is here so that the
training recipe is inspectable and rerunnable rather than only described.

Expect a few hours on a laptop: twelve arm-by-group combinations, each running
one leave-one-subject-out loop, and each fold running an inner five-fold search
for the number of boosting rounds.

    python scripts/00_train_glucose_models.py --out data/stage2/predictions_rerun

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import ARMS, FEATURES, GROUP_LABEL, PREDICTIONS
from nutrilens.training import load_features, run_loso


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PREDICTIONS,
                        help='where to write the per-fold predictions')
    parser.add_argument('--arms', nargs='*', default=list(ARMS),
                        help='subset of model arms to train')
    args = parser.parse_args()

    df = load_features(FEATURES)
    print(f'{len(df):,} meal events, {df.subject_id.nunique()} participants\n')

    for arm in args.arms:
        for group in GROUP_LABEL:
            started = time.time()
            result = run_loso(df, arm, group)
            folder = args.out / f'{arm}-{group}'
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'all_folds.npz', **result)
            print(f'  {ARMS[arm]:<42s} {group:<15s} '
                  f'{len(result["y_true"]):4d} held-out meals  '
                  f'{time.time() - started:6.0f} s')

    print(f'\nwrote per-fold predictions to {args.out}')


if __name__ == '__main__':
    main()
