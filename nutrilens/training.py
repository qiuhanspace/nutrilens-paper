"""Leave-one-subject-out training of the postprandial glucose model.

The recipe, in one place:

  * Models are trained leave-one-subject-out WITHIN a metabolic group, so
    training and test participants always share a glycaemic state and no
    participant contributes to both sides of a fold.
  * The number of boosting rounds is chosen inside each training fold by
    grouped five-fold cross-validation with early stopping, grouping by
    participant, and the median best-round count is used to refit on the whole
    training fold. Nothing is selected using the held-out participant.
  * For the held-out participant, the first 30 % of that participant's records
    form the calibration window and the remaining 70 % the test set. The
    per-timepoint median residual over the calibration window, clipped to
    +/- 30 mg/dL, becomes that participant's offset.
  * Four prediction fields are stored per fold, which is what makes the
    ingredient analysis possible without retraining anything.

Running this script reproduces the stored predictions under
``data/stage2/predictions/``. It takes a few hours on a laptop; the analysis
scripts read the stored predictions and do not require it.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from .config import (CALIB_FRACTION, CATEGORICAL_FEATURES,
                     EARLY_STOPPING_ROUNDS, N_INNER_CV_SPLITS,
                     NUM_BOOST_ROUND, OFFSET_CLIP_MGDL, TARGETS, XGB_PARAMS,
                     arm_features)


def load_features(path) -> pd.DataFrame:
    """The preprocessed meal-event table, with categoricals declared as such."""
    df = pd.read_csv(path)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def calibration_mask(subject_rows: pd.DataFrame,
                     fraction: float = CALIB_FRACTION) -> np.ndarray:
    """First `fraction` of a participant's meal events, in time order.

    The window is defined on records rather than on days, so a participant who
    logs more densely gets a shorter calendar window. That is intentional: what
    the offset needs is a number of meals, not a number of days.
    """
    epochs = subject_rows['meal_epoch_id'].to_numpy()
    unique = pd.unique(epochs)
    n = int(fraction * len(unique)) if fraction < 1 else int(fraction)
    calib = set(unique[:max(1, min(n, len(unique)))])
    return np.array([e in calib for e in epochs], dtype=bool)


def calibration_offset(y_cal: np.ndarray, pred_cal: np.ndarray) -> np.ndarray:
    """Per-timepoint median residual, clipped.

    Clipping bounds the damage a single anomalous calibration window can do;
    without it one atypical onboarding period can shift every later prediction
    for that participant.
    """
    offsets = np.median(np.asarray(y_cal, float) - np.asarray(pred_cal, float),
                        axis=0)
    return np.clip(offsets, -OFFSET_CLIP_MGDL, OFFSET_CLIP_MGDL)


def _fit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> xgb.Booster:
    """Fit one training fold, choosing the number of rounds inside it."""
    params = dict(XGB_PARAMS)
    if y.shape[1] > 1:
        params['multi_strategy'] = 'multi_output_tree'
        params['num_target'] = y.shape[1]

    best_rounds = []
    for train_idx, val_idx in GroupKFold(n_splits=N_INNER_CV_SPLITS).split(X, y, groups):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y[train_idx],
                             enable_categorical=True)
        dval = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx],
                           enable_categorical=True)
        history: dict = {}
        xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                  evals=[(dval, 'eval')], evals_result=history,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                  verbose_eval=False)
        best_rounds.append(len(history['eval']['rmse']))

    rounds = int(np.median(best_rounds))
    dfull = xgb.DMatrix(X, label=y, enable_categorical=True)
    return xgb.train(params, dfull, num_boost_round=rounds, verbose_eval=False)


def run_loso(df: pd.DataFrame, arm: str, group: str) -> dict:
    """Leave-one-subject-out predictions for one arm within one group.

    Returns the same five arrays that are stored in the shipped `.npz` files.
    """
    features = arm_features(arm)
    block = df[df['t2d_status'] == group].reset_index(drop=True)
    subjects = pd.unique(block['subject_id'])

    y_true, raw, adjusted, naive_group, naive_individual, sids = [], [], [], [], [], []

    for held_out in subjects:
        is_test = block['subject_id'] == held_out
        train, test = block[~is_test], block[is_test].reset_index(drop=True)

        model = _fit(train[features], train[TARGETS].to_numpy(float),
                     train['subject_id'].to_numpy())
        pred = model.predict(xgb.DMatrix(test[features], enable_categorical=True))

        y_subject = test[TARGETS].to_numpy(float)
        cal = calibration_mask(test)
        if cal.all():                      # too few meals to leave any for test
            continue
        offset = calibration_offset(y_subject[cal], pred[cal])

        keep = ~cal
        y_true.append(y_subject[keep])
        raw.append(pred[keep])
        adjusted.append(pred[keep] + offset)
        naive_group.append(np.tile(np.median(train[TARGETS].to_numpy(float), axis=0),
                                   (keep.sum(), 1)))
        naive_individual.append(np.tile(np.median(y_subject[cal], axis=0),
                                        (keep.sum(), 1)))
        sids.append(np.repeat(str(held_out), keep.sum()))

    return dict(
        y_true=np.vstack(y_true),
        y_pred_raw=np.vstack(raw),
        y_pred_adjusted=np.vstack(adjusted),
        y_pred_naive_group=np.vstack(naive_group),
        y_pred_naive_individual=np.vstack(naive_individual),
        subject_ids=np.concatenate(sids),
    )
