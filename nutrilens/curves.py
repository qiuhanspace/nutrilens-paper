"""Trajectory reconstruction, per-participant error, and clinical readouts.

The model predicts glucose at nine anchor points. Everything downstream —
the primary error metric, every clinical readout, every decision rule — is
computed from the continuous trajectory reconstructed from those nine points,
so that one prediction serves every decision.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

from .config import (ARMS, GROUP_LABEL, PREDICTIONS, T_ANCHORS, T_DENSE)


# ── Reconstruction ───────────────────────────────────────────────────────
def reconstruct(anchors: np.ndarray, method: str = 'cubic') -> np.ndarray:
    """Nine anchor points -> a 121-minute trajectory.

    The natural boundary condition (zero second derivative at the endpoints) is
    used rather than the more common not-a-knot condition, which overshoots at
    the ends of a nine-point curve. Three alternative schemes are provided for
    the reconstruction sensitivity analysis; all four agree to within
    0.1 mg/dL.
    """
    a = np.asarray(anchors, dtype=float)
    if method == 'cubic':
        return CubicSpline(T_ANCHORS, a, axis=1, bc_type='natural')(T_DENSE)
    if method == 'linear':
        return np.vstack([np.interp(T_DENSE, T_ANCHORS, row) for row in a])
    if method == 'pchip':
        return PchipInterpolator(T_ANCHORS, a, axis=1)(T_DENSE)
    if method == 'akima':
        return Akima1DInterpolator(T_ANCHORS, a, axis=1)(T_DENSE)
    raise ValueError(f'unknown reconstruction method: {method!r}')


# ── Loading the stored per-fold predictions ──────────────────────────────
def load_predictions(arm: str, group: str) -> dict:
    """Held-out predictions for one model arm in one metabolic group.

    Keys:
        y_true                    measured glucose at the nine anchor points
        y_pred_raw                model output before personalisation
        y_pred_adjusted           after the per-participant calibration offset
        y_pred_naive_group        the training group's median trajectory
        y_pred_naive_individual   the participant's calibration-window median
        subject_ids               participant identifier per meal event
    """
    path = PREDICTIONS / f'{arm}-{group}' / 'all_folds.npz'
    with np.load(path, allow_pickle=True) as d:
        return {k: np.asarray(d[k]) for k in d.files}


# ── Primary endpoint ─────────────────────────────────────────────────────
def per_subject_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                     subject_ids: np.ndarray,
                     method: str = 'cubic') -> pd.Series:
    """Per-participant RMSE of the reconstructed trajectory.

    Aggregation order matters and is stated explicitly: the error is computed
    per timepoint across a participant's held-out meals and then averaged
    across the 121 timepoints. This weights each part of the trajectory
    equally rather than letting the high-variance peak region dominate.

    Both the measured and the predicted trajectory are reconstructed from their
    nine anchor points, so the metric compares two curves of the same
    construction rather than scoring a curve against the raw sensor trace.
    """
    yt = reconstruct(y_true, method)
    yp = reconstruct(y_pred, method)
    sids = np.asarray(subject_ids)
    out = {}
    for s in pd.unique(sids):
        m = sids == s
        per_timepoint = np.sqrt(np.mean((yt[m] - yp[m]) ** 2, axis=0))
        out[str(s)] = float(per_timepoint.mean())
    return pd.Series(out).sort_index()


def arm_rmse(arm: str, group: str, field: str = 'y_pred_adjusted',
             method: str = 'cubic') -> pd.Series:
    """Per-participant RMSE for one arm, one group, one prediction field."""
    z = load_predictions(arm, group)
    return per_subject_rmse(z['y_true'], z[field], z['subject_ids'], method)


def rmse_table(method: str = 'cubic') -> pd.DataFrame:
    """Mean and SD of per-participant RMSE for every arm and group (Table 1)."""
    rows = []
    for arm, label in ARMS.items():
        for group, glabel in GROUP_LABEL.items():
            s = arm_rmse(arm, group, method=method)
            rows.append(dict(arm=arm, model=label, group=glabel,
                             n_subjects=len(s), rmse_mean=float(s.mean()),
                             rmse_sd=float(s.std(ddof=1)),
                             rmse_median=float(s.median())))
    return pd.DataFrame(rows)


# ── Clinical readouts, all from the same trajectory ──────────────────────
def readouts(curves: np.ndarray) -> pd.DataFrame:
    """The eight readouts derived from an (n_meals, 121) trajectory matrix.

    Both incremental-area definitions are returned. They differ materially in
    type 2 diabetes, where the trajectory spends time below its own starting
    value; the positive-only convention is the one glycaemic-index research
    uses, the signed convention credits those excursions.
    """
    curves = np.asarray(curves, dtype=float)
    baseline = curves[:, 0][:, None]
    excursion = curves - baseline
    return pd.DataFrame({
        'Peak glucose': curves.max(axis=1),
        'Time to peak': T_DENSE[curves.argmax(axis=1)],
        'Glucose at 2 h': curves[:, -1],
        'AUC 0-2 h': np.trapezoid(curves, T_DENSE, axis=1),
        'iAUC, net signed': np.trapezoid(excursion, T_DENSE, axis=1),
        'iAUC, positive only': np.trapezoid(np.clip(excursion, 0, None),
                                            T_DENSE, axis=1),
        'Time in range 140-180': ((curves >= 140) & (curves <= 180)).sum(axis=1).astype(float),
        'Time above 180': (curves > 180).sum(axis=1).astype(float),
    })


READOUT_UNITS = {
    'Peak glucose': 'mg/dL', 'Time to peak': 'min', 'Glucose at 2 h': 'mg/dL',
    'AUC 0-2 h': 'mg/dL*min', 'iAUC, net signed': 'mg/dL*min',
    'iAUC, positive only': 'mg/dL*min', 'Time in range 140-180': 'min',
    'Time above 180': 'min',
}


def rule_breached(curves: np.ndarray, spec: dict,
                  tol: float = 1e-9) -> np.ndarray:
    """Whether each trajectory breaches a decision rule at that rule's window.

    The tolerance matters for exactly one meal in this cohort, whose 2-hour
    value is exactly 140 mg/dL. Interpolation reproduces an anchor point to
    within about 1e-14, which is enough to flip a bare `>=` comparison. A
    threshold decision should not be settled by floating-point arithmetic, so a
    value within `tol` of the threshold counts as reaching it.
    """
    value = rule_value(curves, spec['window'], spec['stat'])
    return value >= spec['threshold'] - tol


def rule_value(curves: np.ndarray, window: tuple[int, int],
               stat: str = 'max') -> np.ndarray:
    """The value a clinical decision rule reads, at its own measurement window.

    A rule specified as "the value 1-2 hours after the start of the meal" is
    not the same as "the maximum over the postprandial window": in this cohort
    the true peak often arrives before 60 minutes, so scoring the former as the
    latter over-counts breaches by about 14 %.
    """
    lo, hi = window
    segment = np.asarray(curves, dtype=float)[:, lo:hi + 1]
    return segment.max(axis=1) if stat == 'max' else segment.mean(axis=1)


# ── Error grid ───────────────────────────────────────────────────────────
def clarke_zone(reference: float, predicted: float) -> str:
    """Clarke consensus error-grid zone for one predicted-measured pair.

    Zone A: clinically accurate. Zone B: benign error. Zone C: unnecessary
    treatment. Zone D: failure to detect. Zone E: erroneous treatment.
    """
    r, p = float(reference), float(predicted)
    if (r <= 70 and p <= 70) or abs(p - r) <= 0.2 * r:
        return 'A'
    if (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
        return 'E'
    if ((70 <= r <= 290 and p >= r + 110) or (130 <= r <= 180 and p <= 7 / 5 * r - 182)):
        return 'C'
    if ((r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180)
            or (175 / 3 <= r <= 70 and p >= 6 / 5 * r)):
        return 'D'
    return 'B'


def clarke_summary(y_true: np.ndarray, y_pred: np.ndarray,
                   method: str = 'cubic') -> pd.Series:
    """Percentage of predicted-measured pairs in each Clarke zone."""
    yt = reconstruct(y_true, method).ravel()
    yp = reconstruct(y_pred, method).ravel()
    zones = pd.Series([clarke_zone(a, b) for a, b in zip(yt, yp)])
    pct = zones.value_counts(normalize=True) * 100
    return pct.reindex(['A', 'B', 'C', 'D', 'E']).fillna(0.0)
