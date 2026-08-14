"""Counterfactual meal modifications: the ranked clinical next step.

The framing is algorithmic recourse. Given a meal whose measured trajectory
breached a clinical threshold, what is the smallest change to the meal that
would have improved it, and by how much?

Three properties make the result interpretable rather than circular.

1. **The measurement defines the problem, not the model.** A meal needs a next
   step when its MEASURED trajectory reached the threshold. The model is never
   asked whether a meal was unsafe.
2. **The counterfactual is anchored on the measured trajectory.**
   ``cf(t) = measured(t) + [predicted_modified(t) - predicted_original(t)]``.
   Any per-meal offset between predicted and measured cancels exactly, so no
   meal is counted as improved merely because the model under-predicted it.
   What remains model-derived is the SIZE of the response to a composition
   change, which is the quantity the model is being asked for.
3. **Derived features are recomputed, never perturbed.** A substitution that
   changes carbohydrate must also move the densities and buffering ratios that
   depend on it, using the exact training-time formulas. Perturbing the
   primitives and leaving the derived features stale would ask the model a
   question it was never trained on.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import (CALIB_FRACTION, MODELS, OFFSET_CLIP_MGDL, REFERENCE,
                     TARGETS, arm_features)

EPS = 1e-3

# ── Action magnitudes, each anchored to a food or a guideline ────────────
CARB_EXCHANGE_G = 15.0     # one carbohydrate choice, ADA exchange lists
CARB_ALLOWANCE_G = 45.0    # three exchanges, the standard meal allowance
WHOLEGRAIN_FIBRE_G = 1.7   # brown vs white rice, whole-wheat vs white bread
OIL_TSP_FAT_G = 5.0        # one teaspoon of oil, about 45 kcal
# Shrinking a starchy portion removes food mass as well as carbohydrate;
# cooked staples run 25-35 % carbohydrate by weight, so about 3 g of food per
# gram of net carbohydrate.
FOOD_PER_CARB_G = 3.0
# ADA non-starchy vegetable exchange: half a cup cooked, about 80 g.
VEGETABLE_SERVING = dict(mass=80.0, carb=3.0, fiber=2.0, protein=2.0, fat=0.0)
SWAP_WINDOW_H = 2.0        # "a meal at a similar time of day"
# Replacing the meal replaces its COMPOSITION only. Meal timing and the
# preceding-meal context belong to the occasion, not to the dish, and carrying
# them over from the donor meal would credit the swap with an effect that is
# really a change of time of day.
COMPOSITION = ['mass', 'calories', 'carbs', 'fiber', 'fat', 'protein',
               'energy_density', 'carb_density', 'fat_density',
               'protein_density', 'fiber_buffering', 'fat_buffering',
               'protein_buffering']
FEASIBILITY_PERCENTILE = 95   # a lever may not push a nutrient past this

ACTIONS = [
    ('S1_wholegrain_swap', 'S1', 'Substitute',
     'swap 1 grain serving for whole grain',
     'brown vs white rice, whole-wheat vs white bread: +1.7 g fibre per serving'),
    ('S2_carb_to_protein', 'S2', 'Substitute',
     'swap 1 carbohydrate serving for protein',
     'ADA exchange lists (15 g); isocaloric-substitution trials'),
    ('A1_add_veg_1', 'A1', 'Add',
     'add 1 serving non-starchy vegetables',
     'ADA non-starchy vegetable exchange, half a cup cooked'),
    ('R2_portion_20', 'R2', 'Reduce',
     'reduce the starch portion by 20 %',
     'the practical ceiling: only about a third would accept a 20 % cut permanently'),
    ('R3_carb_allowance', 'R3', 'Reduce',
     'cut carbohydrate to the standard meal allowance (<= 45 g)',
     'ADA carbohydrate counting, three exchanges; target-based'),
    ('R4_cut_fat_1tsp', 'R4', 'Reduce',
     'cut added fat by 1 tsp oil',
     'targets fat density: -5 g fat, -45 kcal'),
    ('X1_meal_swap', 'X1', 'Replace',
     'eat a different meal at a similar time of day',
     "the participant's own best-predicted meal within +/- 2 h of the same "
     'clock time; meal selection rather than modification'),
]


def feasibility_caps(df: pd.DataFrame) -> dict:
    """Cohort percentile for each nutrient an action can increase.

    An action must not propose a meal the cohort never eats.
    """
    return {n: float(np.percentile(df[n].to_numpy(float), FEASIBILITY_PERCENTILE))
            for n in ('fiber', 'protein', 'fat')}


def apply_deltas(block: pd.DataFrame, d_carb=0.0, d_fiber=0.0, d_fat=0.0,
                 d_protein=0.0, d_mass=0.0) -> pd.DataFrame:
    """Apply macronutrient deltas and re-derive every dependent feature.

    Energy and mass are updated by delta rather than recomputed from scratch,
    so a delta of zero reproduces the observed row exactly. Fibre carries no
    available energy in this feature table, because `carbs` is already net of
    fibre, so a carbohydrate-to-fibre swap reduces energy by 4 kcal per gram.
    """
    r = block.copy()
    z = np.zeros(len(block))
    dc, dfib, dfat, dpro, dm = (np.asarray(x, float) + z for x in
                                (d_carb, d_fiber, d_fat, d_protein, d_mass))
    r['carbs'] = np.maximum(r['carbs'].to_numpy(float) + dc, 0.0)
    r['fiber'] = np.maximum(r['fiber'].to_numpy(float) + dfib, 0.0)
    r['fat'] = np.maximum(r['fat'].to_numpy(float) + dfat, 0.0)
    r['protein'] = np.maximum(r['protein'].to_numpy(float) + dpro, 0.0)
    r['mass'] = np.maximum(r['mass'].to_numpy(float) + dm, 1.0)
    r['calories'] = np.maximum(
        r['calories'].to_numpy(float) + 4.0 * dc + 4.0 * dpro + 9.0 * dfat, 0.0)

    cal = r['calories'].to_numpy(float)
    mass = r['mass'].to_numpy(float)
    carb = r['carbs'].to_numpy(float)
    r['energy_density'] = 100.0 * cal / (mass + EPS)
    r['carb_density'] = 100.0 * carb / (cal + EPS)
    r['fat_density'] = 100.0 * r['fat'].to_numpy(float) / (cal + EPS)
    r['protein_density'] = 100.0 * r['protein'].to_numpy(float) / (cal + EPS)
    r['fiber_buffering'] = np.log2(1.0 + r['fiber'].to_numpy(float) / (carb + EPS))
    r['fat_buffering'] = np.log2(1.0 + r['fat'].to_numpy(float) / (carb + EPS))
    r['protein_buffering'] = np.log2(1.0 + r['protein'].to_numpy(float) / (carb + EPS))
    return r


def apply_action(block: pd.DataFrame, key: str, caps: dict,
                 mass_cap: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Returns (modified meals, availability mask, grams of carbohydrate moved).

    An action is available only where it is physically possible: one cannot
    remove starch a meal does not contain, and the fibre a whole-grain swap
    adds is bounded by the grain the meal actually contains.
    """
    carb = block['carbs'].to_numpy(float)
    fib = block['fiber'].to_numpy(float)
    pro = block['protein'].to_numpy(float)
    fat = block['fat'].to_numpy(float)
    mass = block['mass'].to_numpy(float)
    n = len(block)
    moved, available, deltas = np.zeros(n), np.ones(n, bool), {}

    if key == 'S1_wholegrain_swap':
        moved = np.full(n, WHOLEGRAIN_FIBRE_G)
        available = (carb > WHOLEGRAIN_FIBRE_G) & (fib + WHOLEGRAIN_FIBRE_G <= caps['fiber'])
        deltas = dict(d_carb=-moved, d_fiber=moved)
    elif key == 'S2_carb_to_protein':
        moved = np.full(n, CARB_EXCHANGE_G)
        available = (carb > CARB_EXCHANGE_G) & (pro + CARB_EXCHANGE_G <= caps['protein'])
        deltas = dict(d_carb=-moved, d_protein=moved)
    elif key == 'A1_add_veg_1':
        v = VEGETABLE_SERVING
        available = mass + v['mass'] <= mass_cap
        deltas = dict(d_carb=v['carb'], d_fiber=v['fiber'],
                      d_protein=v['protein'], d_mass=v['mass'])
    elif key == 'R2_portion_20':
        moved = 0.20 * carb
        available = carb > 0
        deltas = dict(d_carb=-moved, d_mass=-FOOD_PER_CARB_G * moved)
    elif key == 'R3_carb_allowance':
        moved = np.maximum(carb - CARB_ALLOWANCE_G, 0.0)
        available = carb > CARB_ALLOWANCE_G
        deltas = dict(d_carb=-moved, d_mass=-FOOD_PER_CARB_G * moved)
    elif key == 'R4_cut_fat_1tsp':
        available = fat > OIL_TSP_FAT_G
        deltas = dict(d_fat=np.full(n, -OIL_TSP_FAT_G),
                      d_mass=np.full(n, -OIL_TSP_FAT_G))
    else:
        raise ValueError(f'{key} is not applied row-wise (meal swap is handled '
                         'in the fold loop, because it needs the other meals)')
    return apply_deltas(block, **deltas), available, moved


def counterfactual_on_measured(measured: np.ndarray, predicted_base: np.ndarray,
                               predicted_modified: np.ndarray) -> np.ndarray:
    """The counterfactual trajectory, anchored on the measured trace."""
    return (np.asarray(measured, float)
            + np.asarray(predicted_modified, float)
            - np.asarray(predicted_base, float))


def _dmatrix(block: pd.DataFrame) -> xgb.DMatrix:
    X = block[arm_features(REFERENCE)].copy().reset_index(drop=True)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype('category')
    return xgb.DMatrix(X, enable_categorical=True)


def iter_folds(df: pd.DataFrame, group: str):
    """Yield (participant, model, test meals, offset, measured anchors) per fold.

    Re-predicting counterfactuals requires the model that produced the original
    prediction for that participant, so the saved per-fold boosters are used
    rather than a refit. Row order and fold order follow the training run
    exactly, which is asserted below; the reconstructed predictions reproduce
    the stored ones to 0.0 mg/dL.
    """
    status = df.groupby('subject_id')['t2d_status'].first()
    subjects = sorted(status[status == group].index.tolist())
    block = df[df.subject_id.isin(subjects)].reset_index(drop=True)

    folder = MODELS / f'{REFERENCE}-{group}'
    meta = json.loads((folder / 'metadata.json').read_text())
    assert meta['test_ids'] == subjects, f'{group}: fold order mismatch'

    for i, held_out in enumerate(meta['test_ids']):
        booster = xgb.Booster()
        booster.load_model(str(folder / f'fold_{i:03d}_subject_{held_out}.json'))
        rows = block[block.subject_id == held_out].reset_index(drop=True)
        pred = booster.predict(_dmatrix(rows))
        y = rows[TARGETS].to_numpy(float)

        epochs = pd.unique(rows['meal_epoch_id'])
        n_cal = int(CALIB_FRACTION * len(epochs))
        cal = rows['meal_epoch_id'].isin(set(epochs[:n_cal])).to_numpy()
        offset = np.clip(np.median(y[cal] - pred[cal], axis=0),
                         -OFFSET_CLIP_MGDL, OFFSET_CLIP_MGDL)
        yield held_out, booster, rows.loc[~cal].reset_index(drop=True), offset, y[~cal]
