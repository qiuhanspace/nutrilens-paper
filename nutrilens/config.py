"""Paths, cohort definitions and locked analysis constants.

Every constant that the manuscript depends on is defined once, here, so that a
recipe cannot drift between scripts.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Layout ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
STAGE1 = DATA / 'stage1'
STAGE2 = DATA / 'stage2'
PREDICTIONS = STAGE2 / 'predictions'
MODELS = STAGE2 / 'models'
SHAP = STAGE2 / 'shap'
FEATURES = STAGE2 / 'meal_events_features.csv'

OUT = ROOT / 'outputs'
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'
for _d in (OUT, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ── Cohort ───────────────────────────────────────────────────────────────
# Metabolic groups, in ascending severity. Assignment uses the American
# Diabetes Association dual criterion (HbA1c >= 6.5 % or fasting glucose
# >= 126 mg/dL -> type 2 diabetes; else HbA1c >= 5.7 % or fasting glucose
# >= 100 mg/dL -> prediabetes; else normoglycaemia), which is the criterion
# clinical care uses and which differs from the HbA1c-only grouping of the
# original CGMacros descriptor.
GROUPS = ['normoglycemic', 'prediabetic', 'diabetic']
GROUP_LABEL = {'normoglycemic': 'Normoglycemic',
               'prediabetic': 'Prediabetic',
               'diabetic': 'Type 2 Diabetic'}
GROUP_SHORT = {'normoglycemic': 'Normo', 'prediabetic': 'Pre', 'diabetic': 'T2D'}

# ── Model arms ───────────────────────────────────────────────────────────
# The four arms reported in the main text. All four are trained with the same
# model class, the same hyperparameters and the same calibration offset, and
# differ only in which features they receive.
REFERENCE = 'image_logs-demo_nutrition'
ARMS = {
    'image_logs-demo_nutrition': 'NutriLens',
    'manual_logs-demo_nutrition': 'Manual Log',
    'image_logs-demo_nutrition_wearable_metabolic':
        'Full (NutriLens + wearable + metabolic)',
    'baseline-demo': 'Baseline (Demographics)',
}
COMPARATORS = [a for a in ARMS if a != REFERENCE]

# ── Prediction target and reconstruction grid ────────────────────────────
# Nine anchor points at 15-minute spacing: the coarsest grid that still
# resolves the rise, the peak and the 2-hour value.
TARGETS = [f'postprandial_glucose_t{t:03d}' for t in range(0, 121, 15)]
T_ANCHORS = np.arange(0, 121, 15, dtype=float)
T_DENSE = np.arange(0, 121, dtype=float)

# ── Personalisation ──────────────────────────────────────────────────────
CALIB_FRACTION = 0.30      # first 30 % of a participant's records (~2 days)
OFFSET_CLIP_MGDL = 30.0    # bound on the per-timepoint calibration offset

# ── Statistics ───────────────────────────────────────────────────────────
# Equivalence margin, derived in Methods from three published anchors. It is
# NOT an established threshold and must not be described as one.
MARGIN_MGDL = 3.0
EQUIV_ALPHA = 0.05         # two one-sided tests read the 90 % interval
N_CONTRASTS = 3            # Bonferroni across the three within-group contrasts
N_PERMUTATIONS = 200_000   # permutation null for the trend test
SEED = 42
SEED_PERMUTATION = 20260813

# ── Clinical decision rules ──────────────────────────────────────────────
# Each rule is evaluated at the measurement window its source specifies. The
# windows differ, and scoring a rule at the wrong window changes which meals it
# selects. The level-2 rule is a per-meal DERIVATION of a day-level threshold
# and is labelled as such wherever it appears.
DECISION_RULES = {
    'ADA: 1-2 h value >= 180 mg/dL': dict(window=(60, 120), stat='max', threshold=180),
    'IDF: 1-2 h value >= 160 mg/dL': dict(window=(60, 120), stat='max', threshold=160),
    'AACE: value at 2 h >= 140 mg/dL': dict(window=(120, 120), stat='max', threshold=140),
    'Level 2 (derived): reaches 250 mg/dL': dict(window=(0, 120), stat='max', threshold=250),
}

# Threshold above which a meal is treated as needing a next step. 180 mg/dL is
# the most permissive of the guideline ceilings, so the resulting count of
# flagged meals is conservative.
ACTION_THRESHOLD_MGDL = 180.0

# ── XGBoost ──────────────────────────────────────────────────────────────
# Fixed a priori for small-sample stability and held constant across every arm,
# so that differences between arms reflect their inputs and not their tuning.
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'seed': 0,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'eta': 0.02,
    'alpha': 0.0,
    'lambda': 50.0,
    'gamma': 0.0,
    'max_bin': 256,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50
N_INNER_CV_SPLITS = 5

# ── Features ─────────────────────────────────────────────────────────────
DEMOGRAPHIC_FEATURES = ['age', 'gender', 'bmi']
MEAL_CONTEXT_FEATURES = ['meal_onset_time_sin', 'meal_onset_time_cos', 'uncertain']
NUTRITION_FEATURES = [
    'mass', 'calories', 'carbs', 'fiber', 'fat', 'protein',
    'energy_density', 'carb_density', 'fat_density', 'protein_density',
    'fiber_buffering', 'fat_buffering', 'protein_buffering',
    'past_0to4hr_meals_calories', 'past_0to4hr_meals_fat',
    'past_0to4hr_meals_protein', 'past_0to4hr_meals_fiber',
    'past_0to4hr_meals_energy_density', 'past_0to4hr_meals_fat_density',
    'past_0to4hr_meals_protein_density', 'past_0to4hr_meals_fiber_buffering',
]
WEARABLE_FEATURES = [
    'hr_min_time_sin', 'hr_min_time_cos',
    'past_0to1hr_hr_mean_over_baseline', 'past_0to1hr_hr_variance',
    'past_1to4hr_hr_mean_over_baseline', 'past_1to4hr_hr_variance',
    'past_0to1hr_mets_time_active', 'past_0to1hr_mets_variance',
    'past_1to4hr_mets_time_active', 'past_1to4hr_mets_variance',
]
METABOLIC_FEATURES = [
    'hba1c', 'fglu', 'insulin', 'triglycerides', 'hdl', 'non_hdl',
    'butyrate_production', 'bile_acid_metabolism', 'lps_biosynthesis',
    'inflammatory_activity', 'gut_lining', 'tma_production',
    'protein_fermentation',
]

# Features that are categorical and must be declared as such. A categorical
# feature silently cast to a number would impose an ordering the scale does not
# have; the microbiome scores are the case that matters.
CATEGORICAL_FEATURES = [
    'gender', 'uncertain', 'is_liquid_meal', 'has_caffeine', 'has_alcohol',
    'manual_log_is_liquid_meal', 'manual_log_has_caffeine',
    'manual_log_has_alcohol',
    'gut_lining', 'lps_biosynthesis', 'tma_production', 'butyrate_production',
    'bile_acid_metabolism', 'inflammatory_activity', 'protein_fermentation',
]


def arm_features(arm: str) -> list[str]:
    """The feature list each model arm receives."""
    image = MEAL_CONTEXT_FEATURES + NUTRITION_FEATURES
    log = MEAL_CONTEXT_FEATURES + [f'manual_log_{f}' for f in NUTRITION_FEATURES]
    return {
        'image_logs-demo_nutrition': DEMOGRAPHIC_FEATURES + image,
        'manual_logs-demo_nutrition': DEMOGRAPHIC_FEATURES + log,
        'image_logs-demo_nutrition_wearable_metabolic':
            DEMOGRAPHIC_FEATURES + image + WEARABLE_FEATURES + METABOLIC_FEATURES,
        'baseline-demo': DEMOGRAPHIC_FEATURES,
    }[arm]


# ── Feature taxonomy, for attribution reporting ──────────────────────────
# The five kinds of information a photograph supplies. Attribution is summed
# within these groups, so the question "which kind of information does the
# photograph contribute?" has one answer rather than 24.
IMAGE_SUBGROUPS = {
    'Meal macronutrients': ['mass', 'calories', 'carbs', 'fiber', 'fat',
                            'protein'],
    'Composition ratios': ['energy_density', 'carb_density', 'fat_density',
                           'protein_density', 'fiber_buffering',
                           'fat_buffering', 'protein_buffering'],
    'Preceding meals (0-4 h)': ['past_0to4hr_meals_calories',
                                'past_0to4hr_meals_fat',
                                'past_0to4hr_meals_protein',
                                'past_0to4hr_meals_fiber',
                                'past_0to4hr_meals_energy_density',
                                'past_0to4hr_meals_fat_density',
                                'past_0to4hr_meals_protein_density',
                                'past_0to4hr_meals_fiber_buffering'],
    'Meal timing': ['meal_onset_time_sin', 'meal_onset_time_cos'],
    'Photograph-recognisability flag': ['uncertain'],
}

# Display names for the current-meal nutrition features, which are the ones a
# dietary recommendation can act on.
NUTRITION_DISPLAY = {
    'mass': 'mass', 'calories': 'energy', 'carbs': 'net carbohydrate',
    'fiber': 'fibre', 'fat': 'fat', 'protein': 'protein',
    'energy_density': 'energy density', 'carb_density': 'carbohydrate density',
    'fat_density': 'fat density', 'protein_density': 'protein density',
    'fiber_buffering': 'fibre : carbohydrate ratio',
    'fat_buffering': 'fat : carbohydrate ratio',
    'protein_buffering': 'protein : carbohydrate ratio',
}
NUTRITION_KIND = {f: ('composition ratio' if f in
                      IMAGE_SUBGROUPS['Composition ratios'] else 'macronutrient')
                  for f in NUTRITION_DISPLAY}

# The arm retrained without age, sex and body mass index, used for the
# demographics ablation cost in Fig. 4a.
ARM_NO_DEMOGRAPHICS = 'image_nutrition_no_demographics'
