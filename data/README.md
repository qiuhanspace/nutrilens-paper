# Data

Only preprocessed data are distributed here. The raw sources are public and are
cited below; nothing in this directory permits re-identification of a
participant.

## `stage1/` — image to nutrition

| Path | Contents |
|---|---|
| `cleaned_v4/{val,test}_{n5k,january}/metadata.csv` | the cleaned evaluation cohort: dish identifier, source, and reference mass, energy, carbohydrate, net carbohydrate, fibre, fat and protein. 3,028 Nutrition5k dishes and 869 January dishes, 3,897 in total. |
| `parsed/{method}_{val,test}_per_dish.csv` | per-dish estimates from NutriLens and each comparator, already parsed from the model responses into the same schema as the reference. Files with `_gemini` in the name are the backbone-controlled re-run, which is not the comparison reported in the paper. |
| `human/annotator{1,2}_per_dish_totals.csv` | the two certified nutritionists' per-dish totals for the 80-dish panel. |
| `human/n5k40_name_mapping.csv` | maps the 40 Nutrition5k panel dishes to their identifiers. |
| `train_metadata.csv` | the 100 dishes used to optimise the prompt. 95 of them fall inside the evaluation pool, which is why Supplementary Table 11 exists. |

Sources. Nutrition5k: Thames Q et al., *Proc. IEEE/CVF CVPR* 2021. January Food
Benchmark: Hosseinian A et al., arXiv:2508.09966, 2025. The 40 everyday-meal
photographs of the human panel come from the GLOW UP study and are not
redistributed here; only the derived per-dish totals are.

## `stage2/` — nutrition to glucose trajectory

| Path | Contents |
|---|---|
| `meal_events_features.csv` | the preprocessed meal-event table: one row per 2-hour meal event, with participant identifier, metabolic group, every model feature, and the nine measured glucose anchor points. 1,205 events from 45 participants. |
| `predictions/{arm}-{group}/all_folds.npz` | held-out predictions from the leave-one-subject-out loop, one file per model arm and metabolic group. Arrays: `y_true`, `y_pred_raw`, `y_pred_adjusted`, `y_pred_naive_group`, `y_pred_naive_individual`, `subject_ids`. Only the 867 held-out events appear; calibration-window events are excluded by construction. |
| `models/image_logs-demo_nutrition-{group}/` | the saved per-fold boosters for the NutriLens arm, plus `metadata.json` giving the fold order. Needed only by script 06, which has to re-predict counterfactuals with the model that produced the original prediction. |
| `shap/{group}/` | TreeSHAP values from the separate single-output training run used for attribution. |

Source. CGMacros: Das A et al., *Sci Data* 2025;12:1557, available via
PhysioNet. The raw continuous-glucose traces, wearable streams and meal
photographs are not redistributed here; obtain them from PhysioNet under its
terms.

## What is deliberately absent

- Raw continuous-glucose, heart-rate and metabolic-equivalent time series.
- Meal photographs from any cohort.
- Individual laboratory results and gut-microbiome reports beyond the derived
  features already in the meal-event table.

The preprocessing that turns the raw sources into `meal_events_features.csv` is
described in the Methods and summarised in Supplementary Fig. 2; it is not
rerunnable from this repository, because it requires the raw data.
