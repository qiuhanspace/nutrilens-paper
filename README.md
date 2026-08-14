# Image-based digital biomarkers predict postprandial glucose responses across glycaemic states

Analysis code and derived data for the paper of the same name, submitted to
*npj Digital Medicine* (collection: Digital Biomarkers for Enabling Proactive
Clinical Decisions).

Qiuhan Jin, Victoria Brügger, Magdalena Fuchs, Benjamin Wirth, Stefan Bilz,
Michael Brändle, Elgar Fleisch, Tobias Kowatsch\*, Mia Jovanova\*
*(\*shared senior authors)*

**Correspondence:** Qiuhan Jin — <qiuhan.jin@unisg.ch>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

## Summary

NutriLens estimates meal composition from a single smartphone photograph with a
vision-language model, and predicts the individual's 2-hour postprandial glucose
trajectory from that estimate plus age, sex and body mass index. The prediction
is anchored to the person once, by an approximately two-day continuous glucose
monitoring onboarding; after that it uses no sensor, wearable or food-log input.
Every clinical readout — peak, 2-hour value, area, time in range, and the
decision rules built on them — is derived from the one predicted trajectory.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_all.py          # about 30 seconds
```

Tables land in `outputs/tables/`, figure panels in `outputs/figures/`.

## What runs, and what it produces

| Script | Produces |
|---|---|
| `00_train_glucose_models.py` | the per-fold predictions. **Optional** — they ship in `data/`, and this takes hours |
| `01_table1_and_cross_model_tests.py` | **Table 1**, Supplementary Tables 6 and 8 |
| `02_severity_gradient.py` | Supplementary Table 12 |
| `03_readouts_and_error_grid.py` | Supplementary Table 9, and the Clarke zone percentages of Fig. 3c |
| `04_ingredient_ladder.py` | Supplementary Table 13 |
| `05_decision_rules_and_alerts.py` | Supplementary Tables 15, 16 and 18 |
| `06_recommendation_actions.py` | Supplementary Table 17 |
| `07_nutrition_benchmark.py` | Supplementary Tables 1 and 11, and the values behind Fig. 2a |
| `09_nutritionist_agreement.py` | Supplementary Table 2, and the values behind Fig. 2b |
| `10_feature_attribution.py` | the values behind Fig. 4a, 4b and 4c |
| `11_sensitivity_and_ceilings.py` | the values behind Supplementary Figs 3, 4 and 5 |
| `12_figures.py` | every data panel, main and supplementary |

The manuscript figures are composed canvases: these panels plus hand-drawn
schematics, assembled in a vector editor. Panels regenerated here reproduce the
numbers, not the final typesetting. Figs 1 and 5 and Supplementary Figs 1 and 2
are schematics and flow charts with no data panel, so nothing here draws them.

Two panels are worth reading with their construction in mind. The
calibration-window sweep (Supplementary Fig. 3b) re-cuts the window and refits
the offset at each fraction, so it covers the NutriLens arm only, and its 30 %
column equals Table 1 by construction — that equality is the panel's internal
check. The reproducibility ceiling (Supplementary Fig. 5a) pairs each meal with
the most nutritionally similar other meal the same participant ate, so it is
computed on all 1,205 meal events from the dense measured trace and never
involves the model.

## The library

| Module | Responsibility |
|---|---|
| `nutrilens/config.py` | paths, cohort definitions, and every locked analysis constant — thresholds, margins, hyperparameters, seeds |
| `nutrilens/curves.py` | trajectory reconstruction, per-participant error, clinical readouts, error grid |
| `nutrilens/stats.py` | paired difference and equivalence testing, effect size, bootstrap, trend test |
| `nutrilens/training.py` | leave-one-subject-out training and per-participant calibration |
| `nutrilens/recourse.py` | counterfactual meal modifications |

Constants live in `config.py` and nowhere else, so a recipe cannot drift between
scripts.

## The recipe, in short

**Training.** Gradient-boosted trees with a multi-output objective predict all
nine anchor points jointly. Models are trained leave-one-subject-out *within* a
metabolic group, so training and test participants always share a glycaemic
state. The number of boosting rounds is chosen inside each training fold by
grouped five-fold cross-validation with early stopping; nothing is selected
using the held-out participant. Hyperparameters are fixed a priori and held
constant across arms, so differences between arms reflect their inputs rather
than their tuning.

**Personalisation.** For each held-out participant, the first 30 % of their
records form a calibration window and the remaining 70 % the test set. The
per-timepoint median residual over that window, clipped to ±30 mg/dL, becomes
the participant's offset. Every accuracy figure is computed on the 867 held-out
events.

**Primary endpoint.** Per-participant root-mean-squared error of the
reconstructed trajectory. Aggregation order is deliberate: error is computed per
timepoint across a participant's meals, then averaged across the 121 timepoints.
Both the measured and the predicted trajectory are reconstructed from their nine
anchors, so the metric compares two curves of the same construction.

**Inference.** Three contrasts are pre-specified per group, each comparator
against NutriLens, with Bonferroni correction within group. Every contrast is
reported as a difference test *and* an equivalence test against a ±3 mg/dL
margin, with the minimum detectable effect beside it, because a non-significant
difference is not evidence of sameness and a significant difference smaller than
the margin is not a meaningful one. One further test, pre-specified and
uncorrected, checks whether the advantage grows across the three groups.

## Data

Preprocessed data only; see `data/README.md` for the contents and provenance of
every file, and for what is deliberately absent. The raw continuous-glucose
traces and meal photographs are not redistributed — obtain CGMacros from
PhysioNet and Nutrition5k and the January Food Benchmark from their releases.

## Reproducibility notes

- Seeds are fixed throughout: 42 for modelling and bootstrap resampling, 20260813
  for the permutation null of the trend test.
- The nutrition front end queries a proprietary vision-language model through a
  public interface, which can change without notice. The per-dish estimates it
  produced are therefore distributed as parsed CSVs rather than regenerated, and
  the model identifier, the winning prompt and the decoding parameters are
  recorded in the Methods.
- One meal has a 2-hour value of exactly 140 mg/dL, which is one of the decision
  thresholds. Interpolation reproduces an anchor to about 1e-14, enough to flip a
  bare comparison, so rule tests use a tolerance. See `curves.rule_breached`.
- Percentages of the gap to safety are computed from glucose values rounded to
  0.1 mg/dL, an order of magnitude below the sensor's own resolution. See
  `pct_gap_closed` in script 06.

## Data availability

Only **derived** data are distributed here; `data/README.md` documents every
file and its provenance. Raw continuous-glucose traces, wearable streams and
meal photographs are **not** redistributed.

| Source | Where to obtain it |
|---|---|
| CGMacros (45 adults, CGM + meals) | PhysioNet, under its data use agreement — Das et al., *Sci Data* 2025;12:1557 |
| Nutrition5k | Thames et al., *Proc. IEEE/CVF CVPR* 2021 |
| January Food Benchmark | Hosseinian et al., arXiv:2508.09966, 2025 |
| GLOW UP meal photographs | Not public; de-identified per-dish expert totals only are included, under BASEC 2025-00972 |

## Repository structure

```
nutrilens/     the library: constants, curves, statistics, training, recourse
scripts/       numbered analysis scripts, plus run_all.py
data/          derived inputs (see data/README.md)
outputs/       tables and figure panels, regenerated by run_all.py
```

## License

Code: MIT (see `LICENSE`). Derived data remain subject to the terms of their
sources, which `LICENSE` lists.

## Citation

If you use this code or the derived data, please cite the paper. Machine-readable
metadata is in `CITATION.cff`.

> Jin Q, Brügger V, Fuchs M, Wirth B, Bilz S, Brändle M, Fleisch E, Kowatsch T,
> Jovanova M. Image-based digital biomarkers predict postprandial glucose
> responses across glycaemic states. *npj Digital Medicine* (2026).

## Provenance

This repository is the **released subset** of the study's analysis. The complete
working tree — including sensitivity analyses, superseded variants and the
manuscript build — is maintained separately; the code here is the same lineage,
reduced to what is needed to reproduce the published figures and tables.
