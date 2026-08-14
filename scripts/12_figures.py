"""Every data panel of the main and supplementary figures.

The manuscript figures are composed canvases: these panels plus hand-drawn
schematics, assembled in a vector editor. This script regenerates every panel
that is computed from data, using the tables written by scripts 01 to 11, so
that a reader can check that the numbers in the figures are the numbers in the
tables.

Main figures
    fig2a_nutrition_estimation   error against four published methods
    fig2b_nutritionist_agreement model and expert agreement, read against the
                                 agreement between the two experts
    fig3a_cross_model            difference and equivalence, three contrasts
    fig3b_incremental_value      per-participant advantage over the baseline
    fig3c_clarke_error_grid      predicted against measured, by zone
    fig4a_attribution_ablation   what the model uses against what it needs
    fig4b_image_subgroups        which kind of information the photograph gives
    fig4c_nutrition_ranking      individual nutrition features, ranked
    fig4d_alert_and_next_step    the alert (ROC) and the ranked next step

Supplementary figures
    suppfig3_sensitivity         four discretionary choices, swept
    suppfig4_per_participant     one panel per participant
    suppfig5_information_ceilings  the two ceilings

Figs 1 and 5, and Supplementary Figs 1 and 2, are schematics and flow charts
rather than data panels, and are drawn by hand.

Author: Qiuhan Jin <qiuhan.jin@unisg.ch>
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nutrilens.config import (ARMS, COMPARATORS, DECISION_RULES, FIGURES,
                              GROUP_LABEL, GROUP_SHORT, IMAGE_SUBGROUPS,
                              MARGIN_MGDL, REFERENCE, TABLES)
from nutrilens.curves import (arm_rmse, clarke_zone, load_predictions,
                              reconstruct, rule_breached, rule_value)

# Okabe-Ito, so the figures survive greyscale and the common colour-vision
# deficiencies. Group colours ascend with metabolic severity throughout.
GROUP_COLOUR = {'Normoglycemic': '#0072B2', 'Prediabetic': '#E69F00',
                'Type 2 Diabetic': '#D55E00'}
FAMILY_COLOUR = {'Replace': '#0072B2', 'Substitute': '#009E73',
                 'Add': '#56B4E9', 'Reduce': '#CC79A7'}
STRATEGY_ORDER = ['none', 'training-group median curve',
                  'own calibration-window median curve',
                  'per-participant offset (used throughout)']
RECON_ORDER = ['linear', 'pchip', 'cubic', 'akima']

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})


def save(fig, stem: str) -> None:
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES / f'{stem}.{ext}')
    plt.close(fig)
    print(f'  {stem}.pdf / .png')


def fig2a() -> None:
    d = pd.read_csv(TABLES / 'supptable1_nutrition_by_dataset.csv')
    d = d[d.dataset == 'Combined']
    quantities = ['mass (g)', 'calories (kcal)', 'total carbs (g)',
                  'fat (g)', 'protein (g)']
    fig, axes = plt.subplots(1, 5, figsize=(9.5, 2.4))
    for ax, q in zip(axes, quantities):
        sub = d[d.nutrient == q].sort_values('MAE')
        colours = ['#D55E00' if m == 'NutriLens' else '0.55' for m in sub.method]
        ax.barh(range(len(sub)), sub.MAE, color=colours, height=0.68)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.method if q == quantities[0] else [])
        ax.invert_yaxis()
        ax.set_title(q)
        ax.set_xlabel('mean absolute error')
    fig.suptitle('Image-based nutrition estimation, 3,897 pooled dishes, '
                 'each method on its own published backbone', y=1.06)
    save(fig, 'fig2a_nutrition_estimation')


def fig3a() -> None:
    d = pd.read_csv(TABLES / 'supptable6_cross_model_tests.csv')
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axvspan(-MARGIN_MGDL, MARGIN_MGDL, color='0.90', zorder=0,
               label=f'equivalence margin (± {MARGIN_MGDL:g} mg/dL)')
    ax.axvline(0, color='0.35', lw=0.8, zorder=1)

    labels, y = [], 0
    for group in GROUP_LABEL.values():
        for comparator in COMPARATORS:
            r = d[(d.group == group) & (d.comparator == ARMS[comparator])].iloc[0]
            ax.plot([r.ci90_lo, r.ci90_hi], [y, y], color=GROUP_COLOUR[group], lw=1.6)
            ax.plot(r.mean_diff_mgdl, y, 'o', ms=6, color=GROUP_COLOUR[group],
                    mfc=GROUP_COLOUR[group] if r.equivalent_tost90 else 'white',
                    mew=1.4, zorder=3)
            if r.p_bonferroni3 < 0.05:
                ax.text(r.ci90_lo - 0.35, y, '*', va='center', ha='right', fontsize=11)
            labels.append(f'{ARMS[comparator].split(" (")[0]}')
            y += 1
        labels.append('')
        y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Δ per-participant RMSE, NutriLens − comparator (mg/dL)\n'
                  '← favours NutriLens')
    handles = [plt.Line2D([], [], marker='o', ls='', color='0.3', mfc='0.3',
                          label='equivalent (90 % CI inside the margin)'),
               plt.Line2D([], [], marker='o', ls='', color='0.3', mfc='white',
                          label='not equivalent'),
               plt.Line2D([], [], marker='$*$', ls='', color='k',
                          label='different (Wilcoxon, Bonferroni-3)')]
    ax.legend(handles=handles, loc='upper center', frameon=False,
              bbox_to_anchor=(0.5, -0.28), ncol=3)
    for i, group in enumerate(GROUP_LABEL.values()):
        ax.text(ax.get_xlim()[1], i * 4, group, ha='right', va='center',
                color=GROUP_COLOUR[group], fontweight='bold')
    save(fig, 'fig3a_cross_model')


def fig3b() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8), sharey=True)
    for ax, (group, label) in zip(axes, GROUP_LABEL.items()):
        a = arm_rmse(REFERENCE, group)
        b = arm_rmse('baseline-demo', group)
        idx = a.index.intersection(b.index)
        advantage = (b.loc[idx] - a.loc[idx]).to_numpy(float)
        x = np.random.default_rng(0).normal(0, 0.05, len(advantage))
        ax.axhline(0, color='0.35', lw=0.8)
        ax.plot(x, advantage, 'o', ms=4, alpha=0.75, color=GROUP_COLOUR[label])
        mean = advantage.mean()
        se = advantage.std(ddof=1) / np.sqrt(len(advantage))
        ax.plot([-0.22, 0.22], [mean, mean], color='k', lw=1.8)
        ax.plot([0, 0], [mean - 1.96 * se, mean + 1.96 * se], color='k', lw=1.2)
        ax.set_title(f'{label}\nn = {len(advantage)}')
        ax.set_xticks([])
        ax.set_xlim(-0.45, 0.45)
    axes[0].set_ylabel('advantage over Baseline (Demographics)\n(mg/dL, positive favours NutriLens)')
    fig.suptitle('Incremental value of the photograph, per participant', y=1.04)
    save(fig, 'fig3b_incremental_value')


def fig3c() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharex=True, sharey=True)
    for ax, (group, label) in zip(axes, GROUP_LABEL.items()):
        z = load_predictions(REFERENCE, group)
        measured = reconstruct(z['y_true']).ravel()
        predicted = reconstruct(z['y_pred_adjusted']).ravel()
        keep = np.random.default_rng(0).choice(
            len(measured), size=min(12_000, len(measured)), replace=False)
        zones = np.array([clarke_zone(a, b)
                          for a, b in zip(measured[keep], predicted[keep])])
        ax.plot([0, 400], [0, 400], color='0.5', lw=0.7, zorder=1)
        for zone, colour in [('A', GROUP_COLOUR[label]), ('B', '0.45'),
                             ('C', '#E69F00'), ('D', '#CC0000'), ('E', 'k')]:
            m = zones == zone
            if m.any():
                ax.plot(measured[keep][m], predicted[keep][m], '.', ms=1.2,
                        alpha=0.35, color=colour)
        pct = pd.read_csv(TABLES / 'fig3c_clarke_error_grid.csv')
        row = pct[pct.group == label].iloc[0]
        ax.set_title(f'{label}\nA+B {row.zone_AB_pct:.1f} %   D {row.zone_D_pct:.1f} %')
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_xlabel('measured glucose (mg/dL)')
    axes[0].set_ylabel('predicted glucose (mg/dL)')
    save(fig, 'fig3c_clarke_error_grid')


def fig4d() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4),
                             gridspec_kw=dict(width_ratios=[1, 1.35]))

    # Left: the alert, pooled across groups.
    ax = axes[0]
    measured, predicted = [], []
    for group in GROUP_SHORT:
        z = load_predictions(REFERENCE, group)
        measured.append(reconstruct(z['y_true']))
        predicted.append(reconstruct(z['y_pred_adjusted']))
    measured, predicted = np.vstack(measured), np.vstack(predicted)

    st15 = pd.read_csv(TABLES / 'supptable15_alert_discrimination.csv')
    ax.plot([0, 1], [0, 1], color='0.6', lw=0.8, ls='--')
    for (name, spec), colour in zip(DECISION_RULES.items(),
                                    ['#D55E00', '#0072B2', '#009E73', '#CC79A7']):
        y = rule_breached(measured, spec).astype(int)
        score = rule_value(predicted, spec['window'], spec['stat'])
        fpr, tpr, _ = roc_curve(y, score)
        auroc = st15[(st15.rule == name) & (st15.group == 'All')].AUROC.iloc[0]
        short = name.split(':')[0]
        ax.plot(fpr, tpr, color=colour, lw=1.5, label=f'{short}  {auroc:.3f}')
    ax.set_xlabel('false-positive rate')
    ax.set_ylabel('true-positive rate')
    ax.set_title('The alert: flagging a breach from the photograph,\nbefore the meal')
    ax.legend(title='AUROC', loc='lower right', frameon=False)

    # Right: the ranked next step. Labels are kept short and availability is
    # annotated on the bar, because an action that helps greatly but applies
    # rarely is not the same proposition as one that helps modestly and always.
    ax = axes[1]
    short = {'X1': 'a different meal, similar time',
             'S2': '1 carb serving → protein',
             'A1': '+1 serving non-starchy veg',
             'S1': '1 grain serving → whole grain',
             'R3': 'carbohydrate → 45 g allowance',
             'R2': 'starch portion −20 %',
             'R4': 'cut 1 tsp added fat'}
    st17 = pd.read_csv(TABLES / 'supptable17a_action_effects.csv')
    st17 = st17.sort_values('pct_gap_closed_peak')
    y = np.arange(len(st17))
    colours = [FAMILY_COLOUR[f] for f in st17.family]
    ax.barh(y + 0.19, st17.pct_gap_closed_peak, height=0.36, color=colours)
    ax.barh(y - 0.19, st17.pct_gap_closed_area, height=0.36, color=colours,
            alpha=0.5)
    for yi, (peak, area, avail) in enumerate(zip(st17.pct_gap_closed_peak,
                                                 st17.pct_gap_closed_area,
                                                 st17.available_pct)):
        ax.text(max(peak, area) + 0.3, yi, f'{avail:.0f} %', va='center',
                fontsize=6, color='0.35')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{c}  {short[c]}' for c in st17.code])
    ax.set_xlim(0, max(st17.pct_gap_closed_area) * 1.22)
    ax.set_xlabel('median % of the gap to safety closed\n'
                  '(grey figure = share of flagged meals the action applies to)')
    ax.set_title('The next step: seven guideline-anchored actions,\nranked')
    handles = [mpl.patches.Patch(color='0.35', label='peak glucose'),
               mpl.patches.Patch(color='0.35', alpha=0.5,
                                 label='area above 180 mg/dL')]
    handles += [mpl.patches.Patch(color=c, label=f)
                for f, c in FAMILY_COLOUR.items()]
    ax.legend(handles=handles, loc='lower right', frameon=False, ncol=2)
    fig.subplots_adjust(wspace=0.55)
    save(fig, 'fig4d_alert_and_next_step')




def fig2b() -> None:
    """Three rater pairs by six quantities. The bottom row is the reference."""
    d = pd.read_csv(TABLES / 'supptable2_nutritionist_agreement.csv')
    d = d[d.panel == 'Combined']
    order = ['NutriLens – Annotator 1', 'NutriLens – Annotator 2',
             'Annotator 1 – Annotator 2']
    grid = d.pivot_table(index='rater_pair', columns='nutrient',
                         values='lins_ccc').reindex(order)
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    im = ax.imshow(grid.to_numpy(float), cmap='viridis', vmin=0.4, vmax=0.85,
                   aspect='auto')
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=30, ha='right')
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.to_numpy(float)[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if v < 0.68 else 'black')
    # The expert-expert row is the reference the model rows are read against.
    ax.axhline(1.5, color='k', lw=1.4)
    ax.set_title("Lin's concordance, 80-dish panel\n"
                 'bottom row bounds what expert agreement looks like')
    fig.colorbar(im, ax=ax, shrink=0.85, label="Lin's CCC")
    save(fig, 'fig2b_nutritionist_agreement')


def fig4a() -> None:
    """Attribution share against ablation cost, back to back."""
    d = pd.read_csv(TABLES / 'fig4a_attribution_and_ablation.csv')
    arms = ['image-derived features', 'demographic features',
            'calibration offset']
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6), sharey=True)
    for ax, (group, label) in zip(axes, GROUP_SHORT.items()):
        sub = d[d.group == label].set_index('arm').loc[arms]
        y = np.arange(len(arms))
        ax.barh(y + 0.19, sub.share_pct, height=0.36, color='0.55')
        ax2 = ax.twiny()
        ax2.barh(y - 0.19, sub.ablation_cost_mgdl, height=0.36,
                 color=GROUP_COLOUR[GROUP_LABEL[group]])
        ax2.axvline(0, color='0.3', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([a.replace(' features', '').replace(' offset', '')
                            for a in arms])
        ax.invert_yaxis()
        ax.set_xlabel('share of attribution (%)', color='0.35')
        ax2.set_xlabel('cost of removing it (mg/dL)',
                       color=GROUP_COLOUR[GROUP_LABEL[group]])
        ax2.text(0.5, 1.28, GROUP_LABEL[group], transform=ax.transAxes,
                 ha='center', fontweight='bold',
                 color=GROUP_COLOUR[GROUP_LABEL[group]])
    fig.suptitle('What the model uses (grey) against what it needs (colour).  '
                 'A negative cost means the arm is worth removing.', y=1.30)
    fig.subplots_adjust(wspace=0.3)
    save(fig, 'fig4a_attribution_ablation')


def fig4b() -> None:
    """Which kind of information the photograph supplies."""
    d = pd.read_csv(TABLES / 'fig4b_image_subgroups.csv')
    order = list(IMAGE_SUBGROUPS)
    grid = d.pivot_table(index='group', columns='subgroup',
                         values='share_of_image_arm_pct')
    grid = grid.reindex(index=['Normo', 'Pre', 'T2D'], columns=order)
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    left = np.zeros(len(grid))
    shades = ['#0072B2', '#56B4E9', '#009E73', '#E69F00', '#999999']
    for subgroup, colour in zip(order, shades):
        ax.barh(range(len(grid)), grid[subgroup], left=left, height=0.6,
                color=colour, label=subgroup)
        for i, (v, l) in enumerate(zip(grid[subgroup], left)):
            if v > 6:
                ax.text(l + v / 2, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=7, color='white')
        left += grid[subgroup].to_numpy()
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels([GROUP_LABEL[g] for g in GROUP_SHORT])
    ax.invert_yaxis()
    ax.set_xlabel('share of the image-derived arm (%)')
    ax.set_xlim(0, 100)
    ax.set_title('Inside the image arm: meal timing grows with severity')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3,
              frameon=False)
    save(fig, 'fig4b_image_subgroups')


def fig4c() -> None:
    """Individual nutrition features, ranked by mean rank across groups."""
    d = pd.read_csv(TABLES / 'fig4c_nutrition_feature_ranking.csv')
    d = d.sort_values('mean_rank', ascending=False)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    y = np.arange(len(d))
    for short, colour in zip(GROUP_SHORT.values(),
                             [GROUP_COLOUR[g] for g in GROUP_LABEL.values()]):
        ax.plot(d[f'rank_{short}'], y, 'o', ms=5, color=colour, alpha=0.85,
                label=short)
    ax.plot(d.mean_rank, y, '|', ms=12, color='k', label='mean rank')
    ax.set_yticks(y)
    ax.set_yticklabels(d.display)
    ax.set_xlabel('rank within group (1 = most important)')
    ax.set_title('Carbohydrate quality outranks carbohydrate quantity')
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'fig4c_nutrition_ranking')


def suppfig3() -> None:
    """The four discretionary choices, one row each."""
    fig, axes = plt.subplots(4, 3, figsize=(8.6, 10.0))
    day = pd.read_csv(TABLES / 'suppfig3a_day_drift.csv')
    win = pd.read_csv(TABLES / 'suppfig3b_calibration_window.csv')
    strat = pd.read_csv(TABLES / 'suppfig3c_calibration_strategy.csv')
    recon = pd.read_csv(TABLES / 'suppfig3d_reconstruction.csv')

    for col, (group, label) in enumerate(GROUP_LABEL.items()):
        short = GROUP_SHORT[group]
        colour = GROUP_COLOUR[label]

        ax = axes[0, col]
        g = day[day.group == short]
        ax.plot(g.day, g.median_rmse, '-o', ms=3, color=colour)
        ax.fill_between(g.day, g.q25, g.q75, color=colour, alpha=0.2)
        ax.set_title(label)
        ax.set_xlabel('day since first test meal')

        ax = axes[1, col]
        g = win[win.group == short]
        ax.plot(g.calibration_fraction, g.mean_rmse, '-o', ms=3, color=colour)
        ax.axvline(0.30, color='0.4', ls='--', lw=0.8)
        ax.set_xlabel('calibration-window fraction')

        ax = axes[2, col]
        g = strat[strat.group == label]
        for model, gg in g.groupby('model'):
            gg = gg.set_index('strategy').loc[list(STRATEGY_ORDER)]
            ax.plot(range(len(gg)), gg.rmse_mean, '-o', ms=3,
                    label=model.split(' (')[0])
        ax.set_xticks(range(len(STRATEGY_ORDER)))
        ax.set_xticklabels(['none', 'group\nmedian', 'own\nmedian',
                            'personal\noffset'], fontsize=6)

        ax = axes[3, col]
        g = recon[recon.group == label]
        for model, gg in g.groupby('model'):
            gg = gg.set_index('reconstruction').loc[RECON_ORDER]
            ax.plot(range(len(gg)), gg.rmse_mean, '-o', ms=3,
                    label=model.split(' (')[0])
        ax.set_xticks(range(len(RECON_ORDER)))
        ax.set_xticklabels(RECON_ORDER, fontsize=6)

    for row, name in enumerate(['a  day of prediction', 'b  calibration window',
                                'c  calibration strategy',
                                'd  reconstruction scheme']):
        axes[row, 0].set_ylabel(f'{name}\nRMSE (mg/dL)')
    axes[3, 2].legend(frameon=False, fontsize=6, loc='center left',
                      bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save(fig, 'suppfig3_sensitivity')


def suppfig4() -> None:
    """One small panel per participant, blocked by group."""
    d = pd.read_csv(TABLES / 'suppfig4_per_participant_curves.csv')
    subjects = d[['group', 'subject_id']].drop_duplicates()
    order = sorted(subjects.itertuples(index=False),
                   key=lambda r: (list(GROUP_LABEL.values()).index(r.group),
                                  r.subject_id))
    ncol = 8
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 1.5 * nrow),
                             sharex=True, sharey=True)
    for ax, (group, subject) in zip(axes.ravel(), order):
        g = d[(d.group == group) & (d.subject_id == subject)]
        colour = GROUP_COLOUR[group]
        ax.plot(g.minute, g.measured_mean, color='0.25', lw=1.2)
        ax.fill_between(g.minute, g.measured_mean - g.measured_ci,
                        g.measured_mean + g.measured_ci, color='0.25', alpha=0.18)
        ax.plot(g.minute, g.predicted_mean, color=colour, lw=1.2)
        ax.fill_between(g.minute, g.predicted_mean - g.predicted_ci,
                        g.predicted_mean + g.predicted_ci, color=colour, alpha=0.22)
        ax.set_title(f'{subject}  (n = {int(g.n_meals.iloc[0])})', fontsize=6)
    for ax in axes.ravel()[len(order):]:
        ax.axis('off')
    fig.suptitle('Measured (grey) against predicted (colour) mean trajectory, '
                 'per participant', y=1.0)
    fig.supxlabel('minutes from meal onset')
    fig.supylabel('glucose (mg/dL)')
    fig.tight_layout()
    save(fig, 'suppfig4_per_participant')


def suppfig5() -> None:
    """The two information ceilings."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0),
                             gridspec_kw=dict(width_ratios=[1, 1.3]))

    ax = axes[0]
    d = pd.read_csv(TABLES / 'suppfig5a_iauc_ceiling.csv')
    y = np.arange(len(d))
    for i, r in d.iterrows():
        ax.plot([r.ceiling_ci_lo, r.ceiling_ci_hi], [i, i], color='0.5', lw=1.4)
        ax.plot(r.ceiling_r, i, 'o', ms=7, mfc='white', mec='0.3', mew=1.4)
        ax.plot(r.model_r, i, 'o', ms=7, color=GROUP_COLOUR[r.group])
    ax.set_yticks(y)
    ax.set_yticklabels(d.group)
    ax.invert_yaxis()
    ax.set_xlabel('Pearson r for incremental area')
    ax.set_title('a  the model (filled) against the in-cohort\n'
                 'reproducibility proxy (open)')

    ax = axes[1]
    d = pd.read_csv(TABLES / 'suppfig5b_shape_modes.csv')
    modes = ['peak amplitude', 'peak timing', 'tail shape']
    width = 0.26
    for i, (group, label) in enumerate(GROUP_LABEL.items()):
        sub = (d[(d.group == label) & (d.model == ARMS[REFERENCE])]
               .set_index('mode').loc[modes])
        x = np.arange(len(modes)) + (i - 1) * width
        ax.bar(x, sub.variance_share, width * 0.45, color='0.75')
        ax.bar(x + width * 0.45, sub.recoverable_r2, width * 0.45,
               color=GROUP_COLOUR[label])
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylabel('share of within-participant variance (grey)\n'
                  'and fraction recoverable (colour)')
    ax.set_title('b  only amplitude is recoverable')
    fig.tight_layout()
    save(fig, 'suppfig5_information_ceilings')


def main() -> None:
    print(f'writing panels to {FIGURES}')
    for panel in (fig2a, fig2b, fig3a, fig3b, fig3c, fig4a, fig4b, fig4c,
                  fig4d, suppfig3, suppfig4, suppfig5):
        panel()


if __name__ == '__main__':
    main()
