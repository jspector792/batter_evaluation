"""
suite_diagnostics.py
=====================
Diagnostic plots and fit metrics for the stage-1 regressors.

The headline output is the predicted-vs-actual HEX PLOT, one small-multiple
grid per target:

    rows = model variant            (m1, m2, m3)
    cols = scoring context          (2025 out-of-fold | 2026 carry | 2026 context)

Hexbin rather than a scatter because there are ~330k points per panel and a
scatter would be a solid blob -- density is the whole point. Counts are log
scaled: the offset target is ~78% exact zeros, so on a linear count scale that
one row of cells saturates and everything else reads as empty.

Colour follows the sequential rule -- a single hue, light to dark, for the
density ramp. The two overlays (y = x identity, and the least-squares fit) are
distinguished by LINESTYLE as well as hue, so they are not separated by colour
alone.

Also written
------------
  stage1_fit_metrics.csv   R^2 / RMSE / MAE / Spearman per variant x target x
                           context, plus a whiffs-only cut for the offset
                           target (the population where miss_distance is
                           actually tracked rather than assigned 0)
  residual_hist_*.png      residual distributions, which is where the
                           zero-inflation shows up most starkly
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite_config import (VARIANT_ORDER, TARGETS, TRAIN_SEASON, EVAL_SEASON,
                          OUT_DIR, PLOT_DIR, oof_path, eval_scored_path)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=0.95)

# Repo palette. DENSITY_CMAP is a single-hue light->dark ramp, per the
# sequential-colour rule; never a rainbow map.
BLUE, RED, GRAY = '#2563EB', '#DC2626', '#6B7280'
DENSITY_CMAP = 'Blues'

ACTUAL_COL = {'timing': 'int_y', 'offset': 'miss_distance_t'}
AXIS_LABEL = {'timing': 'contact depth int_y (in)',
              'offset': 'miss distance (in, contact = 0)'}

CONTEXTS = [('oof', f'{TRAIN_SEASON} out-of-fold'),
            ('carry', f'{EVAL_SEASON} carry 2025 RE'),
            ('context', f'{EVAL_SEASON} RE suppressed')]


def load_panels(variant: str) -> dict:
    """(context_key, target) -> DataFrame with 'actual' and 'pred'."""
    panels = {}
    oof = pd.read_parquet(oof_path(variant))
    ev = pd.read_parquet(eval_scored_path(variant))

    for target in TARGETS:
        a = ACTUAL_COL[target]
        if f'predicted_{target}' in oof.columns:
            d = oof[[a, f'predicted_{target}', 'is_contact']].dropna()
            panels[('oof', target)] = d.rename(
                columns={a: 'actual', f'predicted_{target}': 'pred'})
        for mode in ('carry', 'context'):
            col = f'predicted_{target}_{mode}'
            if col in ev.columns:
                d = ev[[a, col, 'is_contact']].dropna()
                panels[(mode, target)] = d.rename(
                    columns={a: 'actual', col: 'pred'})
    return panels


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    ok = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[ok], pred[ok]
    if len(actual) < 10:
        return dict(n=len(actual), r2=np.nan, rmse=np.nan, mae=np.nan,
                    spearman=np.nan, bias=np.nan)
    resid = actual - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return dict(
        n=int(len(actual)),
        r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        mae=float(np.mean(np.abs(resid))),
        spearman=float(stats.spearmanr(actual, pred).statistic),
        bias=float(np.mean(resid)),
    )


def hex_panel(ax, actual, pred, xlim, ylim, gridsize=60):
    hb = ax.hexbin(pred, actual, gridsize=gridsize, cmap=DENSITY_CMAP,
                   norm=LogNorm(vmin=1), linewidths=0,
                   extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    # Identity line: dashed. Fit line: solid. Distinct in linestyle as well as
    # hue, so the two are not told apart by colour alone.
    ax.plot([lo, hi], [lo, hi], ls='--', lw=1.5, color=RED, zorder=3,
            label='y = x')
    ok = np.isfinite(actual) & np.isfinite(pred)
    if ok.sum() > 10:
        slope, intercept = np.polyfit(pred[ok], actual[ok], 1)
        xs = np.array([xlim[0], xlim[1]])
        ax.plot(xs, slope * xs + intercept, ls='-', lw=1.5, color=GRAY,
                zorder=3, label=f'fit (slope {slope:.2f})')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, linewidth=0.5)
    return hb


def plot_target(target: str, all_panels: dict, path: str):
    variants = [v for v in VARIANT_ORDER
                if any((c, target) in all_panels.get(v, {}) for c, _ in CONTEXTS)]
    if not variants:
        print(f'  no panels for {target}, skipping')
        return

    # Shared limits across every panel so the grid is actually comparable.
    vals = np.concatenate([
        np.concatenate([all_panels[v][(c, target)]['actual'].values,
                        all_panels[v][(c, target)]['pred'].values])
        for v in variants for c, _ in CONTEXTS if (c, target) in all_panels[v]])
    lo, hi = np.nanpercentile(vals, [0.2, 99.8])
    pad = 0.04 * (hi - lo)
    lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(len(variants), len(CONTEXTS),
                             figsize=(4.6 * len(CONTEXTS), 4.3 * len(variants)),
                             squeeze=False, constrained_layout=True)
    hb = None
    for ri, v in enumerate(variants):
        for ci, (ckey, clabel) in enumerate(CONTEXTS):
            ax = axes[ri][ci]
            key = (ckey, target)
            if key not in all_panels[v]:
                ax.set_visible(False)
                continue
            d = all_panels[v][key]
            hb = hex_panel(ax, d['actual'].values, d['pred'].values, lim, lim)
            m = metrics(d['actual'].values, d['pred'].values)
            ax.set_title(f'{v}  |  {clabel}', fontsize=10, pad=6)
            ax.text(0.03, 0.97,
                    f"$R^2$={m['r2']:.3f}\nRMSE={m['rmse']:.2f}\n"
                    f"MAE={m['mae']:.2f}\nn={m['n']:,}",
                    transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.35', fc='white',
                              ec=GRAY, alpha=0.85))
            if ri == len(variants) - 1:
                ax.set_xlabel(f'predicted {AXIS_LABEL[target]}')
            if ci == 0:
                ax.set_ylabel(f'actual {AXIS_LABEL[target]}')
            if ri == 0 and ci == 0:
                ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    if hb is not None:
        cb = fig.colorbar(hb, ax=axes, location='right', shrink=0.6, pad=0.01)
        cb.set_label('swings per hex (log scale)', fontsize=9)
    fig.suptitle(f'Predicted vs actual — {target} '
                 f'({AXIS_LABEL[target]})', fontsize=13)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def plot_residual_hists(all_panels: dict, target: str, path: str):
    variants = [v for v in VARIANT_ORDER if v in all_panels]
    fig, axes = plt.subplots(len(variants), len(CONTEXTS),
                             figsize=(4.2 * len(CONTEXTS), 2.9 * len(variants)),
                             squeeze=False, constrained_layout=True)
    for ri, v in enumerate(variants):
        for ci, (ckey, clabel) in enumerate(CONTEXTS):
            ax = axes[ri][ci]
            key = (ckey, target)
            if key not in all_panels[v]:
                ax.set_visible(False)
                continue
            d = all_panels[v][key]
            resid = d['actual'].values - d['pred'].values
            ax.hist(resid, bins=80, color=BLUE, alpha=0.85)
            ax.axvline(0, color=RED, ls='--', lw=1.2)
            ax.set_title(f'{v} | {clabel}', fontsize=9)
            ax.set_yscale('log')
            ax.grid(alpha=0.25, linewidth=0.5)
            if ri == len(variants) - 1:
                ax.set_xlabel(f'residual (actual - predicted), {target}')
    fig.suptitle(f'Residual distribution — {target} (log count axis)',
                 fontsize=12)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def main():
    ap = argparse.ArgumentParser(description='Stage-1 diagnostics + hex plots')
    ap.add_argument('--variants', nargs='*', default=VARIANT_ORDER)
    args = ap.parse_args()

    all_panels, rows = {}, []
    for v in args.variants:
        if not os.path.exists(oof_path(v)):
            print(f'SKIP {v}: stage-1 output missing')
            continue
        all_panels[v] = load_panels(v)
        for (ckey, target), d in all_panels[v].items():
            m = metrics(d['actual'].values, d['pred'].values)
            rows.append(dict(variant=v, target=target, context=ckey,
                             population='all_swings', **m))
            if target == 'offset':
                # The zero mass is assigned, not measured. Whiffs are the
                # population where miss_distance is genuinely tracked, so the
                # fit there is the one that speaks to measurement rather than
                # to how well the model recovers the contact/whiff split.
                w = d[~d['is_contact']]
                mw = metrics(w['actual'].values, w['pred'].values)
                rows.append(dict(variant=v, target=target, context=ckey,
                                 population='whiffs_only', **mw))

    if not all_panels:
        raise SystemExit('No stage-1 outputs found -- run suite_stage1.py first.')

    for target in TARGETS:
        plot_target(target, all_panels,
                    os.path.join(PLOT_DIR, f'hex_pred_vs_actual_{target}.png'))
        plot_residual_hists(all_panels, target,
                            os.path.join(PLOT_DIR, f'residual_hist_{target}.png'))

    met = pd.DataFrame(rows).sort_values(
        ['target', 'population', 'context', 'variant'])
    out_csv = os.path.join(OUT_DIR, 'stage1_fit_metrics.csv')
    met.to_csv(out_csv, index=False)
    print(f'-> {out_csv}')
    print()
    print(met.to_string(index=False,
                        float_format=lambda x: f'{x:,.4f}'))


if __name__ == '__main__':
    main()
