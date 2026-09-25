"""
suite_prob_diagnostics.py
==========================
Diagnoses each variant's stage-2 output AS A PROBABILITY GENERATOR.

What is actually being diagnosed
---------------------------------
There is no auxiliary probability head anywhere in the pipeline. The stage-1
models (MERF or RF) are squared-error REGRESSORS emitting two continuous
quantities in inches -- predicted_timing and predicted_offset. Probability is
produced downstream by a separate binary classifier:

    XGBClassifier(eval_metric='logloss')  fit on is_contact,
      features = X_raw + pitch-type dummies + curated interactions
                 + [predicted_timing, predicted_offset]
      -> predict_proba
      -> isotonic CalibratedClassifierCV, fit on a held-out 15% split

So every variant shares one classifier architecture and differs only in the
two stage-1 feature columns it is handed. `raw_baseline` gets neither and is
the floor.

Metrics
-------
Overall and within bins:

    bce        binary cross-entropy (log loss), the headline
    brier      Brier score
    auc        ranking quality, insensitive to calibration
    ece        expected calibration error (10 quantile bins)
    skill      1 - bce / bce_base, where bce_base is the log loss of always
               predicting THAT BIN's own base rate

Why `skill` matters for the binned view
----------------------------------------
Raw BCE is not comparable across bins. A bin whose contact rate is near 50%
is intrinsically harder than one at 90%, so it posts a higher loss even for a
perfect model. Ranking bins by raw BCE therefore mostly rediscovers where the
base rate is near one half. `skill` divides that out: it asks how much better
the model is than knowing only the bin's base rate, so a low-skill bin is one
the model genuinely handles badly. Both are reported; read `skill` to find
weak regions, `bce` to size their cost.

Outputs
-------
  prob_diagnostics_overall.csv
  prob_diagnostics_binned.csv
  plots/prob_diag_overall.png
  plots/prob_diag_binned_skill.png
  plots/prob_diag_reliability.png
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
import seaborn as sns
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import (VARIANT_ORDER, OUT_DIR, PLOT_DIR, TRAIN_SEASON,
                          EVAL_SEASON, probs_path)
from suite_data import load_train_and_eval
from suite_stage2 import RAW_TAG
from model_utils import CONTINUOUS_RAW

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE, RED, GREEN, GRAY, PURPLE = ('#2563EB', '#DC2626', '#16A34A', '#6B7280',
                                  '#7C3AED')
EPS = 1e-6
N_BINS = 10

# Binned by every continuous raw feature the classifier actually consumes,
# plus the two stage-1 outputs (a variant can be weak precisely where its own
# stage-1 prediction is unreliable).
BIN_VARS = CONTINUOUS_RAW + ['predicted_timing', 'predicted_offset']

MODEL_LABEL = {
    'm1_merf_base': 'm1 MERF base',
    'm2_merf_battrack': 'm2 MERF + bat track',
    'm3_rf_nore': 'm3 RF no RE',
    'm4_rf_missonly': 'm4 RF miss-only',
    RAW_TAG: 'raw baseline (no stage-1)',
}


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def bce(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins=N_BINS):
    """Expected calibration error over equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < n_bins * 2:
        return np.nan
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def prob_metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    base = float(y.mean())
    loss = bce(y, p)
    # Log loss of predicting this population's own base rate everywhere.
    base_loss = bce(y, np.full_like(y, np.clip(base, EPS, 1 - EPS)))
    out = dict(
        n=int(len(y)), base_rate=base, bce=loss, bce_baserate=base_loss,
        skill=float(1 - loss / base_loss) if base_loss > 0 else np.nan,
        brier=float(np.mean((p - y) ** 2)),
        ece=ece(y, p), mean_pred=float(p.mean()),
    )
    out['auc'] = (float(roc_auc_score(y, p))
                  if 0 < y.sum() < len(y) else np.nan)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────────────────

def load_scored(model: str, season: int, feat: pd.DataFrame,
                stage1: pd.DataFrame | None) -> dict:
    """Return {context_label: frame with y, p and every binning variable}."""
    path = probs_path(model, season)
    if not os.path.exists(path):
        return {}
    pr = pd.read_parquet(path)

    cols = [c for c in feat.columns
            if c in set(BIN_VARS) | {'row_key', 'is_contact'}]
    base = pr.merge(feat[cols], on='row_key', how='inner',
                    suffixes=('', '_f'))
    if stage1 is not None:
        base = base.merge(stage1, on='row_key', how='left')

    out = {}
    if season == TRAIN_SEASON and 'p_contact' in base.columns:
        out['2025_oof'] = base.assign(p=base['p_contact'])
    for mode in ('carry', 'context'):
        col = f'p_contact_{mode}'
        if col in base.columns:
            out[f'{season}_{mode}'] = base.assign(p=base[col])
    return out


def stage1_cols(model: str, season: int) -> pd.DataFrame | None:
    """predicted_timing / predicted_offset for the binning views."""
    if model == RAW_TAG:
        return None
    if season == TRAIN_SEASON:
        p = os.path.join(OUT_DIR, f'stage1_oof_{model}.parquet')
        if not os.path.exists(p):
            return None
        d = pd.read_parquet(p, columns=['row_key', 'predicted_timing',
                                        'predicted_offset'])
        return d
    p = os.path.join(OUT_DIR, f'stage1_eval{season}_{model}.parquet')
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p, columns=['row_key', 'predicted_timing_context',
                                    'predicted_offset_context'])
    return d.rename(columns={'predicted_timing_context': 'predicted_timing',
                             'predicted_offset_context': 'predicted_offset'})


def binned_table(df: pd.DataFrame, model: str, context: str) -> pd.DataFrame:
    rows = []
    y = df['is_contact'].astype(float).to_numpy()
    p = df['p'].to_numpy()
    for var in BIN_VARS:
        if var not in df.columns:
            continue
        v = df[var].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 2000:
            continue
        vv, yy, pp = v[ok], y[ok], p[ok]
        # Equal-count bins so every row carries the same weight; `zone` is
        # discrete and may collapse to fewer bins, which is fine.
        edges = np.unique(np.quantile(vv, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            continue
        idx = np.clip(np.digitize(vv, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 200:
                continue
            met = prob_metrics(yy[m], pp[m])
            rows.append(dict(model=model, context=context, variable=var,
                             bin=b, bin_lo=float(edges[b]),
                             bin_hi=float(edges[b + 1]),
                             bin_mid=float(np.median(vv[m])), **met))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────

def plot_overall(ov: pd.DataFrame, path: str):
    d = ov[ov['context'] == f'{EVAL_SEASON}_context'].copy()
    if d.empty:
        d = ov.copy()
    d['label'] = d['model'].map(MODEL_LABEL).fillna(d['model'])
    d = d.sort_values('bce')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, (col, title, better) in zip(axes, [
            ('bce', 'Binary cross-entropy', 'lower is better'),
            ('skill', 'Skill vs base rate', 'higher is better'),
            ('ece', 'Expected calibration error', 'lower is better')]):
        colors = [RED if m == RAW_TAG else BLUE for m in d['model']]
        ax.barh(d['label'], d[col], color=colors, height=0.62)
        for i, v in enumerate(d[col]):
            ax.text(v, i, f'  {v:.4f}', va='center', fontsize=9)
        ax.set_title(f'{title}\n({better})', fontsize=11)
        ax.grid(alpha=0.3, axis='x')
        ax.margins(x=0.22)
    fig.suptitle(f'Stage-2 classifiers as probability generators — '
                 f'{EVAL_SEASON}, random effects suppressed',
                 fontsize=13, fontweight='bold')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def plot_binned(bn: pd.DataFrame, path: str, metric='skill'):
    d = bn[bn['context'] == f'{EVAL_SEASON}_context']
    if d.empty:
        d = bn
    variables = [v for v in BIN_VARS if v in set(d['variable'])]
    models = [m for m in list(MODEL_LABEL) if m in set(d['model'])]
    colors = dict(zip(models, [BLUE, RED, GREEN, PURPLE, GRAY]))

    ncol = 4
    nrow = int(np.ceil(len(variables) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                             squeeze=False, constrained_layout=True)
    for i, var in enumerate(variables):
        ax = axes[i // ncol][i % ncol]
        for m in models:
            s = d[(d['variable'] == var) & (d['model'] == m)].sort_values('bin')
            if s.empty:
                continue
            ax.plot(s['bin_mid'], s[metric], marker='o', ms=3.5, lw=1.6,
                    color=colors[m], label=MODEL_LABEL[m])
        ax.set_title(var, fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    for j in range(len(variables), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    axes[0][0].set_ylabel(f'{metric} (vs bin base rate)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(models),
               fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f'Where the probability models are weak — {metric} by decile '
                 f'of each raw feature ({EVAL_SEASON})',
                 fontsize=13, fontweight='bold')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def plot_reliability(frames: dict, path: str):
    models = [m for m in MODEL_LABEL if m in frames]
    colors = dict(zip(models, [BLUE, RED, GREEN, PURPLE, GRAY]))
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], ls='--', color='black', lw=1.4,
            label='perfectly calibrated')
    for m in models:
        d = frames[m]
        y = d['is_contact'].astype(float).to_numpy()
        p = d['p'].to_numpy()
        edges = np.quantile(p, np.linspace(0, 1, 13))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() < 100:
                continue
            xs.append(p[sel].mean())
            ys.append(y[sel].mean())
        ax.plot(xs, ys, marker='o', ms=5, lw=1.8, color=colors[m],
                label=MODEL_LABEL[m])
    ax.set_xlabel('mean predicted P(contact)')
    ax.set_ylabel('observed contact rate')
    ax.set_title(f'Reliability — {EVAL_SEASON}, RE suppressed', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Probability-generator diagnostics')
    ap.add_argument('--models', nargs='*',
                    default=VARIANT_ORDER + [RAW_TAG])
    ap.add_argument('--seasons', nargs='*', type=int,
                    default=[TRAIN_SEASON, EVAL_SEASON])
    args = ap.parse_args()

    train, eval_df = load_train_and_eval(TRAIN_SEASON, EVAL_SEASON,
                                         verbose=False)
    feats = {TRAIN_SEASON: train, EVAL_SEASON: eval_df}

    overall, binned, rel_frames = [], [], {}
    for model in args.models:
        for season in args.seasons:
            s1 = stage1_cols(model, season)
            frames = load_scored(model, season, feats[season], s1)
            for ctx, df in frames.items():
                df = df.dropna(subset=['p', 'is_contact'])
                if df.empty:
                    continue
                met = prob_metrics(df['is_contact'].astype(float), df['p'])
                overall.append(dict(model=model, context=ctx, **met))
                binned.append(binned_table(df, model, ctx))
                if ctx == f'{EVAL_SEASON}_context':
                    rel_frames[model] = df
            print(f'  {model:22s} season {season}: '
                  f'{len(frames)} context(s)', flush=True)

    ov = pd.DataFrame(overall)
    bn = pd.concat([b for b in binned if not b.empty], ignore_index=True)
    ov.to_csv(os.path.join(OUT_DIR, 'prob_diagnostics_overall.csv'),
              index=False)
    bn.to_csv(os.path.join(OUT_DIR, 'prob_diagnostics_binned.csv'),
              index=False)

    plot_overall(ov, os.path.join(PLOT_DIR, 'prob_diag_overall.png'))
    plot_binned(bn, os.path.join(PLOT_DIR, 'prob_diag_binned_skill.png'),
                'skill')
    if rel_frames:
        plot_reliability(rel_frames,
                         os.path.join(PLOT_DIR, 'prob_diag_reliability.png'))

    pd.set_option('display.width', 240)
    print('\n=== Overall ===\n')
    show = ov[['model', 'context', 'n', 'base_rate', 'bce', 'skill', 'brier',
               'auc', 'ece']]
    print(show.to_string(index=False, float_format=lambda v: f'{v:,.5f}'))

    print('\n=== Weakest bins by skill '
          f'({EVAL_SEASON}, RE suppressed) ===\n')
    w = bn[bn['context'] == f'{EVAL_SEASON}_context']
    print(w.nsmallest(12, 'skill')[
        ['model', 'variable', 'bin_lo', 'bin_hi', 'n', 'base_rate',
         'bce', 'skill']].to_string(
        index=False, float_format=lambda v: f'{v:,.4f}'))


if __name__ == '__main__':
    main()
