"""
suite_timing_re_test.py
========================
Does the batter random effect earn its keep in the TIMING model, or can we
switch to a plain random forest?

Why the existing variants cannot answer this
---------------------------------------------
m1 (MERF timing) and m3 (RF timing) differ in three ways at once: the random
intercept, the bat-tracking feature block, and the batter-mean aggregates.
Any gap between them is unattributable. Bat tracking is also excluded here on
purpose -- `attack_direction` correlates -0.913 with `int_y` and explains
R^2 = 0.834 of it alone, so adding it to a timing model restates the target
rather than predicting it.

The design: a 2x2 on TIMING_BASE features only
------------------------------------------------
                        | batter-mean aggregates | no aggregates
    MERF (random int.)  |  A  (= m1's timing)    |  D
    RF   (no random e.) |  B                     |  C

    A vs B  isolates the random effect, holding aggregates fixed
    D vs C  isolates the random effect with no batter info at all
    A vs D  isolates the aggregates within MERF
    B vs C  isolates the aggregates within RF

A is not refit -- m1's cached timing predictions already are exactly this
configuration, so it is read from disk.

Note the batter-mean aggregates (mean bat_speed, mean release_speed faced)
ARE batter identity in another form. A model carrying them is not
batter-agnostic, which matters if the reason for dropping MERF is to remove
batter dependence rather than just to save compute.

Scoring
-------
2025 out-of-fold (within-season, where a random effect is legitimately
available) and 2026 (fit on 2025 only). For MERF, 2026 is scored twice:
`carry` applies each batter's 2025 intercept, `context` suppresses it. That
is the decisive comparison -- if MERF-carry does not beat RF on 2026, the
random effect is not transferring across seasons and is not worth its cost.

Also reported: the between-batter calibration slope (mean actual on mean
predicted per batter; 1.0 = calibrated), because the earlier offset-model
finding was that suppressing a MERF's random intercept leaves the fixed
effects over-dispersed.

Outputs
-------
  timing_re_test.csv
  plots/timing_re_test.png
"""

import os
import sys
import time
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import (TIMING_BASE, TIMING_AGG, XGB_TIMING, N_FOLDS, SEED,
                          TRAIN_SEASON, EVAL_SEASON, OUT_DIR, PLOT_DIR,
                          oof_path, eval_scored_path)
from suite_data import load_train_and_eval
from suite_stage1 import (build_train_design, build_score_design, fit_model,
                          predict_model)

warnings.filterwarnings('ignore')
logging.getLogger('merf').setLevel(logging.WARNING)
sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE, RED, GREEN, GRAY = '#2563EB', '#DC2626', '#16A34A', '#6B7280'

MIN_SWINGS = 100


def cfg(agg):
    return dict(name='timing', outcome_col='int_y', features=list(TIMING_BASE),
                batter_agg_features=list(TIMING_AGG) if agg else [],
                xgb_params=XGB_TIMING, max_iter=15, train_subset='all')


ARMS = {
    # key: (label, kind, uses batter-mean aggregates)
    'B_rf_agg':    ('RF + batter aggregates',      'rf',   True),
    'C_rf_noagg':  ('RF, no batter info',          'rf',   False),
    'D_merf_noagg': ('MERF, no batter aggregates', 'merf', False),
    # MERF's own fixed-effects learner, standalone. Separates "random effect"
    # from "boosting vs forest" -- without this arm, A-vs-B conflates the two.
    'E_xgb_noagg': ('XGBoost, no random effect',   'xgb',  False),
}
A_KEY = 'A_merf_agg'
A_LABEL = 'MERF + batter aggregates (= m1)'


def metrics(actual, pred):
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    resid = a - p
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return dict(n=int(len(a)),
                r2=float(1 - np.sum(resid ** 2) / ss_tot),
                rmse=float(np.sqrt(np.mean(resid ** 2))),
                mae=float(np.mean(np.abs(resid))),
                bias=float(np.mean(resid)))


def batter_calibration(df, actual_col, pred_col):
    """Between-batter calibration slope; 1.0 = calibrated."""
    d = df.dropna(subset=[actual_col, pred_col])
    g = d.groupby('batter')
    b = pd.DataFrame({'a': g[actual_col].mean(), 'p': g[pred_col].mean(),
                      'n': g.size()})
    b = b[b['n'] >= MIN_SWINGS]
    if len(b) < 20:
        return np.nan
    return float(np.polyfit(b['p'], b['a'], 1)[0])


def run_arm(key, train, eval_df, folds):
    label, kind, agg = ARMS[key]
    c = cfg(agg)
    print(f'\n--- {key}: {label} ({kind}) ---', flush=True)

    # 2025 out-of-fold
    oof_parts = []
    for fold in sorted(train['fold'].dropna().unique()):
        t0 = time.time()
        tr = (train['fold'] != fold) & train['fold'].notna()
        te = train['fold'] == fold
        X, Z, cl, y, x_cols, _, a = build_train_design(train, c, tr)
        model = fit_model(kind, c, X, Z, cl, y)
        Xte, Zte, clte, sub = build_score_design(train, c, te, x_cols, a)
        pred = predict_model(kind, model, Xte, Zte, clte, 'carry')
        oof_parts.append(pd.DataFrame({'row_key': sub['row_key'].values,
                                       'pred': np.asarray(pred)}))
        print(f'  fold {int(fold)}: n={len(y):,} [{time.time()-t0:.0f}s]',
              flush=True)
    oof = pd.concat(oof_parts, ignore_index=True)

    # full 2025 fit -> score 2026
    t0 = time.time()
    allm = pd.Series(True, index=train.index)
    X, Z, cl, y, x_cols, _, a = build_train_design(train, c, allm)
    model = fit_model(kind, c, X, Z, cl, y)
    print(f'  full fit n={len(y):,} [{time.time()-t0:.0f}s]', flush=True)

    evm = pd.Series(True, index=eval_df.index)
    Xe, Ze, cle, sube = build_score_design(eval_df, c, evm, x_cols, a)
    scored = pd.DataFrame({'row_key': sube['row_key'].values})
    for mode in ('carry', 'context'):
        scored[mode] = np.asarray(predict_model(kind, model, Xe, Ze, cle, mode))
    return oof, scored


def collect(key, label, oof, scored, train, eval_df):
    rows = []
    t = train[['row_key', 'batter', 'int_y']].merge(oof, on='row_key',
                                                    how='inner')
    m = metrics(t['int_y'], t['pred'])
    rows.append(dict(arm=key, label=label, context=f'{TRAIN_SEASON}_oof',
                     **m, calib_slope=batter_calibration(t, 'int_y', 'pred')))

    e = eval_df[['row_key', 'batter', 'int_y']].merge(scored, on='row_key',
                                                      how='inner')
    for mode in ('carry', 'context'):
        m = metrics(e['int_y'], e[mode])
        rows.append(dict(arm=key, label=label,
                         context=f'{EVAL_SEASON}_{mode}', **m,
                         calib_slope=batter_calibration(e, 'int_y', mode)))
    return rows


def load_arm_A(train, eval_df):
    """m1's cached timing predictions -- already exactly configuration A."""
    o = pd.read_parquet(oof_path('m1_merf_base'),
                        columns=['row_key', 'predicted_timing'])
    o = o.rename(columns={'predicted_timing': 'pred'})
    s = pd.read_parquet(eval_scored_path('m1_merf_base'),
                        columns=['row_key', 'predicted_timing_carry',
                                 'predicted_timing_context'])
    s = s.rename(columns={'predicted_timing_carry': 'carry',
                          'predicted_timing_context': 'context'})
    return collect(A_KEY, A_LABEL, o, s, train, eval_df)


def plot(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    order = [A_KEY] + list(ARMS)
    labels = {A_KEY: A_LABEL, **{k: v[0] for k, v in ARMS.items()}}
    ctxs = [f'{TRAIN_SEASON}_oof', f'{EVAL_SEASON}_carry',
            f'{EVAL_SEASON}_context']
    colors = {ctxs[0]: GRAY, ctxs[1]: RED, ctxs[2]: BLUE}

    for ax, (col, title, note) in zip(axes, [
            ('r2', 'R²  (higher better)', ''),
            ('rmse', 'RMSE, inches  (lower better)', ''),
            ('calib_slope', 'Between-batter calibration slope',
             '1.0 = calibrated')]):
        x = np.arange(len(order))
        w = 0.26
        for j, ctx in enumerate(ctxs):
            vals = [res[(res['arm'] == a) & (res['context'] == ctx)][col].mean()
                    for a in order]
            ax.bar(x + (j - 1) * w, vals, w, color=colors[ctx], label=ctx)
        if col == 'calib_slope':
            ax.axhline(1.0, color='black', ls='--', lw=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[a].replace(' (= m1)', '\n(= m1)')
                            for a in order], fontsize=8.5, rotation=18,
                           ha='right')
        ax.set_title(f'{title}\n{note}' if note else title, fontsize=11)
        ax.grid(alpha=0.3, axis='y')
    axes[0].legend(fontsize=9)
    fig.suptitle('Timing model: does the batter random effect earn its keep?',
                 fontsize=14, fontweight='bold')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def main():
    ap = argparse.ArgumentParser(description='Timing-model random-effect test')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--folds', type=int, default=N_FOLDS)
    args = ap.parse_args()

    train, eval_df = load_train_and_eval(TRAIN_SEASON, EVAL_SEASON,
                                         verbose=False)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    train = train.copy()
    train['fold'] = np.nan
    for i, (_, te) in enumerate(kf.split(train)):
        train.loc[train.index[te], 'fold'] = i

    rows = load_arm_A(train, eval_df)
    for key in args.arms:
        oof, scored = run_arm(key, train, eval_df, args.folds)
        rows += collect(key, ARMS[key][0], oof, scored, train, eval_df)

    res = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, 'timing_re_test.csv')

    # Arms are run separately (the MERF arm costs ~1h, the RF arms ~7min each),
    # so merge with whatever is already on disk instead of clobbering it.
    # Rows for arms just recomputed are replaced, not duplicated.
    if os.path.exists(path):
        prev = pd.read_csv(path)
        prev = prev[~prev['arm'].isin(set(res['arm']))]
        res = pd.concat([prev, res], ignore_index=True)
    order = {a: i for i, a in enumerate([A_KEY] + list(ARMS))}
    res = res.sort_values(['arm', 'context'],
                          key=lambda s: s.map(order) if s.name == 'arm' else s)
    res.to_csv(path, index=False)
    plot(res, os.path.join(PLOT_DIR, 'timing_re_test.png'))

    pd.set_option('display.width', 220)
    print('\n=== Timing model: random effect vs plain RF ===\n')
    print(res[['arm', 'label', 'context', 'n', 'r2', 'rmse', 'mae', 'bias',
               'calib_slope']].to_string(
        index=False, float_format=lambda v: f'{v:,.4f}'))
    print(f'\n-> {path}')


if __name__ == '__main__':
    main()
