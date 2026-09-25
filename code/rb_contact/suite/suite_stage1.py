"""
suite_stage1.py
================
Stage 1 of the miss-distance suite: the two continuous regressors, for each
of the three variants.

    timing  ->  int_y            (contact depth, inches)
    offset  ->  miss_distance_t  (tracked miss distance, 0 on contact)

For every variant this produces two things:

  1. Out-of-fold predictions on the TRAINING season (2025), via 5-fold CV with
     row-level folds shared between both regressors. These are what stage 2
     consumes as features and what the 2025-side residual diagnostics use.

  2. A full-2025 fit, applied to the EVALUATION season (2026) twice:
       carry    -- batter keeps the random intercept learned from their 2025
                   swings (unseen 2026 batters fall through to fixed effects)
       context  -- random intercept forced to zero for everyone
     For the RF variant there is no random effect, so the two modes are equal
     by construction; both columns are still written so downstream code stays
     uniform.

Leakage controls carried over from the original pipeline
---------------------------------------------------------
* Batter-mean aggregate features are computed on TRAINING rows only and joined
  onto held-out rows, so a row's own bat_speed never enters the feature used
  to predict it.
* Pitch-type dummy columns are aligned to the training design; a category
  absent from a fold is added as an all-zero column rather than silently
  shifting the matrix.
* No 2026 row takes part in any fit.
"""

import os
import sys
import time
import json
import logging
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from merf import MERF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite_config import (VARIANTS, VARIANT_ORDER, TARGETS, N_FOLDS, SEED,
                          TRAIN_SEASON, EVAL_SEASON, RF_PARAMS, MODEL_DIR,
                          OUT_DIR, PLACEHOLDER_BATTER, oof_path,
                          eval_scored_path)
from suite_data import load_train_and_eval

warnings.filterwarnings('ignore')
logging.getLogger('merf').setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────
# Design assembly
# ──────────────────────────────────────────────────────────────────────────

def build_train_design(df: pd.DataFrame, cfg: dict, mask: pd.Series):
    """
    Rows usable for TRAINING: non-null target and all features.
    Returns the batter-aggregate table so held-out scoring can reuse it
    instead of recomputing (which would leak the held-out row's own values).
    """
    needed = [cfg['outcome_col'], 'batter'] + cfg['features']
    sub = df.loc[mask]

    # train_subset='miss_only' restricts the FIT to whiffs, where
    # miss_distance is genuinely tracked rather than assigned 0. Scoring is
    # unaffected -- build_score_design still returns every swing.
    if cfg.get('train_subset', 'all') == 'miss_only':
        sub = sub[~sub['is_contact']]

    sub = sub.dropna(subset=needed).copy()

    agg = None
    if cfg['batter_agg_features']:
        agg = (sub.groupby('batter')[cfg['batter_agg_features']]
                  .mean().add_prefix('batter_mean_').reset_index())
        sub = sub.merge(agg, on='batter', how='left')

    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt',
                         drop_first=False)
    pt_cols = sorted(c for c in sub.columns if c.startswith('pt_'))
    x_cols = cfg['features'] + pt_cols
    if cfg['batter_agg_features']:
        x_cols += [f'batter_mean_{f}' for f in cfg['batter_agg_features']]

    X = sub[x_cols].astype(float).reset_index(drop=True)
    y = sub[cfg['outcome_col']].reset_index(drop=True)
    clusters = sub['batter'].reset_index(drop=True)
    Z = np.ones((len(sub), 1))
    return X, Z, clusters, y, x_cols, sub, agg


def build_score_design(df: pd.DataFrame, cfg: dict, mask: pd.Series,
                       x_cols: list, train_agg: pd.DataFrame | None):
    """Rows to SCORE: features must be present, target may be null."""
    needed = ['batter'] + cfg['features']
    sub = df.loc[mask].dropna(subset=needed).copy()

    if cfg['batter_agg_features']:
        sub = sub.merge(train_agg, on='batter', how='left')
        for f in cfg['batter_agg_features']:
            col = f'batter_mean_{f}'
            # A batter with no training rows gets the population mean rather
            # than a null feature.
            sub[col] = sub[col].fillna(train_agg[col].mean())

    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt',
                         drop_first=False)
    for c in x_cols:
        if c not in sub.columns:
            sub[c] = 0.0
    X = sub[x_cols].astype(float).reset_index(drop=True)
    clusters = sub['batter'].reset_index(drop=True)
    Z = np.ones((len(sub), 1))
    return X, Z, clusters, sub


# ──────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────

def fit_model(kind: str, cfg: dict, X, Z, clusters, y):
    if kind == 'merf':
        mrf = MERF(fixed_effects_model=XGBRegressor(**cfg['xgb_params']),
                   max_iterations=cfg['max_iter'],
                   gll_early_stop_threshold=1e-4)
        mrf.fit(X, Z, clusters, y)
        return mrf
    if kind == 'rf':
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X, y)
        return rf
    if kind == 'xgb':
        # Plain gradient boosting, no random effect: MERF's own fixed-effects
        # learner run standalone. Exists so that MERF-vs-RF comparisons can
        # separate "random effect" from "boosting vs forest".
        gb = XGBRegressor(**cfg['xgb_params'])
        gb.fit(X, y)
        return gb
    raise ValueError(f'unknown model kind {kind!r}')


def predict_model(kind: str, model, X, Z, clusters, re_mode: str = 'carry'):
    """
    re_mode only bites for MERF:
      carry   -- pass the real batter labels; unseen ones fall to fixed effects
      context -- pass a placeholder label so every random intercept is zero
    """
    if kind in ('rf', 'xgb'):
        return model.predict(X)
    if re_mode == 'context':
        clusters = pd.Series([PLACEHOLDER_BATTER] * len(X))
    return model.predict(X, Z, clusters)


# ──────────────────────────────────────────────────────────────────────────
# Out-of-fold on the training season
# ──────────────────────────────────────────────────────────────────────────

def run_oof(train: pd.DataFrame, variant: dict, target: str) -> pd.DataFrame:
    cfg = variant[target]
    kind = variant['kind']
    print(f'\n--- OOF {variant["name"]} / {target} '
          f'({kind}, target={cfg["outcome_col"]}) ---', flush=True)

    out = []
    for fold in sorted(train['fold'].dropna().unique()):
        t0 = time.time()
        tr_mask = (train['fold'] != fold) & train['fold'].notna()
        te_mask = train['fold'] == fold

        X, Z, cl, y, x_cols, _, agg = build_train_design(train, cfg, tr_mask)
        model = fit_model(kind, cfg, X, Z, cl, y)

        Xte, Zte, clte, sub_te = build_score_design(train, cfg, te_mask,
                                                    x_cols, agg)
        pred = predict_model(kind, model, Xte, Zte, clte, 'carry')

        out.append(pd.DataFrame({'row_key': sub_te['row_key'].values,
                                 f'predicted_{target}': np.asarray(pred)}))
        print(f'  fold {int(fold)}: train={len(y):,} score={len(pred):,} '
              f'pred mean={np.mean(pred):.3f} sd={np.std(pred):.3f} '
              f'[{time.time() - t0:.0f}s]', flush=True)

    return pd.concat(out, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# Full fit + evaluation-season scoring
# ──────────────────────────────────────────────────────────────────────────

def run_full_fit_and_score(train: pd.DataFrame, eval_df: pd.DataFrame,
                           variant: dict, target: str) -> tuple[pd.DataFrame, dict]:
    cfg = variant[target]
    kind = variant['kind']
    print(f'\n--- FULL FIT {variant["name"]} / {target} '
          f'-> score {EVAL_SEASON} ---', flush=True)

    t0 = time.time()
    all_mask = pd.Series(True, index=train.index)
    X, Z, cl, y, x_cols, _, agg = build_train_design(train, cfg, all_mask)
    model = fit_model(kind, cfg, X, Z, cl, y)
    print(f'  fit on {len(y):,} {TRAIN_SEASON} rows [{time.time() - t0:.0f}s]',
          flush=True)

    joblib.dump({'model': model, 'x_cols': x_cols, 'agg': agg, 'kind': kind},
                os.path.join(MODEL_DIR, f'{variant["name"]}_{target}.joblib'))

    ev_mask = pd.Series(True, index=eval_df.index)
    Xe, Ze, cle, sub_e = build_score_design(eval_df, cfg, ev_mask, x_cols, agg)

    res = pd.DataFrame({'row_key': sub_e['row_key'].values})
    for mode in ('carry', 'context'):
        res[f'predicted_{target}_{mode}'] = np.asarray(
            predict_model(kind, model, Xe, Ze, cle, mode))

    delta = res[f'predicted_{target}_carry'] - res[f'predicted_{target}_context']
    known = cle.isin(set(train['batter'])).values
    info = dict(variant=variant['name'], target=target, kind=kind,
                n_train=int(len(y)), n_eval=int(len(res)),
                eval_rows_known_batter=int(known.sum()),
                eval_rows_new_batter=int((~known).sum()),
                re_delta_sd=float(delta.std()),
                re_delta_absmax=float(delta.abs().max()))
    print(f'  scored {len(res):,} {EVAL_SEASON} rows | '
          f'known-batter rows {known.sum():,} / new {(~known).sum():,} | '
          f'carry-context sd={delta.std():.4f}', flush=True)

    if kind == 'merf':
        # The whole carry/context contrast is meaningless if suppression is a
        # no-op, so fail loudly rather than reporting two identical columns.
        assert delta.abs().max() > 1e-8, (
            f'{variant["name"]}/{target}: carry == context for every row; '
            f'random-effect suppression did nothing')
    return res, info


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────

def assign_folds(train: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    train = train.copy()
    train['fold'] = np.nan
    for fold_id, (_, test_idx) in enumerate(kf.split(train)):
        train.loc[train.index[test_idx], 'fold'] = fold_id
    print(f'Fold sizes: {train["fold"].value_counts().sort_index().to_dict()}')
    return train


def main():
    ap = argparse.ArgumentParser(description='Stage 1 of the miss-distance suite')
    ap.add_argument('--variants', nargs='*', default=VARIANT_ORDER,
                    help='subset of variants to run')
    ap.add_argument('--targets', nargs='*', default=TARGETS)
    ap.add_argument('--folds', type=int, default=N_FOLDS)
    ap.add_argument('--sample-n', type=int, default=None,
                    help='subsample the training season for a smoke test')
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip a variant whose outputs are already on disk')
    args = ap.parse_args()

    train, eval_df = load_train_and_eval(TRAIN_SEASON, EVAL_SEASON)
    if args.sample_n:
        train = train.sample(n=min(args.sample_n, len(train)),
                             random_state=SEED).reset_index(drop=True)
        eval_df = eval_df.sample(n=min(args.sample_n, len(eval_df)),
                                 random_state=SEED).reset_index(drop=True)
        print(f'SMOKE TEST: train={len(train):,} eval={len(eval_df):,}')

    train = assign_folds(train, args.folds)

    id_cols = ['row_key', 'batter', 'pitcher', 'is_contact', 'description',
               'game_date', 'game_pk', 'at_bat_number', 'pitch_number',
               'int_y', 'miss_distance_t', 'in_zone', 'pitch_type']

    infos = []
    for vname in args.variants:
        variant = VARIANTS[vname]
        print(f'\n{"=" * 70}\nVARIANT {vname} ({variant["kind"]})\n{"=" * 70}',
              flush=True)

        if args.skip_existing and os.path.exists(oof_path(vname)) \
                and os.path.exists(eval_scored_path(vname)):
            print('  outputs already present, skipping')
            continue

        oof = train[id_cols + ['fold']].copy()
        scored = eval_df[id_cols].copy()

        for target in args.targets:
            oof = oof.merge(run_oof(train, variant, target),
                            on='row_key', how='left')
            res, info = run_full_fit_and_score(train, eval_df, variant, target)
            scored = scored.merge(res, on='row_key', how='left')
            infos.append(info)

        oof.to_parquet(oof_path(vname), index=False)
        scored.to_parquet(eval_scored_path(vname), index=False)
        print(f'\n-> {oof_path(vname)} ({len(oof):,} rows)')
        print(f'-> {eval_scored_path(vname)} ({len(scored):,} rows)')

    if infos:
        path = os.path.join(OUT_DIR, 'stage1_fit_info.json')
        prev = []
        if os.path.exists(path):
            with open(path) as f:
                prev = json.load(f)
        with open(path, 'w') as f:
            json.dump(prev + infos, f, indent=2)
        print(f'\n-> {path}')


if __name__ == '__main__':
    main()
