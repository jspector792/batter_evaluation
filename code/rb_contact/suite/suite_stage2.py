"""
suite_stage2.py
================
Stage 2 of the miss-distance suite: the is_contact classifier.

The classifier itself is UNCHANGED from the original pipeline -- it reuses
model_utils.assemble_design / make_gbm / calibrate / evaluate verbatim. The
only thing that varies across variants is which stage-1 predictions it is
handed as the two extra features.

Per variant, two sets of contact probabilities are produced:

  {train_season}  out-of-fold, on the same folds stage 1 used, so a swing's
                  p_contact never comes from a model that trained on it.

  {eval_season}   from a classifier trained on ALL of the training season
                  (using stage 1's out-of-fold features, so the classifier
                  never sees in-sample stage-1 predictions), applied to the
                  evaluation season. Produced twice, once from the `carry`
                  stage-1 columns and once from `context`, so the batter
                  random-effect contribution can be traced all the way
                  through to the final probability.

A `raw` baseline -- X_raw only, no stage-1 features at all -- is also written
once. It does not depend on any variant and is the honest floor for judging
whether the stage-1 predictions add anything.
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import (VARIANTS, VARIANT_ORDER, TRAIN_SEASON, EVAL_SEASON,
                          OUT_DIR, MODEL_DIR, SEED, oof_path, eval_scored_path,
                          probs_path)
from suite_data import load_train_and_eval
from model_utils import (assemble_design, make_gbm, calibrate, evaluate,
                         CONTINUOUS_RAW, BINARY_RAW, X_HYBRID_EXTRA)

warnings.filterwarnings('ignore')

RAW_TAG = 'raw_baseline'


def _fit_and_score(train_df, score_df, extra, pt_categories, tag):
    """Fit an isotonic-calibrated GBM on train_df, return p_contact on score_df."""
    model = make_gbm()

    # Calibration must be fit on rows the GBM did not train on.
    from sklearn.model_selection import train_test_split
    fit_part, calib_part = train_test_split(
        train_df, test_size=0.15, random_state=SEED,
        stratify=train_df['is_contact'])
    model.fit(assemble_design(fit_part, extra, pt_categories),
              fit_part['is_contact'].astype(int))
    cal = calibrate(model, assemble_design(calib_part, extra, pt_categories),
                    calib_part['is_contact'].astype(int), method='isotonic')

    p = cal.predict_proba(assemble_design(score_df, extra, pt_categories))[:, 1]
    joblib.dump(cal, os.path.join(MODEL_DIR, f'stage2_{tag}.joblib'))
    return p, cal


def run_variant(vname: str, train: pd.DataFrame, eval_df: pd.DataFrame):
    print(f'\n{"=" * 70}\nSTAGE 2  {vname}\n{"=" * 70}', flush=True)

    oof = pd.read_parquet(oof_path(vname))[
        ['row_key', 'fold', 'predicted_timing', 'predicted_offset']]
    tr = train.merge(oof, on='row_key', how='inner')

    scored = pd.read_parquet(eval_scored_path(vname))
    ev = eval_df.merge(scored[['row_key'] + [c for c in scored.columns
                                             if c.startswith('predicted_')]],
                       on='row_key', how='inner')

    extra = X_HYBRID_EXTRA
    needed_tr = CONTINUOUS_RAW + BINARY_RAW + extra + ['pitch_type',
                                                       'is_contact', 'fold']
    tr = tr.dropna(subset=needed_tr).copy()
    pt_categories = sorted(set(tr['pitch_type']) | set(ev['pitch_type']))

    # ── Out-of-fold probabilities on the training season ──────────────────
    parts = []
    for fold in sorted(tr['fold'].unique()):
        fit_df, test_df = tr[tr['fold'] != fold], tr[tr['fold'] == fold]
        p, _ = _fit_and_score(fit_df, test_df, extra, pt_categories,
                              f'{vname}_fold{int(fold)}')
        parts.append(pd.DataFrame({'row_key': test_df['row_key'].values,
                                   'p_contact': p}))
        print(f'  fold {int(fold)}: n={len(test_df):,}', flush=True)
    tr_probs = tr[['row_key', 'batter', 'game_date', 'game_pk',
                   'at_bat_number', 'pitch_number', 'is_contact']].merge(
        pd.concat(parts, ignore_index=True), on='row_key', how='left')
    tr_probs.to_parquet(probs_path(vname, TRAIN_SEASON), index=False)
    print(f'-> {probs_path(vname, TRAIN_SEASON)} ({len(tr_probs):,} rows)')

    # ── Evaluation season, once per random-effect mode ────────────────────
    ev_out = ev[['row_key', 'batter', 'game_date', 'game_pk', 'at_bat_number',
                 'pitch_number', 'is_contact']].copy()
    for mode in ('carry', 'context'):
        cols = {f'predicted_timing_{mode}': 'predicted_timing',
                f'predicted_offset_{mode}': 'predicted_offset'}
        ev_mode = ev.rename(columns=cols)
        need = CONTINUOUS_RAW + BINARY_RAW + extra + ['pitch_type', 'is_contact']
        ev_mode = ev_mode.dropna(subset=need)
        p, _ = _fit_and_score(tr, ev_mode, extra, pt_categories,
                              f'{vname}_full_{mode}')
        ev_out = ev_out.merge(
            pd.DataFrame({'row_key': ev_mode['row_key'].values,
                          f'p_contact_{mode}': p}),
            on='row_key', how='left')
        print(f'  {EVAL_SEASON}/{mode}: n={len(ev_mode):,} '
              f'mean p={p.mean():.4f}', flush=True)

    ev_out.to_parquet(probs_path(vname, EVAL_SEASON), index=False)
    print(f'-> {probs_path(vname, EVAL_SEASON)} ({len(ev_out):,} rows)')


def run_raw_baseline(train: pd.DataFrame, eval_df: pd.DataFrame):
    """X_raw only -- no stage-1 feature, identical for every variant."""
    print(f'\n{"=" * 70}\nSTAGE 2  {RAW_TAG}\n{"=" * 70}', flush=True)

    # Reuse any variant's fold assignment so the raw baseline is comparable
    # fold-for-fold with the hybrid models. Every variant shares one fold
    # split, so whichever file exists will do -- don't assume it is the first
    # variant's, since the suite can be run for a subset.
    src = next((v for v in VARIANT_ORDER if os.path.exists(oof_path(v))), None)
    if src is None:
        print('SKIP raw baseline: no stage-1 output exists to take folds from')
        return
    folds = pd.read_parquet(oof_path(src))[['row_key', 'fold']]
    tr = train.merge(folds, on='row_key', how='inner')
    need = CONTINUOUS_RAW + BINARY_RAW + ['pitch_type', 'is_contact', 'fold']
    tr = tr.dropna(subset=need).copy()
    ev = eval_df.dropna(subset=CONTINUOUS_RAW + BINARY_RAW +
                        ['pitch_type', 'is_contact']).copy()
    pt_categories = sorted(set(tr['pitch_type']) | set(ev['pitch_type']))

    parts = []
    for fold in sorted(tr['fold'].unique()):
        fit_df, test_df = tr[tr['fold'] != fold], tr[tr['fold'] == fold]
        p, _ = _fit_and_score(fit_df, test_df, [], pt_categories,
                              f'{RAW_TAG}_fold{int(fold)}')
        parts.append(pd.DataFrame({'row_key': test_df['row_key'].values,
                                   'p_contact': p}))
        print(f'  fold {int(fold)}: n={len(test_df):,}', flush=True)
    out_tr = tr[['row_key', 'batter', 'game_date', 'game_pk', 'at_bat_number',
                 'pitch_number', 'is_contact']].merge(
        pd.concat(parts, ignore_index=True), on='row_key', how='left')
    out_tr.to_parquet(probs_path(RAW_TAG, TRAIN_SEASON), index=False)

    p, _ = _fit_and_score(tr, ev, [], pt_categories, f'{RAW_TAG}_full')
    out_ev = ev[['row_key', 'batter', 'game_date', 'game_pk', 'at_bat_number',
                 'pitch_number', 'is_contact']].copy()
    # Same probabilities under both modes: this model has no stage-1 input,
    # so there is no random effect anywhere to carry or suppress.
    out_ev['p_contact_carry'] = p
    out_ev['p_contact_context'] = p
    out_ev.to_parquet(probs_path(RAW_TAG, EVAL_SEASON), index=False)
    print(f'  {EVAL_SEASON}: n={len(ev):,} mean p={p.mean():.4f}')
    print(f'-> {probs_path(RAW_TAG, EVAL_SEASON)}')


def main():
    ap = argparse.ArgumentParser(description='Stage 2 of the miss-distance suite')
    ap.add_argument('--variants', nargs='*', default=VARIANT_ORDER)
    ap.add_argument('--skip-raw', action='store_true')
    args = ap.parse_args()

    train, eval_df = load_train_and_eval(TRAIN_SEASON, EVAL_SEASON)

    for vname in args.variants:
        if not os.path.exists(oof_path(vname)):
            print(f'SKIP {vname}: {oof_path(vname)} missing (run stage 1 first)')
            continue
        run_variant(vname, train, eval_df)

    if not args.skip_raw:
        run_raw_baseline(train, eval_df)


if __name__ == '__main__':
    main()
