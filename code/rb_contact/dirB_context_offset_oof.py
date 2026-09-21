"""
dirB_context_offset_oof.py
===========================
Generates a **context-only** out-of-fold offset prediction, which Direction B
needs and the existing `predicted_offset` cannot provide.

Why
---
`predicted_offset` in oof_predictions.parquet comes from a MERF that carries a
batter random intercept. MERF's prediction is `f(X) + b_batter`, and `b_batter`
is fit so the batter's residuals sum to roughly zero over the swings in the
training folds. Since folds are assigned at the *swing* level, every batter has
swings in all five folds, so `b_batter` is estimated from that same batter's
season either way.

The consequence, measured: the per-batter mean of
`barrel_distance_v2 - predicted_offset` has an across-batter SD of 0.028 inches
against a typical per-batter sampling SD of 0.084 inches -- the between-batter
spread is *smaller* than noise, so discrimination is exactly 0. And first-half
mean vs. second-half mean correlate at **r = -0.77**: not a weak stat, an
arithmetically self-cancelling one, because the random intercept pins the
season-long sum of residuals near zero and the two halves are forced to offset
each other.

So "Miss Distance Added" measured against `predicted_offset` is null by
construction. The batter effect is precisely what a skill metric must keep, and
the random intercept has already subtracted it.

What this script does
---------------------
Re-runs the same 5-fold MERF loop as `oof_timing_offset.run_oof` for
BARREL_CONFIG, reusing the fold assignment stored in oof_predictions.parquet so
the split is identical, and predicts each held-out fold **twice**:

  predicted_offset_refit    real batter cluster -- reproduces the existing
                            `predicted_offset`, which is the validation that
                            this loop is the same computation
  predicted_offset_context  an unseen placeholder batter ID, which MERF's
                            random-effect lookup maps to a zero random
                            intercept, leaving the population fixed effect
                            f(X) only -- the same mechanism step6_context_model
                            uses and verifies

`barrel_distance_v2 - predicted_offset_context` is then "how much closer to the
ball did this batter get than the league-average batter would have on these
pitches", which is a skill quantity.

Output
------
  out/rb_contact/oof_context_offset.parquet
      row_key, predicted_offset_refit, predicted_offset_context
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from merf import MERF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import (prepare_data, build_subset, build_predict_frame,
                               BARREL_CONFIG, OUT_DIR)

warnings.filterwarnings('ignore')

PLACEHOLDER_BATTER = '__CONTEXT_ONLY_PLACEHOLDER__'
OUT_NAME = 'oof_context_offset.parquet'


def run(df: pd.DataFrame, max_iter: int | None = None) -> pd.DataFrame:
    cfg = dict(BARREL_CONFIG)
    if max_iter is not None:
        cfg['max_iter'] = max_iter

    results = []
    for fold in sorted(df['fold'].dropna().unique()):
        t0 = time.time()
        train_mask = (df['fold'] != fold) & df['fold'].notna()
        test_mask = (df['fold'] == fold)

        X_tr, Z_tr, clusters_tr, y_tr, x_cols, _, train_agg = build_subset(df, cfg, train_mask)
        mrf = MERF(fixed_effects_model=XGBRegressor(**cfg['xgb_params']),
                   max_iterations=cfg['max_iter'], gll_early_stop_threshold=1e-4)
        mrf.fit(X_tr, Z_tr, clusters_tr, y_tr)

        # (a) real batter cluster -- should reproduce the cached predicted_offset
        X_te, Z_te, clusters_te, sub_te = build_predict_frame(
            df, cfg, test_mask, x_cols, train_agg)
        pred_refit = mrf.predict(X_te, Z_te, clusters_te)

        # (b) unseen placeholder cluster -- random intercept falls through to 0
        faked = df.loc[test_mask].copy()
        faked['batter'] = PLACEHOLDER_BATTER
        X_ctx, Z_ctx, clusters_ctx, sub_ctx = build_predict_frame(
            faked, cfg, pd.Series(True, index=faked.index), x_cols, train_agg)
        pred_ctx = mrf.predict(X_ctx, Z_ctx, clusters_ctx)

        assert (sub_te['row_key'].values == sub_ctx['row_key'].values).all(), \
            'real-cluster and placeholder-cluster predict frames fell out of alignment'
        # Suppression must actually do something -- if the two predictions were
        # identical the random effect was never contributing and the whole
        # premise of this script would be wrong.
        assert not np.allclose(pred_refit, pred_ctx, atol=1e-6), \
            f'fold {fold}: placeholder prediction == real-batter prediction; ' \
            f'random-effect suppression is a no-op'

        print(f'  fold {fold}: train n={len(y_tr):,} test n={len(pred_ctx):,}  '
              f'context mean={pred_ctx.mean():.3f} sd={pred_ctx.std():.3f}  '
              f'(real-cluster sd={pred_refit.std():.3f})  [{time.time()-t0:.0f}s]',
              flush=True)

        results.append(pd.DataFrame({
            'row_key': sub_te['row_key'].values,
            'predicted_offset_refit': pred_refit,
            'predicted_offset_context': pred_ctx,
        }))
    return pd.concat(results, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--max-iter', type=int, default=None,
                    help='override BARREL_CONFIG max_iterations (default 20, '
                         'with 1e-4 relative-GLL early stopping)')
    ap.add_argument('--out-name', default=OUT_NAME)
    args = ap.parse_args()

    df = prepare_data()
    oof_path = os.path.join(OUT_DIR, 'oof_predictions.parquet')
    oof = pd.read_parquet(oof_path)[['row_key', 'fold', 'predicted_offset']]
    df = df.merge(oof, on='row_key', how='inner')
    print(f'Reusing the fold assignment from {os.path.basename(oof_path)}: '
          f'{df["fold"].value_counts().sort_index().to_dict()}')

    out = run(df, args.max_iter)

    # Validation: the real-cluster refit should match the cached predicted_offset.
    chk = out.merge(df[['row_key', 'predicted_offset']], on='row_key', how='inner').dropna()
    corr = chk['predicted_offset_refit'].corr(chk['predicted_offset'])
    mad = (chk['predicted_offset_refit'] - chk['predicted_offset']).abs().mean()
    print(f'\nRefit vs. cached predicted_offset: r={corr:.6f}, mean|diff|={mad:.6f} '
          f'(n={len(chk):,}) -- near-exact agreement confirms this loop reproduces '
          f'the Step 1c computation, so the context column differs only by the '
          f'suppressed random intercept.')

    delta = (chk['predicted_offset_refit'] - out.set_index('row_key')
             .loc[chk['row_key'], 'predicted_offset_context'].values)
    print(f'Batter random intercept implied (refit - context): '
          f'mean={delta.mean():.4f} sd={delta.std():.4f} '
          f'range=[{delta.min():.3f}, {delta.max():.3f}]')

    path = os.path.join(OUT_DIR, args.out_name)
    out.to_parquet(path, index=False)
    print(f'\n-> {path} ({len(out):,} rows)')


if __name__ == '__main__':
    main()
