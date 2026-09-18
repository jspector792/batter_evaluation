"""
step6_context_model.py
=========================
RB-Contact% Step 6: xContact% and Contact Skill Added.

Trains ONE full-population MERF model per architecture (timing, offset --
no fold holdout, since this model's role is "what would an average batter
produce against this pitch", not out-of-fold scoring) and ONE full-population
contact-probability GBM on the resulting context-only features. For each
batter's actually-faced pitches, generates a context-only prediction by
querying the MERF models with an unseen/placeholder batter ID, which MERF's
non-centered random-effects design maps to a purely fixed-effect (population
average) prediction with the random intercept term at exactly zero.

Explicit verification (spec 6.1): confirms this suppression is real (not a
silent no-op) by checking three things on held-out rows:
  (a) predictions using two DIFFERENT unseen placeholder batter IDs are
      identical -- proves no leakage through the cluster label itself
  (b) predictions using an unseen ID differ from predictions using each row's
      REAL batter ID -- proves the batter's random effect was actually
      contributing something (not already zero)
  (c) varying pitch context (e.g. release_speed) while holding the
      placeholder ID fixed changes the prediction -- proves context
      sensitivity survived the suppression (it isn't just returning a
      constant)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor, XGBClassifier
from merf import MERF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import prepare_data, build_subset, build_predict_frame, INT_Y_CONFIG, BARREL_CONFIG, OUT_DIR
from model_utils import assemble_design, make_gbm, CONTINUOUS_RAW, BINARY_RAW, X_HYBRID_EXTRA

warnings.filterwarnings('ignore')

PLACEHOLDER_BATTER = '__CONTEXT_ONLY_PLACEHOLDER__'
PLACEHOLDER_BATTER_2 = '__CONTEXT_ONLY_PLACEHOLDER_2__'  # a second, different one, for check (a)


def fit_full_population_models(df: pd.DataFrame) -> dict:
    """One MERF fit per architecture on ALL eligible rows -- see module docstring."""
    models = {}
    for cfg in [INT_Y_CONFIG, BARREL_CONFIG]:
        all_mask = pd.Series(True, index=df.index)
        X, Z, clusters, y, x_cols, _, agg = build_subset(df, cfg, all_mask)
        print(f'Fitting full-population {cfg["name"]} model on {len(y):,} rows...')
        xgb = XGBRegressor(**cfg['xgb_params'])
        mrf = MERF(fixed_effects_model=xgb, max_iterations=cfg['max_iter'],
                  gll_early_stop_threshold=1e-4)
        mrf.fit(X, Z, clusters, y)
        # agg = batter-level means (bat_speed/release_speed) computed on ALL
        # rows here (there's no held-out fold for a full-population model) --
        # reused below so an unseen placeholder batter ID correctly falls
        # back to the population mean instead of erroring on a null merge.
        models[cfg['name']] = dict(model=mrf, x_cols=x_cols, cfg=cfg, train_agg=agg)
        path = os.path.join(OUT_DIR, 'models', f'{cfg["name"]}_full_population_model.joblib')
        joblib.dump(mrf, path, compress=3)
        print(f'  -> {path}')
    return models


def context_only_predict(df: pd.DataFrame, model_info: dict, row_mask: pd.Series,
                         placeholder: str = PLACEHOLDER_BATTER) -> pd.DataFrame:
    """Predict with `batter` overwritten to an unseen placeholder ID, so
    MERF's random-intercept lookup falls through to zero (fixed effect only)."""
    cfg, mrf, x_cols = model_info['cfg'], model_info['model'], model_info['x_cols']
    faked = df.loc[row_mask].copy()
    faked['batter'] = placeholder
    X, Z, clusters, sub = build_predict_frame(faked, cfg, pd.Series(True, index=faked.index),
                                              x_cols, train_agg=model_info['train_agg'])
    fitted = mrf.predict(X, Z, clusters)
    return pd.DataFrame({'row_key': sub['row_key'].values, f'context_{cfg["name"]}': fitted})


def verify_suppression(df: pd.DataFrame, models: dict, n_check: int = 2000) -> dict:
    """Spec 6.1's explicit verification. Returns pass/fail per check."""
    sample = df.sample(n=min(n_check, len(df)), random_state=42)
    results = {}
    for name, info in models.items():
        cfg = info['cfg']
        real_mask = pd.Series(True, index=sample.index)

        # Real batter ID
        X_real, Z_real, clusters_real, sub_real = build_predict_frame(
            sample, cfg, real_mask, info['x_cols'], train_agg=info['train_agg'])
        p_real = info['model'].predict(X_real, Z_real, clusters_real)

        # Placeholder A
        pred_a = context_only_predict(sample, info, real_mask, PLACEHOLDER_BATTER)
        # Placeholder B (different unseen ID)
        pred_b = context_only_predict(sample, info, real_mask, PLACEHOLDER_BATTER_2)

        check_a_vs_b = np.allclose(pred_a[f'context_{name}'].values,
                                   pred_b[f'context_{name}'].values, atol=1e-8)

        # p_real is aligned to `sample`'s row order via build_predict_frame's
        # internal dropna -- re-merge on row_key to compare safely
        real_df = pd.DataFrame({'row_key': sub_real['row_key'].values, 'p_real': p_real})
        merged = real_df.merge(pred_a, on='row_key')
        check_real_differs = not np.allclose(merged['p_real'], merged[f'context_{name}'], atol=1e-6)

        # Context sensitivity: perturb release_speed_c and confirm the
        # placeholder-ID prediction changes
        perturbed = sample.copy()
        perturbed['release_speed_c'] = perturbed['release_speed_c'] + 2.0  # +2 SD
        pred_perturbed = context_only_predict(perturbed, info, real_mask, PLACEHOLDER_BATTER)
        merged2 = pred_a.merge(pred_perturbed, on='row_key', suffixes=('_orig', '_perturbed'))
        check_context_sensitive = not np.allclose(
            merged2[f'context_{name}_orig'], merged2[f'context_{name}_perturbed'], atol=1e-6)

        results[name] = dict(
            placeholder_a_equals_b=bool(check_a_vs_b),
            placeholder_differs_from_real=bool(check_real_differs),
            context_perturbation_changes_prediction=bool(check_context_sensitive),
        )
        print(f'[{name}] suppression checks: {results[name]}')
        assert check_a_vs_b, f'{name}: two different placeholder IDs gave different predictions -- suppression is leaking'
        assert check_real_differs, f'{name}: placeholder prediction == real-batter prediction -- suppression is a no-op'
        assert check_context_sensitive, f'{name}: perturbing pitch context did not change the prediction'
    return results


def main():
    df = prepare_data()
    models = fit_full_population_models(df)

    print('\nVerifying random-effect suppression (spec 6.1)...')
    checks = verify_suppression(df, models)

    all_mask = pd.Series(True, index=df.index)
    timing_ctx = context_only_predict(df, models['int_y'], all_mask)
    offset_ctx = context_only_predict(df, models['barrel'], all_mask)

    out = (df[['row_key', 'batter']]
           .merge(timing_ctx, on='row_key', how='left')
           .merge(offset_ctx, on='row_key', how='left')
           .rename(columns={'context_int_y': 'predicted_timing', 'context_barrel': 'predicted_offset'}))

    # ── Full-population contact-probability model, fit on the SAME features
    #    the winning Step 2/4 model used, then applied to context-only inputs ──
    full_df = df.merge(out[['row_key', 'predicted_timing', 'predicted_offset']], on='row_key')
    needed = CONTINUOUS_RAW + BINARY_RAW + X_HYBRID_EXTRA + ['pitch_type', 'is_contact']
    sub = full_df.dropna(subset=needed).copy()
    pt_categories = sorted(sub['pitch_type'].unique())
    X_full = assemble_design(sub, X_HYBRID_EXTRA, pt_categories)
    contact_model = make_gbm()
    contact_model.fit(X_full, sub['is_contact'].astype(int))
    joblib.dump(contact_model, os.path.join(OUT_DIR, 'models', 'contact_model_full_population.joblib'))

    x_contact_p = contact_model.predict_proba(X_full)[:, 1]
    sub = sub.assign(x_contact_p=x_contact_p)

    x_contact_pct = sub.groupby('batter')['x_contact_p'].mean().rename('x_contact_pct')

    rb_path = os.path.join(OUT_DIR, 'rb_contact_pct_by_batter.parquet')
    if os.path.exists(rb_path):
        rb = pd.read_parquet(rb_path).set_index('batter')
        rb = rb.join(x_contact_pct)
        rb['contact_skill_added'] = rb['rb_contact_pct'] - rb['x_contact_pct']
        league_mean_skill_added = rb['contact_skill_added'].mean()
        print(f'\nLeague-average Contact Skill Added: {league_mean_skill_added:.5f} '
              f'(sanity check: should be ~0)')
        rb.reset_index().to_parquet(rb_path, index=False)
        print(f'-> updated {rb_path} with x_contact_pct, contact_skill_added')
    else:
        print(f'\n{rb_path} not found yet -- run step5 first to merge Contact Skill Added in. '
              f'x_contact_pct computed for {len(x_contact_pct)} batters, not yet saved.')


if __name__ == '__main__':
    main()
