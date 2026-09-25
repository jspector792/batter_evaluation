"""
suite_partial_mda_v2.py
========================
Step 1 of the MDA follow-up: partial out pitch mix and the model's own
context-only expectation, on top of contact rate.

The confound being tested
-------------------------
MDA survived partialling out contact rate. But discrimination separates signal
from *sampling* noise only -- it cannot tell true batter skill from a stable,
batter-specific MODEL ERROR. Supporting evidence that this is a live worry:
sd(mean predicted_offset_context) = 0.319 in exceeds sd(mean actual
miss_distance) = 0.290 in, i.e. the context-only model attributes more
between-batter spread to pitch selection than outcomes actually show. If the
model systematically mishandles a particular pitch mix, a batter who
persistently faces that mix gets a stable non-zero MDA that has nothing to do
with their skill.

So: regress MDA jointly on contact rate, the batter's mean predicted offset,
and a compact pitch-mix summary, then re-test discrimination on the residual.

Why the variance correction is EXACT here, not approximate
-----------------------------------------------------------
Every predictor is a within-batter mean of a swing-level quantity:

    contact_pct             mean of 1[is_contact]
    mean_pred_offset        mean of predicted_offset_context
    mean_release_speed_c    mean of release_speed_c
    mean_abs_plate_x        mean of |plate_x_bat_flip|
    mean_plate_z            mean of plate_z
    fb_share / bb_share     mean of 1[fastball] / 1[breaking]

So the partialled residual is itself a within-batter mean of one composite
swing-level variable

    w_j = offset_residual_j - sum_k b_k * z_jk

giving  Var_sampling(e_i) = Var_within(w_j) / n_i  exactly. Every
cross-covariance term (including the mechanical coupling between MDA and
mean_pred_offset, which share the prediction) is absorbed by construction --
no term can be forgotten. This reduces to the validated single-variable
formula when there is one predictor, and is cross-checked against a
swing-level bootstrap with coefficients held fixed.

Coupling caveat on the mean_pred_offset coefficient
----------------------------------------------------
MDA = actual - predicted, so regressing it on mean(predicted) shares a term
across both sides: Cov(A-P, P) = Cov(A,P) - Var(P). A negative coefficient is
therefore expected even under a well-calibrated model whenever the
between-batter spread in mean(P) is partly sampling noise. The COEFFICIENT is
thus not a clean miscalibration test on its own; the gated quantity is the
residual's discrimination, which is computed correctly regardless.

Outputs
-------
  dirB_mda_partialled_v2.csv
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import (VARIANT_ORDER, OUT_DIR, TRAIN_SEASON, EVAL_SEASON)
from suite_data import load_train_and_eval
from followup_utils import discrimination

warnings.filterwarnings('ignore')

MIN_SWINGS_METRIC = 100
N_BOOT = 400
SEED = 42

FASTBALL = {'FF', 'SI', 'FC', 'FA', 'FT'}
BREAKING = {'SL', 'CU', 'KC', 'ST', 'SV', 'CS', 'SC', 'KN'}
# Offspeed (CH, FS, FO, EP) is the reference category and is deliberately
# omitted: the three shares sum to ~1, so including all of them alongside an
# intercept would be collinear.

RESID_COL = 'offset_residual_context'

# (per-batter predictor name, swing-level column it is the mean of)
PREDICTORS = [
    ('contact_pct',          '_z_contact'),
    ('mean_pred_offset',     'predicted_offset_context'),
    ('mean_release_speed_c', 'release_speed_c'),
    ('mean_abs_plate_x',     '_z_abs_plate_x'),
    ('mean_plate_z',         'plate_z'),
    ('fb_share',             '_z_fb'),
    ('bb_share',             '_z_bb'),
]

# Pitch-mix controls that are pitch DESCRIPTORS, not the model's own output.
# These are not mechanically coupled to MDA, so they are the clean confound
# test. mean_pred_offset is held separate because MDA = actual - predicted
# shares a term with it (see the coupling caveat in the module docstring).
PITCHMIX_ONLY = ['contact_pct', 'mean_release_speed_c', 'mean_abs_plate_x',
                 'mean_plate_z', 'fb_share', 'bb_share']

MODELS = {
    'contact_only': ['contact_pct'],
    'contact_plus_pitchmix_no_pred': PITCHMIX_ONLY,
    'contact_plus_pitchmix': [p for p, _ in PREDICTORS],
}


def build_swings(variant: str, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Swing-level frame with the residual and every predictor's source column."""
    resid = pd.read_parquet(
        os.path.join(OUT_DIR, f'dirC_swing_residuals_{variant}.parquet'))
    keep = ['row_key', 'batter', RESID_COL, 'predicted_offset_context',
            'miss_distance_t', 'is_contact', 'pitch_type']
    df = resid[keep].merge(
        eval_df[['row_key', 'release_speed_c', 'plate_x_bat_flip', 'plate_z']],
        on='row_key', how='inner')

    df['_z_contact'] = df['is_contact'].astype(float)
    df['_z_abs_plate_x'] = df['plate_x_bat_flip'].abs()
    df['_z_fb'] = df['pitch_type'].isin(FASTBALL).astype(float)
    df['_z_bb'] = df['pitch_type'].isin(BREAKING).astype(float)

    need = [RESID_COL] + [z for _, z in PREDICTORS]
    df = df.dropna(subset=need).copy()

    n = df.groupby('batter')['batter'].transform('size')
    return df[n >= MIN_SWINGS_METRIC].copy()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby('batter')
    out = pd.DataFrame({'mda': g[RESID_COL].mean(), 'n': g.size()})
    for name, z in PREDICTORS:
        out[name] = g[z].mean()
    return out


def fit_ols(y: np.ndarray, X: np.ndarray):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ beta
    resid = y - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot)
    n, k = len(y), X.shape[1]
    adj = float(1 - (1 - r2) * (n - 1) / (n - k - 1))
    return beta, resid, r2, adj


def exact_sampling_var(df: pd.DataFrame, bat: pd.DataFrame,
                       cols: list[str], beta: np.ndarray) -> np.ndarray:
    """
    Var_within(w_j)/n_i where w_j = resid_j - sum_k b_k z_jk.
    Exact: all cross-covariances are inside Var(w).
    """
    z_cols = [dict(PREDICTORS)[c] for c in cols]
    w = df[RESID_COL].to_numpy(dtype=float).copy()
    for b_k, z in zip(beta[1:], z_cols):
        w -= b_k * df[z].to_numpy(dtype=float)
    tmp = pd.DataFrame({'batter': df['batter'].to_numpy(), 'w': w})
    gv = tmp.groupby('batter')['w'].var(ddof=1)
    gn = tmp.groupby('batter')['w'].size()
    return (gv / gn).reindex(bat.index).to_numpy()


def bootstrap_sampling_var(df: pd.DataFrame, bat: pd.DataFrame,
                           cols: list[str], beta: np.ndarray,
                           n_boot: int, seed: int) -> np.ndarray:
    """Resample each batter's swings; recompute e* with FIXED coefficients."""
    rng = np.random.default_rng(seed)
    z_cols = [dict(PREDICTORS)[c] for c in cols]
    arrs = {b: (g[RESID_COL].to_numpy(float),
                g[z_cols].to_numpy(float))
            for b, g in df.groupby('batter')}
    out = np.full(len(bat), np.nan)
    for i, b in enumerate(bat.index):
        r, Z = arrs[b]
        n = len(r)
        idx = rng.integers(0, n, size=(n_boot, n))
        mda_star = r[idx].mean(axis=1)
        z_star = Z[idx].mean(axis=1)           # (n_boot, k)
        e_star = mda_star - (beta[0] + z_star @ beta[1:])
        out[i] = e_star.var(ddof=1)
    return out


def pure_noise_reference(bat: pd.DataFrame, var: np.ndarray,
                         seed: int) -> dict:
    """
    Discrimination of a metric with no between-batter signal: each value drawn
    from its own sampling distribution around a single common mean. Reported
    unclipped too, since discrimination() floors at 0 and the raw numerator
    shows how tight the floor really is.
    """
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0, np.sqrt(np.clip(var, 0, None)))
    total = float(np.var(vals, ddof=1))
    raw = (total - float(np.nanmean(var))) / total if total > 0 else np.nan
    return dict(disc=discrimination(vals, var), disc_unclipped=raw)


def run_variant(variant: str, eval_df: pd.DataFrame, n_boot: int) -> list[dict]:
    df = build_swings(variant, eval_df)
    bat = aggregate(df)
    y = bat['mda'].to_numpy(float)

    rows = []
    for label, cols in MODELS.items():
        X = bat[cols].to_numpy(float)
        beta, e, r2, adj = fit_ols(y, X)

        var_exact = exact_sampling_var(df, bat, cols, beta)
        var_boot = bootstrap_sampling_var(df, bat, cols, beta, n_boot, SEED)

        d_exact = discrimination(e, var_exact)
        d_boot = discrimination(e, var_boot)
        null = pure_noise_reference(bat, var_exact, SEED)

        row = dict(
            variant=variant, partialling=label, n_batters=int(len(bat)),
            n_predictors=len(cols), r2=r2, adj_r2=adj,
            disc_resid_exact=d_exact, disc_resid_bootstrap=d_boot,
            disc_pure_noise=null['disc'],
            disc_pure_noise_unclipped=null['disc_unclipped'],
            resid_sd=float(np.std(e, ddof=1)),
            resid_sd_over_mda_sd=float(np.std(e, ddof=1) / np.std(y, ddof=1)),
            mean_sampling_var_exact=float(np.nanmean(var_exact)),
        )
        # Raw and standardised coefficients, so magnitudes are comparable.
        sd_y = float(np.std(y, ddof=1))
        for c, b_k in zip(cols, beta[1:]):
            row[f'b_{c}'] = float(b_k)
            row[f'beta_std_{c}'] = float(b_k * np.std(bat[c].to_numpy(float),
                                                      ddof=1) / sd_y)
        rows.append(row)

    # How much of the between-batter spread in mean_pred_offset is REAL vs
    # pitch-sample noise? If reliability is high, the large negative
    # coefficient on it reflects genuine miscalibration rather than
    # regression attenuation, and controlling for it is legitimate.
    gp = df.groupby('batter')['predicted_offset_context']
    var_p_samp = (gp.var(ddof=1) / gp.size()).reindex(bat.index).to_numpy()
    var_p_obs = float(np.var(bat['mean_pred_offset'].to_numpy(float), ddof=1))
    rel_p = float(max(0.0, var_p_obs - np.nanmean(var_p_samp)) / var_p_obs)
    # Between-batter calibration of the context model: regress each batter's
    # mean ACTUAL miss distance on their mean PREDICTED. Slope 1.0 = the
    # model's between-batter spread matches reality; slope < 1 = it
    # over-disperses (predicts more batter-to-batter difference than occurs).
    ga = df.groupby('batter')['miss_distance_t'].mean().reindex(bat.index)
    cal_slope = float(np.polyfit(bat['mean_pred_offset'].to_numpy(float),
                                 ga.to_numpy(float), 1)[0])
    # Slope of MDA on mean predicted, and the value expected from attenuation
    # alone under perfect calibration. Observed << expected => real
    # miscalibration, so controlling for mean_pred_offset is legitimate
    # rather than over-control.
    mda_pred_slope = float(np.polyfit(bat['mean_pred_offset'].to_numpy(float),
                                      bat['mda'].to_numpy(float), 1)[0])
    for r in rows:
        r['mean_pred_offset_reliability'] = rel_p
        r['mean_pred_offset_sd_between'] = float(np.sqrt(var_p_obs))
        r['mean_pred_offset_sd_sampling'] = float(np.sqrt(np.nanmean(var_p_samp)))
        r['calibration_slope_actual_on_pred'] = cal_slope
        r['slope_mda_on_pred'] = mda_pred_slope
        r['slope_expected_if_calibrated'] = rel_p - 1.0

    # Raw MDA discrimination, same cohort, for the side-by-side.
    g = df.groupby('batter')[RESID_COL]
    var_mda = (g.var(ddof=1) / g.size()).reindex(bat.index).to_numpy()
    rows.insert(0, dict(
        variant=variant, partialling='none_raw_MDA', n_batters=int(len(bat)),
        n_predictors=0, r2=np.nan, adj_r2=np.nan,
        disc_resid_exact=discrimination(y, var_mda),
        disc_resid_bootstrap=np.nan,
        disc_pure_noise=pure_noise_reference(bat, var_mda, SEED)['disc'],
        disc_pure_noise_unclipped=pure_noise_reference(
            bat, var_mda, SEED)['disc_unclipped'],
        resid_sd=float(np.std(y, ddof=1)), resid_sd_over_mda_sd=1.0,
        mean_sampling_var_exact=float(np.nanmean(var_mda))))
    return rows


def main():
    ap = argparse.ArgumentParser(description='Step 1: pitch-mix partialling')
    ap.add_argument('--variants', nargs='*', default=VARIANT_ORDER)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args()

    _, eval_df = load_train_and_eval(TRAIN_SEASON, EVAL_SEASON, verbose=False)

    rows = []
    for v in args.variants:
        rows += run_variant(v, eval_df, args.n_boot)
    out = pd.DataFrame(rows)

    path = os.path.join(OUT_DIR, 'dirB_mda_partialled_v2.csv')
    out.to_csv(path, index=False)

    pd.set_option('display.width', 260)
    core = ['variant', 'partialling', 'n_batters', 'r2', 'adj_r2',
            'disc_resid_exact', 'disc_resid_bootstrap', 'disc_pure_noise',
            'resid_sd_over_mda_sd']
    print('=== Step 1: discrimination after partialling ===\n')
    print(out[core].to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    print('\n=== Standardised coefficients (joint model) ===\n')
    jm = out[out['partialling'] == 'contact_plus_pitchmix']
    bcols = [c for c in out.columns if c.startswith('beta_std_')]
    print(jm[['variant'] + bcols].to_string(
        index=False, float_format=lambda v: f'{v:,.3f}'))
    print(f'\n-> {path}')


if __name__ == '__main__':
    main()
