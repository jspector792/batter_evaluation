"""
oof_timing_offset.py
=====================
RB-Contact% Step 1c: generate genuinely out-of-fold predicted_timing (int_y) and
predicted_offset (barrel_distance_v2) for every swing, by k-fold retraining the
same MERF architectures used in final_merf_models.py (INT_Y_CONFIG, BARREL_CONFIG).

Why this exists instead of reusing int_y_predicted / barrel_distance_predicted
already sitting in data/all_pitches_2025/*.parquet: those columns are IN-SAMPLE
predictions (final_merf_models.py trains on the full dataset and predicts back
onto the rows it trained on). Using them as features in the RB-Contact% pipeline
would leak each swing's own outcome-adjacent signal into its own prediction.
See out/rb_contact/data_audit.md sections 2 and 4.

Fold assignment is at the swing (row) level, not batter level, per spec Step 1c/2.1
-- a batter's swings can land in multiple folds. Each model's training subset for
fold k excludes fold k's rows; each model still generates a prediction for every
row in fold k that has valid X_raw features, even if that row's own real target
happens to be null (e.g. a contact event missing launch_angle) -- the prediction
only depends on pre-outcome features, never on the target.

Outputs
-------
  out/rb_contact/oof_predictions.parquet
    one row per swing: row key, batter, pitcher, is_contact label, fold_id,
    predicted_timing, predicted_offset (both OOF)
"""

import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from merf import MERF

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'all_pitches_2025')
OUT_DIR  = os.path.join(BASE_DIR, 'out', 'rb_contact')
os.makedirs(OUT_DIR, exist_ok=True)

INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

MISS       = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL       = {'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip'}
IN_PLAY    = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
BUNT_DESC  = {'missed_bunt', 'foul_bunt', 'bunt_foul_tip'}
ALL_SWINGS = MISS | FOUL | IN_PLAY
CONTACT    = FOUL | IN_PLAY

BAT_DIAMETER, BALL_DIAMETER = 2.61, 2.90
C = (BAT_DIAMETER / 2) + (BALL_DIAMETER / 2)
LA_CENTER = 20.0

N_FOLDS = 5
SEED    = 42

# Same architectures as final_merf_models.py -- reused verbatim so predicted_timing
# and predicted_offset mean the same thing they do in the rest of the project.
INT_Y_CONFIG = dict(
    name='int_y', outcome_col='int_y',
    features=['release_speed_c', 'plate_x_bat_flip', 'plate_z',
              'pfx_x_bat_flip', 'pfx_z', 'same_hand'],
    batter_agg_features=['bat_speed', 'release_speed'],
    xgb_params=dict(n_estimators=400, max_depth=5, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                    verbosity=0),
    max_iter=15,
)
BARREL_CONFIG = dict(
    name='barrel', outcome_col='barrel_distance_v2',
    features=['release_speed_c', 'plate_x_bat_flip', 'plate_z', 'intercept_y'],
    batter_agg_features=[],
    xgb_params=dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                    verbosity=0),
    max_iter=20,
)


def compute_barrel_distance_v2(df: pd.DataFrame) -> pd.Series:
    bd = pd.Series(np.nan, index=df.index, dtype=float)
    is_miss = df['description'].isin(MISS)
    is_contact = df['description'].isin(CONTACT)
    valid_miss = is_miss & df['miss_distance'].notna() & (df['miss_distance'] >= 0)
    bd.loc[valid_miss] = df.loc[valid_miss, 'miss_distance'] + C
    valid_contact = is_contact & df['launch_angle'].notna()
    la_rad = np.deg2rad(df.loc[valid_contact, 'launch_angle'])
    bd.loc[valid_contact] = np.abs(C * np.sin(la_rad - np.deg2rad(LA_CENTER)))
    return bd


def prepare_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    frames = []
    for f in files:
        part = pd.read_parquet(f)
        part['_source_file'] = f
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    print(f'Loaded {len(df):,} total pitches from {len(files)} file(s)')

    df['row_key'] = (df['game_pk'].astype(str) + '_' +
                      df['at_bat_number'].astype(str) + '_' +
                      df['pitch_number'].astype(str))
    assert df['row_key'].is_unique, 'row_key not unique -- check source files for overlap'

    # ── Swing population, bunts excluded (see data_audit.md §1) ─────────────────
    df = df[df['description'].isin(ALL_SWINGS) & ~df['description'].isin(BUNT_DESC)].copy()
    df['is_contact'] = df['description'].isin(CONTACT)
    print(f'Swing population (bunts excluded): {len(df):,} '
          f'(whiffs={len(df) - df["is_contact"].sum():,}, contact={df["is_contact"].sum():,})')

    for col in ['batter', 'pitcher', 'pitch_type', 'stand']:
        df[col] = df[col].astype(str)

    df['plate_x_bat_flip'] = df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
    df['pfx_x_bat_flip']   = df['pfx_x']   * df['stand'].map({'R': -1, 'L': 1}).fillna(1)

    rs_mean, rs_std = df['release_speed'].mean(), df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std

    df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})
    df['int_y'] = df['intercept_y']

    df['same_hand'] = (
        ((df['stand'] == 'R') & (df['p_throws'] == 'R')) |
        ((df['stand'] == 'L') & (df['p_throws'] == 'L'))
    ).astype(float)

    df['barrel_distance_v2'] = compute_barrel_distance_v2(df)

    return df.reset_index(drop=True)


def build_subset(df: pd.DataFrame, cfg: dict, row_mask: pd.Series):
    """
    Rows usable for TRAINING this model: need non-null target + all features.
    Returns the batter-aggregate table too (computed from TRAINING rows only)
    so build_predict_frame can reuse it on the held-out fold without leaking
    that fold's own bat_speed/release_speed values into its own features.
    """
    needed = [cfg['outcome_col'], 'batter'] + cfg['features']
    sub = df.loc[row_mask].dropna(subset=needed).copy()

    agg = None
    if cfg['batter_agg_features']:
        agg = (sub.groupby('batter')[cfg['batter_agg_features']]
                  .mean().add_prefix('batter_mean_').reset_index())
        sub = sub.merge(agg, on='batter', how='left')

    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt', drop_first=False)
    pt_dummies = [c for c in sub.columns if c.startswith('pt_')]
    x_cols = cfg['features'] + pt_dummies
    if cfg['batter_agg_features']:
        x_cols += [f'batter_mean_{f}' for f in cfg['batter_agg_features']]

    X = sub[x_cols].astype(float).reset_index(drop=True)
    Z = np.ones((len(sub), 1))
    clusters = sub['batter'].reset_index(drop=True)
    y = sub[cfg['outcome_col']].reset_index(drop=True)
    return X, Z, clusters, y, x_cols, sub, agg


def build_predict_frame(df: pd.DataFrame, cfg: dict, row_mask: pd.Series,
                        x_cols: list, train_agg: pd.DataFrame | None):
    """
    Rows to PREDICT for: need valid X_raw features only, target can be null.
    train_agg (batter-level means, computed on training rows only by
    build_subset) is used here instead of recomputing from `df` -- recomputing
    from the full df would let a held-out row's own bat_speed/release_speed
    leak into the batter_mean_* feature used to predict that very row.
    """
    needed = ['batter'] + cfg['features']
    sub = df.loc[row_mask].dropna(subset=needed).copy()

    if cfg['batter_agg_features']:
        sub = sub.merge(train_agg, on='batter', how='left')
        # Batters with zero training-fold rows (rare given row-level random
        # folds) get the population mean rather than a null feature.
        for f in cfg['batter_agg_features']:
            col = f'batter_mean_{f}'
            sub[col] = sub[col].fillna(train_agg[col].mean())

    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt', drop_first=False)
    for c in x_cols:
        if c not in sub.columns:
            sub[c] = 0.0
    X = sub[x_cols].astype(float).reset_index(drop=True)
    Z = np.ones((len(sub), 1))
    clusters = sub['batter'].reset_index(drop=True)
    return X, Z, clusters, sub


def run_oof(df: pd.DataFrame, cfg: dict, fold_col: str) -> pd.DataFrame:
    print(f'\n{"="*70}\nOOF generation: {cfg["name"]} -> predicted_{cfg["name"]}\n{"="*70}')
    results = []
    for fold in sorted(df[fold_col].dropna().unique()):
        train_mask = (df[fold_col] != fold) & df[fold_col].notna()
        test_mask  = (df[fold_col] == fold)

        X_tr, Z_tr, clusters_tr, y_tr, x_cols, _, train_agg = build_subset(df, cfg, train_mask)
        print(f'  fold {fold}: train n={len(y_tr):,}')

        xgb = XGBRegressor(**cfg['xgb_params'])
        # gll_early_stop_threshold: a real production run showed GLL for the
        # timing model oscillating in a ~0.006% band from iteration 3 onward
        # (1348422-1348506) while burning ~4+ min/iteration on the full data --
        # 5 folds x 15-20 max_iterations at that plateaued rate would take
        # 9-11+ hours for no real improvement. 1e-4 (0.01%) relative GLL
        # change stops once it's genuinely converged instead of grinding to
        # the fixed iteration ceiling regardless.
        mrf = MERF(fixed_effects_model=xgb, max_iterations=cfg['max_iter'],
                  gll_early_stop_threshold=1e-4)
        mrf.fit(X_tr, Z_tr, clusters_tr, y_tr)

        X_te, Z_te, clusters_te, sub_te = build_predict_frame(
            df, cfg, test_mask, x_cols, train_agg)
        fitted = mrf.predict(X_te, Z_te, clusters_te)
        print(f'          test n={len(fitted):,}  predicted_{cfg["name"]} '
              f'mean={fitted.mean():.3f} sd={fitted.std():.3f}')

        results.append(pd.DataFrame({
            'row_key': sub_te['row_key'].values,
            f'predicted_{cfg["name"]}': fitted,
        }))
    return pd.concat(results, ignore_index=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate OOF predicted_timing/predicted_offset')
    parser.add_argument('--sample-n', type=int, default=None,
                        help='Subsample N swings for a quick smoke test')
    parser.add_argument('--folds', type=int, default=N_FOLDS,
                        help=f'Number of CV folds (default {N_FOLDS})')
    parser.add_argument('--out-name', default='oof_predictions.parquet',
                        help='Output filename (default oof_predictions.parquet -- '
                             'use a different name for smoke-test runs so they '
                             "don't overwrite the real artifact)")
    args = parser.parse_args()

    df = prepare_data()
    if args.sample_n and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=SEED).reset_index(drop=True)
        print(f'Subsampled to {len(df):,} rows for smoke test')

    # ── Fold assignment: swing-level random split, same folds reused for both
    #    models so a given row's predicted_timing and predicted_offset always
    #    come from a model that excluded the SAME held-out rows. ─────────────
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    df['fold'] = np.nan
    for fold_id, (_, test_idx) in enumerate(kf.split(df)):
        df.loc[df.index[test_idx], 'fold'] = fold_id
    print(f'\nFold sizes: {df["fold"].value_counts().sort_index().to_dict()}')

    timing_oof = run_oof(df, INT_Y_CONFIG, 'fold')
    offset_oof = run_oof(df, BARREL_CONFIG, 'fold')

    out = (df[['row_key', 'batter', 'pitcher', 'is_contact', 'description', 'fold']]
           .merge(timing_oof, on='row_key', how='left')
           .merge(offset_oof, on='row_key', how='left'))

    print(f'\nFinal OOF frame: {len(out):,} rows')
    print(f'  predicted_int_y non-null: {out["predicted_int_y"].notna().sum():,}')
    print(f'  predicted_barrel non-null: {out["predicted_barrel"].notna().sum():,}')

    path = os.path.join(OUT_DIR, args.out_name)
    out.to_parquet(path, index=False)
    print(f'\n-> {path}')


if __name__ == '__main__':
    main()
