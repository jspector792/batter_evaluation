"""
write_barrel_swing_predictions.py
==================================
Loads the three saved barrel distance swing variant MERF models and writes
their predictions as new columns into all parquet files in DATA_DIR.

Models loaded
-------------
  var_1.joblib  — base + attack_angle
  var_2.joblib  — base + swing_path_tilt
  var_3.joblib  — base + attack_angle + swing_path_tilt

New columns written to each parquet file
-----------------------------------------
  barrel_pred_v1   — fitted values from var_1
  barrel_pred_v2   — fitted values from var_2
  barrel_pred_v3   — fitted values from var_3

All prediction columns are NaN on non-swing rows and on swing rows missing
any required feature for that variant.

Preprocessing applied (must match training exactly)
----------------------------------------------------
  release_speed_c     = (release_speed - pop_mean) / pop_sd
                        mean and sd computed on the FULL dataset before
                        any filtering, matching prepare_data() in the
                        variants script
  plate_x_bat_flip    = plate_x * stand.map({'R': -1, 'L': 1})
  intercept_y         = renamed from INTERCEPT_Y_COL
  pt_* dummies        = pd.get_dummies(pitch_type, prefix='pt')
                        any pitch type not seen in training is filled with 0
  attack_angle        = used as-is (natural scale, degrees)
  swing_path_tilt     = used as-is (natural scale, degrees)

Parquet writeback strategy
--------------------------
Each file is tracked via _source_file during the full-dataset load.
Predictions are assigned using .values positional alignment after
resetting index within each file's row block, avoiding the index
mismatch bug that caused all-NaN predictions previously.

Usage
-----
  python write_barrel_swing_predictions.py [--dry-run]

  --dry-run   Compute predictions and print coverage but do not write files.

Requires: merf, xgboost, joblib, pandas, numpy
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data", "all_pitches_2025")
CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "out", "exploratory", "final_models_variants")

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

MISS       = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL       = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY    = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS = IN_PLAY | FOUL | MISS

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    'v1': os.path.join(CODE_DIR, 'var_1.joblib'),
    'v2': os.path.join(CODE_DIR, 'var_2.joblib'),
    'v3': os.path.join(CODE_DIR, 'var_3.joblib'),
}

PRED_COLS = {
    'v1': 'barrel_pred_v1',
    'v2': 'barrel_pred_v2',
    'v3': 'barrel_pred_v3',
}

# ── Base features shared by all variants ──────────────────────────────────────
BASE_FEATURES = [
    'release_speed_c',
    'plate_x_bat_flip',
    'plate_z',
    'intercept_y',
]

# ── Extra features per variant ────────────────────────────────────────────────
EXTRA_FEATURES = {
    'v1': ['attack_angle'],
    'v2': ['swing_path_tilt'],
    'v3': ['attack_angle', 'swing_path_tilt'],
}

CLUSTER_COL = 'batter'


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load all parquet files into a single DataFrame, tracking source file
    for writeback. Returns (df, files).

    ignore_index=True here because we use _source_file + positional alignment
    for writeback rather than index-based joining — avoids the all-NaN bug.
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    frames = []
    for f in files:
        part = pd.read_parquet(f)
        part['_source_file'] = f
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    print(f'Loaded {len(df):,} rows from {len(files)} file(s)')
    return df, files


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations in the same order as prepare_data()
    in barrel_distance_merf_swing_variants.py.

    release_speed_c is computed from the FULL dataset population mean and sd
    before any row filtering — this is critical to match training exactly.
    """
    df = df.copy()

    # ── String types ──────────────────────────────────────────────────────────
    for col in ['batter', 'pitch_type', 'stand']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── plate_x_bat_flip ──────────────────────────────────────────────────────
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    print('plate_x_bat_flip: applied')

    # ── release_speed_c ───────────────────────────────────────────────────────
    # Computed on the full dataset (all rows, no filtering) to match training.
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.4f}, sd={rs_std:.4f}')

    # ── intercept_y ───────────────────────────────────────────────────────────
    if INTERCEPT_Y_COL in df.columns:
        df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})
        print('intercept_y: renamed from raw column')
    elif 'intercept_y' in df.columns:
        print('intercept_y: already present')
    else:
        print('WARNING: intercept_y not found — predictions will be NaN '
              'for rows requiring it')

    # ── pt_ dummies ───────────────────────────────────────────────────────────
    # Computed on the full dataset so all pitch types are represented.
    # Missing dummies will be filled with 0 per-variant after loading the model.
    df = pd.get_dummies(df, columns=['pitch_type'], prefix='pt',
                        drop_first=False)
    pt_cols = [c for c in df.columns if c.startswith('pt_')]
    print(f'pt_ dummies: {len(pt_cols)} pitch types encoded')

    # ── attack_angle, swing_path_tilt ─────────────────────────────────────────
    # Used on natural scale (degrees) — no transformation needed.
    for col in ['attack_angle', 'swing_path_tilt']:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f'{col}: {n:,} non-null ({100*n/len(df):.1f}%)')
        else:
            print(f'WARNING: {col!r} not found — V1/V3 predictions will be NaN')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREDICT WITH ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def predict_variant(df: pd.DataFrame, variant: str, mrf) -> pd.Series:
    """
    Generate predictions for one variant on the full dataset.
    Only swing rows with all required features get a prediction; others NaN.

    Uses positional alignment (.values) when writing back into the result
    Series to avoid index mismatch issues.
    """
    extra    = EXTRA_FEATURES[variant]
    features = BASE_FEATURES + extra

    # Get the pt_ columns the model was trained on from its XGBoost internals
    # feature_names_in_ is set by sklearn-compatible estimators on fit
    try:
        trained_pt_cols = [
            c for c in mrf.trained_fe_model.feature_names_in_
            if c.startswith('pt_')
        ]
    except AttributeError:
        # Fallback: use whatever pt_ cols are in df
        trained_pt_cols = [c for c in df.columns if c.startswith('pt_')]
        print(f'  [{variant}] WARNING: could not read trained feature names '
              f'from model — using all pt_ cols in data')

    all_features = features + trained_pt_cols

    # Add any missing pt_ columns as zero (pitch type not seen in this subset)
    for col in trained_pt_cols:
        if col not in df.columns:
            df[col] = 0
            print(f'  [{variant}] Added missing dummy column: {col}')

    # Swing rows only
    swing_mask = df['description'].isin(ALL_SWINGS)

    # Valid rows: swing + all features present
    valid_mask = swing_mask & df[all_features].notna().all(axis=1)
    n_swing    = swing_mask.sum()
    n_valid    = valid_mask.sum()
    n_missing  = n_swing - n_valid
    print(f'  [{variant}] swing rows={n_swing:,}  '
          f'valid={n_valid:,}  missing features={n_missing:,}')

    result = pd.Series(np.nan, index=df.index, name=PRED_COLS[variant])

    if n_valid == 0:
        print(f'  [{variant}] WARNING: no valid rows — check feature columns')
        return result

    sub_valid = df[valid_mask]

    X        = sub_valid[all_features].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub_valid), 1))
    clusters = sub_valid[CLUSTER_COL].astype(str).reset_index(drop=True)

    fitted = mrf.predict(X, Z, clusters)

    # Assign using positional index of valid rows — .values avoids
    # the label-alignment bug that caused all-NaN predictions previously
    result.iloc[np.where(valid_mask)[0]] = fitted

    print(f'  [{variant}] R² check: {_r2(df, result, variant):.4f}')
    return result


def _r2(df: pd.DataFrame, pred: pd.Series, variant: str) -> float:
    """Quick R² against barrel_distance_v2 where both are non-null."""
    outcome = 'barrel_distance_v2'
    if outcome not in df.columns:
        return float('nan')
    mask = pred.notna() & df[outcome].notna()
    if mask.sum() < 2:
        return float('nan')
    y      = df.loc[mask, outcome].values
    y_hat  = pred.loc[mask].values
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ══════════════════════════════════════════════════════════════════════════════
# 4. WRITE PREDICTIONS BACK TO PARQUET
# ══════════════════════════════════════════════════════════════════════════════

def write_to_parquet(df: pd.DataFrame, files: list[str], dry_run: bool):
    """
    Write prediction columns back to each source parquet file.
    Uses _source_file + positional alignment (.values) to correctly
    map rows in the concatenated df back to their source file.
    """
    pred_col_names = list(PRED_COLS.values())

    print(f'\nWriting predictions to {len(files)} parquet file(s)...')
    for fpath in files:
        part     = pd.read_parquet(fpath)
        mask     = df['_source_file'] == fpath
        df_part  = df[mask].reset_index(drop=True)

        if len(df_part) != len(part):
            print(f'  WARNING: row count mismatch for {os.path.basename(fpath)} '
                  f'(parquet={len(part):,}, df_part={len(df_part):,}) — skipping')
            continue

        for col in pred_col_names:
            if col in df_part.columns:
                # .values ensures positional alignment, no index label matching
                part[col] = df_part[col].values

        if not dry_run:
            part.to_parquet(fpath, index=False)

        for col in pred_col_names:
            n = part[col].notna().sum() if col in part.columns else 0
            print(f'  {os.path.basename(fpath)} | {col}: {n:,} non-null')

    if dry_run:
        print('\n[dry-run] no files written')


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Write barrel swing variant predictions to parquet files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and report predictions without writing')
    args = parser.parse_args()

    # ── Verify model files exist ───────────────────────────────────────────────
    for variant, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Model file not found: {path}\n'
                f'Expected saved models in CODE_DIR: {CODE_DIR}'
            )
    print('Model files found:')
    for variant, path in MODEL_PATHS.items():
        size_mb = os.path.getsize(path) / 1e6
        print(f'  {variant}: {path}  ({size_mb:.1f} MB)')

    # ── Load data ──────────────────────────────────────────────────────────────
    df, files = load_data()

    # ── Preprocess ────────────────────────────────────────────────────────────
    print('\nPreprocessing...')
    df = preprocess(df)

    # ── Load models and predict ────────────────────────────────────────────────
    for variant, model_path in MODEL_PATHS.items():
        print(f'\n{"="*60}')
        print(f'Variant {variant.upper()} — {model_path}')
        print('='*60)

        mrf = joblib.load(model_path)
        print(f'  Model loaded  '
              f'(EM iterations run: {len(mrf.gll_history)}, '
              f'final GLL: {mrf.gll_history[-1]:.2f})')

        pred = predict_variant(df, variant, mrf)
        df[PRED_COLS[variant]] = pred

    # ── Write back ────────────────────────────────────────────────────────────
    write_to_parquet(df, files, dry_run=args.dry_run)

    print('\nDone.')


if __name__ == '__main__':
    main()