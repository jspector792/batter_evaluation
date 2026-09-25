"""
suite_data.py
==============
Season loading and feature construction for the miss-distance model suite.

Differences from oof_timing_offset.prepare_data() that matter
--------------------------------------------------------------
* `bat_speed >= 50` filter. Swings with NO tracked bat speed are also removed,
  because NaN fails the comparison. That is ~26k swings/season on top of the
  ~10k genuinely under 50 mph, and it is a real population change, so
  prepare_season() reports both counts rather than letting them vanish.

* Target `miss_distance_t`: 0 on contact, tracked `miss_distance` on whiffs.
  A whiff whose miss_distance is missing (~40/season) stays NaN and is dropped
  from offset-model training, rather than being silently coerced to 0 -- 0
  means "made contact" here, so imputing it would invert the label.

* `release_speed_c` is standardised with the TRAINING season's mean/sd, passed
  in via `scaler`. Re-standardising 2026 on its own moments would quietly
  redefine the feature between fit and evaluation.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd

from suite_config import (DATA_ROOT, ALL_SWINGS, BUNT_DESC, CONTACT,
                          MIN_BAT_SPEED, INTERCEPT_Y_COL, BAT_TRACKING)

warnings.filterwarnings('ignore')


def season_dir(season: int) -> str:
    return os.path.join(DATA_ROOT, f'all_pitches_{season}')


def prepare_season(season: int, scaler: dict | None = None,
                   verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Load one season's pitches and return (swing frame, scaler).

    `scaler` holds the release_speed mean/sd to apply. Pass None for the
    training season (moments are computed and returned); pass the training
    season's returned scaler for the evaluation season.
    """
    files = sorted(glob.glob(os.path.join(season_dir(season), '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {season_dir(season)}')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if verbose:
        print(f'[{season}] loaded {len(df):,} pitches from {len(files)} file(s)')

    df['row_key'] = (df['game_pk'].astype(str) + '_' +
                     df['at_bat_number'].astype(str) + '_' +
                     df['pitch_number'].astype(str))
    assert df['row_key'].is_unique, f'{season}: row_key not unique'

    # ── Swing population ──────────────────────────────────────────────────
    df = df[df['description'].isin(ALL_SWINGS) &
            ~df['description'].isin(BUNT_DESC)].copy()
    n_swings = len(df)

    n_missing_bs = int(df['bat_speed'].isna().sum())
    n_slow = int((df['bat_speed'] < MIN_BAT_SPEED).sum())
    df = df[df['bat_speed'] >= MIN_BAT_SPEED].copy()

    df['is_contact'] = df['description'].isin(CONTACT)
    if verbose:
        print(f'[{season}] swings (bunts excluded): {n_swings:,}')
        print(f'[{season}]   dropped {n_slow:,} under {MIN_BAT_SPEED:g} mph, '
              f'{n_missing_bs:,} with no tracked bat speed')
        print(f'[{season}]   kept {len(df):,} '
              f'(contact {df["is_contact"].sum():,}, '
              f'whiff {(~df["is_contact"]).sum():,})')

    for col in ['batter', 'pitcher', 'pitch_type', 'stand']:
        df[col] = df[col].astype(str)

    # pandas 3.0's astype(str) PRESERVES missing values rather than rendering
    # them as the string 'nan', so pitch_type can still hold NaN here. That
    # makes sorted(set(pitch_type)) raise on mixed float/str, and leaves the
    # one-hot encoder silently emitting an all-zero row. Give the handful of
    # untyped pitches an explicit category instead of dropping the swings
    # (2 rows in 2025, 80 in 2026).
    n_untyped = int(df['pitch_type'].isna().sum())
    if n_untyped:
        df['pitch_type'] = df['pitch_type'].fillna('UNK')
        if verbose:
            print(f'[{season}]   pitch_type: {n_untyped:,} untyped -> "UNK"')

    # ── Handedness-flipped geometry ───────────────────────────────────────
    flip = df['stand'].map({'R': -1, 'L': 1}).fillna(1)
    df['plate_x_bat_flip'] = df['plate_x'] * flip
    df['pfx_x_bat_flip'] = df['pfx_x'] * flip

    # ── Release speed, standardised on the TRAINING season ────────────────
    if scaler is None:
        scaler = dict(release_speed_mean=float(df['release_speed'].mean()),
                      release_speed_sd=float(df['release_speed'].std()))
        if verbose:
            print(f'[{season}] release_speed scaler fit: '
                  f'mean={scaler["release_speed_mean"]:.3f} '
                  f'sd={scaler["release_speed_sd"]:.3f}')
    df['release_speed_c'] = ((df['release_speed'] - scaler['release_speed_mean'])
                             / scaler['release_speed_sd'])

    df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})
    df['int_y'] = df['intercept_y']

    df['same_hand'] = (
        ((df['stand'] == 'R') & (df['p_throws'] == 'R')) |
        ((df['stand'] == 'L') & (df['p_throws'] == 'L'))
    ).astype(float)

    # ── Offset target: tracked miss distance, contact = 0 ─────────────────
    df['miss_distance_t'] = np.where(df['is_contact'], 0.0, df['miss_distance'])
    n_bad = int((~df['is_contact']) .sum() - df.loc[~df['is_contact'],
                                                    'miss_distance'].notna().sum())
    if verbose:
        zero_frac = float((df['miss_distance_t'] == 0).mean())
        print(f'[{season}]   miss_distance_t: {zero_frac:.1%} exactly zero, '
              f'{n_bad:,} whiff(s) left NaN (untracked)')
        for c in BAT_TRACKING:
            miss = int(df[c].isna().sum())
            if miss:
                print(f'[{season}]   {c}: {miss:,} null')

    df['game_date'] = pd.to_datetime(df['game_date'])
    df['season'] = season
    df['in_zone'] = df['zone'] <= 9

    return df.reset_index(drop=True), scaler


def load_train_and_eval(train_season: int, eval_season: int,
                        verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training season first (fits the scaler), evaluation season second."""
    train, scaler = prepare_season(train_season, None, verbose)
    eval_df, _ = prepare_season(eval_season, scaler, verbose)

    if verbose:
        shared = set(train['batter']) & set(eval_df['batter'])
        only_eval = set(eval_df['batter']) - set(train['batter'])
        n_rows_new = int(eval_df['batter'].isin(only_eval).sum())
        print(f'\nBatter overlap {train_season}->{eval_season}: '
              f'{len(shared):,} shared, {len(only_eval):,} new in {eval_season} '
              f'({n_rows_new:,} swings = {n_rows_new / len(eval_df):.1%}). '
              f'New batters have no random intercept and fall through to '
              f'fixed effects even in "carry" mode.')
    return train, eval_df
