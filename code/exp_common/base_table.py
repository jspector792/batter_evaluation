"""
base_table.py
==============
Builds the shared swing-level base table that all four exploratory
directions read. Written once to out/exp_common/, so the four analyses cannot
drift from each other's inputs.

Model selection (fixed by decision, not configurable here)
------------------------------------------------------------
    timing        m1_merf_base   -- MERF (XGBoost fixed effects + per-batter
                                    random intercept). The random-effect test
                                    showed the intercept is worth +0.057 R^2
                                    on held-out 2026 and transfers across
                                    seasons, so it is kept.
    miss distance m4_rf_missonly -- plain RandomForest fit on WHIFFS ONLY.
                                    Unbiased on the population where miss
                                    distance is genuinely tracked (bias
                                    +0.004 in vs +0.818 in for the
                                    all-swings model).

Out-of-fold discipline
----------------------
No in-sample prediction appears anywhere. For 2025 the columns are 5-fold
out-of-fold. For 2026 the models were fit on 2025 only and applied, so every
2026 row is fully held out.

carry vs context
----------------
    carry    the batter's 2025 random intercept is applied
    context  the intercept is suppressed; league-average batter

For the miss-distance model these are IDENTICAL by construction -- it is a
plain RF with no random effect. Both columns are still emitted so downstream
code can treat timing and offset uniformly. For 2025 only a single out-of-fold
column exists per target (a within-season OOF MERF prediction necessarily
carries the batter's own effect), so `*_context` is absent for that season and
the carry column is named `*_oof`.

The miss-distance target, and what is deliberately NOT here
-------------------------------------------------------------
`actual_miss_distance` is populated for WHIFFS ONLY, from the tracked Statcast
`miss_distance`. Contact events are left NaN rather than 0.

The launch-angle-derived pseudo miss distance (|C*sin(la - 20deg)|) used by
the original barrel_distance_v2 is NOT computed and NOT present. Per the
standing leakage rule it may not be a model input; since nothing downstream
needs it descriptively yet, omitting it entirely removes the chance of it
being picked up by accident.

Consequently `offset_residual_*` is defined on whiffs only. Any per-batter
offset aggregate is therefore a whiff-population statistic, and its
denominator differs from the timing residual's.
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
SUITE_CODE = os.path.join(BASE_DIR, 'code', 'rb_contact', 'suite')
RB_CODE = os.path.join(BASE_DIR, 'code', 'rb_contact')
for _p in (SUITE_CODE, RB_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from suite_config import (TRAIN_SEASON, EVAL_SEASON, OUT_DIR as SUITE_OUT,
                          DATA_ROOT, oof_path, eval_scored_path)
from suite_data import prepare_season
from followup_utils import walk_counts

warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(BASE_DIR, 'out', 'exp_common')
os.makedirs(OUT_DIR, exist_ok=True)

TIMING_MODEL = 'm1_merf_base'
OFFSET_MODEL = 'm4_rf_missonly'

CONTEXT_COLS = ['pitch_type', 'release_speed', 'pfx_x', 'pfx_z',
                'plate_x', 'plate_z', 'zone', 'in_zone',
                'plate_x_bat_flip', 'pfx_x_bat_flip', 'release_speed_c',
                'same_hand',
                # Spin is not in the "pitch context" list the directions ask
                # for, but it IS in model_utils.CONTINUOUS_RAW, so any
                # downstream refit of the stage-2 classifier needs it.
                'release_spin_rate', 'spin_axis']
SWING_COLS = ['bat_speed', 'swing_length', 'attack_angle',
              'attack_direction', 'swing_path_tilt']
ID_COLS = ['row_key', 'batter', 'pitcher', 'game_date', 'game_pk',
           'at_bat_number', 'pitch_number', 'description', 'is_contact',
           'stand', 'p_throws']


def table_path(season: int) -> str:
    return os.path.join(OUT_DIR, f'base_table_{season}.parquet')


def reconstruct_counts(season: int) -> pd.DataFrame:
    """Ball/strike count entering each pitch. Reuses the validated walker."""
    cols = ['game_pk', 'at_bat_number', 'pitch_number', 'description']
    files = sorted(glob.glob(os.path.join(DATA_ROOT, f'all_pitches_{season}',
                                          '*.parquet')))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                   ignore_index=True)
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'],
                        kind='mergesort')
    ab = (df['game_pk'].astype(np.int64) * 1000 +
          df['at_bat_number'].astype(np.int64)).values
    b, s = walk_counts(df['description'].values, ab)
    df['balls'], df['strikes'] = b, s
    df['count_str'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    # Ahead / even / behind from the BATTER's perspective.
    df['count_state'] = np.select(
        [df['balls'] > df['strikes'], df['balls'] == df['strikes']],
        ['ahead', 'even'], default='behind')
    df['row_key'] = (df['game_pk'].astype(str) + '_' +
                     df['at_bat_number'].astype(str) + '_' +
                     df['pitch_number'].astype(str))
    return df[['row_key', 'balls', 'strikes', 'count_str', 'count_state']]


def load_predictions(season: int) -> pd.DataFrame:
    """predicted_timing_* from the MERF, predicted_offset_* from the RF."""
    if season == TRAIN_SEASON:
        t = pd.read_parquet(oof_path(TIMING_MODEL),
                            columns=['row_key', 'predicted_timing'])
        t = t.rename(columns={'predicted_timing': 'predicted_timing_oof'})
        o = pd.read_parquet(oof_path(OFFSET_MODEL),
                            columns=['row_key', 'predicted_offset'])
        o = o.rename(columns={'predicted_offset': 'predicted_offset_oof'})
        return t.merge(o, on='row_key', how='outer')

    t = pd.read_parquet(eval_scored_path(TIMING_MODEL),
                        columns=['row_key', 'predicted_timing_carry',
                                 'predicted_timing_context'])
    o = pd.read_parquet(eval_scored_path(OFFSET_MODEL),
                        columns=['row_key', 'predicted_offset_carry',
                                 'predicted_offset_context'])
    return t.merge(o, on='row_key', how='outer')


def build(season: int, verbose: bool = True) -> pd.DataFrame:
    # The evaluation season must be standardised with the TRAINING season's
    # release-speed moments, or release_speed_c means something different
    # between fit and use.
    if season == TRAIN_SEASON:
        df, _ = prepare_season(season, None, verbose)
    else:
        _, scaler = prepare_season(TRAIN_SEASON, None, False)
        df, _ = prepare_season(season, scaler, verbose)

    keep = ID_COLS + CONTEXT_COLS + SWING_COLS + ['int_y', 'miss_distance_t']
    df = df[[c for c in keep if c in df.columns]].copy()

    df = df.rename(columns={'int_y': 'actual_timing'})
    # Tracked miss distance, whiffs only. Contact stays NaN -- never 0, and
    # never the launch-angle-derived proxy.
    df['actual_miss_distance'] = np.where(df['is_contact'], np.nan,
                                          df['miss_distance_t'])
    df = df.drop(columns=['miss_distance_t'])

    df = df.merge(reconstruct_counts(season), on='row_key', how='left')
    df = df.merge(load_predictions(season), on='row_key', how='left')

    modes = ['oof'] if season == TRAIN_SEASON else ['carry', 'context']
    for m in modes:
        tp, op = f'predicted_timing_{m}', f'predicted_offset_{m}'
        if tp in df.columns:
            df[f'timing_residual_{m}'] = df['actual_timing'] - df[tp]
        if op in df.columns:
            # Whiffs only: actual_miss_distance is NaN on contact, so this
            # propagates NaN there by construction rather than by a filter
            # someone could forget to apply.
            df[f'offset_residual_{m}'] = df['actual_miss_distance'] - df[op]

    df['season'] = season
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number',
                         'pitch_number'], kind='mergesort').reset_index(drop=True)

    if verbose:
        print(f'\n[{season}] base table: {len(df):,} swings, '
              f'{len(df.columns)} columns, {df["batter"].nunique()} batters')
        for m in modes:
            for q in ('timing', 'offset'):
                c = f'{q}_residual_{m}'
                if c in df.columns:
                    print(f'  {c}: {df[c].notna().sum():,} non-null')
        if season != TRAIN_SEASON:
            d = (df['predicted_offset_carry'] -
                 df['predicted_offset_context']).abs().max()
            print(f'  offset carry vs context max|diff| = {d:.2e} '
                  f'(0 expected: RF has no random effect)')
            d2 = (df['predicted_timing_carry'] -
                  df['predicted_timing_context']).std()
            print(f'  timing carry-context sd = {d2:.4f} in '
                  f'(non-zero expected: MERF random intercept)')
    return df


def load(season: int) -> pd.DataFrame:
    """Read the cached table, building it first if absent."""
    p = table_path(season)
    if not os.path.exists(p):
        build(season).to_parquet(p, index=False)
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser(description='Build the shared base table')
    ap.add_argument('--seasons', nargs='*', type=int,
                    default=[TRAIN_SEASON, EVAL_SEASON])
    args = ap.parse_args()
    for s in args.seasons:
        df = build(s)
        df.to_parquet(table_path(s), index=False)
        print(f'-> {table_path(s)}')


if __name__ == '__main__':
    main()
