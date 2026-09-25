"""
hitters.py
===========
Shared test-hitter selection for Directions 1 and 4, so both describe the
same players.

Selection is deliberately spread rather than top-N by playing time: the point
is to cover a range of swing types, not to build a leaderboard. Among batters
clearing a swing-count floor, the pool is crossed by handedness x bat-speed
tercile x contact-rate tercile, and the highest-volume batter in each
non-empty combination is taken until the quota is filled. That guarantees
both hands, sluggers and contact types, and fast and slow bats.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(BASE_DIR, 'data')

MIN_SWINGS = 250


def batter_names(season: int) -> pd.Series:
    """batter id (str) -> display name."""
    p = os.path.join(DATA_DIR, f'batter_stats_{season}.csv')
    if not os.path.exists(p):
        return pd.Series(dtype=object)
    d = pd.read_csv(p, usecols=['batter_id', 'batter_name'], low_memory=False)
    d = d.dropna(subset=['batter_id', 'batter_name']).drop_duplicates('batter_id')
    d['batter_id'] = d['batter_id'].astype('Int64').astype(str)
    return d.set_index('batter_id')['batter_name']


def select_test_hitters(df: pd.DataFrame, names: pd.Series,
                        n: int = 16, min_swings: int = MIN_SWINGS,
                        seed: int = 42) -> pd.DataFrame:
    g = df.groupby('batter')
    prof = pd.DataFrame({
        'n_swings': g.size(),
        'contact_pct': g['is_contact'].mean(),
        'bat_speed': g['bat_speed'].mean(),
        'stand': g['stand'].agg(lambda s: s.mode().iat[0]),
    })
    prof = prof[prof['n_swings'] >= min_swings].copy()
    mapped = pd.Series(prof.index.map(names), index=prof.index)
    prof['name'] = mapped.where(mapped.notna(),
                                'batter ' + prof.index.astype(str))

    prof['bs_t'] = pd.qcut(prof['bat_speed'], 3, labels=['slow', 'mid', 'fast'])
    prof['ct_t'] = pd.qcut(prof['contact_pct'], 3,
                           labels=['low-contact', 'mid', 'high-contact'])

    picks, rng = [], np.random.default_rng(seed)
    combos = [(h, b, c) for h in ['R', 'L']
              for b in ['fast', 'mid', 'slow']
              for c in ['low-contact', 'mid', 'high-contact']]
    rng.shuffle(combos)
    # One pass taking the highest-volume batter per cell, then fill by volume.
    for h, b, c in combos:
        if len(picks) >= n:
            break
        cell = prof[(prof['stand'] == h) & (prof['bs_t'] == b) &
                    (prof['ct_t'] == c)]
        cell = cell[~cell.index.isin(picks)]
        if not cell.empty:
            picks.append(cell['n_swings'].idxmax())
    if len(picks) < n:
        rest = prof[~prof.index.isin(picks)].nlargest(n - len(picks),
                                                      'n_swings')
        picks += list(rest.index)

    out = prof.loc[picks, ['name', 'stand', 'n_swings', 'contact_pct',
                           'bat_speed', 'bs_t', 'ct_t']]
    out = out.reset_index().rename(columns={'index': 'batter'})
    if 'batter' not in out.columns:
        out = out.rename(columns={out.columns[0]: 'batter'})
    return out.sort_values('name').reset_index(drop=True)
