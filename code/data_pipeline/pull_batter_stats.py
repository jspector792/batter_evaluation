"""
pull_batter_stats.py
====================
Pulls and merges batter-level statistics from multiple pybaseball endpoints
for a given date range, outputting a single wide CSV with both batter_id
(MLBAM) and batter_name columns.

Data sources
------------
1. FanGraphs batting stats (batting_stats)
      Traditional + advanced: AVG, OBP, SLG, wOBA, wRC+, WAR, K%, BB%,
      BABIP, ISO, LD%, GB%, FB%, Hard%, Soft%, Spd, etc.
      Accepts start_dt / end_dt directly.

2. Statcast exit velocity / barrels (statcast_batter_exitvelo_barrels)
      avg_hit_speed, max_hit_speed, avg_hit_angle, brl_percent,
      brl_pa, anglesweetspotpercent, etc.
      Season-level only (year argument). If the date range spans two
      seasons both are pulled and the appropriate season is kept.

3. Statcast expected stats (statcast_batter_expected_stats)
      xBA, xSLG, xwOBA, xERA (as batter), exit_velocity_avg,
      launch_angle_avg, sweet_spot_percent, barrel_batted_rate.
      Season-level only.

4. Statcast percentile ranks (statcast_batter_percentile_ranks)
      Percentile ranks for EV, Barrel%, Hard Hit%, xBA, xSLG, xwOBA,
      Chase%, Whiff%, Sprint Speed, etc.
      Season-level only.

ID bridging
-----------
FanGraphs uses its own integer player IDs (IDfg). Statcast uses MLBAM IDs.
The Chadwick register bridges them: key_fangraphs → key_mlbam.
Both batter_id (MLBAM) and batter_name are added to all sources before merging.

Usage
-----
  python pull_batter_stats.py --start 2025-04-01 --end 2025-09-30
  python pull_batter_stats.py --start 2025-04-01 --end 2025-09-30 --min-pa 100
  python pull_batter_stats.py --start 2024-04-01 --end 2024-09-30 --out my_stats.csv

Arguments
---------
  --start     Start date (YYYY-MM-DD), required
  --end       End date (YYYY-MM-DD), required
  --min-pa    Minimum plate appearances to include (default: 50)
  --out       Output CSV filename (default: batter_stats_{start}_{end}.csv)
  --no-cache  Re-download Chadwick register even if cached

Outputs
-------
  {OUT_DIR}/batter_stats_{start}_{end}.csv    merged wide stats table
  {OUT_DIR}/batter_stats_{start}_{end}_coverage.txt  source coverage report

Requires: pybaseball, pandas, numpy
  pip install pybaseball
"""

import os, argparse, warnings, time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pybaseball
from pybaseball import (
    batting_stats,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_batter_percentile_ranks,
    chadwick_register,
)

warnings.filterwarnings('ignore')
pybaseball.cache.enable()   # cache API responses to avoid re-downloading

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Minimum PA for FanGraphs pull ─────────────────────────────────────────────
# This is the pybaseball qual argument; can be overridden by --min-pa
DEFAULT_MIN_PA = 50


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHADWICK REGISTER — ID BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

def load_register() -> pd.DataFrame:
    """
    Load the Chadwick register and return a clean bridge DataFrame with:
      key_mlbam    — MLBAM integer ID (used by Statcast)
      key_fangraphs — FanGraphs integer ID
      full_name    — 'FirstName LastName'
    Rows missing either key are dropped.
    """
    print('Loading Chadwick register...')
    reg = chadwick_register()

    reg = reg[['key_mlbam', 'key_fangraphs',
               'name_first', 'name_last']].copy()
    reg = reg.dropna(subset=['key_mlbam', 'key_fangraphs'])
    reg['key_mlbam']     = reg['key_mlbam'].astype(int)
    reg['key_fangraphs'] = reg['key_fangraphs'].astype(int)
    reg['full_name'] = (reg['name_first'].str.strip() + ' '
                        + reg['name_last'].str.strip())

    print(f'  Register loaded: {len(reg):,} entries with both MLBAM and FG IDs')
    return reg


def add_ids(df: pd.DataFrame, reg: pd.DataFrame,
            join_key: str, id_col_out: str = 'batter_id') -> pd.DataFrame:
    """
    Add batter_id (MLBAM) and batter_name to df by joining on join_key.

    join_key: column in df containing the key to join on
              'key_fangraphs' → df has FG IDs, we add MLBAM
              'key_mlbam'     → df has MLBAM IDs, we add name only
    """
    if join_key == 'key_fangraphs':
        bridge = reg[['key_fangraphs', 'key_mlbam', 'full_name']].rename(
            columns={'key_mlbam': id_col_out, 'full_name': 'batter_name'})
        df = df.merge(bridge, on='key_fangraphs', how='left')
    elif join_key == 'key_mlbam':
        bridge = reg[['key_mlbam', 'full_name']].rename(
            columns={'key_mlbam': id_col_out, 'full_name': 'batter_name'})
        df = df.merge(bridge, on=id_col_out, how='left')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FANGRAPHS BATTING STATS
# ══════════════════════════════════════════════════════════════════════════════

def _finish_bref_season_fallback(fg: pd.DataFrame,
                                  reg: pd.DataFrame) -> pd.DataFrame:
    """
    Finish the batting_stats_bref fallback used when FanGraphs is unreachable.

    batting_stats_bref returns season totals keyed by 'mlbID' (an MLBAM ID),
    so batter_id comes straight off that column and only the name needs to be
    joined from the register. Season stats are BRef's, not FanGraphs' — the
    advanced FanGraphs-only columns (wRC+, WAR, Spd, batted-ball rates) are
    NOT recoverable this way, so expect a much narrower fg_ block than a
    successful FanGraphs pull.
    """
    id_col = next((c for c in ['mlbID', 'mlb_ID', 'mlbam_id']
                   if c in fg.columns), None)
    if id_col is None:
        print(f'  WARNING: BRef fallback has no MLBAM ID column. '
              f'Columns: {fg.columns.tolist()}')
        print('  NOTE: output will have ZERO fg_ columns.')
        return pd.DataFrame()

    fg = fg.rename(columns={id_col: 'batter_id'})
    fg['batter_id'] = pd.to_numeric(fg['batter_id'],
                                     errors='coerce').astype('Int64')
    fg = fg.dropna(subset=['batter_id'])

    # Aggregate away multi-team rows: BRef lists a traded player once per
    # stint, and a duplicated batter_id would fan out the downstream merge.
    pa_col = next((c for c in ['PA', 'G'] if c in fg.columns), None)
    if pa_col:
        fg = (fg.sort_values(pa_col, ascending=False)
                .drop_duplicates(subset='batter_id', keep='first')
                .reset_index(drop=True))

    fg = add_ids(fg, reg, join_key='key_mlbam')

    identity = {'batter_id', 'batter_name', 'Name', 'Team', 'Tm',
                'Season', 'Age', 'Lev'}
    fg = fg.rename(columns={
        c: f'fg_{c}' for c in fg.columns
        if c not in identity and not c.startswith('fg_')
    })

    print(f'  BRef fallback usable: {len(fg):,} rows, '
          f'{len([c for c in fg.columns if c.startswith("fg_")])} fg_ columns')
    print('  NOTE: these are Baseball Reference season stats, NOT FanGraphs. '
          'wRC+, WAR and FanGraphs batted-ball rates are unavailable.')
    return fg


def pull_fangraphs(start_dt: str, end_dt: str,
                   min_pa: int, reg: pd.DataFrame) -> pd.DataFrame:
    """
    Pull FanGraphs batting stats for the season(s) covered by the date range.
    batting_stats() accepts season integers only, not date strings.
    If the range spans two seasons both are pulled as separate rows.
    """
    start_year = datetime.strptime(start_dt, '%Y-%m-%d').year
    end_year   = datetime.strptime(end_dt,   '%Y-%m-%d').year

    print(f'\nPulling FanGraphs batting stats (seasons {start_year}–{end_year})...')
    try:
        fg = batting_stats(
            start_season = start_year,
            end_season   = end_year,
            qual         = min_pa,
            ind          = 1,
        )
        print(f'  FanGraphs: {len(fg):,} rows, {len(fg.columns)} columns')
    except Exception as e:
        print(f'  FanGraphs (legacy endpoint) failed: {e}')
        print('  Trying FanGraphs via batting_stats_bref fallback...')
        try:
            from pybaseball import batting_stats_bref
            fg = batting_stats_bref(start_year)
            fg['Season'] = start_year
            print(f'  BRef season fallback: {len(fg):,} rows')
        except Exception as e2:
            print(f'  FanGraphs fallback also failed: {e2}')
            print('  NOTE: output will have ZERO fg_ columns.')
            return pd.DataFrame()

        # The fallback is keyed by MLBAM ID ('mlbID'), not by a FanGraphs ID,
        # so it cannot go through the key_fangraphs bridge below. Handle it
        # here and return early. Columns still get the fg_ prefix so the
        # merge and the coverage report treat this as the same source slot.
        return _finish_bref_season_fallback(fg, reg)

    # FanGraphs uses 'IDfg' as the player ID column
    id_col = None
    for candidate in ['IDfg', 'idfg', 'playerid', 'PlayerID']:
        if candidate in fg.columns:
            id_col = candidate
            break

    if id_col is None:
        print(f'  WARNING: could not find FG ID column. Columns: {fg.columns.tolist()}')
        print('  NOTE: output will have ZERO fg_ columns.')
        return pd.DataFrame()

    fg = fg.rename(columns={id_col: 'key_fangraphs'})
    fg['key_fangraphs'] = pd.to_numeric(fg['key_fangraphs'],
                                         errors='coerce').astype('Int64')
    fg = fg.dropna(subset=['key_fangraphs'])
    fg['key_fangraphs'] = fg['key_fangraphs'].astype(int)

    # Add MLBAM ID and name
    fg = add_ids(fg, reg, join_key='key_fangraphs')

    # Prefix FG-specific columns to avoid collision on merge
    # Keep identity columns unprefixed
    identity = {'batter_id', 'batter_name', 'key_fangraphs', 'Name',
                 'Team', 'Season', 'Age'}
    fg = fg.rename(columns={
        c: f'fg_{c}' for c in fg.columns
        if c not in identity and not c.startswith('fg_')
    })

    n_matched = fg['batter_id'].notna().sum()
    print(f'  Matched to MLBAM ID: {n_matched:,} / {len(fg):,}')
    return fg

def pull_bref_range(start_dt: str, end_dt: str,
                    reg: pd.DataFrame) -> pd.DataFrame:
    """
    Pull Baseball Reference batting stats for the exact date range.
    Returns traditional slash stats (AVG, OBP, SLG, HR, RBI, etc.)
    keyed by player name — no FG or MLBAM ID available directly.
    We match on name via the register as a best-effort bridge.
    """
    from pybaseball import batting_stats_range

    print(f'\nPulling BRef batting stats range ({start_dt} → {end_dt})...')
    try:
        br = batting_stats_range(start_dt, end_dt)
        print(f'  BRef: {len(br):,} rows')
    except Exception as e:
        print(f'  BRef range pull failed: {e}')
        return pd.DataFrame()

    # BRef has no numeric player ID — match on full name as fallback
    # Build name → MLBAM map from register
    name_map = reg.set_index('full_name')['key_mlbam'].to_dict()
    br['batter_id']   = br['Name'].map(name_map)
    br['batter_name'] = br['Name']

    identity = {'batter_id', 'batter_name', 'Name', 'Age', 'Team'}
    br = br.rename(columns={
        c: f'br_{c}' for c in br.columns
        if c not in identity and not c.startswith('br_')
    })

    matched = br['batter_id'].notna().sum()
    print(f'  Matched to MLBAM ID: {matched:,} / {len(br):,} '
          f'(unmatched rows kept with NaN batter_id)')
    # Dedup on batter_id keeping the row with more PA
    # Players who changed teams appear once per team in BRef range data
    pa_col = next((c for c in br.columns if c in ('br_PA', 'PA')), None)
    if pa_col and br['batter_id'].notna().any():
        br = (br.sort_values(pa_col, ascending=False)
                .drop_duplicates(subset='batter_id', keep='first')
                .reset_index(drop=True))
        print(f'  After dedup: {len(br):,} rows')

    return br


# ══════════════════════════════════════════════════════════════════════════════
# 3. STATCAST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def years_in_range(start_dt: str, end_dt: str) -> list[int]:
    """Return list of calendar years covered by the date range."""
    start_year = datetime.strptime(start_dt, '%Y-%m-%d').year
    end_year   = datetime.strptime(end_dt,   '%Y-%m-%d').year
    return list(range(start_year, end_year + 1))


def pull_statcast_ev_barrels(years: list[int],
                              reg: pd.DataFrame) -> pd.DataFrame:
    """
    statcast_batter_exitvelo_barrels: EV, barrels, sweet spot, hard hit.
    Key columns: player_id (MLBAM), avg_hit_speed, max_hit_speed,
                 avg_hit_angle, brl_percent, brl_pa,
                 anglesweetspotpercent, ev95percent, ev95plus_pa.
    """
    print('\nPulling Statcast EV/barrels...')
    frames = []
    for yr in years:
        try:
            df = statcast_batter_exitvelo_barrels(yr, minBBE=1)
            df['season'] = yr
            frames.append(df)
            print(f'  {yr}: {len(df):,} rows')
            time.sleep(0.5)   # be polite to the API
        except Exception as e:
            print(f'  {yr} failed: {e}')

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Standardise ID column name
    for candidate in ['player_id', 'batter', 'IDfg']:
        if candidate in out.columns:
            out = out.rename(columns={candidate: 'batter_id'})
            break
    out['batter_id'] = pd.to_numeric(out['batter_id'],
                                      errors='coerce').astype('Int64')

    out = add_ids(out, reg, join_key='key_mlbam')

    # Prefix
    identity = {'batter_id', 'batter_name', 'season',
                 'last_name', 'first_name', 'player_name'}
    out = out.rename(columns={
        c: f'ev_{c}' for c in out.columns
        if c not in identity and not c.startswith('ev_')
    })
    print(f'  Total rows: {len(out):,}')
    return out


def pull_statcast_expected(years: list[int],
                            reg: pd.DataFrame) -> pd.DataFrame:
    """
    statcast_batter_expected_stats: xBA, xSLG, xwOBA, exit_velocity_avg,
    launch_angle_avg, sweet_spot_percent, barrel_batted_rate.
    """
    print('\nPulling Statcast expected stats...')
    frames = []
    for yr in years:
        try:
            df = statcast_batter_expected_stats(yr, minPA=1)
            df['season'] = yr
            frames.append(df)
            print(f'  {yr}: {len(df):,} rows')
            time.sleep(0.5)
        except Exception as e:
            print(f'  {yr} failed: {e}')

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    for candidate in ['player_id', 'batter', 'IDfg']:
        if candidate in out.columns:
            out = out.rename(columns={candidate: 'batter_id'})
            break
    out['batter_id'] = pd.to_numeric(out['batter_id'],
                                      errors='coerce').astype('Int64')

    out = add_ids(out, reg, join_key='key_mlbam')

    identity = {'batter_id', 'batter_name', 'season',
                 'last_name', 'first_name', 'player_name'}
    out = out.rename(columns={
        c: f'xst_{c}' for c in out.columns
        if c not in identity and not c.startswith('xst_')
    })
    print(f'  Total rows: {len(out):,}')
    return out


def pull_statcast_percentiles(years: list[int],
                               reg: pd.DataFrame) -> pd.DataFrame:
    """
    statcast_batter_percentile_ranks: percentile ranks for EV, Barrel%,
    Hard Hit%, xBA, xSLG, xwOBA, Chase%, Whiff%, Sprint Speed, etc.
    """
    print('\nPulling Statcast percentile ranks...')
    frames = []
    for yr in years:
        try:
            df = statcast_batter_percentile_ranks(yr)
            df['season'] = yr
            frames.append(df)
            print(f'  {yr}: {len(df):,} rows')
            time.sleep(0.5)
        except Exception as e:
            print(f'  {yr} failed: {e}')

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    for candidate in ['player_id', 'batter', 'IDfg']:
        if candidate in out.columns:
            out = out.rename(columns={candidate: 'batter_id'})
            break
    out['batter_id'] = pd.to_numeric(out['batter_id'],
                                      errors='coerce').astype('Int64')

    out = add_ids(out, reg, join_key='key_mlbam')

    identity = {'batter_id', 'batter_name', 'season',
                 'last_name', 'first_name', 'player_name'}
    out = out.rename(columns={
        c: f'pct_{c}' for c in out.columns
        if c not in identity and not c.startswith('pct_')
    })
    print(f'  Total rows: {len(out):,}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_all(fg: pd.DataFrame,
              br: pd.DataFrame,
              ev: pd.DataFrame,
              xst: pd.DataFrame,
              pct: pd.DataFrame,
              years: list[int]) -> pd.DataFrame:
    """
    Left-join all sources on batter_id (+ season where available).
    FanGraphs is the spine — every batter in FG appears in the output.
    Statcast sources add columns where available; NaN where not matched.

    Multi-year handling: if the date range spans multiple seasons, each
    source may have multiple rows per batter (one per season). The merge
    uses batter_id + season as the join key in that case.
    """
    print('\nMerging sources...')

    if fg.empty:
        print('  WARNING: FanGraphs data empty — output will be Statcast only')
        # Fall back to EV as the spine
        base = ev.copy() if not ev.empty else xst.copy()
    else:
        base = fg.copy()

    # Normalise season column name for FG
    if 'Season' in base.columns and 'season' not in base.columns:
        base['season'] = base['Season'].astype(int)

    def safe_merge(left: pd.DataFrame, right: pd.DataFrame,
                   label: str) -> pd.DataFrame:
        if right.empty:
            print(f'  Skipping {label} (empty)')
            return left

        use_season = 'season' in right.columns and 'season' in left.columns \
                     and len(years) > 1
        join_keys  = ['batter_id', 'season'] if use_season else ['batter_id']

        # Drop columns that will cause collisions — name variants and season
        # when not used as a join key
        drop_cols = ['batter_name', 'last_name', 'first_name',
                     'player_name', 'Name', 'last_name, first_name']
        if 'season' not in join_keys:
            drop_cols.append('season')

        right = right.drop(columns=[c for c in drop_cols if c in right.columns],
                           errors='ignore')

        n_before = len(left)
        merged = left.merge(right, on=join_keys, how='left')
        n_after = len(merged)

        if n_after != n_before:
            print(f'  WARNING: {label} merge changed row count '
                  f'{n_before} → {n_after} (possible duplicate IDs in source)')

        right_cols_in_merged = [c for c in right.columns
                                 if c in merged.columns and c not in join_keys]
        if right_cols_in_merged:
            matched = merged[right_cols_in_merged].notna().any(axis=1).sum()
            print(f'  {label}: {matched:,} / {n_before:,} batters matched')
        return merged
    
    merged = safe_merge(base, br,  'BRef range stats')    # ← add after FG becomes base
    merged = safe_merge(merged, ev,  'EV/barrels')
    merged = safe_merge(merged, xst, 'expected stats')
    merged = safe_merge(merged, pct, 'percentile ranks')

    # ── Clean up duplicate/redundant columns ──────────────────────────────────
    # Drop suffixed duplicates created by merge (_x / _y)
    for col in list(merged.columns):
        if col.endswith('_y'):
            merged = merged.drop(columns=[col])
        elif col.endswith('_x'):
            merged = merged.rename(columns={col: col[:-2]})

    # Ensure batter_id and batter_name are the first two columns
    cols = merged.columns.tolist()
    front = [c for c in ['batter_id', 'batter_name'] if c in cols]
    rest  = [c for c in cols if c not in front]
    merged = merged[front + rest]

    print(f'\nFinal merged table: {len(merged):,} rows, '
          f'{len(merged.columns)} columns')
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 5. COVERAGE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def coverage_report(merged: pd.DataFrame, out_path: str):
    """
    Write a text file summarising column coverage (non-null %) per source.
    Useful for quickly seeing which batters are missing from which sources.
    """
    prefixes = {
        'FanGraphs':         'fg_',
        'BRef range':        'br_',
        'EV/barrels':        'ev_',
        'Expected stats':    'xst_',
        'Percentile ranks':  'pct_',
    }

    lines = [
        f'Coverage report',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Total batters: {len(merged):,}',
        f'Total columns: {len(merged.columns)}',
        '',
    ]

    for source, prefix in prefixes.items():
        cols = [c for c in merged.columns if c.startswith(prefix)]
        if not cols:
            lines.append(f'{source}: no columns found')
            continue
        pct_nonull = merged[cols].notna().mean().mean() * 100
        lines.append(f'{source} ({len(cols)} columns): '
                     f'{pct_nonull:.1f}% non-null on average')

    # Batters missing MLBAM ID
    missing_id = merged['batter_id'].isna().sum()
    lines.append(f'\nBatters missing MLBAM ID: {missing_id:,}')

    # Batters missing name
    missing_name = merged['batter_name'].isna().sum()
    lines.append(f'Batters missing name: {missing_name:,}')

    report = '\n'.join(lines)
    print(f'\n{report}')

    with open(out_path, 'w') as f:
        f.write(report)
    print(f'\nCoverage report → {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pull and merge batter stats from FanGraphs + Statcast')
    parser.add_argument('--start',    required=True,
                        help='Start date YYYY-MM-DD')
    parser.add_argument('--end',      required=True,
                        help='End date YYYY-MM-DD')
    parser.add_argument('--min-pa',   type=int, default=DEFAULT_MIN_PA,
                        help=f'Minimum PA (default {DEFAULT_MIN_PA})')
    parser.add_argument('--out',      default=None,
                        help='Output CSV filename (default: auto-named)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable pybaseball cache')
    args = parser.parse_args()

    # Validate dates
    try:
        start = datetime.strptime(args.start, '%Y-%m-%d')
        end   = datetime.strptime(args.end,   '%Y-%m-%d')
    except ValueError as e:
        print(f'Invalid date format: {e}')
        raise SystemExit(1)

    if end < start:
        print('--end must be after --start')
        raise SystemExit(1)

    if args.no_cache:
        pybaseball.cache.disable()

    years = years_in_range(args.start, args.end)
    print(f'Date range: {args.start} → {args.end}')
    print(f'Calendar years: {years}')
    print(f'Minimum PA: {args.min_pa}')

    # ── Output paths ───────────────────────────────────────────────────────────
    slug = f'{args.start}_{args.end}'.replace('-', '')
    out_csv  = args.out or os.path.join(OUT_DIR, f'batter_stats_{slug}.csv')
    out_cov  = out_csv.replace('.csv', '_coverage.txt')

    # ── Pull data ──────────────────────────────────────────────────────────────
    reg = load_register()

    fg  = pull_fangraphs(args.start, args.end, args.min_pa, reg)
    br  = pull_bref_range(args.start, args.end, reg)      # ← add
    ev  = pull_statcast_ev_barrels(years, reg)
    xst = pull_statcast_expected(years, reg)
    pct = pull_statcast_percentiles(years, reg)

    merged = merge_all(fg, br, ev, xst, pct, years)       # ← pass br

    # ── Merge ──────────────────────────────────────────────────────────────────
    # merged = merge_all(fg, ev, xst, pct, years)

    # ── Save ───────────────────────────────────────────────────────────────────
    merged.to_csv(out_csv, index=False)
    print(f'\nStats written to {out_csv}')

    coverage_report(merged, out_cov)

    # ── Quick preview ─────────────────────────────────────────────────────────
    print('\nColumn groups in output:')
    for prefix, label in [('fg_', 'FanGraphs'), ('br_', 'BRef range'),
                           ('ev_', 'EV/barrels'),
                           ('xst_', 'Expected'), ('pct_', 'Percentile')]:
        cols = [c for c in merged.columns if c.startswith(prefix)]
        print(f'  {label}: {len(cols)} columns')

    print('\nSample (first 5 rows, identity columns only):')
    id_cols = [c for c in ['batter_id', 'batter_name', 'season',
                            'fg_PA', 'fg_AVG', 'fg_wRC+',
                            'xst_xwoba', 'ev_brl_percent']
               if c in merged.columns]
    print(merged[id_cols].head().to_string(index=False))


if __name__ == '__main__':
    main()