"""
run_fingerprint.py
===================
Direction 4: scouting-report-style weakness fingerprint per hitter.

Richer than Direction 1 on purpose: cross-tabs over pitch type, zone,
velocity band and count state (plate-discipline context), rather than the
single 2D pitch-region grid.

A single 4-way cell was tried first and abandoned: pitch x zone x velo x
count allows 882 combinations against ~1,100 swings per hitter, and only 12
cells league-wide cleared the count floor. Several coarser 2- and 3-way
VIEWS are used instead, each leaving cells large enough to read.

Weak spot = cell where the hitter's mean residual is worst, subject to a
minimum swing count. Residuals are measured against the LEAGUE mean for the
same cell, so "weak" means weak relative to what that cell does to everyone,
not merely that the cell is hard.

The honesty gate carried over from Direction 1
------------------------------------------------
Direction 1 established that a hitter's single worst cell in a continuous
plate_z x pfx_z grid does not survive a random 50/50 split of that hitter's
own swings (worst-cell percentile ~0.5 for 31 of 32 hitter x metric
combinations). Every weak spot flagged here is therefore re-tested by split
half AND against a permutation null, because "stays in the worst quartile"
is a softer bar than "is the argmax", and the spot was selected on the same
data it is then scored on. Only the lift over the null is evidence.
Flagged-but-unstable spots are reported as such rather than quietly dropped.

On the scouting-narrative comparison (step 3)
-----------------------------------------------
Not attempted as specified. Verifying flagged weak spots against beat-writer
coverage for the current season would mean sourcing and subjectively scoring
narrative claims; without a reliable, checkable source for this season I
would be inventing the comparison, which is worse than not running it. What
IS reported instead is an objective, data-derived profile for each hitter
(handedness, chase rate, zone-contact rate, bat speed, overall contact rate)
so a reader who does know the player can judge face validity themselves.

Outputs (out/exp4_weakness_fingerprint/)
-----------------------------------------
  fingerprint_cells.csv     every hitter x cell with n and league-relative residual
  fingerprint_summary.csv   top weak spots per hitter, with stability flag
  fingerprint_<name>.md     one-page-per-hitter summary
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE_DIR, 'code', 'exp_common'))
from base_table import load as load_base  # noqa: E402
from hitters import select_test_hitters, batter_names  # noqa: E402

warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(BASE_DIR, 'out', 'exp4_weakness_fingerprint')
os.makedirs(OUT_DIR, exist_ok=True)

SEASON = 2026
MIN_CELL = 25
N_SPLITS = 150
TOP_K = 3
SEED = 42

VELO_EDGES = [-np.inf, 88, 93, np.inf]
VELO_LAB = ['soft', 'medium', 'hard']

METRICS = {'timing': 'timing_residual_context',
           'offset': 'offset_residual_context'}

# Statcast zones: 1-9 in the strike zone (rows of three), 11-14 the shadow
# quadrants outside it.
# A single 4-way cell (pitch x zone x velo x count) allows 882 combinations
# against ~1,100 swings per hitter -- about one swing per cell. Several
# coarser 2- and 3-way views are used instead, each chosen to leave cells
# large enough to read.
VIEWS = [
    ('pitch_family', 'zone_coarse'),
    ('pitch_family', 'count_state'),
    ('zone_coarse', 'velo_band'),
    ('zone_coarse', 'count_state'),
    ('pitch_family', 'zone_coarse', 'count_state'),
]

ZONE_COARSE = {
    'up-in': 'upper', 'up-mid': 'upper', 'up-away': 'upper',
    'mid-in': 'inner', 'low-in': 'lower', 'low-mid': 'lower',
    'mid-away': 'outer', 'low-away': 'lower', 'middle': 'middle',
    'chase up-in': 'chase-up', 'chase up-away': 'chase-up',
    'chase low-in': 'chase-low', 'chase low-away': 'chase-low',
}

ZONE_REGION = {1: 'up-in', 2: 'up-mid', 3: 'up-away',
               4: 'mid-in', 5: 'middle', 6: 'mid-away',
               7: 'low-in', 8: 'low-mid', 9: 'low-away',
               11: 'chase up-in', 12: 'chase up-away',
               13: 'chase low-in', 14: 'chase low-away'}


def prep(df):
    df = df.copy()
    df['velo_band'] = pd.cut(df['release_speed'], VELO_EDGES, labels=VELO_LAB)
    df['zone_region'] = df['zone'].map(ZONE_REGION).fillna('other')
    # Coarser zone grouping for the cross-tabs. The 13-way Statcast zone
    # crossed with anything else leaves cells of ~1 swing.
    df['zone_coarse'] = df['zone_region'].map(ZONE_COARSE).fillna('other')
    # Group the long tail of pitch types so cells are not shredded.
    fam = {'FF': 'fastball', 'SI': 'sinker', 'FC': 'cutter', 'FA': 'fastball',
           'SL': 'slider', 'ST': 'sweeper', 'CU': 'curve', 'KC': 'curve',
           'SV': 'slider', 'CS': 'curve', 'KN': 'other',
           'CH': 'change', 'FS': 'splitter', 'FO': 'other', 'EP': 'other'}
    df['pitch_family'] = df['pitch_type'].map(fam).fillna('other')
    return df


def add_cell(df, view):
    """Cell key for one cross-tab view."""
    out = df.copy()
    out['cell'] = out[view[0]].astype(str)
    for k in view[1:]:
        out['cell'] = out['cell'] + ' | ' + out[k].astype(str)
    out['view'] = ' x '.join(view)
    return out


def league_map(df, col):
    s = df.dropna(subset=[col])
    return s.groupby('cell')[col].mean()


def league_maps(df, views):
    """{(view, metric): cell -> league mean}"""
    out = {}
    for v in views:
        dv = add_cell(df, v)
        for m, c in METRICS.items():
            out[(' x '.join(v), m)] = league_map(dv, c)
    return out


def split_half_cell(d, col, cell, rng, n_splits=N_SPLITS, permute=False):
    """
    How often does this cell stay in the hitter's worst quartile on a random
    half of their own swings?

    `permute=True` shuffles the residuals across the hitter's swings first,
    destroying any real cell structure. That null is essential: the flagged
    cell was SELECTED as the worst on this same data, and "worst quartile" is
    a weak bar when a view has few cells, so a high retention rate can arise
    with no signal at all. Only the gap between observed and null is
    evidence.
    """
    s = d.dropna(subset=[col])
    vals = s[col].to_numpy()
    if permute:
        vals = rng.permutation(vals)
    cells = s['cell'].to_numpy()
    in_cell = cells == cell
    if in_cell.sum() < 2 * 8:
        return np.nan
    keeps = 0
    used = 0
    for _ in range(n_splits):
        m = rng.random(len(s)) < 0.5
        for half in (m, ~m):
            v, c = vals[half], cells[half]
            if (c == cell).sum() < 8:
                continue
            g = pd.DataFrame({'c': c, 'v': v}).groupby('c')['v']
            means = g.mean()[g.size() >= 8]
            if len(means) < 4 or cell not in means.index:
                continue
            used += 1
            if means.loc[cell] >= means.quantile(0.75):
                keeps += 1
    return float(keeps / used) if used else np.nan


def main():
    ap = argparse.ArgumentParser(description='Direction 4: fingerprints')
    ap.add_argument('--n-hitters', type=int, default=16)
    ap.add_argument('--season', type=int, default=SEASON)
    args = ap.parse_args()

    df = prep(load_base(args.season))
    names = batter_names(args.season)
    hitters = select_test_hitters(df, names, args.n_hitters)
    rng = np.random.default_rng(SEED)
    lg = league_maps(df, VIEWS)

    all_cells, summary = [], []
    for _, h in hitters.iterrows():
        bid, nm = h['batter'], h['name']
        d = df[df['batter'] == bid]
        prof = dict(
            stand=h['stand'], n_swings=int(h['n_swings']),
            contact_pct=float(h['contact_pct']),
            bat_speed=float(h['bat_speed']),
            chase_pct=float(d['zone_region'].str.startswith('chase').mean()),
            zone_contact=float(d.loc[d['in_zone'], 'is_contact'].mean()))

        lines = [f'# {nm} — weakness fingerprint ({args.season})\n',
                 f'- bats **{prof["stand"]}**, {prof["n_swings"]:,} swings, '
                 f'contact {prof["contact_pct"]:.1%}, '
                 f'zone-contact {prof["zone_contact"]:.1%}, '
                 f'chase-zone swings {prof["chase_pct"]:.1%}, '
                 f'bat speed {prof["bat_speed"]:.1f} mph\n']

        for metric, col in METRICS.items():
            pooled = []
            for view in VIEWS:
                vname = ' x '.join(view)
                dv = add_cell(d, view)
                s = dv.dropna(subset=[col])
                if s.empty:
                    continue
                g = s.groupby('cell')[col]
                t = pd.DataFrame({'mean_residual': g.mean(), 'n': g.size()})
                t = t[t['n'] >= MIN_CELL]
                if t.empty:
                    continue
                t['league'] = lg[(vname, metric)].reindex(t.index)
                t['vs_league'] = t['mean_residual'] - t['league']
                t['view'] = vname
                t['batter'], t['name'], t['metric'] = bid, nm, metric
                all_cells.append(t.reset_index())
                pooled.append((vname, dv, t))
            if not pooled:
                lines.append(f'\n## {metric}\n\nNo cell in any view reaches '
                             f'n >= {MIN_CELL}.\n')
                continue
            t_all = pd.concat([t.assign(_v=v) for v, _, t in pooled])
            t_all = t_all.sort_values('vs_league', ascending=False)

            lines.append(f'\n## {metric} — top {TOP_K} weak spots '
                         f'(worse than league for the same cell)\n')
            lines.append('| view | cell | n | mean resid | vs league | stability |')
            lines.append('|---|---|---|---|---|---|')
            for cell, r in t_all.head(TOP_K).iterrows():
                dv = dict((v, dd) for v, dd, _ in pooled)[r['_v']]
                stab = split_half_cell(dv, col, cell, rng)
                null = split_half_cell(dv, col, cell, rng, permute=True)
                lift = ((stab - null) if (stab is not None and null is not None
                                          and np.isfinite(stab)
                                          and np.isfinite(null)) else np.nan)
                verdict = ('stable' if (np.isfinite(lift) and lift >= 0.20
                                        and (stab or 0) >= 0.7)
                           else 'UNSTABLE — likely noise')
                summary.append(dict(batter=bid, name=nm, metric=metric,
                                    view=r['_v'],
                                    cell=cell, n=int(r['n']),
                                    mean_residual=r['mean_residual'],
                                    vs_league=r['vs_league'],
                                    stability=stab, stability_null=null,
                                    stability_lift=lift, verdict=verdict))
                lines.append(f"| {r['_v']} | {cell} | {int(r['n'])} | "
                             f"{r['mean_residual']:+.2f} | "
                             f"{r['vs_league']:+.2f} | "
                             f"{'n/a' if stab is None or np.isnan(stab) else f'{stab:.2f}'}"
                             f" — {verdict} |")

        lines.append('\n> Stability = how often this cell stays in the '
                     'hitter\'s own worst quartile across random halves of '
                     'their swings. Below ~0.7 the spot is not reproducible '
                     'and should not be read as a scouting finding.\n')
        with open(os.path.join(OUT_DIR,
                               f'fingerprint_{nm.replace(" ", "_")}.md'),
                  'w') as f:
            f.write('\n'.join(lines) + '\n')

    cells = pd.concat(all_cells, ignore_index=True)
    summ = pd.DataFrame(summary)
    cells.to_csv(os.path.join(OUT_DIR, 'fingerprint_cells.csv'), index=False)
    summ.to_csv(os.path.join(OUT_DIR, 'fingerprint_summary.csv'), index=False)

    pd.set_option('display.width', 220)
    print(f'hitters: {len(hitters)} | cells: {len(cells):,} | '
          f'flagged weak spots: {len(summ)}')
    print('\n=== Stability of flagged weak spots ===')
    print(summ.groupby('metric')[
        ['stability', 'stability_null', 'stability_lift']].mean().to_string(
        float_format=lambda v: f'{v:,.3f}'))
    print(f"\nstable (obs >= 0.70 AND lift >= 0.20 over permutation null): "
          f"{(summ['verdict'] == 'stable').sum()} "
          f"of {summ['stability'].notna().sum()} testable")
    print('\n=== Sample of flagged spots ===')
    print(summ.head(14)[['name', 'metric', 'cell', 'n', 'vs_league',
                         'stability', 'stability_null',
                         'stability_lift']].to_string(
        index=False, float_format=lambda v: f'{v:,.3f}'))
    print(f'\n-> {OUT_DIR}')


if __name__ == '__main__':
    main()
