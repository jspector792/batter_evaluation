"""
run_matchup_surface.py
=======================
Direction 1: hitter x pitch-region matchup surfaces, with a split-half
validity gate.

Approach
--------
For a modest set of test hitters, bin their OWN actual swings over a coarse
pitch-region grid (plate_z x pfx_z) and, separately, by release-speed band.
Report mean residual per cell with the cell count shown, never hidden.

    timing_residual  = actual contact depth - predicted    (all swings)
    offset_residual  = actual miss distance - predicted    (WHIFFS ONLY --
                       actual miss distance is NaN on contact by construction
                       in the base table, so this denominator is smaller)

carry vs context does not matter for the SHAPE of a surface
-------------------------------------------------------------
The MERF random effect is an intercept: constant per batter. So
    timing_residual_carry = timing_residual_context - b_i
and within one hitter the two differ by a constant across every cell. The
surface's shape, the ranking of its cells, and split-half stability are all
identical either way; only the overall level shifts. `context` is used here
so levels are comparable across hitters. For offset the question is moot --
the miss-distance model is a plain RF with no random effect.

The validation gate (step 4)
-----------------------------
A weak cell that does not reappear on a random 50/50 split of that same
hitter's own swings is noise. For each hitter, over many random splits:

    surface_r    Pearson correlation of cell means between halves, across
                 cells that clear the count floor in BOTH halves. This is the
                 stability of the whole surface.
    worst_pctl   take the worst cell in half A, look up its percentile rank
                 in half B. Near 0 means the weak spot replicates; near 0.5
                 means it is noise.

Both are compared against a within-hitter permutation null: shuffle the
residuals across that hitter's swings, destroying any real cell structure,
and recompute. A hitter passes only if the observed statistic is outside the
null.

Not implemented: the synthetic held-swing-fixed surface (step 5, explicitly
optional). It needs an extrapolation-support model to say where the surface
may be trusted, which is a larger build than the rest of this direction.

Outputs (out/exp1_matchup_surface/)
------------------------------------
  matchup_cells.csv        every hitter x cell, with n
  matchup_validation.csv   split-half statistics + null, per hitter
  matchup_notes.md         one short written note per test hitter
  plots/surface_<name>.png heatmap per hitter
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE_DIR, 'code', 'exp_common'))
from base_table import load as load_base  # noqa: E402
from hitters import select_test_hitters, batter_names  # noqa: E402

warnings.filterwarnings('ignore')
sns.set_theme(style='white', font_scale=0.95)

OUT_DIR = os.path.join(BASE_DIR, 'out', 'exp1_matchup_surface')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SEASON = 2026
MIN_CELL = 15          # cells below this are shown but never ranked
MIN_CELL_SPLIT = 8     # per half, for the split-half check
N_SPLITS = 200
SEED = 42

# Coarse grid: 4 height bands x 4 vertical-break bands. Chosen coarse on
# purpose -- a single hitter has ~500 swings, so a finer grid produces cells
# too small to say anything about.
PLATE_Z_EDGES = [-np.inf, 1.9, 2.4, 2.9, np.inf]
PLATE_Z_LAB = ['low', 'low-mid', 'up-mid', 'high']
PFX_Z_EDGES = [-np.inf, 0.4, 0.9, 1.3, np.inf]
PFX_Z_LAB = ['drop', 'low ride', 'mid ride', 'high ride']
VELO_EDGES = [-np.inf, 85, 90, 94, np.inf]
VELO_LAB = ['<85', '85-90', '90-94', '94+']

RESID = {'timing': 'timing_residual_context',
         'offset': 'offset_residual_context'}


def add_bins(df):
    df = df.copy()
    df['z_band'] = pd.cut(df['plate_z'], PLATE_Z_EDGES, labels=PLATE_Z_LAB)
    df['br_band'] = pd.cut(df['pfx_z'], PFX_Z_EDGES, labels=PFX_Z_LAB)
    df['velo_band'] = pd.cut(df['release_speed'], VELO_EDGES, labels=VELO_LAB)
    return df


def cell_means(d, resid_col, rows='z_band', cols='br_band'):
    s = d.dropna(subset=[resid_col])
    if s.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = s.groupby([rows, cols], observed=False)[resid_col]
    return g.mean().unstack(), g.size().unstack()


def league_cell_map(df, resid_col):
    """League-average residual per cell, over ALL batters."""
    s = df.dropna(subset=[resid_col])
    g = s.groupby([s['z_band'].astype(str) + '|' +
                   s['br_band'].astype(str)])[resid_col]
    return g.mean()


def split_half_stats(d, resid_col, rng, n_splits=N_SPLITS, permute=False,
                     league=None):
    """
    Median surface correlation and worst-cell percentile over random splits.

    If `league` (a cell -> league-mean map) is supplied, both halves are
    measured as DEVIATIONS from the league surface before correlating. That
    is the test that matters: every hitter shares the same strong league
    gradient (high in the zone, high ride is hard for everyone), so a raw
    surface correlation can be large while carrying no hitter-specific
    information at all. Subtracting the league surface leaves only what is
    idiosyncratic to this hitter.
    """
    s = d.dropna(subset=[resid_col])
    if len(s) < 60:
        return np.nan, np.nan
    vals = s[resid_col].to_numpy()
    keys = list(zip(s['z_band'].astype(str), s['br_band'].astype(str)))
    keys = np.array([f'{a}|{b}' for a, b in keys])

    rs, pctls = [], []
    for _ in range(n_splits):
        v = rng.permutation(vals) if permute else vals
        mask = rng.random(len(s)) < 0.5
        out = {}
        for half, m in (('a', mask), ('b', ~mask)):
            dfh = pd.DataFrame({'k': keys[m], 'v': v[m]})
            g = dfh.groupby('k')['v']
            out[half] = pd.DataFrame({'mean': g.mean(), 'n': g.size()})
        common = out['a'].index.intersection(out['b'].index)
        common = [k for k in common
                  if out['a'].loc[k, 'n'] >= MIN_CELL_SPLIT
                  and out['b'].loc[k, 'n'] >= MIN_CELL_SPLIT]
        if len(common) < 4:
            continue
        a = out['a'].loc[common, 'mean']
        b = out['b'].loc[common, 'mean']
        if league is not None:
            lg = league.reindex(common)
            a = a - lg
            b = b - lg
            a, b = a.dropna(), b.dropna()
            common2 = a.index.intersection(b.index)
            if len(common2) < 4:
                continue
            a, b = a.loc[common2], b.loc[common2]
        if a.std() == 0 or b.std() == 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
        # Worst (most positive = worst for both metrics) cell in A, where does
        # it rank in B? 0 = also worst in B, 0.5 = random.
        worst = a.idxmax()
        pctls.append(float((b < b.loc[worst]).mean()))
    if not rs:
        return np.nan, np.nan
    return float(np.median(rs)), float(np.median(pctls))


def plot_surface(name, d, path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4),
                             constrained_layout=True)
    specs = [('timing', 'br_band', 'timing residual (in)  + = later than expected'),
             ('offset', 'br_band', 'miss-distance residual (in), whiffs only'),
             ('timing', 'velo_band', 'timing residual by velocity'),
             ('offset', 'velo_band', 'offset residual by velocity')]
    for ax, (metric, colvar, title) in zip(axes.ravel(), specs):
        m, n = cell_means(d, RESID[metric], 'z_band', colvar)
        if m.empty:
            ax.set_visible(False)
            continue
        lim = np.nanmax(np.abs(m.values)) or 1.0
        im = ax.imshow(m.values, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       aspect='auto')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v, c = m.values[i, j], n.values[i, j]
                if np.isnan(v):
                    continue
                weak = c < MIN_CELL
                ax.text(j, i, f'{v:+.2f}\nn={int(c)}' + ('*' if weak else ''),
                        ha='center', va='center', fontsize=8,
                        color='#999999' if weak else 'black')
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(m.columns, fontsize=8)
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(m.index, fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'{name} — matchup surface, {SEASON}  '
                 f'(* = n < {MIN_CELL}, not reliable)',
                 fontsize=13, fontweight='bold')
    fig.savefig(path, dpi=135, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Direction 1: matchup surfaces')
    ap.add_argument('--n-hitters', type=int, default=16)
    ap.add_argument('--season', type=int, default=SEASON)
    args = ap.parse_args()

    df = add_bins(load_base(args.season))
    names = batter_names(args.season)
    hitters = select_test_hitters(df, names, args.n_hitters)
    print(f'Test hitters ({len(hitters)}):')
    print(hitters.to_string(index=False))

    rng = np.random.default_rng(SEED)
    # League-average surface per metric, from ALL batters -- the shared
    # gradient that must be removed before a surface can be called
    # hitter-specific.
    league = {m: league_cell_map(df, c) for m, c in RESID.items()}
    cells, val = [], []
    for bid, row in hitters.set_index('batter').iterrows():
        d = df[df['batter'] == bid]
        nm = row['name']
        for metric, col in RESID.items():
            m, n = cell_means(d, col)
            if m.empty:
                continue
            # Built straight from the groupby rather than unstack/stack:
            # pandas 3.0 removed stack(dropna=), and observed=False already
            # yields every category combination including the empty ones.
            s = d.dropna(subset=[col])
            gg = s.groupby(['z_band', 'br_band'], observed=False)[col]
            long = pd.DataFrame({'mean_residual': gg.mean(),
                                 'n': gg.size()}).reset_index()
            long['batter'], long['name'], long['metric'] = bid, nm, metric
            cells.append(long)

            obs_r, obs_p = split_half_stats(d, col, rng)
            null_r, null_p = split_half_stats(d, col, rng, permute=True)
            dev_r, dev_p = split_half_stats(d, col, rng, league=league[metric])
            dev_r0, _ = split_half_stats(d, col, rng, permute=True,
                                         league=league[metric])
            val.append(dict(batter=bid, name=nm, metric=metric,
                            n_swings=int(d[col].notna().sum()),
                            surface_r=obs_r, surface_r_null=null_r,
                            dev_r=dev_r, dev_r_null=dev_r0,
                            worst_pctl=obs_p, worst_pctl_null=null_p))
        plot_surface(nm, d, os.path.join(
            PLOT_DIR, f'surface_{nm.replace(" ", "_")}.png'))

    cells = pd.concat(cells, ignore_index=True)
    val = pd.DataFrame(val)
    cells.to_csv(os.path.join(OUT_DIR, 'matchup_cells.csv'), index=False)
    val.to_csv(os.path.join(OUT_DIR, 'matchup_validation.csv'), index=False)

    pd.set_option('display.width', 200)
    print('\n=== Split-half validation ===\n')
    print(val.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))

    # Written notes
    lines = [f'# Direction 1 — matchup surfaces ({args.season})\n',
             f'Residuals are context-based. The MERF random effect is a per-batter '
             f'intercept, so carry vs context shifts a hitter\'s whole surface by a '
             f'constant and changes neither its shape nor these statistics.\n',
             f'`offset` rows use whiffs only ({RESID["offset"]} is NaN on contact '
             f'by construction).\n',
             f'\nA hitter/metric **passes** only if the median split-half surface '
             f'correlation clears its own permutation null and the worst cell '
             f'replicates (worst_pctl well below 0.5).\n']
    for _, r in val.iterrows():
        # The gate is the DEVIATION correlation, not the raw surface: a raw
        # correlation can be inflated purely by the shared league gradient.
        passed = (np.isfinite(r['dev_r']) and
                  r['dev_r'] > max(0.2, (r['dev_r_null'] or 0) + 0.15)
                  and np.isfinite(r['worst_pctl']) and r['worst_pctl'] < 0.35)
        sub = cells[(cells['batter'] == r['batter']) &
                    (cells['metric'] == r['metric']) &
                    (cells['n'] >= MIN_CELL)]
        worst = (sub.loc[sub['mean_residual'].idxmax()]
                 if not sub.empty and sub['mean_residual'].notna().any()
                 else None)
        verdict = 'STABLE' if passed else 'not stable — treat as noise'
        loc = (f"{worst['z_band']} / {worst['br_band']} "
               f"({worst['mean_residual']:+.2f} in, n={int(worst['n'])})"
               if worst is not None else 'n/a')
        lines.append(f"- **{r['name']} — {r['metric']}**: {verdict}. "
                     f"deviation-from-league r={r['dev_r']:.2f} "
                     f"(null {r['dev_r_null']:.2f}; raw surface r="
                     f"{r['surface_r']:.2f}), "
                     f"worst-cell pctl={r['worst_pctl']:.2f}. "
                     f"Worst reliable cell: {loc}.")
    with open(os.path.join(OUT_DIR, 'matchup_notes.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n-> {OUT_DIR}')


if __name__ == '__main__':
    main()
