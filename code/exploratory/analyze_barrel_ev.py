"""
Barrel placement with exit velocity folded in - all-in-play reference only.

Timing proxy: intercept_y - population median (inches).
  - Timing-adjusted LW computed from ALL swings (in-play + fouls + misses).
  - Barrel deviations computed for in-play only (launch metrics required).

For each in-play fastball, the *intended* outcome is the population mean of
(launch_angle, launch_speed, estimated_woba) for swings with the same
swing shape (swing_path_tilt x attack_angle, 20x20 quantile bins).
Timing is kept as the timing variable and is NOT used as a shape dimension.

Three deviation metrics:
  dev_la    = actual_LA    - intended_LA    (launch angle only)
  dev_ls    = actual_LS    - intended_LS    (exit velocity only)
  dev_xwoba = actual_xwoba - intended_xwoba (speed-angle surface - both)

Primary figure: 1x3 hex plots of each deviation vs timing-adjusted LW
  (mean delta_run_exp for the swing's timing bin, 40 equal-width bins)
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY, INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw, TIMING_AXIS_LABEL)

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "misc")
SHAPE_BINS  = 20
TIMING_BINS = 40

# Columns for all-swing timing-LW base (in-play + fouls + misses).
COLS_ALLSWING = [
    "pitch_type", "description",
    INTERCEPT_X,
    INTERCEPT_Y,
    "delta_run_exp",
]
COLS_INPLAY = [
    "pitch_type", "description",
    INTERCEPT_X,
    INTERCEPT_Y,
    "delta_run_exp",
    "swing_path_tilt", "attack_angle",
    "launch_angle", "launch_speed",
    "estimated_woba_using_speedangle",
]

# ── Load ALL swings for timing-adjusted LW ────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

# Load minimal columns for all swings (timing LW computation)
full_df = pd.concat(
    [pd.read_csv(f, usecols=COLS_ALLSWING) for f in files], ignore_index=True
)
full_df = full_df[full_df['pitch_type'].isin(FASTBALL_TYPES) &
                  full_df['description'].isin(ALL_SWINGS)].copy()
full_df = full_df.dropna(subset=[INTERCEPT_Y, 'delta_run_exp'])
TIMING_CENTER = add_timing(full_df)
print(f"All swings (timing LW base): {len(full_df):,}")

# Timing bins from all swings
timing_edges = np.linspace(full_df['timing'].quantile(0.005),
                            full_df['timing'].quantile(0.995),
                            TIMING_BINS + 1)
full_df['timing_bin'] = pd.cut(full_df['timing'], bins=timing_edges)
timing_lw = (full_df.groupby('timing_bin', observed=True)['delta_run_exp']
               .mean().rename('timing_lw'))

# ── Load in-play subset (with launch metrics) ─────────────────────────────────
df = pd.concat(
    [pd.read_csv(f, usecols=COLS_INPLAY) for f in files], ignore_index=True
)
df = df[df['pitch_type'].isin(FASTBALL_TYPES) &
        df['description'].isin(IN_PLAY)].copy()
df = df.dropna(subset=['delta_run_exp', INTERCEPT_Y,
                        'swing_path_tilt', 'attack_angle', 'launch_angle', 'launch_speed'])
print(f"In-play fastballs with all core fields: {len(df):,}")
n_xwoba = df['estimated_woba_using_speedangle'].notna().sum()
print(f"  xwoba available: {n_xwoba:,} ({n_xwoba/len(df):.1%})")

# ── Join timing-adjusted LW (centred on the all-swing population) ─────────────
add_timing(df, center=TIMING_CENTER)
df['timing_bin'] = pd.cut(df['timing'], bins=timing_edges)
df = df.join(timing_lw, on='timing_bin')

# ── Swing-shape bins: 20×20 quantile bins on swing_path_tilt × attack_angle ──
df['tilt_bin']  = pd.qcut(df['swing_path_tilt'], q=SHAPE_BINS, duplicates='drop')
df['angle_bin'] = pd.qcut(df['attack_angle'],    q=SHAPE_BINS, duplicates='drop')
df['shape_key'] = list(zip(df['tilt_bin'].astype(str), df['angle_bin'].astype(str)))

# ── Intended outcomes (all-in-play means per tilt bin) ────────────────────────
intended = (df.groupby('shape_key', observed=True)
              .agg(intended_la   =('launch_angle', 'mean'),
                   intended_ls   =('launch_speed',  'mean'),
                   intended_xwoba=('estimated_woba_using_speedangle', 'mean'))
              .rename_axis('shape_key'))
df = df.join(intended, on='shape_key')

# ── Deviations ────────────────────────────────────────────────────────────────
df['dev_la']    = df['launch_angle']                    - df['intended_la']
df['dev_ls']    = df['launch_speed']                    - df['intended_ls']
df['dev_xwoba'] = df['estimated_woba_using_speedangle'] - df['intended_xwoba']

core = df.dropna(subset=['dev_la', 'dev_ls', 'timing_lw'])
core_x = core.dropna(subset=['dev_xwoba'])

print(f"\nSwings with LA+LS deviations: {len(core):,}")
print(f"Swings with xwoba deviation:  {len(core_x):,}")

for col, label in [('dev_la','dev_LA'), ('dev_ls','dev_LS'), ('dev_xwoba','dev_xwoba')]:
    sub = core_x if col == 'dev_xwoba' else core
    r_t, p_t = pearsonr(sub[col], sub['timing_lw'])
    r_r, p_r = pearsonr(sub[col], sub['delta_run_exp'])
    print(f"  {label:<12}  r(timing_lw)={r_t:+.3f} p={p_t:.4f}  "
          f"r(raw_lw)={r_r:+.3f} p={p_r:.4f}")

def binned_means(x, y, n=40):
    bins = pd.qcut(pd.Series(x), q=n, duplicates='drop')
    agg = pd.DataFrame({'x': x, 'y': y, 'bin': bins}).groupby('bin', observed=True)
    return agg['x'].mean().values, agg['y'].mean().values

# ── Primary figure: 1×3 hex plots ────────────────────────────────────────────
CONFIGS = [
    ('dev_la',    core,   'Launch Angle Deviation (deg)',
     'actual LA - intended LA for this swing shape bin (tilt x attack_angle)\n<- hit low         hit high ->',
     'blue'),
    ('dev_ls',    core,   'Exit Velocity Deviation (mph)',
     'actual EV - intended EV for this swing shape bin (tilt x attack_angle)\n<- weaker         harder ->',
     'darkorange'),
    ('dev_xwoba', core_x, 'xwOBA Deviation',
     'actual xwOBA - intended xwOBA for this swing shape bin (tilt x attack_angle)\n<- worse quality       better quality ->',
     'darkgreen'),
]

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, (col, data, xlabel_short, xlabel_full, accent) in zip(axes, CONFIGS):
    x = data[col].values
    y = data['timing_lw'].values

    x_lo, x_hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    mask = (x >= x_lo) & (x <= x_hi)
    xp, yp = x[mask], y[mask]

    r, p = pearsonr(xp, yp)

    hb = ax.hexbin(xp, yp, gridsize=50, cmap='YlOrRd',
                   mincnt=5, linewidths=0.15)
    fig.colorbar(hb, ax=ax, pad=0.02).set_label("Count", fontsize=8)

    mx, my = binned_means(xp, yp)
    ax.plot(mx, my, color='royalblue', lw=2.5, zorder=5, label='Binned mean')

    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7)
    ax.axhline(0, color='grey',  lw=0.8, ls=':',  alpha=0.6)

    pstr = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    ax.set_xlabel(xlabel_full, fontsize=9)
    ax.set_ylabel("Timing-Adjusted LW\n"
                  "(mean delta run exp for this timing bin, all swings)", fontsize=9)
    ax.set_title(f"{xlabel_short}\nr = {r:+.3f}   p = {pstr}   n = {mask.sum():,}", fontsize=11)
    ax.legend(fontsize=8)

fig.suptitle("Barrel Placement Deviations vs Timing-Distance-Adjusted Run Value\n"
             "All-in-play reference  ·  20x20 swing shape bins (tilt x attack_angle)  ·  2025 MLB fastballs",
             fontsize=13, y=1.01)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'barrel_ev_hex.png')
fig.savefig(out1, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out1}")

# ── Supplement: 2D scatter dev_la vs dev_ls coloured by mean delta_run_exp ───
NBINS2D = 25
core2 = core.copy()
core2['dla_bin'] = pd.cut(core2['dev_la'], bins=NBINS2D)
core2['dls_bin'] = pd.cut(core2['dev_ls'], bins=NBINS2D)
surface = (core2.groupby(['dla_bin', 'dls_bin'], observed=True)
                .agg(mean_lw=('delta_run_exp', 'mean'),
                     n      =('delta_run_exp', 'count'),
                     mid_la =('dev_la', 'median'),
                     mid_ls =('dev_ls', 'median'))
                .reset_index()
                .dropna()
                .query('n >= 10'))

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

ax = axes2[0]
sc = ax.scatter(surface['mid_la'], surface['mid_ls'],
                c=surface['mean_lw'], cmap='RdYlGn',
                s=surface['n'] / surface['n'].max() * 200 + 10,
                edgecolors='k', linewidths=0.2,
                vmin=surface['mean_lw'].quantile(0.05),
                vmax=surface['mean_lw'].quantile(0.95))
fig2.colorbar(sc, ax=ax, pad=0.02).set_label("Mean Delta Run Expectancy", fontsize=9)
ax.axvline(0, color='black', lw=1, ls='--', alpha=0.6)
ax.axhline(0, color='black', lw=1, ls='--', alpha=0.6)
ax.set_xlabel("Launch Angle Deviation from Intended (deg)\n<- below intended   above intended ->", fontsize=10)
ax.set_ylabel("Exit Velocity Deviation from Intended (mph)\n<- softer than intended   harder than intended ->", fontsize=10)
ax.set_title("Mean Run Value by (dev_LA, dev_EV)\n"
             "Colour = run value, size ∝ count  (min 10 swings per cell)", fontsize=11)
_nmax0 = surface['n'].max()
for _n_ref in [50, 200, 500]:
    ax.scatter([], [], s=_n_ref / _nmax0 * 200 + 10, c='gray', alpha=0.6,
               edgecolors='k', linewidths=0.2, label=f'n = {_n_ref}')
ax.legend(title='Swings per cell', fontsize=8, framealpha=0.85)

core3 = core_x.copy()
core3['dla_bin'] = pd.cut(core3['dev_la'], bins=NBINS2D)
core3['dls_bin'] = pd.cut(core3['dev_ls'], bins=NBINS2D)
surface3 = (core3.groupby(['dla_bin', 'dls_bin'], observed=True)
                  .agg(mean_xwoba_dev=('dev_xwoba', 'mean'),
                       n             =('dev_xwoba', 'count'),
                       mid_la        =('dev_la', 'median'),
                       mid_ls        =('dev_ls', 'median'))
                  .reset_index().dropna().query('n >= 10'))

ax = axes2[1]
sc2 = ax.scatter(surface3['mid_la'], surface3['mid_ls'],
                 c=surface3['mean_xwoba_dev'], cmap='RdYlGn',
                 s=surface3['n'] / surface3['n'].max() * 200 + 10,
                 edgecolors='k', linewidths=0.2,
                 vmin=surface3['mean_xwoba_dev'].quantile(0.05),
                 vmax=surface3['mean_xwoba_dev'].quantile(0.95))
fig2.colorbar(sc2, ax=ax, pad=0.02).set_label("Mean xwOBA Deviation", fontsize=9)
ax.axvline(0, color='black', lw=1, ls='--', alpha=0.6)
ax.axhline(0, color='black', lw=1, ls='--', alpha=0.6)
ax.set_xlabel("Launch Angle Deviation from Intended (deg)", fontsize=10)
ax.set_ylabel("Exit Velocity Deviation from Intended (mph)", fontsize=10)
ax.set_title("Mean xwOBA Deviation by (dev_LA, dev_EV)\n"
             "Shows how LA and EV deviations jointly determine quality", fontsize=11)
_nmax1 = surface3['n'].max()
for _n_ref in [50, 200, 500]:
    ax.scatter([], [], s=_n_ref / _nmax1 * 200 + 10, c='gray', alpha=0.6,
               edgecolors='k', linewidths=0.2, label=f'n = {_n_ref}')
ax.legend(title='Swings per cell', fontsize=8, framealpha=0.85)

fig2.suptitle("Joint (dev_LA, dev_EV) Quality Surface — All-In-Play Reference",
              fontsize=12, y=1.01)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'barrel_ev_surface2d.png')
fig2.savefig(out2, dpi=180, bbox_inches='tight')
plt.close(fig2)
print(f"Saved → {out2}")

print("\nDone.")
