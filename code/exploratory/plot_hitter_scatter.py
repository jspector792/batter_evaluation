"""
2×3 grid of timing vs delta_run_exp scatter plots for six specific hitters
across all pitch types. Points are coloured by launch_speed_angle (1–6,
Statcast contact-quality scale). Misses (no launch metrics) get value 0
and are drawn in grey.

Hitters: Aaron Judge, Shohei Ohtani, Chandler Simpson, Nick Sogard,
         Trent Grisham, Isaac Paredes.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pybaseball import playerid_lookup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_utils import (ALL_SWINGS, INTERCEPT_Y, load_pitches, add_timing,
                           weighted_moving_average)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "misc")

TARGETS = [
    ("Aaron",    "Judge"),
    ("Shohei",   "Ohtani"),
    ("Chandler", "Simpson"),
    ("Dominic",  "Canzone"),
    ("Trent",    "Grisham"),
    ("Isaac",    "Paredes"),
]

# ── Resolve names → mlbam IDs ────────────────────────────────────────────────
print("Looking up batter IDs…")
hitters = []
for first, last in TARGETS:
    rows = playerid_lookup(last, first)
    # Pick the most recent / active row by mlb_played_last
    rows = rows.dropna(subset=['key_mlbam'])
    if len(rows) == 0:
        raise SystemExit(f"No mlbam ID found for {first} {last}")
    if len(rows) > 1 and 'mlb_played_last' in rows.columns:
        rows = rows.sort_values('mlb_played_last', ascending=False)
    mlbam = int(rows.iloc[0]['key_mlbam'])
    hitters.append((f"{first} {last}", mlbam))
    print(f"  {first} {last} → {mlbam}")

# ── Load all 2025 swings + centre timing on the GLOBAL median ────────────────
target_ids = [m for _, m in hitters]
COLS = ['batter', 'description', 'pitch_type', INTERCEPT_Y,
        'delta_run_exp', 'launch_speed_angle', 'estimated_woba_using_speedangle']
all_swings = load_pitches(DATA_DIR, cols=COLS, descriptions=ALL_SWINGS)
all_swings = all_swings.dropna(subset=[INTERCEPT_Y])
GLOBAL_CENTER = add_timing(all_swings)
print(f"Global intercept_y median = {GLOBAL_CENTER:.2f} in  (n = {len(all_swings):,} swings)")

# Subset to the six target hitters
df = all_swings[all_swings['batter'].isin(target_ids)].copy()
# y-axis = xwOBA. Misses (no estimated_woba) get 0.
df['xwoba'] = df['estimated_woba_using_speedangle'].fillna(0.0)
print(f"Swings across the six target hitters: {len(df):,}")

# Replace NaN launch_speed_angle (misses) with 0
df['lsa'] = df['launch_speed_angle'].fillna(0).astype(int)

# ── Plot 2×3 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)
axes = axes.flatten()

# LSA-coding:
# 0 = miss (grey), 1-6 = contact quality (colormap)
# Using a 7-step colormap with miss=grey
miss_color = '#cccccc'
lsa_cmap   = plt.get_cmap('viridis', 6)  # 6 colours for LSA 1..6
lsa_labels = {1: 'Weak', 2: 'Topped', 3: 'Under',
              4: 'Flare/Burner', 5: 'Solid', 6: 'Barrel'}

for ax, (name, mlbam) in zip(axes, hitters):
    sub = df[df['batter'] == mlbam].copy()
    # Sort so misses are plotted first (under contact), then by LSA so barrels are on top
    sub = sub.sort_values('lsa')

    miss_sub = sub[sub['lsa'] == 0]
    in_sub   = sub[sub['lsa'] >= 1]

    ax.scatter(miss_sub['timing'], miss_sub['xwoba'],
               c=miss_color, s=12, alpha=0.5, edgecolors='none',
               label=f"Miss  (n={len(miss_sub)})")
    if len(in_sub) > 0:
        sc = ax.scatter(in_sub['timing'], in_sub['xwoba'],
                        c=in_sub['lsa'], cmap=lsa_cmap, vmin=0.5, vmax=6.5,
                        s=18, alpha=0.85, edgecolors='k', linewidths=0.2)

    # Count-weighted moving average over bins. Bins with <8 pitches are dropped
    # before averaging, and each 5-bin window must contain ≥30 pitches in total
    # — guarantees no single high-leverage swing can pull the curve.
    peak_loc = np.nan
    gx, gy, bin_tbl = weighted_moving_average(
        sub['timing'].values, sub['xwoba'].values,
        n_bins=20, window=5, min_per_bin=8, min_window_total=30,
        quantile_range=(0.05, 0.95))
    if len(gx) >= 4:
        ax.plot(gx, gy, color='black', lw=2.6, zorder=6,
                label=f"Weighted moving avg  (≥30/window)")
        peak_loc = float(gx[np.argmax(gy)])

    ax.axvline(0,  color='black', lw=0.8, ls='--', alpha=0.7,
               label='timing = 0 (pop. median)')
    ax.axvline(12.7, color='royalblue', lw=0.8, ls='--', alpha=0.6,
               label='pop. peak ≈ +12.7 in')
    if np.isfinite(peak_loc):
        ax.axvline(peak_loc, color='crimson', lw=1.5, ls=':', alpha=0.85,
                   label=f"hitter peak = {peak_loc:+.1f} in")
    ax.axhline(0, color='grey', lw=0.6, ls=':', alpha=0.6)
    ax.set_title(f"{name}  (n={len(sub):,})", fontsize=12)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)

for ax in axes[3:]:
    ax.set_xlabel("Timing (in, centred on global median)\n"
                  "← oppo/late                            pull/early →",
                  fontsize=10)
for ax in axes[::3]:
    ax.set_ylabel("xwOBA  (misses = 0)", fontsize=10)

# Single shared colorbar for LSA
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=lsa_cmap, norm=mcolors.BoundaryNorm(np.arange(0.5, 7.5), 6))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(1, 7))
cbar.ax.set_yticklabels([lsa_labels[i] for i in range(1, 7)], fontsize=9)
cbar.set_label('launch_speed_angle (contact quality)', fontsize=10)

# Legend for misses (manually, since they're shown in grey)
fig.text(0.93, 0.10, '⬤ grey = miss\n(no launch metrics)',
         fontsize=9, ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

fig.suptitle("Timing vs xwOBA — Selected Hitters (2025, all pitch types)\n"
             "Misses set to xwOBA = 0  ·  Coloured by launch_speed_angle (contact quality)  ·  "
             "Black = count-weighted moving avg over bins",
             fontsize=12, y=1.00)
plt.tight_layout(rect=[0, 0, 0.92, 0.97])
out_path = os.path.join(OUT_DIR, 'hitter_scatter_6.png')
fig.savefig(out_path, dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out_path}")
print("Done.")
