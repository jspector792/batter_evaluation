"""
Barrel placement via intended launch angle.

Timing proxy: intercept_y - population median (inches).
All-swing filter: in-play + foul + miss for the timing-LW reference; in-play
only for the barrel-deviation computation (launch_angle required).

For each in-play fastball swing, estimate the *intended* launch angle as the
mean launch angle of swings with the same swing shape (swing_path_tilt x
attack_angle, 20x20 quantile bins). Timing is used as the timing variable
(not a shape dimension) so that barrel placement is measured orthogonally
to timing.

Two versions:
  v1 (barrel intent)   - reference = mean LA of barrels (launch_speed_angle==6)
                         in the same shape bin
  v2 (all-play intent) - reference = mean LA of all in-play balls in the bin

Barrel deviation = actual launch angle - intended launch angle

Primary visual:
  X: barrel deviation
  Y: timing-adjusted LW = mean(delta_run_exp) for swings in the same
     timing bin (40 equal-width bins, computed from ALL swings including
     misses)

Timing-adjusted LW uses ALL swings (in-play + fouls + misses) for a larger
sample; barrel deviation is computed for in-play only (launch_angle required).
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY, INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw, TIMING_AXIS_LABEL)

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
SHAPE_BINS  = 20
TIMING_BINS = 40

COLS = [
    "pitch_type", "batter", "stand",
    "delta_run_exp", "description",
    "swing_path_tilt", "attack_angle",
    INTERCEPT_X,
    INTERCEPT_Y,
    "launch_angle", "launch_speed_angle",
]

# ── Load ALL swings (in-play + fouls + misses) ────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
full_df = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
full_df = full_df[full_df['pitch_type'].isin(FASTBALL_TYPES) &
                  full_df['description'].isin(ALL_SWINGS)].copy()
full_df = full_df.dropna(subset=[INTERCEPT_Y, 'delta_run_exp'])
TIMING_CENTER = add_timing(full_df)
print(f"All swings (timing + delta_run_exp): {len(full_df):,}")

# ── Timing bins from ALL swings ──────────────────────────────────────────────
timing_edges = np.linspace(full_df['timing'].quantile(0.005),
                            full_df['timing'].quantile(0.995),
                            TIMING_BINS + 1)
full_df['timing_bin'] = pd.cut(full_df['timing'], bins=timing_edges)
timing_lw = (full_df.groupby('timing_bin', observed=True)['delta_run_exp']
               .mean().rename('timing_lw'))
print(f"Timing LW range: [{timing_lw.min():.4f}, {timing_lw.max():.4f}]")
print(f"Timing bins with data: {timing_lw.notna().sum()}")

# ── In-play subset for barrel analysis ───────────────────────────────────────
df = full_df[full_df['description'].isin(IN_PLAY)].copy()
df = df.dropna(subset=['swing_path_tilt', 'launch_angle'])
df = df.join(timing_lw, on='timing_bin')  # join timing-angle-adjusted LW
print(f"\nIn-play fastballs with tilt + launch_angle: {len(df):,}")
print(f"  timing_lw available: {df['timing_lw'].notna().sum():,}")

# ── Swing-shape bins: 20×20 quantile bins on swing_path_tilt × attack_angle ──
# timing_angle is timing; shape is defined by tilt × attack_angle
df['tilt_bin']  = pd.qcut(df['swing_path_tilt'], q=SHAPE_BINS, duplicates='drop')
df['angle_bin'] = pd.qcut(df['attack_angle'],    q=SHAPE_BINS, duplicates='drop')
df['shape_key'] = list(zip(df['tilt_bin'].astype(str), df['angle_bin'].astype(str)))

print(f"\nShape bins (tilt × attack_angle): {df['shape_key'].nunique()}")
print(f"  In-play per bin (min/median/max): "
      f"{df.groupby('shape_key').size().min()} / "
      f"{df.groupby('shape_key').size().median():.0f} / "
      f"{df.groupby('shape_key').size().max()}")

# ── Version 1: barrel-based intended LA ──────────────────────────────────────
barrels = df[df['launch_speed_angle'] == 6]
barrel_intent = (barrels.groupby('shape_key')['launch_angle']
                         .mean()
                         .rename('intended_la_v1'))
df = df.join(barrel_intent, on='shape_key')
print(f"\nV1 shape bins with barrel data: {barrel_intent.notna().sum()}")
print(f"  Min barrels per bin: "
      f"{barrels.groupby('shape_key').size().min()}")

# ── Version 2: all-in-play intended LA ───────────────────────────────────────
all_intent = (df.groupby('shape_key')['launch_angle']
                .mean()
                .rename('intended_la_v2'))
df = df.join(all_intent, on='shape_key')

# ── Barrel deviations ─────────────────────────────────────────────────────────
df['barrel_dev_v1'] = df['launch_angle'] - df['intended_la_v1']
df['barrel_dev_v2'] = df['launch_angle'] - df['intended_la_v2']

df_v1 = df.dropna(subset=['barrel_dev_v1', 'timing_lw'])
df_v2 = df.dropna(subset=['barrel_dev_v2', 'timing_lw'])
print(f"\nVersion 1 (barrel intent): {len(df_v1):,} swings  "
      f"| barrel_dev mean={df_v1['barrel_dev_v1'].mean():.2f} deg  std={df_v1['barrel_dev_v1'].std():.2f} deg")
print(f"Version 2 (all-play intent): {len(df_v2):,} swings  "
      f"| barrel_dev mean={df_v2['barrel_dev_v2'].mean():.2f} deg  std={df_v2['barrel_dev_v2'].std():.2f} deg")

r_v1, p_v1 = pearsonr(df_v1['barrel_dev_v1'], df_v1['timing_lw'])
r_v2, p_v2 = pearsonr(df_v2['barrel_dev_v2'], df_v2['timing_lw'])
print(f"\nr(barrel_dev_v1, timing_lw) = {r_v1:.3f}  p={p_v1:.4f}")
print(f"r(barrel_dev_v2, timing_lw) = {r_v2:.3f}  p={p_v2:.4f}")

r_raw_v1, p_raw_v1 = pearsonr(df_v1['barrel_dev_v1'], df_v1['delta_run_exp'])
r_raw_v2, p_raw_v2 = pearsonr(df_v2['barrel_dev_v2'], df_v2['delta_run_exp'])
print(f"\nr(barrel_dev_v1, raw delta_run_exp) = {r_raw_v1:.3f}  p={p_raw_v1:.4f}")
print(f"r(barrel_dev_v2, raw delta_run_exp) = {r_raw_v2:.3f}  p={p_raw_v2:.4f}")

def binned_means(x, y, n=40):
    bins = pd.qcut(x, q=n, duplicates='drop')
    g = pd.DataFrame({'x': x, 'y': y, 'bin': bins})
    agg = g.groupby('bin', observed=True).agg(mx=('x','mean'), my=('y','mean'), n=('y','count'))
    return agg['mx'].values, agg['my'].values

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

CONFIGS = [
    (df_v1, 'barrel_dev_v1', r_v1, p_v1,
     "V1: Barrel Intent Reference\n"
     "(intended LA = mean LA of barrels in same swing shape bin (tilt x attack_angle))",
     "Barrel Deviation from Barrel Intent (deg)\nactual launch angle - mean barrel launch angle for this swing shape bin"),
    (df_v2, 'barrel_dev_v2', r_v2, p_v2,
     "V2: All-In-Play Intent Reference\n"
     "(intended LA = mean LA of all in-play balls in same swing shape bin (tilt x attack_angle))",
     "Barrel Deviation from Average Intent (deg)\nactual launch angle - mean in-play launch angle for this swing shape bin"),
]

for ax, (data, dev_col, r, p, title, xlabel) in zip(axes, CONFIGS):
    x = data[dev_col].values
    y = data['timing_lw'].values

    x_lo, x_hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    mask = (x >= x_lo) & (x <= x_hi)
    x_plot, y_plot = x[mask], y[mask]

    hb = ax.hexbin(
        x_plot, y_plot,
        gridsize=50,
        cmap='YlOrRd',
        mincnt=5,
        linewidths=0.15,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Count per hex bin", fontsize=9)

    mx, my = binned_means(x_plot, y_plot, n=40)
    ax.plot(mx, my, color='royalblue', lw=2.5, zorder=5, label='Binned mean')

    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7, label='Zero deviation')
    ax.axhline(0, color='grey',  lw=0.8, ls=':',  alpha=0.6)

    pstr = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Timing-Adjusted LW\n"
                  "(mean delta run expectancy for this timing bin, all swings)",
                  fontsize=10)
    ax.set_title(f"{title}\nr = {r:.3f}   p = {pstr}   n = {mask.sum():,}", fontsize=11)
    ax.legend(fontsize=9)

fig.suptitle("Barrel Deviation vs Timing-Distance-Adjusted Run Value\n"
             "2025 MLB Fastballs  ·  Shape bins = 20x20 tilt x attack_angle quantiles",
             fontsize=13, y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'barrel_intent_hex.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out}")

# ── Supplement: show intended LA surface (V1 and V2) ─────────────────────────
df['tilt_mid']  = df['tilt_bin'].apply(lambda b: b.mid if hasattr(b,'mid') else np.nan)
df['angle_mid'] = df['angle_bin'].apply(lambda b: b.mid if hasattr(b,'mid') else np.nan)

grid_barrel = (df[df['launch_speed_angle']==6]
               .groupby(['tilt_mid', 'angle_mid'], observed=True)['launch_angle']
               .mean().reset_index())
grid_all = (df.groupby(['tilt_mid', 'angle_mid'], observed=True)['launch_angle']
              .mean().reset_index())

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
for ax, grid, title in [
    (axes2[0], grid_barrel, "Intended LA Surface — Barrel Reference\n(mean barrel launch angle per swing shape bin)"),
    (axes2[1], grid_all,    "Intended LA Surface — All In-Play Reference\n(mean launch angle per swing shape bin)"),
]:
    sc = ax.scatter(grid['tilt_mid'], grid['angle_mid'],
                    c=grid['launch_angle'], cmap='RdYlGn',
                    s=80, edgecolors='k', linewidths=0.3, vmin=5, vmax=35)
    cb = fig2.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Mean Launch Angle (deg)", fontsize=9)
    ax.set_xlabel("Swing Path Tilt (deg)", fontsize=10)
    ax.set_ylabel("Attack Angle (deg)", fontsize=10)
    ax.set_title(title, fontsize=11)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'barrel_intent_surface.png')
fig2.savefig(out2, dpi=180, bbox_inches='tight')
plt.close(fig2)
print(f"Saved → {out2}")

print("\nDone.")
