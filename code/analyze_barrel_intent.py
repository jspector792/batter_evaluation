"""
Barrel placement via intended launch angle.

For each in-play fastball swing, estimate the *intended* launch angle as the
mean launch angle of swings with the same shape (swing_path_tilt × attack_angle
neighbourhood — 20×20 quantile grid).

Two versions:
  v1 (barrel intent)   — reference = mean LA of barrels (launch_speed_angle==6)
                         in the same swing-shape bin
  v2 (all-play intent) — reference = mean LA of all in-play balls in the bin

Barrel deviation = actual launch angle − intended launch angle

Primary visual:
  X: barrel deviation
  Y: timing-adjusted LW = mean(delta_run_exp) for swings in the same timing
     angle bin  (40 equal-width bins)

This shows how much barrel placement explains run value once timing is
factored into the y-axis.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
IN_PLAY  = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
SHAPE_BINS   = 20    # quantile grid per axis
TIMING_BINS  = 40    # equal-width timing bins for LW smoothing

COLS = [
    "pitch_type", "batter", "stand",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "delta_run_exp", "description",
    "swing_path_tilt", "attack_angle",
    "launch_angle", "launch_speed_angle",
]

# ── Load ─────────────────────────────────────────────────────────────────────
files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
df = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES) & df['description'].isin(IN_PLAY)].copy()

IX = 'intercept_ball_minus_batter_pos_x_inches'
IY = 'intercept_ball_minus_batter_pos_y_inches'
df = df.dropna(subset=[IX, IY, 'delta_run_exp',
                        'swing_path_tilt', 'attack_angle', 'launch_angle'])
print(f"In-play fastballs with all fields: {len(df):,}")

# ── Timing angle ─────────────────────────────────────────────────────────────
df['timing_raw'] = np.degrees(np.arctan2(df[IX], df[IY]))
TIMING_MED = df['timing_raw'].median()
df['timing'] = df['timing_raw'] - TIMING_MED

# ── Timing-adjusted LW: mean delta_run_exp per timing bin ────────────────────
timing_bin_edges = np.linspace(df['timing'].quantile(0.005),
                               df['timing'].quantile(0.995),
                               TIMING_BINS + 1)
df['timing_bin'] = pd.cut(df['timing'], bins=timing_bin_edges)
timing_lw = (df.groupby('timing_bin', observed=True)['delta_run_exp']
               .mean()
               .rename('timing_lw'))
df = df.join(timing_lw, on='timing_bin')
print(f"Timing LW range: [{df['timing_lw'].min():.4f}, {df['timing_lw'].max():.4f}]")
print(f"Timing bins with data: {df['timing_bin'].nunique()}")

# ── Swing-shape grid: 20×20 quantile bins ────────────────────────────────────
df['tilt_bin']  = pd.qcut(df['swing_path_tilt'], q=SHAPE_BINS, duplicates='drop')
df['angle_bin'] = pd.qcut(df['attack_angle'],    q=SHAPE_BINS, duplicates='drop')
df['shape_key'] = list(zip(df['tilt_bin'].astype(str), df['angle_bin'].astype(str)))

# ── Version 1: barrel-based intended LA ──────────────────────────────────────
barrels = df[df['launch_speed_angle'] == 6]
barrel_intent = (barrels.groupby('shape_key')['launch_angle']
                         .mean()
                         .rename('intended_la_v1'))
df = df.join(barrel_intent, on='shape_key')

# ── Version 2: all-in-play intended LA ───────────────────────────────────────
all_intent = (df.groupby('shape_key')['launch_angle']
                .mean()
                .rename('intended_la_v2'))
df = df.join(all_intent, on='shape_key')

# ── Barrel deviations ─────────────────────────────────────────────────────────
df['barrel_dev_v1'] = df['launch_angle'] - df['intended_la_v1']
df['barrel_dev_v2'] = df['launch_angle'] - df['intended_la_v2']

# Drop swings where either intended LA is missing (no barrels in that cell — shouldn't happen)
df_v1 = df.dropna(subset=['barrel_dev_v1', 'timing_lw'])
df_v2 = df.dropna(subset=['barrel_dev_v2', 'timing_lw'])
print(f"\nVersion 1 (barrel intent): {len(df_v1):,} swings  "
      f"| barrel_dev mean={df_v1['barrel_dev_v1'].mean():.2f}°  std={df_v1['barrel_dev_v1'].std():.2f}°")
print(f"Version 2 (all-play intent): {len(df_v2):,} swings  "
      f"| barrel_dev mean={df_v2['barrel_dev_v2'].mean():.2f}°  std={df_v2['barrel_dev_v2'].std():.2f}°")

# Correlations
r_v1, p_v1 = pearsonr(df_v1['barrel_dev_v1'], df_v1['timing_lw'])
r_v2, p_v2 = pearsonr(df_v2['barrel_dev_v2'], df_v2['timing_lw'])
print(f"\nr(barrel_dev_v1, timing_lw) = {r_v1:.3f}  p={p_v1:.4f}")
print(f"r(barrel_dev_v2, timing_lw) = {r_v2:.3f}  p={p_v2:.4f}")

# Also report: does barrel deviation predict RAW delta_run_exp?
r_raw_v1, p_raw_v1 = pearsonr(df_v1['barrel_dev_v1'], df_v1['delta_run_exp'])
r_raw_v2, p_raw_v2 = pearsonr(df_v2['barrel_dev_v2'], df_v2['delta_run_exp'])
print(f"\nr(barrel_dev_v1, raw delta_run_exp) = {r_raw_v1:.3f}  p={p_raw_v1:.4f}")
print(f"r(barrel_dev_v2, raw delta_run_exp) = {r_raw_v2:.3f}  p={p_raw_v2:.4f}")

# Binned means for overlay line
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
     "(intended LA = mean LA of barrels with same swing shape)",
     "Barrel Deviation from Barrel Intent (°)\nactual launch angle − mean barrel launch angle for this swing shape"),
    (df_v2, 'barrel_dev_v2', r_v2, p_v2,
     "V2: All-In-Play Intent Reference\n"
     "(intended LA = mean LA of all in-play balls with same swing shape)",
     "Barrel Deviation from Average Intent (°)\nactual launch angle − mean in-play launch angle for this swing shape"),
]

for ax, (data, dev_col, r, p, title, xlabel) in zip(axes, CONFIGS):
    x = data[dev_col].values
    y = data['timing_lw'].values

    # Clip extreme outliers for display
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

    # Binned mean overlay
    mx, my = binned_means(x_plot, y_plot, n=40)
    ax.plot(mx, my, color='royalblue', lw=2.5, zorder=5, label='Binned mean')

    # Reference lines
    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7, label='Zero deviation')
    ax.axhline(0, color='grey',  lw=0.8, ls=':',  alpha=0.6)

    pstr = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Timing-Adjusted LW\n(mean Δ run expectancy for this timing angle bin)",
                  fontsize=10)
    ax.set_title(f"{title}\nr = {r:.3f}   p = {pstr}   n = {mask.sum():,}", fontsize=11)
    ax.legend(fontsize=9)

fig.suptitle("Barrel Deviation vs Timing-Adjusted Run Value\n"
             "2025 MLB In-Play Fastballs",
             fontsize=13, y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'barrel_intent_hex.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out}")

# ── Supplement: show intended LA surface for both versions ───────────────────
# 2D heatmap of the swing-shape grid, coloured by intended LA
# Compute numeric bin centres for plotting
df['tilt_mid']  = df['tilt_bin'].apply(lambda b: b.mid if hasattr(b,'mid') else np.nan)
df['angle_mid'] = df['angle_bin'].apply(lambda b: b.mid if hasattr(b,'mid') else np.nan)

grid_barrel = (df[df['launch_speed_angle']==6]
               .groupby(['tilt_mid','angle_mid'], observed=True)['launch_angle']
               .mean().reset_index())
grid_all = (df.groupby(['tilt_mid','angle_mid'], observed=True)['launch_angle']
              .mean().reset_index())

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
for ax, grid, title in [
    (axes2[0], grid_barrel, "Intended LA Surface — Barrel Reference\n(mean barrel launch angle per swing-shape bin)"),
    (axes2[1], grid_all,    "Intended LA Surface — All In-Play Reference\n(mean launch angle per swing-shape bin)"),
]:
    sc = ax.scatter(grid['tilt_mid'], grid['angle_mid'],
                    c=grid['launch_angle'], cmap='RdYlGn',
                    s=120, edgecolors='k', linewidths=0.3, vmin=5, vmax=35)
    cb = fig2.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Mean Launch Angle (°)", fontsize=9)
    ax.set_xlabel("Swing Path Tilt (°)", fontsize=10)
    ax.set_ylabel("Attack Angle (°)", fontsize=10)
    ax.set_title(title, fontsize=11)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'barrel_intent_surface.png')
fig2.savefig(out2, dpi=180, bbox_inches='tight')
plt.close(fig2)
print(f"Saved → {out2}")

print("\nDone.")
