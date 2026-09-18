"""
LW vs timing, split by full-season GMM mode (oppo-peak vs pull-peak).

Timing proxy: intercept_y − population median (inches), pull/early-positive.
Swing filter: ALL_SWINGS (in-play + fouls + swinging strikes / misses).

GMM is fit on ALL spline peaks (significant or not) so the full distribution
shape drives the mode assignment.
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY,
                          INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw,
                          weighted_moving_average,
                          TIMING_AXIS_LABEL, FASTBALL_TYPES, INSIDE_Z, MIDDLE_Z, OUTSIDE_Z, add_zone_and_matchup)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "deprecated")
MIN_PA   = 50
N_BINS   = 30

COLS = [
    "pitch_type", "batter", "stand",
    INTERCEPT_X, INTERCEPT_Y,
    "delta_run_exp", "description",
]

files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
df = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES) & df['description'].isin(ALL_SWINGS)].copy()
df = df.dropna(subset=[INTERCEPT_Y, 'delta_run_exp'])
center = add_timing(df)
print(f"Swings: {len(df):,}  (timing centre = {center:.2f} in)")

# ── Full-season spline peaks (ALL, not just significant) + GMM ────────────────
def peak_all(data, min_n=MIN_PA):
    """Return the spline peak loc + significance flag (loc may be NaN)."""
    if len(data) < min_n:
        return pd.Series({'peak': np.nan, 'peak_sig': False})
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n)
    return pd.Series({'peak': loc, 'peak_sig': bool(sig)})

# Significance-filtered wrapper (kept for compatibility)
def peak_timing(data, min_n=MIN_PA, **kw):
    if len(data) < min_n:
        return np.nan
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n, **kw)
    return loc if sig else np.nan

peak_tbl = df.groupby('batter').apply(peak_all)
peaks = peak_tbl['peak'].dropna()
n_sig = int(peak_tbl['peak_sig'].reindex(peaks.index).fillna(False).sum())
print(f"Hitters with spline peak: {len(peaks)}  "
      f"(significant: {n_sig}, {n_sig/len(peaks):.1%})")

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(peaks.values.reshape(-1, 1))
gm_means = gmm.means_.flatten()
oppo_idx = int(np.argmin(gm_means))

mode_map = {
    b: ('oppo-peak' if np.argmax(gmm.predict_proba([[v]])[0]) == oppo_idx else 'pull-peak')
    for b, v in peaks.items()
}
stand_map = df.groupby('batter')['stand'].agg(lambda s: s.mode()[0])

print(f"oppo-peak:  μ={gm_means[oppo_idx]:.1f} in   n={(pd.Series(mode_map)=='oppo-peak').sum()}")
print(f"pull-peak: μ={gm_means[1-oppo_idx]:.1f} in  n={(pd.Series(mode_map)=='pull-peak').sum()}")

df['mode'] = df['batter'].map(mode_map)
df = df.dropna(subset=['mode'])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

STYLE = {
    'oppo-peak': dict(color='#1f77b4', ls='-',  label_prefix='Oppo-peak'),
    'pull-peak': dict(color='#d62728', ls='--', label_prefix='Pull-peak'),
}

for mode, style in STYLE.items():
    sub = df[df['mode'] == mode]
    color, ls, prefix = style['color'], style['ls'], style['label_prefix']

    # Count-weighted moving average over bins
    gx, gy, bin_tbl = weighted_moving_average(
        sub['timing'].values, sub['delta_run_exp'].values,
        n_bins=N_BINS, window=5,
        min_per_bin=max(15, len(sub) // 500),
        min_window_total=max(50, len(sub) // 80))
    if len(gx) < 4:
        continue

    # Faint dots for raw bin means
    if len(bin_tbl):
        ax.plot(bin_tbl['t_mid'], bin_tbl['y_mean'], color=color, ls='',
                marker='o', ms=2.5, alpha=0.35, zorder=1)
    ax.plot(gx, gy, color=color, lw=2.5, ls=ls,
            label=f"{prefix}  (n={len(sub):,} pitches, {(pd.Series(mode_map)==mode).sum()} hitters)",
            zorder=3)

    # Peak from the moving-average curve
    peak_x = float(gx[np.argmax(gy)])
    peak_y = float(gy.max())
    ax.scatter(peak_x, peak_y, c=color, s=120, zorder=6,
               edgecolors='black', linewidths=1.0)
    ax.annotate(f"peak {peak_x:.1f} in",
                xy=(peak_x, peak_y),
                xytext=(peak_x + 1.5, peak_y + 0.002),
                fontsize=10, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
ax.axvline(0, color='grey',  lw=0.8, ls=':',  alpha=0.6)
ax.set_xlabel(TIMING_AXIS_LABEL, fontsize=11)
ax.set_ylabel("Mean Δ Run Expectancy", fontsize=11)
ax.set_title(f"LW vs Timing by GMM Mode\n"
             f"Oppo-peak (μ={gm_means[oppo_idx]:.1f} in) vs "
             f"Pull-peak (μ={gm_means[1-oppo_idx]:.1f} in) hitters — 2025 fastballs (all swings)",
             fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'lw_by_mode.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out}")
