"""
LW vs timing angle, split by full-season GMM mode (early-peak vs late-peak).
Same binned-mean method as timing_binary_4_h1_matchup.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
IN_PLAY  = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
MIN_PA   = 50
POLY_DEG = 4
N_BINS   = 30

COLS = [
    "pitch_type", "batter", "stand",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "delta_run_exp", "description",
]

files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
df = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES) & df['description'].isin(IN_PLAY)].copy()

IX = 'intercept_ball_minus_batter_pos_x_inches'
IY = 'intercept_ball_minus_batter_pos_y_inches'
df = df.dropna(subset=[IX, IY, 'delta_run_exp'])

df['timing_raw'] = np.degrees(np.arctan2(df[IX], df[IY]))
TIMING_MED = df['timing_raw'].median()
df['timing'] = df['timing_raw'] - TIMING_MED

# ── Full-season peak + GMM ────────────────────────────────────────────────────
def peak_timing(data):
    if len(data) < MIN_PA:
        return np.nan
    x, y = data['timing'].values, data['delta_run_exp'].values
    try:
        p = np.poly1d(np.polyfit(x, y, POLY_DEG))
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        xs = np.linspace(lo, hi, 500)
        return float(xs[np.argmax(p(xs))])
    except Exception:
        return np.nan

peaks = df.groupby('batter').apply(peak_timing).dropna()

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(peaks.values.reshape(-1, 1))
gm_means = gmm.means_.flatten()
early_idx = int(np.argmin(gm_means))

mode_map = {
    b: ('early-peak' if np.argmax(gmm.predict_proba([[v]])[0]) == early_idx else 'late-peak')
    for b, v in peaks.items()
}
stand_map = df.groupby('batter')['stand'].agg(lambda s: s.mode()[0])

print(f"early-peak: μ={gm_means[early_idx]:.1f}°  n={(pd.Series(mode_map)=='early-peak').sum()}")
print(f"late-peak:  μ={gm_means[1-early_idx]:.1f}°  n={(pd.Series(mode_map)=='late-peak').sum()}")

df['mode'] = df['batter'].map(mode_map)
df = df.dropna(subset=['mode'])

# ── Binned LW curve ───────────────────────────────────────────────────────────
def lw_curve(data):
    data = data.dropna(subset=['timing', 'delta_run_exp'])
    bins = np.linspace(data['timing'].quantile(0.005),
                       data['timing'].quantile(0.995), N_BINS + 1)
    data = data.copy()
    data['bin'] = pd.cut(data['timing'], bins=bins)
    return (data.groupby('bin', observed=True)
                .agg(mean_lw=('delta_run_exp', 'mean'),
                     sem_lw =('delta_run_exp', 'sem'),
                     mid    =('timing', 'median'),
                     n      =('delta_run_exp', 'count'))
                .dropna().reset_index())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

STYLE = {
    'early-peak': dict(color='#1f77b4', ls='-',  label_prefix='Early-peak'),
    'late-peak':  dict(color='#d62728', ls='--', label_prefix='Late-peak'),
}

for mode, style in STYLE.items():
    sub = df[df['mode'] == mode]
    g = lw_curve(sub)
    color, ls, prefix = style['color'], style['ls'], style['label_prefix']

    ax.fill_between(g['mid'],
                    g['mean_lw'] - 1.96 * g['sem_lw'],
                    g['mean_lw'] + 1.96 * g['sem_lw'],
                    alpha=0.12, color=color)
    ax.plot(g['mid'], g['mean_lw'], color=color, lw=2.5, ls=ls, marker='o', ms=3,
            label=f"{prefix}  (n={len(sub):,} pitches, {(pd.Series(mode_map)==mode).sum()} hitters)")

    # mark the peak of a smoothed polynomial through the binned means
    try:
        p = np.poly1d(np.polyfit(g['mid'], g['mean_lw'], 4))
        xs = np.linspace(g['mid'].min(), g['mid'].max(), 1000)
        peak_x = float(xs[np.argmax(p(xs))])
        peak_y = float(p(peak_x))
        ax.scatter(peak_x, peak_y, c=color, s=120, zorder=6,
                   edgecolors='black', linewidths=1.0)
        ax.annotate(f"peak {peak_x:+.1f}°",
                    xy=(peak_x, peak_y),
                    xytext=(peak_x + 1.5, peak_y + 0.002),
                    fontsize=10, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    except Exception:
        pass

ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
ax.axvline(0, color='grey',  lw=0.8, ls=':',  alpha=0.6)
ax.set_xlabel("Timing Angle (°, centred on population median)\n"
              "← early (pull-side contact)          late (oppo-side contact) →",
              fontsize=11)
ax.set_ylabel("Mean Δ Run Expectancy", fontsize=11)
ax.set_title("LW vs Timing Angle by GMM Mode\n"
             "Early-peak (μ=−10.4°) vs Late-peak (μ=+12.3°) hitters — in-play fastballs 2025",
             fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'lw_by_mode.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out}")
