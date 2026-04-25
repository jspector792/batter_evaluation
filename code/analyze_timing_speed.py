"""
Does pitch speed shift the optimal timing angle?

Hypothesis: faster pitches force batters to make contact earlier (more negative
timing angle), so the bimodal peak-timing distribution may reflect hitters
facing different speed profiles rather than two distinct swing archetypes.

Tests:
  1. Population-level LW-vs-timing curves by speed quartile — does the peak shift?
  2. Within-hitter paired comparison (Q1 slow vs Q4 fast peaks) — does the SAME
     hitter peak earlier on faster pitches?
  3. Per-hitter peak distributions by speed quartile — does the bimodality change?
  4. Speed sensitivity (Q4 − Q1 peak shift) vs hitter features — who adjusts most?

Speed quartiles (population, in-play fastballs):
  Q1 ≤ 91.8 mph | Q2 91.8–93.8 | Q3 93.8–95.7 | Q4 > 95.7 mph
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, ttest_rel, ttest_ind
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
IN_PLAY  = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
MIN_PA_SEASON = 50   # full-season GMM
MIN_PA_BIN    = 15   # per-hitter per speed bin
POLY_SEASON   = 4
POLY_BIN      = 3    # lower degree for smaller samples
N_BINS        = 30   # bins for LW-vs-timing curves

COLS = [
    "pitch_type", "batter", "stand", "p_throws",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "delta_run_exp", "description", "release_speed",
    "bat_speed", "attack_angle", "swing_path_tilt",
    "launch_speed", "launch_angle", "launch_speed_angle",
]

# ── Load ─────────────────────────────────────────────────────────────────────
files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
chunks = [pd.read_csv(f, usecols=COLS) for f in files]
df = pd.concat(chunks, ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES)].copy()
df = df[df['description'].isin(IN_PLAY)].copy()

IX = 'intercept_ball_minus_batter_pos_x_inches'
IY = 'intercept_ball_minus_batter_pos_y_inches'
df = df.dropna(subset=[IX, IY, 'delta_run_exp', 'stand', 'release_speed'])
print(f"In-play fastballs: {len(df):,}")

# Timing angle
df['timing_raw'] = np.degrees(np.arctan2(df[IX], df[IY]))
TIMING_MED = df['timing_raw'].median()
df['timing'] = df['timing_raw'] - TIMING_MED

# Speed quartiles (population boundaries)
Q_BOUNDS = df['release_speed'].quantile([.25, .5, .75]).values
labels = [
    f"Q1 (≤{Q_BOUNDS[0]:.1f})",
    f"Q2 ({Q_BOUNDS[0]:.1f}–{Q_BOUNDS[1]:.1f})",
    f"Q3 ({Q_BOUNDS[1]:.1f}–{Q_BOUNDS[2]:.1f})",
    f"Q4 (>{Q_BOUNDS[2]:.1f})",
]
df['speed_q'] = pd.cut(
    df['release_speed'],
    bins=[-np.inf] + list(Q_BOUNDS) + [np.inf],
    labels=labels, ordered=True,
)
print(f"\nSpeed quartile breakdown:")
print(df.groupby('speed_q', observed=True)['pitch_type']
        .value_counts().unstack(fill_value=0))

# ── Peak-finding helper ───────────────────────────────────────────────────────
def peak_timing(data, min_n, degree):
    if len(data) < min_n:
        return np.nan
    x = data['timing'].values
    y = data['delta_run_exp'].values
    try:
        p = np.poly1d(np.polyfit(x, y, degree))
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        xs = np.linspace(lo, hi, 500)
        return float(xs[np.argmax(p(xs))])
    except Exception:
        return np.nan

# ── LW-vs-timing curve helper ─────────────────────────────────────────────────
def lw_curve(data, n_bins=N_BINS):
    data = data.dropna(subset=['timing', 'delta_run_exp'])
    if len(data) < 200:
        return None
    bins = np.linspace(data['timing'].quantile(0.005),
                       data['timing'].quantile(0.995), n_bins + 1)
    d = data.copy()
    d['bin'] = pd.cut(d['timing'], bins=bins)
    g = (d.groupby('bin', observed=True)
          .agg(mean_lw=('delta_run_exp', 'mean'),
               sem_lw =('delta_run_exp', 'sem'),
               mid    =('timing', 'median'),
               n      =('delta_run_exp', 'count'))
          .dropna().reset_index())
    return g

# ── Full-season peaks + GMM ───────────────────────────────────────────────────
print("\nFitting full-season GMM...")
season_peaks = (df.groupby('batter')
                  .apply(lambda g: peak_timing(g, MIN_PA_SEASON, POLY_SEASON))
                  .dropna()
                  .rename('peak'))
gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
gmm.fit(season_peaks.values.reshape(-1, 1))
gm_means = gmm.means_.flatten()
early_idx = int(np.argmin(gm_means))
late_idx  = int(np.argmax(gm_means))

def assign_mode(val):
    if np.isnan(val): return np.nan
    return 'early' if np.argmax(gmm.predict_proba([[val]])[0]) == early_idx else 'late'

season_df = pd.DataFrame({
    'peak_season': season_peaks,
    'mode_season': season_peaks.apply(assign_mode),
    'stand': df.groupby('batter')['stand'].agg(lambda s: s.mode()[0]),
})
print(f"  early-peak: μ={gm_means[early_idx]:.1f}°")
print(f"  late-peak:  μ={gm_means[late_idx]:.1f}°")

# ── Population-level peak per speed bin ──────────────────────────────────────
pop_peaks = {}
for q in labels:
    sub = df[df['speed_q'] == q]
    pk = peak_timing(sub, min_n=200, degree=POLY_SEASON)
    pop_peaks[q] = pk
    print(f"  Population peak [{q}]: {pk:.1f}°  (n={len(sub):,})")

# ── Per-hitter peaks by speed bin ────────────────────────────────────────────
print("\nComputing per-hitter peaks by speed quartile...")
bin_peaks = {}
for q in labels:
    sub = df[df['speed_q'] == q]
    peaks_q = (sub.groupby('batter')
                  .apply(lambda g: peak_timing(g, MIN_PA_BIN, POLY_BIN))
                  .dropna()
                  .rename('peak'))
    bin_peaks[q] = peaks_q
    print(f"  {q}: {len(peaks_q)} hitters  "
          f"mean={peaks_q.mean():.1f}°  median={peaks_q.median():.1f}°  "
          f"std={peaks_q.std():.1f}°")

# Within-hitter Q1 vs Q4 paired comparison
q1_label, q4_label = labels[0], labels[3]
paired_speed = (
    pd.DataFrame({'Q1': bin_peaks[q1_label], 'Q4': bin_peaks[q4_label]})
    .dropna()
    .join(season_df[['mode_season', 'stand']], how='inner')
)
paired_speed['shift'] = paired_speed['Q4'] - paired_speed['Q1']
t_pair, p_pair = ttest_rel(paired_speed['Q1'], paired_speed['Q4'])
r_q1q4, p_r = pearsonr(paired_speed['Q1'], paired_speed['Q4'])
print(f"\nWithin-hitter Q1 vs Q4 paired comparison (n={len(paired_speed)}):")
print(f"  Q1 (slow) mean={paired_speed['Q1'].mean():.1f}°  "
      f"Q4 (fast) mean={paired_speed['Q4'].mean():.1f}°")
print(f"  Mean shift (Q4 − Q1) = {paired_speed['shift'].mean():.1f}°")
print(f"  Paired t-test: t={t_pair:.3f}  p={p_pair:.4f}")
print(f"  Cross-speed correlation: r={r_q1q4:.3f}  p={p_r:.4f}")
print(f"  % hitters peaking earlier on fast pitches: "
      f"{(paired_speed['shift'] < 0).mean():.1%}")

# ── Per-hitter features (for sensitivity analysis) ────────────────────────────
feat_cols = ['bat_speed', 'attack_angle', 'swing_path_tilt',
             'launch_speed', 'launch_angle', 'launch_speed_angle']
feat_means = df.groupby('batter')[feat_cols].mean().add_suffix('_mean')
barrel_rate = (df.assign(barrel=(df['launch_speed_angle']==6).astype(float))
                 .groupby('batter')['barrel'].mean().rename('barrel_rate'))
mean_speed_faced = df.groupby('batter')['release_speed'].mean().rename('mean_speed_faced')
paired_speed = paired_speed.join(feat_means).join(barrel_rate).join(mean_speed_faced)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = ['#2166ac', '#74add1', '#f46d43', '#d73027']  # cool→warm for slow→fast
MODE_COLORS = {'early': '#1f77b4', 'late': '#d62728'}

# ── Fig 1: Population LW-vs-timing curves by speed quartile ──────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

ax = axes1[0]
for q, color in zip(labels, COLORS):
    g = lw_curve(df[df['speed_q'] == q])
    if g is None:
        continue
    ax.fill_between(g['mid'],
                    g['mean_lw'] - 1.96 * g['sem_lw'],
                    g['mean_lw'] + 1.96 * g['sem_lw'],
                    alpha=0.10, color=color)
    ax.plot(g['mid'], g['mean_lw'], color=color, lw=2.2, marker='o', ms=3,
            label=f"{q}  (n={len(df[df['speed_q']==q]):,})")
    # mark peak
    pk = pop_peaks[q]
    if not np.isnan(pk):
        pk_lw = g.loc[(g['mid'] - pk).abs().idxmin(), 'mean_lw']
        ax.scatter(pk, pk_lw, c=color, s=80, zorder=5, edgecolors='k', linewidths=0.8)

ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(0, color='grey', lw=0.8, ls=':')
ax.set_xlabel("Timing Angle (°, centred on population median)\n← early (pull-side)   late (oppo-side) →",
              fontsize=10)
ax.set_ylabel("Mean Δ Run Expectancy", fontsize=10)
ax.set_title("LW vs Timing by Pitch Speed Quartile\n"
             "(dots = estimated population peak)", fontsize=11)
ax.legend(fontsize=8)

# Panel 2: bar chart of population peak by speed quartile
ax = axes1[1]
pk_vals = [pop_peaks[q] for q in labels]
bars = ax.bar(range(4), pk_vals, color=COLORS, edgecolor='k', lw=0.5, alpha=0.85)
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=9, rotation=10)
ax.set_ylabel("Population Peak Timing Angle (°)", fontsize=10)
ax.set_title("Population Peak Timing by Speed Quartile\n"
             "← more early = further below 0", fontsize=11)
for i, (bar, val) in enumerate(zip(bars, pk_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, val + (0.3 if val >= 0 else -0.8),
            f"{val:.1f}°", ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'speed_1_population_curves.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\nSaved speed_1_population_curves.png")

# ── Fig 2: Within-hitter Q1 vs Q4 scatter + shift distribution ───────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: Q1 vs Q4 scatter coloured by full-season mode
ax = axes2[0]
for mode, color, marker in [('early', '#1f77b4', 'o'), ('late', '#d62728', '^')]:
    sub = paired_speed[paired_speed['mode_season'] == mode]
    ax.scatter(sub['Q1'], sub['Q4'], c=color, alpha=0.4, s=18, marker=marker,
               label=f"{mode}-peak (n={len(sub)})")
lo = min(paired_speed[['Q1','Q4']].min()) - 2
hi = max(paired_speed[['Q1','Q4']].max()) + 2
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x (no shift)')
ax.axhline(0, color='grey', lw=0.6, ls=':')
ax.axvline(0, color='grey', lw=0.6, ls=':')
mean_shift = paired_speed['shift'].mean()
ax.set_xlabel(f"Peak Timing — Slow pitches {q1_label} (°)", fontsize=10)
ax.set_ylabel(f"Peak Timing — Fast pitches {q4_label} (°)", fontsize=10)
ax.set_title(f"Within-Hitter: Slow vs Fast Peak Timing\n"
             f"r={r_q1q4:.3f}  mean shift={mean_shift:+.1f}°  "
             f"paired t p={p_pair:.4f}  n={len(paired_speed)}", fontsize=10)
ax.legend(fontsize=8)

# Panel 2: distribution of shift (Q4 - Q1)
ax = axes2[1]
shifts = paired_speed['shift'].values
ax.hist(shifts, bins=30, edgecolor='k', lw=0.4, color='slategrey', alpha=0.75, density=True)
xs = np.linspace(shifts.min(), shifts.max(), 300)
ax.plot(xs, gaussian_kde(shifts)(xs), 'k-', lw=2)
ax.axvline(0, color='black', lw=1.2, ls='--', label='No shift')
ax.axvline(shifts.mean(), color='red', lw=2, ls=':',
           label=f"Mean shift = {shifts.mean():+.1f}°")
pct_earlier = (shifts < 0).mean()
ax.set_xlabel("Peak Timing Shift: Fast − Slow (°)\n< 0 = earlier on fast pitches", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"Distribution of Within-Hitter Speed Shift\n"
             f"{pct_earlier:.0%} of hitters peak earlier on fast pitches", fontsize=10)
ax.legend(fontsize=9)

# Panel 3: all 4 quartile means per hitter (paired lines for a sample)
ax = axes2[2]
# Compute all 4 quartile peaks per hitter
all_q_wide = pd.DataFrame({q: bin_peaks[q] for q in labels}).dropna(how='all')
# For a random sample of 40 hitters, draw lines
np.random.seed(42)
sample = all_q_wide.dropna(thresh=3).sample(min(60, len(all_q_wide.dropna(thresh=3))))
x_pos = range(4)
for batter in sample.index:
    row = sample.loc[batter]
    valid = row.dropna()
    if len(valid) < 2:
        continue
    valid_x = [labels.index(l) for l in valid.index]
    ax.plot(valid_x, valid.values, color='grey', alpha=0.2, lw=0.8)

# Overlay per-quartile mean with CI
means_q = [bin_peaks[q].mean() for q in labels]
sems_q  = [bin_peaks[q].sem()  for q in labels]
ax.errorbar(x_pos, means_q, yerr=[1.96*s for s in sems_q],
            color='black', lw=2.5, capsize=5, marker='o', ms=7,
            label='Mean ± 95% CI (all hitters)', zorder=5)
# Also show by mode
for mode, color in MODE_COLORS.items():
    mode_batters = season_df[season_df['mode_season'] == mode].index
    mode_means = [bin_peaks[q].reindex(mode_batters).dropna().mean() for q in labels]
    ax.plot(x_pos, mode_means, color=color, lw=2, ls='--', marker='s', ms=6,
            label=f"{mode}-peak mean")
ax.axhline(0, color='grey', lw=0.8, ls=':')
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=8, rotation=10)
ax.set_ylabel("Per-Hitter Peak Timing Angle (°)", fontsize=10)
ax.set_title("Peak Timing vs Speed Quartile\n"
             "Grey = individual hitters, black = population mean", fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'speed_2_within_hitter.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("Saved speed_2_within_hitter.png")

# ── Fig 3: Per-hitter peak distributions by speed quartile — bimodality ───────
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: KDE per speed quartile overlaid
ax = axes3[0]
for q, color in zip(labels, COLORS):
    vals = bin_peaks[q].values
    if len(vals) < 10:
        continue
    xs = np.linspace(vals.min(), vals.max(), 400)
    ax.plot(xs, gaussian_kde(vals)(xs), color=color, lw=2.5,
            label=f"{q}  μ={vals.mean():.1f}°  σ={vals.std():.1f}°")
    # GMM fit per quartile
    if len(vals) >= 20:
        gq = GaussianMixture(n_components=2, random_state=42)
        gq.fit(vals.reshape(-1, 1))
        from scipy.stats import norm
        gq_means = gq.means_.flatten()
        gq_weights = gq.weights_
        gq_covs = gq.covariances_.flatten()
        for m, w, v in zip(gq_means, gq_weights, gq_covs):
            ax.plot(xs, w * norm.pdf(xs, m, np.sqrt(v)),
                    color=color, lw=1, ls=':', alpha=0.5)

ax.axvline(0, color='black', lw=1, ls='--', label='Neutral (0°)')
ax.set_xlabel("Per-Hitter Peak Timing Angle (°)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Per-Hitter Peak Distribution by Speed Quartile\n"
             "Dotted = GMM 2-component fit per quartile", fontsize=11)
ax.legend(fontsize=8)

# Panel 2: violin plot of per-hitter peaks by quartile
ax = axes3[1]
data_vio = pd.concat(
    [pd.DataFrame({'peak': bin_peaks[q], 'quartile': q}) for q in labels],
    ignore_index=True,
)
sns.violinplot(data=data_vio, x='quartile', y='peak', palette=COLORS,
               ax=ax, inner='box', cut=0, order=labels)
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_xticklabels(labels, rotation=12, fontsize=8)
ax.set_ylabel("Per-Hitter Peak Timing Angle (°)", fontsize=10)
ax.set_xlabel("Speed Quartile", fontsize=10)
ax.set_title("Distribution of Per-Hitter Peaks by Speed Quartile\n"
             "(box = IQR, white dot = median)", fontsize=11)

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'speed_3_distributions.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved speed_3_distributions.png")

# ── Fig 4: Speed sensitivity vs hitter features ───────────────────────────────
# Speed sensitivity = Q4 peak − Q1 peak (more negative = bigger early shift on fast)
SENSE_FEATS = [
    ('bat_speed_mean',        'Bat Speed (mph)'),
    ('attack_angle_mean',     'Attack Angle (°)'),
    ('swing_path_tilt_mean',  'Swing Path Tilt (°)'),
    ('launch_speed_mean',     'Exit Velocity (mph)'),
    ('launch_speed_angle_mean','Contact Quality'),
    ('barrel_rate',           'Barrel Rate'),
    ('mean_speed_faced',      'Mean Speed Faced (mph)'),
]

print("\nCorrelations with speed sensitivity (Q4 − Q1 shift):")
print(f"  {'Variable':<35} {'r':>8} {'p':>8}")
corr_rows = []
for col, label in SENSE_FEATS:
    sub = paired_speed[['shift', col]].dropna()
    if len(sub) < 10:
        continue
    r, p = pearsonr(sub['shift'], sub[col])
    print(f"  {label:<35} {r:>8.3f} {p:>8.4f}")
    corr_rows.append({'col': col, 'label': label, 'r': r, 'p': p})
corr_sense = pd.DataFrame(corr_rows)

NCOLS_F, NROWS_F = 4, 2
fig4, axes4 = plt.subplots(NROWS_F, NCOLS_F, figsize=(6*NCOLS_F, 5*NROWS_F))
axes4 = axes4.flatten()

for i, (col, label) in enumerate(SENSE_FEATS):
    ax = axes4[i]
    sub = paired_speed[['shift', col, 'mode_season', 'stand']].dropna()
    for mode, color, marker in [('early','#1f77b4','o'), ('late','#d62728','^')]:
        s = sub[sub['mode_season'] == mode]
        ax.scatter(s[col], s['shift'], c=color, alpha=0.35, s=14, marker=marker,
                   label=f"{mode} (n={len(s)})")
    # regression
    x_all = sub[col].values
    y_all = sub['shift'].values
    m, b = np.polyfit(x_all, y_all, 1)
    xs_fit = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(xs_fit, m * xs_fit + b, 'k-', lw=1.5)
    r, p = pearsonr(x_all, y_all)
    pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Speed Sensitivity: Q4 − Q1 (°)\n< 0 = earlier on fast pitches", fontsize=8)
    ax.set_title(f"{label}\nr={r:.3f}  p={pstr}", fontsize=9)
    if i == 0:
        ax.legend(fontsize=7)

for j in range(i + 1, len(axes4)):
    axes4[j].set_visible(False)

fig4.suptitle("Who Adjusts Timing Most for Pitch Speed?\n"
              "Speed sensitivity = peak timing Q4 (fast) − Q1 (slow)",
              fontsize=13, y=1.01)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'speed_4_sensitivity.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved speed_4_sensitivity.png")

print("\nDone.")
