"""
Analyze what drives the bimodal distribution of per-hitter peak timing.

Timing proxy: intercept_y − population median (inches), pull/early-positive.
Swing filter: ALL_SWINGS (in-play + fouls + swinging strikes / misses).

Note: For the GMM fit we use ALL spline peaks (significant or not) so we can
characterise the full distribution. A `peak_sig` flag is carried alongside;
downstream analyses use all peaks but visualise significant ones more saturated.

Steps:
  1. Per-hitter spline peak (all peaks + significance flag)
  2. Compute per-hitter means for swing mechanics & outcome variables
  3. Fit a 2-component GMM to identify two hitter modes
  4. Figure 1: KDE with GMM components
  5. Figure 2: Scatter grid — peak timing vs each variable
  6. Figure 3: Violin grid — each variable by GMM mode
  7. Figure 4: Correlation heatmap + ranked bar chart
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, spearmanr, ttest_ind
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY,
                          INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw,
                          TIMING_AXIS_LABEL, FASTBALL_TYPES, INSIDE_Z, MIDDLE_Z, OUTSIDE_Z, add_zone_and_matchup)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "deprecated")
MIN_PA   = 50

COLS = [
    "pitch_type", "batter", "stand", "p_throws",
    "delta_run_exp", "description",
    "bat_speed", "swing_length", "attack_angle",
    INTERCEPT_X, INTERCEPT_Y,
    "swing_path_tilt",
    "launch_speed", "launch_angle", "launch_speed_angle",
]

# ── Load ─────────────────────────────────────────────────────────────────────
files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
chunks = [pd.read_csv(f, usecols=COLS) for f in files]
df = pd.concat(chunks, ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES)].copy()
df = df[df['description'].isin(ALL_SWINGS)].copy()

df = df.dropna(subset=['delta_run_exp', 'stand', 'p_throws'])
df = df.dropna(subset=[INTERCEPT_Y, 'delta_run_exp'])
center = add_timing(df)
print(f"Swings: {len(df):,}  (timing centre = {center:.2f} in)")

# ── bat_speed_rel: per-batter max from this same dataset ─────────────────────
bat_counts = df.groupby('batter')['bat_speed'].count()
bat_max = df.groupby('batter')['bat_speed'].max()
bat_max = bat_max[bat_counts >= 5]
df['bat_speed_rel'] = df['bat_speed'] / df['batter'].map(bat_max)

# ── Per-hitter spline peak + significance flag ───────────────────────────────
def peak_with_sig(data, min_n=MIN_PA):
    """Return (loc, sig) for the spline peak — loc returned even when not sig."""
    if len(data) < min_n:
        return pd.Series({'peak': np.nan, 'peak_sig': False})
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n)
    return pd.Series({'peak': loc, 'peak_sig': bool(sig)})

# Convenience wrapper (significance-filtered) — kept for callers that want it
def peak_timing(data, min_n=MIN_PA, **kw):
    if len(data) < min_n:
        return np.nan
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n, **kw)
    return loc if sig else np.nan

print("Computing per-hitter spline peaks (with significance flag)...")
peak_table = df.groupby('batter').apply(peak_with_sig)
all_peaks = peak_table['peak'].dropna().rename('peak')
peak_sig  = peak_table['peak_sig'].reindex(all_peaks.index).fillna(False).astype(bool)
n_any = len(all_peaks)
n_sig = int(peak_sig.sum())
print(f"  Hitters with any spline peak:    {n_any}")
print(f"  Hitters with significant peak:   {n_sig}  "
      f"({(n_sig/n_any if n_any else 0):.1%} of those)")

# ── Per-hitter feature means ──────────────────────────────────────────────────
# attack_angle_mean = per-hitter average attack angle (vs peak = optimal angle)
MECH_VARS = [
    'bat_speed', 'bat_speed_rel',
    'attack_angle', 'swing_path_tilt',
    'swing_length', 'timing',
    'launch_angle', 'launch_speed', 'launch_speed_angle',
]
# Only compute means for in-play swings for outcome metrics
in_play_mask = df['description'].isin(IN_PLAY)
outcome_vars = ['launch_angle', 'launch_speed', 'launch_speed_angle']
hitter_means_contact = df.groupby('batter')[
    [v for v in MECH_VARS if v not in outcome_vars]
].mean().add_suffix('_mean')
hitter_means_inplay = df[in_play_mask].groupby('batter')[outcome_vars].mean().add_suffix('_mean')
hitter_means = hitter_means_contact.join(hitter_means_inplay)

barrel_rate = (
    df[in_play_mask]
    .assign(is_barrel=(df[in_play_mask]['launch_speed_angle'] == 6).astype(float))
    .groupby('batter')['is_barrel'].mean()
    .rename('barrel_rate')
)
hard_hit_rate = (
    df[in_play_mask]
    .assign(is_hard=(df[in_play_mask]['launch_speed'] >= 95).astype(float))
    .groupby('batter')['is_hard'].mean()
    .rename('hard_hit_rate')
)

stand_col = df.groupby('batter')['stand'].agg(lambda s: s.mode()[0])
n_col = df.groupby('batter').size().rename('n')

hitters = (pd.DataFrame({'peak': all_peaks, 'peak_sig': peak_sig})
             .join(hitter_means)
             .join(barrel_rate)
             .join(hard_hit_rate)
             .join(stand_col)
             .join(n_col)
             .dropna(subset=['peak']))
hitters = hitters[hitters['n'] >= MIN_PA].copy()
print(f"Hitters for mode analysis (all peaks, sig or not): {len(hitters)}  "
      f"(significant: {int(hitters['peak_sig'].sum())})")

# ── 2-component GMM on peak attack angle ─────────────────────────────────────
gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
gmm.fit(hitters[['peak']])
hitters['gmm_mode'] = gmm.predict(hitters[['peak']])

gm_means = gmm.means_.flatten()
oppo_idx = int(np.argmin(gm_means))
pull_idx = int(np.argmax(gm_means))
hitters['mode_label'] = hitters['gmm_mode'].map({oppo_idx: 'oppo-peak', pull_idx: 'pull-peak'})

n_low  = (hitters['mode_label'] == 'oppo-peak').sum()
n_high = (hitters['mode_label'] == 'pull-peak').sum()
print(f"\nGMM clusters (fit on ALL spline peaks):")
print(f"  oppo-peak (oppo/late):    mean={gm_means[oppo_idx]:.1f} in  n={n_low}")
print(f"  pull-peak (pull/early):   mean={gm_means[pull_idx]:.1f} in  n={n_high}")

# ── All feature columns (with readable labels) ────────────────────────────────
FEAT_COLS = [
    ('bat_speed_mean',        'Mean Bat Speed (mph)'),
    ('bat_speed_rel_mean',    'Mean Bat Speed (relative to own max)'),
    ('attack_angle_mean',     'Mean Attack Angle (°)'),
    ('swing_path_tilt_mean',  'Mean Swing Path Tilt (°)'),
    ('swing_length_mean',     'Mean Swing Length (ft)'),
    ('timing_mean',           'Mean Timing (in)'),
    ('launch_angle_mean',     'Mean Launch Angle (°) [in-play only]'),
    ('launch_speed_mean',     'Mean Exit Velocity (mph) [in-play only]'),
    ('launch_speed_angle_mean','Mean Contact Quality 1–6 [in-play only]'),
    ('barrel_rate',           'Barrel Rate (% in-play)'),
    ('hard_hit_rate',         'Hard-Hit Rate (≥95 mph, % in-play)'),
]
feat_names = [c for c, _ in FEAT_COLS]
feat_labels = {c: l for c, l in FEAT_COLS}

# ── Print correlation table ───────────────────────────────────────────────────
print("\nCorrelations with peak timing (in):")
print(f"  {'Variable':<40} {'Pearson r':>10} {'p':>8} {'Spearman ρ':>12} {'p':>8}")
corr_rows = []
for col, label in FEAT_COLS:
    if col not in hitters.columns:
        continue
    sub = hitters[['peak', col]].dropna()
    r, p_r = pearsonr(sub['peak'], sub[col])
    rho, p_rho = spearmanr(sub['peak'], sub[col])
    corr_rows.append({'col': col, 'label': label, 'r': r, 'p_r': p_r, 'rho': rho, 'p_rho': p_rho})
    print(f"  {label:<40} {r:>10.3f} {p_r:>8.4f} {rho:>12.3f} {p_rho:>8.4f}")
corr_df = pd.DataFrame(corr_rows).set_index('col')

# ── Figure 1: KDE with GMM components overlaid ───────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))

ax = axes1[0]
xs1d = np.linspace(hitters['peak'].min(), hitters['peak'].max(), 500)
kde_all = gaussian_kde(hitters['peak'].values)
ax.plot(xs1d, kde_all(xs1d), 'k-', lw=2.5, label='All hitters KDE')

weights = gmm.weights_
covs    = gmm.covariances_.flatten()
from scipy.stats import norm
for i, (m, v, w) in enumerate(zip(gm_means, covs, weights)):
    sd = np.sqrt(v)
    is_oppo = (i == oppo_idx)
    label = f"{'oppo' if is_oppo else 'pull'}-peak (μ={m:.1f} in, n={n_low if is_oppo else n_high})"
    color = '#1f77b4' if is_oppo else '#d62728'
    ax.plot(xs1d, w * norm.pdf(xs1d, m, sd), lw=2, ls='--', color=color, label=label)
    ax.fill_between(xs1d, 0, w * norm.pdf(xs1d, m, sd), alpha=0.10, color=color)

ax.axvline(0, color='black', lw=1, ls=':')
ax.set_xlabel("Per-Hitter Peak Timing (in, centred on median)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Peak Timing Distribution: 2-Component GMM\n"
             f"(all {len(hitters)} spline peaks; *** = significant side preference, n={int(hitters['peak_sig'].sum())})",
             fontsize=11)
ax.legend(fontsize=9)

ax = axes1[1]
for hand, color, ls in [('L', '#1f77b4', '-'), ('R', '#d62728', '--')]:
    sub = hitters[hitters['stand'] == hand]['peak'].values
    if len(sub) > 5:
        kde_h = gaussian_kde(sub)
        ax.plot(xs1d, kde_h(xs1d), color=color, lw=2, ls=ls,
                label=f"{hand}HH (n={len(sub)})")
        ax.fill_between(xs1d, 0, kde_h(xs1d), alpha=0.07, color=color)
for i, (m, w) in enumerate(zip(gm_means, weights)):
    color = '#1f77b4' if i == oppo_idx else '#d62728'
    ax.axvline(m, color=color, lw=1.5, ls=':', alpha=0.8)
ax.axvline(0, color='black', lw=1, ls=':')
ax.set_xlabel("Per-Hitter Peak Timing (in, centred on median)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Peak Timing KDE by Batter Handedness\nwith GMM mode centres (dotted)", fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'timing_modes_1_gmm.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\nSaved timing_modes_1_gmm.png")

# ── Figure 2: Scatter grid — peak attack angle vs each variable ───────────────
FEAT_COLS_AVAIL = [(c, l) for c, l in FEAT_COLS if c in hitters.columns]
N_FEATS = len(FEAT_COLS_AVAIL)
NCOLS = 3
NROWS = int(np.ceil(N_FEATS / NCOLS))

fig2, axes2 = plt.subplots(NROWS, NCOLS, figsize=(6 * NCOLS, 5 * NROWS))
axes2 = axes2.flatten()

stand_colors = {'L': '#1f77b4', 'R': '#d62728'}

for i, (col, label) in enumerate(FEAT_COLS_AVAIL):
    ax = axes2[i]
    sub = hitters[['peak', col, 'stand', 'peak_sig']].dropna(subset=['peak', col, 'stand'])
    for hand, color in stand_colors.items():
        s = sub[sub['stand'] == hand]
        s_ns = s[~s['peak_sig'].astype(bool)]
        s_si = s[ s['peak_sig'].astype(bool)]
        ax.scatter(s_ns['peak'], s_ns[col], c=color, alpha=0.18, s=10,
                   label=f'{hand}HH n.s. (n={len(s_ns)})')
        ax.scatter(s_si['peak'], s_si[col], c=color, alpha=0.85, s=18,
                   edgecolors='black', linewidths=0.3,
                   label=f'{hand}HH *** (n={len(s_si)})')
    x_all = sub['peak'].values
    y_all = sub[col].values
    m, b = np.polyfit(x_all, y_all, 1)
    xs_fit = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(xs_fit, m * xs_fit + b, 'k-', lw=1.5, zorder=5)
    r, p = pearsonr(x_all, y_all)
    ax.axvline(0, color='grey', lw=0.8, ls=':')
    for k, (gm, gw) in enumerate(zip(gm_means, weights)):
        c = '#1f77b4' if k == oppo_idx else '#d62728'
        ax.axvline(gm, color=c, lw=1.2, ls='--', alpha=0.6)
    pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
    ax.set_title(f"{label}\nr={r:.3f}  p={pstr}", fontsize=9)
    ax.set_xlabel("Peak Timing (in)", fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    if i == 0:
        ax.legend(fontsize=6)

for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle("Per-Hitter Peak Timing (in) vs Swing Mechanics & Outcome Variables\n"
              "Blue=LHH, Red=RHH  |  Faded=non-significant peak, saturated=*** sig side preference (p<0.05)\n"
              "Dashed verticals = GMM mode centres",
              fontsize=12, y=1.02)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'timing_modes_2_scatter.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("Saved timing_modes_2_scatter.png")

# ── Figure 3: Violin grid — each variable by GMM mode ────────────────────────
fig3, axes3 = plt.subplots(NROWS, NCOLS, figsize=(6 * NCOLS, 5 * NROWS))
axes3 = axes3.flatten()

palette = {'oppo-peak': '#1f77b4', 'pull-peak': '#d62728'}

for i, (col, label) in enumerate(FEAT_COLS_AVAIL):
    ax = axes3[i]
    sub = hitters[['mode_label', 'stand', col]].dropna()

    plot_data = []
    tick_labels = []
    positions = []
    pos = 0
    for mode, mc in [('oppo-peak', '#1f77b4'), ('pull-peak', '#d62728')]:
        for hand in ['L', 'R']:
            vals = sub[(sub['mode_label'] == mode) & (sub['stand'] == hand)][col].values
            if len(vals) >= 3:
                vp = ax.violinplot(vals, positions=[pos], widths=0.7,
                                   showmedians=True, showextrema=False)
                for pc in vp['bodies']:
                    pc.set_facecolor(mc)
                    pc.set_alpha(0.55)
                vp['cmedians'].set_color('black')
                tick_labels.append(f"{mode[:3]}\n{hand}HH\n(n={len(vals)})")
                positions.append(pos)
                pos += 1
        pos += 0.4

    low_vals  = sub[sub['mode_label'] == 'oppo-peak'][col].values
    high_vals = sub[sub['mode_label'] == 'pull-peak'][col].values
    if len(low_vals) >= 3 and len(high_vals) >= 3:
        t, p = ttest_ind(low_vals, high_vals)
        pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
        ax.set_title(f"{label}\noppo μ={low_vals.mean():.2f}  pull μ={high_vals.mean():.2f}"
                     f"\nt={t:.2f}  p={pstr}", fontsize=8)
    else:
        ax.set_title(label, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6.5)
    ax.set_ylabel(label, fontsize=8)

for j in range(i + 1, len(axes3)):
    axes3[j].set_visible(False)

fig3.suptitle("Swing Mechanics & Outcomes by GMM Peak Timing Mode (in)\n"
              "Blue = oppo-peak  |  Red = pull-peak",
              fontsize=13, y=1.01)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'timing_modes_3_violin.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved timing_modes_3_violin.png")

# ── Figure 4: Ranked bar chart of |r| + correlation heatmap ─────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7))

ax = axes4[0]
corr_plot = corr_df.sort_values('r')
colors_bar = ['#d62728' if r < 0 else '#1f77b4' for r in corr_plot['r']]
bars = ax.barh(range(len(corr_plot)), corr_plot['r'].values,
               color=colors_bar, edgecolor='k', linewidth=0.4, alpha=0.8)
ax.set_yticks(range(len(corr_plot)))
ax.set_yticklabels([feat_labels[c] for c in corr_plot.index], fontsize=9)
ax.axvline(0, color='black', lw=1)
for j, (col, row) in enumerate(corr_plot.iterrows()):
    sig = '***' if row['p_r'] < 0.001 else ('**' if row['p_r'] < 0.01 else
          ('*' if row['p_r'] < 0.05 else ''))
    if sig:
        x_off = row['r'] + (0.005 if row['r'] >= 0 else -0.005)
        ax.text(x_off, j, sig, va='center', ha='left' if row['r'] >= 0 else 'right',
                fontsize=9, color='black')
ax.set_xlabel("Pearson r with Peak Timing (in)", fontsize=10)
ax.set_title("Correlations with Peak Timing (in)\n(* p<.05  ** p<.01  *** p<.001)", fontsize=11)

hmap_cols = ['peak'] + [c for c in feat_names if c in hitters.columns]
hmap_sub = hitters[hmap_cols].dropna()
hmap_corr = hmap_sub.corr()
hmap_labels = ['Peak\nTiming (in)'] + [feat_labels[c].replace(' (', '\n(').replace(' %', '\n%')
                                   for c in hmap_cols[1:]]
mask = np.triu(np.ones_like(hmap_corr, dtype=bool), k=1)
sns.heatmap(hmap_corr, ax=axes4[1], mask=mask,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=hmap_labels, yticklabels=hmap_labels,
            linewidths=0.5, annot=True, fmt='.2f', annot_kws={'size': 6},
            square=True, cbar_kws={'shrink': 0.8})
axes4[1].set_title("Pairwise Correlation Heatmap\n(all variables + peak timing in)", fontsize=11)
axes4[1].tick_params(axis='x', labelsize=7, rotation=45)
axes4[1].tick_params(axis='y', labelsize=7, rotation=0)

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'timing_modes_4_correlations.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved timing_modes_4_correlations.png")

print("\nDone.")
