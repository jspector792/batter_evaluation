"""
Analyze what drives the bimodal distribution of per-hitter peak timing angles.

Steps:
  1. Replicate per-hitter peak timing (same filter / method as timing_binary.py)
  2. Compute per-hitter means for swing mechanics & outcome variables
  3. Fit a 2-component GMM to identify early-peak vs late-peak hitters
  4. Figure 1: KDE with GMM components
  5. Figure 2: Scatter grid — peak timing vs each variable, coloured by handedness
  6. Figure 3: Violin grid — each variable by GMM mode
  7. Figure 4: Correlation heatmap + ranked bar chart of correlations with peak timing
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, spearmanr, ttest_ind
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
IN_PLAY  = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
MIN_PA   = 50
POLY_DEG = 4

COLS = [
    "pitch_type", "batter", "stand", "p_throws",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "delta_run_exp", "description",
    "bat_speed", "swing_length", "attack_angle",
    "attack_direction", "swing_path_tilt",
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
df = df.dropna(subset=[IX, IY, 'delta_run_exp', 'stand', 'p_throws'])
print(f"In-play fastballs: {len(df):,}")

# ── Timing angle ─────────────────────────────────────────────────────────────
df['timing_raw'] = np.degrees(np.arctan2(df[IX], df[IY]))
TIMING_MED = df['timing_raw'].median()
df['timing'] = df['timing_raw'] - TIMING_MED

# ── bat_speed_rel: per-batter max from this same dataset ─────────────────────
bat_counts = df.groupby('batter')['bat_speed'].count()
bat_max = df.groupby('batter')['bat_speed'].max()
bat_max = bat_max[bat_counts >= 5]
df['bat_speed_rel'] = df['bat_speed'] / df['batter'].map(bat_max)

# ── Per-hitter peak timing ────────────────────────────────────────────────────
def peak_timing(data, min_n=MIN_PA, degree=POLY_DEG):
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

print("Computing per-hitter peak timing angles...")
all_peaks = (df.groupby('batter')
               .apply(peak_timing)
               .dropna()
               .rename('peak'))

# ── Per-hitter feature means ──────────────────────────────────────────────────
MECH_VARS = [
    'bat_speed', 'bat_speed_rel',
    'attack_angle', 'swing_path_tilt',
    'swing_length', 'attack_direction',
    'launch_angle', 'launch_speed', 'launch_speed_angle',
]
hitter_means = df.groupby('batter')[MECH_VARS].mean().add_suffix('_mean')

# contact quality: % barrels (launch_speed_angle == 6) among in-play
barrel_rate = (
    df.assign(is_barrel=(df['launch_speed_angle'] == 6).astype(float))
      .groupby('batter')['is_barrel'].mean()
      .rename('barrel_rate')
)
# hard-hit rate (launch_speed >= 95 mph)
hard_hit_rate = (
    df.assign(is_hard=(df['launch_speed'] >= 95).astype(float))
      .groupby('batter')['is_hard'].mean()
      .rename('hard_hit_rate')
)

stand_col = df.groupby('batter')['stand'].agg(lambda s: s.mode()[0])
n_col = df.groupby('batter').size().rename('n')

hitters = (pd.DataFrame({'peak': all_peaks})
             .join(hitter_means)
             .join(barrel_rate)
             .join(hard_hit_rate)
             .join(stand_col)
             .join(n_col)
             .dropna(subset=['peak']))
hitters = hitters[hitters['n'] >= MIN_PA].copy()
print(f"Hitters for mode analysis: {len(hitters)}")

# ── 2-component GMM on peak timing ───────────────────────────────────────────
gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
gmm.fit(hitters[['peak']])
hitters['gmm_mode'] = gmm.predict(hitters[['peak']])

gm_means = gmm.means_.flatten()
early_idx = int(np.argmin(gm_means))
late_idx  = int(np.argmax(gm_means))
hitters['mode_label'] = hitters['gmm_mode'].map({early_idx: 'early-peak', late_idx: 'late-peak'})

n_early = (hitters['mode_label'] == 'early-peak').sum()
n_late  = (hitters['mode_label'] == 'late-peak').sum()
print(f"\nGMM clusters:")
print(f"  early-peak: mean={gm_means[early_idx]:.1f}°  n={n_early}")
print(f"  late-peak:  mean={gm_means[late_idx]:.1f}°   n={n_late}")

# ── All feature columns (with readable labels) ────────────────────────────────
FEAT_COLS = [
    ('bat_speed_mean',        'Mean Bat Speed (mph)'),
    ('bat_speed_rel_mean',    'Mean Bat Speed (relative to own max)'),
    ('attack_angle_mean',     'Mean Attack Angle (°)'),
    ('swing_path_tilt_mean',  'Mean Swing Path Tilt (°)'),
    ('swing_length_mean',     'Mean Swing Length (ft)'),
    ('attack_direction_mean', 'Mean Attack Direction (°)'),
    ('launch_angle_mean',     'Mean Launch Angle (°)'),
    ('launch_speed_mean',     'Mean Exit Velocity (mph)'),
    ('launch_speed_angle_mean','Mean Contact Quality (1–6)'),
    ('barrel_rate',           'Barrel Rate (% in-play)'),
    ('hard_hit_rate',         'Hard-Hit Rate (≥95 mph, % in-play)'),
]
feat_names = [c for c, _ in FEAT_COLS]
feat_labels = {c: l for c, l in FEAT_COLS}

# ── Print correlation table ───────────────────────────────────────────────────
print("\nCorrelations with peak timing angle:")
print(f"  {'Variable':<35} {'Pearson r':>10} {'p':>8} {'Spearman ρ':>12} {'p':>8}")
corr_rows = []
for col, label in FEAT_COLS:
    sub = hitters[['peak', col]].dropna()
    r, p_r = pearsonr(sub['peak'], sub[col])
    rho, p_rho = spearmanr(sub['peak'], sub[col])
    corr_rows.append({'col': col, 'label': label, 'r': r, 'p_r': p_r, 'rho': rho, 'p_rho': p_rho})
    print(f"  {label:<35} {r:>10.3f} {p_r:>8.4f} {rho:>12.3f} {p_rho:>8.4f}")
corr_df = pd.DataFrame(corr_rows).set_index('col')

# ── Figure 1: KDE with GMM components overlaid ───────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))

# Panel 1: KDE + GMM
ax = axes1[0]
xs = np.linspace(hitters['peak'].min(), hitters['peak'].max(), 500).reshape(-1, 1)
kde_all = gaussian_kde(hitters['peak'].values)
xs1d = xs.flatten()
ax.plot(xs1d, kde_all(xs1d), 'k-', lw=2.5, label='All hitters KDE')

# GMM component curves
weights = gmm.weights_
covs    = gmm.covariances_.flatten()
from scipy.stats import norm
for i, (m, v, w) in enumerate(zip(gm_means, covs, weights)):
    sd = np.sqrt(v)
    label = f"{'early' if i==early_idx else 'late'}-peak (μ={m:.1f}°, n={n_early if i==early_idx else n_late})"
    color = '#1f77b4' if i == early_idx else '#d62728'
    ax.plot(xs1d, w * norm.pdf(xs1d, m, sd), lw=2, ls='--', color=color, label=label)
    ax.fill_between(xs1d, 0, w * norm.pdf(xs1d, m, sd), alpha=0.10, color=color)

ax.axvline(0, color='black', lw=1, ls=':')
ax.set_xlabel("Per-Hitter Peak Timing Angle (°)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Peak Timing Distribution: 2-Component GMM", fontsize=12)
ax.legend(fontsize=9)

# Panel 2: same KDE, coloured by stand
ax = axes1[1]
for hand, color, ls in [('L', '#1f77b4', '-'), ('R', '#d62728', '--')]:
    sub = hitters[hitters['stand'] == hand]['peak'].values
    if len(sub) > 5:
        kde_h = gaussian_kde(sub)
        ax.plot(xs1d, kde_h(xs1d), color=color, lw=2, ls=ls,
                label=f"{hand}HH (n={len(sub)})")
        ax.fill_between(xs1d, 0, kde_h(xs1d), alpha=0.07, color=color)
for i, (m, w) in enumerate(zip(gm_means, weights)):
    color = '#1f77b4' if i == early_idx else '#d62728'
    ax.axvline(m, color=color, lw=1.5, ls=':', alpha=0.8)
ax.axvline(0, color='black', lw=1, ls=':')
ax.set_xlabel("Per-Hitter Peak Timing Angle (°)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Peak Timing KDE by Batter Handedness\nwith GMM mode centres (dotted)", fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'timing_modes_1_gmm.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\nSaved timing_modes_1_gmm.png")

# ── Figure 2: Scatter grid — peak timing vs each variable ────────────────────
N_FEATS = len(FEAT_COLS)
NCOLS = 3
NROWS = int(np.ceil(N_FEATS / NCOLS))

fig2, axes2 = plt.subplots(NROWS, NCOLS, figsize=(6 * NCOLS, 5 * NROWS))
axes2 = axes2.flatten()

stand_colors = {'L': '#1f77b4', 'R': '#d62728'}

for i, (col, label) in enumerate(FEAT_COLS):
    ax = axes2[i]
    sub = hitters[['peak', col, 'stand']].dropna()
    for hand, color in stand_colors.items():
        s = sub[sub['stand'] == hand]
        ax.scatter(s['peak'], s[col], c=color, alpha=0.35, s=12,
                   label=f'{hand}HH (n={len(s)})')
    # overall regression line
    x_all = sub['peak'].values
    y_all = sub[col].values
    m, b = np.polyfit(x_all, y_all, 1)
    xs_fit = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(xs_fit, m * xs_fit + b, 'k-', lw=1.5, zorder=5)
    r, p = pearsonr(x_all, y_all)
    ax.axvline(0, color='grey', lw=0.8, ls=':')
    # GMM mode markers on x-axis
    for k, (gm, gw) in enumerate(zip(gm_means, weights)):
        c = '#1f77b4' if k == early_idx else '#d62728'
        ax.axvline(gm, color=c, lw=1.2, ls='--', alpha=0.6)
    pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
    ax.set_title(f"{label}\nr={r:.3f}  p={pstr}", fontsize=9)
    ax.set_xlabel("Peak Timing Angle (°)", fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    if i == 0:
        ax.legend(fontsize=7)

# hide unused panels
for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle("Per-Hitter Peak Timing vs Swing Mechanics & Outcome Variables\n"
              "Blue=LHH, Red=RHH  |  Dashed verticals = GMM mode centres",
              fontsize=13, y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'timing_modes_2_scatter.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("Saved timing_modes_2_scatter.png")

# ── Figure 3: Violin grid — each variable by GMM mode ────────────────────────
fig3, axes3 = plt.subplots(NROWS, NCOLS, figsize=(6 * NCOLS, 5 * NROWS))
axes3 = axes3.flatten()

palette = {'early-peak': '#1f77b4', 'late-peak': '#d62728'}

for i, (col, label) in enumerate(FEAT_COLS):
    ax = axes3[i]
    sub = hitters[['mode_label', 'stand', col]].dropna()

    # violin by mode x handedness
    plot_data = []
    tick_labels = []
    positions = []
    pos = 0
    for mode, mc in [('early-peak', '#1f77b4'), ('late-peak', '#d62728')]:
        for hand in ['L', 'R']:
            vals = sub[(sub['mode_label'] == mode) & (sub['stand'] == hand)][col].values
            if len(vals) >= 3:
                vp = ax.violinplot(vals, positions=[pos], widths=0.7,
                                   showmedians=True, showextrema=False)
                for pc in vp['bodies']:
                    pc.set_facecolor(mc)
                    pc.set_alpha(0.55)
                vp['cmedians'].set_color('black')
                tick_labels.append(f"{mode[:5]}\n{hand}HH\n(n={len(vals)})")
                positions.append(pos)
                pos += 1
        pos += 0.4  # gap between modes

    # t-test early vs late (pooled, ignore hand)
    early_vals = sub[sub['mode_label'] == 'early-peak'][col].values
    late_vals  = sub[sub['mode_label'] == 'late-peak'][col].values
    if len(early_vals) >= 3 and len(late_vals) >= 3:
        t, p = ttest_ind(early_vals, late_vals)
        pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
        ax.set_title(f"{label}\nearly μ={early_vals.mean():.2f}  late μ={late_vals.mean():.2f}"
                     f"\nt={t:.2f}  p={pstr}", fontsize=8)
    else:
        ax.set_title(label, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6.5)
    ax.set_ylabel(label, fontsize=8)

for j in range(i + 1, len(axes3)):
    axes3[j].set_visible(False)

fig3.suptitle("Swing Mechanics & Outcomes by GMM Peak-Timing Mode\n"
              "Blue = early-peak  |  Red = late-peak",
              fontsize=13, y=1.01)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'timing_modes_3_violin.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved timing_modes_3_violin.png")

# ── Figure 4: Ranked bar chart of |r| + correlation heatmap ─────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: ranked |r| bar chart
ax = axes4[0]
corr_plot = corr_df.sort_values('r')
colors_bar = ['#d62728' if r < 0 else '#1f77b4' for r in corr_plot['r']]
bars = ax.barh(range(len(corr_plot)), corr_plot['r'].values,
               color=colors_bar, edgecolor='k', linewidth=0.4, alpha=0.8)
ax.set_yticks(range(len(corr_plot)))
ax.set_yticklabels([feat_labels[c] for c in corr_plot.index], fontsize=9)
ax.axvline(0, color='black', lw=1)
# significance markers
for j, (col, row) in enumerate(corr_plot.iterrows()):
    sig = '***' if row['p_r'] < 0.001 else ('**' if row['p_r'] < 0.01 else
          ('*' if row['p_r'] < 0.05 else ''))
    if sig:
        x_off = row['r'] + (0.005 if row['r'] >= 0 else -0.005)
        ax.text(x_off, j, sig, va='center', ha='left' if row['r'] >= 0 else 'right',
                fontsize=9, color='black')
ax.set_xlabel("Pearson r with Peak Timing Angle", fontsize=10)
ax.set_title("Correlations with Peak Timing\n(* p<.05  ** p<.01  *** p<.001)", fontsize=11)

# Panel 2: pairwise correlation heatmap among all variables + peak
hmap_cols = ['peak'] + feat_names
hmap_sub = hitters[hmap_cols].dropna()
hmap_corr = hmap_sub.corr()
hmap_labels = ['Peak\nTiming'] + [feat_labels[c].replace(' (', '\n(').replace(' %', '\n%')
                                   for c in feat_names]
mask = np.triu(np.ones_like(hmap_corr, dtype=bool), k=1)
sns.heatmap(hmap_corr, ax=axes4[1], mask=mask,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=hmap_labels, yticklabels=hmap_labels,
            linewidths=0.5, annot=True, fmt='.2f', annot_kws={'size': 6},
            square=True, cbar_kws={'shrink': 0.8})
axes4[1].set_title("Pairwise Correlation Heatmap\n(all variables + peak timing)", fontsize=11)
axes4[1].tick_params(axis='x', labelsize=7, rotation=45)
axes4[1].tick_params(axis='y', labelsize=7, rotation=0)

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'timing_modes_4_correlations.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved timing_modes_4_correlations.png")

print("\nDone.")
