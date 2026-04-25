"""
Timing mode stability over time.

Question: do hitters consistently fall into the early-peak or late-peak GMM mode,
or do they switch from month to month?

Approach:
  1. Re-fit per-hitter peak timing + 2-component GMM on the full season
     (same method as analyze_timing_modes.py).
  2. Split season into calendar months April–September (core season).
     For each hitter × month with ≥MIN_MONTH PAs, compute per-period peak timing
     using a degree-3 polynomial (lower than full-season to limit over-fit noise
     on smaller samples).
  3. Assign each hitter-month to a GMM mode using the full-season GMM.
  4. Analyze consistency:
       - How often is a hitter in the same mode across months?
       - What is the inter-period correlation of peak timing?
       - Is within-hitter variance smaller than between-hitter variance?
       - Are mode-switchers identifiable by performance features?

Figures:
  stability_1_heatmap.png   — mode assignment grid (hitter × month)
  stability_2_halves.png    — first-half vs second-half peak timing scatter +
                              month-by-month inter-period correlation matrix
  stability_3_consistency.png — within-hitter peak timing distributions +
                                 consistency rate histogram
  stability_4_switchers.png — mode-switcher vs stable hitter feature comparison
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, gaussian_kde
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
IN_PLAY  = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
MIN_PA_SEASON = 50   # for full-season GMM fit
MIN_MONTH     = 20   # minimum in-play fastballs per hitter-month
CORE_MONTHS   = [4, 5, 6, 7, 8, 9]  # April–September
POLY_SEASON   = 4    # degree for full-season peak
POLY_PERIOD   = 3    # degree for per-month peak (smaller samples)

COLS = [
    "game_date", "pitch_type", "batter", "stand", "p_throws",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "delta_run_exp", "description",
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
df = df.dropna(subset=[IX, IY, 'delta_run_exp', 'stand'])
df['game_date'] = pd.to_datetime(df['game_date'])
df['month_num'] = df['game_date'].dt.month
df['half'] = df['game_date'].apply(lambda d: 'H1' if d < pd.Timestamp('2025-07-14') else 'H2')

# Timing angle (centred on full-season median)
df['timing_raw'] = np.degrees(np.arctan2(df[IX], df[IY]))
TIMING_MED = df['timing_raw'].median()
df['timing'] = df['timing_raw'] - TIMING_MED

print(f"In-play fastballs: {len(df):,}")
print(f"Date range: {df['game_date'].min().date()} → {df['game_date'].max().date()}")

# ── Peak-finding helpers ──────────────────────────────────────────────────────
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

# ── Full-season peaks + GMM (identical to timing_modes.py) ──────────────────
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

def assign_mode(peak_val):
    if np.isnan(peak_val):
        return np.nan
    probs = gmm.predict_proba([[peak_val]])[0]
    return 'early' if np.argmax(probs) == early_idx else 'late'

season_df = pd.DataFrame({'peak_season': season_peaks})
season_df['mode_season'] = season_df['peak_season'].apply(assign_mode)
season_df['stand'] = df.groupby('batter')['stand'].agg(lambda s: s.mode()[0])

n_early = (season_df['mode_season'] == 'early').sum()
n_late  = (season_df['mode_season'] == 'late').sum()
print(f"  early-peak: μ={gm_means[early_idx]:.1f}°  n={n_early}")
print(f"  late-peak:  μ={gm_means[late_idx]:.1f}°   n={n_late}")

# ── Per-month peaks ───────────────────────────────────────────────────────────
print("\nComputing per-month peaks (Apr–Sep)...")
core = df[df['month_num'].isin(CORE_MONTHS)].copy()
month_peaks_raw = (
    core.groupby(['batter', 'month_num'])
        .apply(lambda g: peak_timing(g, MIN_MONTH, POLY_PERIOD))
        .reset_index()
)
month_peaks_raw.columns = ['batter', 'month_num', 'peak_month']
month_peaks_raw = month_peaks_raw.dropna(subset=['peak_month'])

# Assign GMM mode to each hitter-month
month_peaks_raw['mode_month'] = month_peaks_raw['peak_month'].apply(assign_mode)

# Summary stats per month
for m in CORE_MONTHS:
    sub = month_peaks_raw[month_peaks_raw['month_num'] == m]
    print(f"  Month {m}: {len(sub)} hitter-months  "
          f"early={( sub['mode_month']=='early').sum()}  "
          f"late={(sub['mode_month']=='late').sum()}")

# ── Consistency analysis ──────────────────────────────────────────────────────
# Hitters with data in ≥2 months
month_counts = month_peaks_raw.groupby('batter')['month_num'].count()
hitters_multi = month_counts[month_counts >= 2].index

stability = []
for batter in hitters_multi:
    rows = month_peaks_raw[month_peaks_raw['batter'] == batter]
    peaks = rows['peak_month'].values
    modes = rows['mode_month'].values
    n_months = len(rows)
    majority_mode = pd.Series(modes).mode()[0]
    pct_majority = (modes == majority_mode).mean()
    always_same = pct_majority == 1.0
    stability.append({
        'batter': batter,
        'n_months': n_months,
        'peak_std': np.std(peaks),
        'peak_mean': np.mean(peaks),
        'majority_mode': majority_mode,
        'pct_majority': pct_majority,
        'always_same': always_same,
        'season_peak': season_df.loc[batter, 'peak_season'] if batter in season_df.index else np.nan,
        'season_mode': season_df.loc[batter, 'mode_season'] if batter in season_df.index else np.nan,
        'stand': season_df.loc[batter, 'stand'] if batter in season_df.index else np.nan,
    })

stab_df = pd.DataFrame(stability).set_index('batter')
print(f"\nHitters with ≥2 qualifying months: {len(stab_df)}")
print(f"  Always in same mode: {stab_df['always_same'].sum()} ({stab_df['always_same'].mean():.1%})")
print(f"  % time in majority mode (median): {stab_df['pct_majority'].median():.1%}")
print(f"  Within-hitter peak std (median): {stab_df['peak_std'].median():.1f}°")
print(f"  Between-hitter peak std (full-season): {season_df['peak_season'].std():.1f}°")

# ── Per-feature means (for switcher analysis) ─────────────────────────────────
feat_cols = ['bat_speed', 'attack_angle', 'swing_path_tilt',
             'launch_speed', 'launch_angle', 'launch_speed_angle']
feat_means = df.groupby('batter')[feat_cols].mean().add_suffix('_mean')
stab_df = stab_df.join(feat_means, how='left')

# Half-season peaks (for scatter)
half_peaks = (
    df.groupby(['batter', 'half'])
      .apply(lambda g: peak_timing(g, 25, POLY_SEASON))
      .reset_index()
)
half_peaks.columns = ['batter', 'half', 'peak_half']
half_peaks = half_peaks.dropna(subset=['peak_half'])
half_wide = half_peaks.pivot(index='batter', columns='half', values='peak_half').dropna()
half_wide.columns = ['H1_peak', 'H2_peak']
r_half, p_half = pearsonr(half_wide['H1_peak'], half_wide['H2_peak'])
print(f"\nFirst-half vs second-half peak timing: r={r_half:.3f}  p={p_half:.4f}  n={len(half_wide)}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
MONTH_NAMES = {4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep'}
MODE_COLORS = {'early': '#1f77b4', 'late': '#d62728'}

# ── Fig 1: Mode assignment heatmap (hitter × month) ──────────────────────────
# Build a pivot table: rows=batter, cols=month, value=mode
# Keep only hitters with ≥3 qualifying months; sort by season peak timing

pivot_mode = month_peaks_raw.pivot_table(
    index='batter', columns='month_num', values='mode_month', aggfunc='first'
)
pivot_mode = pivot_mode.reindex(columns=CORE_MONTHS)
pivot_peak = month_peaks_raw.pivot_table(
    index='batter', columns='month_num', values='peak_month', aggfunc='first'
)
pivot_peak = pivot_peak.reindex(columns=CORE_MONTHS)

# Filter to hitters with >= 3 months and a full-season peak
valid_hitters = (pivot_mode.notna().sum(axis=1) >= 3) & pivot_mode.index.isin(season_df.index)
pivot_mode = pivot_mode[valid_hitters]
pivot_peak = pivot_peak[valid_hitters]

# Sort by full-season peak timing (early to late)
sort_order = season_df.loc[pivot_mode.index, 'peak_season'].sort_values()
pivot_mode = pivot_mode.loc[sort_order.index]
pivot_peak = pivot_peak.loc[sort_order.index]

# Encode mode as numeric for colour: early=0, late=1, NaN=0.5 (grey)
mode_num = pivot_mode.map(lambda x: 0 if x == 'early' else (1 if x == 'late' else np.nan))

# Add consistency column (fraction of months in majority mode)
stab_sorted = stab_df.reindex(sort_order.index)

fig1, axes1 = plt.subplots(1, 2, figsize=(20, max(8, len(pivot_mode) * 0.18 + 2)),
                           gridspec_kw={'width_ratios': [6, 1]})

ax = axes1[0]
# Custom colormap: blue=early, red=late
cmap = mcolors.LinearSegmentedColormap.from_list(
    'mode', ['#1f77b4', '#d62728'], N=2)
im = ax.imshow(mode_num.values, aspect='auto', cmap=cmap, vmin=0, vmax=1,
               interpolation='none')

ax.set_xticks(range(len(CORE_MONTHS)))
ax.set_xticklabels([MONTH_NAMES[m] for m in CORE_MONTHS], fontsize=10)
ax.set_yticks(range(len(pivot_mode)))
ax.set_yticklabels(
    [f"{b} ({season_df.loc[b,'peak_season']:.0f}°)" for b in pivot_mode.index],
    fontsize=5)
ax.set_title(
    f"Mode Assignment by Hitter × Month\n"
    f"Blue=early-peak  Red=late-peak  Grey=insufficient data\n"
    f"(n={len(pivot_mode)} hitters, ≥3 qualifying months, sorted by full-season peak)",
    fontsize=11)
ax.set_xlabel("Month", fontsize=10)

# Panel 2: consistency bar per hitter
ax2 = axes1[1]
y_pos = np.arange(len(pivot_mode))
consis = stab_sorted['pct_majority'].reindex(pivot_mode.index).fillna(np.nan)
colors_bar = [MODE_COLORS.get(stab_sorted.loc[b, 'majority_mode'], 'grey')
              for b in pivot_mode.index]
ax2.barh(y_pos, consis.values, color=colors_bar, alpha=0.75, edgecolor='none', height=0.9)
ax2.axvline(1.0, color='black', lw=0.8, ls='--')
ax2.set_xlim(0, 1.05)
ax2.set_xticks([0, 0.5, 1.0])
ax2.set_xticklabels(['0', '.5', '1'], fontsize=8)
ax2.set_yticks([])
ax2.set_title("% months\nin mode", fontsize=9)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'stability_1_heatmap.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\nSaved stability_1_heatmap.png")

# ── Fig 2: First-half vs second-half scatter + month-pair correlation matrix ─
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: H1 vs H2 scatter
ax = axes2[0]
hw = half_wide.join(season_df[['mode_season', 'stand']], how='inner').dropna()
for mode, color, marker in [('early', '#1f77b4', 'o'), ('late', '#d62728', '^')]:
    sub = hw[hw['mode_season'] == mode]
    ax.scatter(sub['H1_peak'], sub['H2_peak'], c=color, alpha=0.4, s=20,
               marker=marker, label=f"{mode}-peak (n={len(sub)})")
lo = min(hw['H1_peak'].min(), hw['H2_peak'].min()) - 2
hi = max(hw['H1_peak'].max(), hw['H2_peak'].max()) + 2
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x')
ax.axhline(0, color='grey', lw=0.6, ls=':')
ax.axvline(0, color='grey', lw=0.6, ls=':')
ax.set_xlabel("Peak Timing — First Half (°)", fontsize=11)
ax.set_ylabel("Peak Timing — Second Half (°)", fontsize=11)
ax.set_title(f"First Half vs Second Half Peak Timing\nr={r_half:.3f}  p={p_half:.4f}  n={len(hw)}",
             fontsize=11)
ax.legend(fontsize=9)

# Panel 2: month-pair Pearson r matrix
months_avail = CORE_MONTHS
r_mat = pd.DataFrame(np.nan, index=CORE_MONTHS, columns=CORE_MONTHS)
n_mat = pd.DataFrame(0,      index=CORE_MONTHS, columns=CORE_MONTHS)
for i, m1 in enumerate(CORE_MONTHS):
    for j, m2 in enumerate(CORE_MONTHS):
        if m1 == m2:
            r_mat.loc[m1, m2] = 1.0
            continue
        sub1 = month_peaks_raw[month_peaks_raw['month_num'] == m1][['batter','peak_month']].set_index('batter')
        sub2 = month_peaks_raw[month_peaks_raw['month_num'] == m2][['batter','peak_month']].set_index('batter')
        merged = sub1.join(sub2, lsuffix='_1', rsuffix='_2').dropna()
        if len(merged) >= 10:
            r, _ = pearsonr(merged['peak_month_1'], merged['peak_month_2'])
            r_mat.loc[m1, m2] = r
            n_mat.loc[m1, m2] = len(merged)

ax = axes2[1]
labels = [MONTH_NAMES[m] for m in CORE_MONTHS]
mask = np.triu(np.ones_like(r_mat, dtype=bool), k=1)
sns.heatmap(r_mat.astype(float), ax=ax, mask=mask,
            cmap='RdBu_r', center=0, vmin=-0.5, vmax=1,
            xticklabels=labels, yticklabels=labels,
            annot=True, fmt='.2f', annot_kws={'size': 9},
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
ax.set_title("Inter-Month Peak Timing Correlation (Pearson r)\nacross hitters", fontsize=11)
ax.tick_params(labelsize=10)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'stability_2_halves.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("Saved stability_2_halves.png")

# ── Fig 3: Consistency histogram + within-hitter timing distributions ─────────
fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: histogram of % months in majority mode
ax = axes3[0]
ax.hist(stab_df['pct_majority'] * 100, bins=20, edgecolor='k', lw=0.4,
        color='steelblue', alpha=0.75)
ax.axvline(stab_df['pct_majority'].median() * 100, color='red', lw=2,
           label=f"Median = {stab_df['pct_majority'].median():.0%}")
ax.axvline(100, color='green', lw=1.5, ls='--', label='Always same mode')
ax.set_xlabel("% Qualifying Months in Majority Mode", fontsize=11)
ax.set_ylabel("Number of Hitters", fontsize=11)
ax.set_title(f"Mode Consistency per Hitter\n"
             f"(n={len(stab_df)} hitters with ≥2 qualifying months)", fontsize=11)
ax.legend(fontsize=9)

# Panel 2: within-hitter vs between-hitter std
ax = axes3[1]
within_std = stab_df['peak_std'].dropna()
between_std_vals = season_df['peak_season'].dropna().values

xs_within = np.linspace(within_std.min(), within_std.max(), 300)
xs_between = np.linspace(between_std_vals.min(), between_std_vals.max(), 300)
ax.plot(xs_within,
        gaussian_kde(within_std)(xs_within),
        color='steelblue', lw=2.5, label=f"Within-hitter monthly std\n(median={within_std.median():.1f}°)")
ax.plot(xs_between,
        gaussian_kde(between_std_vals)(xs_between),
        color='tomato', lw=2.5, ls='--',
        label=f"Between-hitter full-season peak\n(std={between_std_vals.std():.1f}°)")
ax.set_xlabel("Timing Angle Spread (°)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Signal vs Noise:\nWithin-hitter monthly spread vs between-hitter variation", fontsize=11)
ax.legend(fontsize=9)

# Panel 3: strip plot of all monthly peaks, sorted by season peak
# Sample ~30 hitters spanning the full range for readability
sample_batters = sort_order.index[np.linspace(0, len(sort_order)-1, 40, dtype=int)]
monthly_sample = month_peaks_raw[month_peaks_raw['batter'].isin(sample_batters)].copy()
monthly_sample['rank'] = monthly_sample['batter'].map(
    {b: i for i, b in enumerate(sort_order.index)})
monthly_sample['mode_month_color'] = monthly_sample['mode_month'].map(MODE_COLORS)
# per-hitter season peak
ax = axes3[2]
for batter in sample_batters:
    sub = monthly_sample[monthly_sample['batter'] == batter]
    if sub.empty:
        continue
    rank = sub['rank'].iloc[0]
    for _, row in sub.iterrows():
        ax.scatter(row['peak_month'], rank,
                   c=row['mode_month_color'], alpha=0.65, s=25, zorder=3)
    # season peak marker
    if batter in season_df.index:
        sp = season_df.loc[batter, 'peak_season']
        ax.scatter(sp, rank, c='black', marker='|', s=80, zorder=4, linewidths=1.5)

ax.axvline(0, color='grey', lw=0.8, ls=':')
ax.set_xlabel("Monthly Peak Timing Angle (°)\nBlack tick = full-season peak", fontsize=10)
ax.set_ylabel("Hitter rank (early peak → late peak)", fontsize=10)
ax.set_title("Monthly Peak Timing for 40 Sample Hitters\n"
             "Blue=early-mode  Red=late-mode", fontsize=11)

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'stability_3_consistency.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved stability_3_consistency.png")

# ── Fig 4: Mode-switchers vs stable hitters ───────────────────────────────────
stab_df['is_stable'] = stab_df['pct_majority'] >= 0.85
stable = stab_df[stab_df['is_stable']]
switchers = stab_df[~stab_df['is_stable']]
print(f"\nStable hitters (≥85% in one mode): {len(stable)} ({len(stable)/len(stab_df):.1%})")
print(f"Switchers (<85% in one mode):       {len(switchers)} ({len(switchers)/len(stab_df):.1%})")

COMPARE_FEATS = [
    ('bat_speed_mean',        'Bat Speed (mph)'),
    ('attack_angle_mean',     'Attack Angle (°)'),
    ('swing_path_tilt_mean',  'Swing Path Tilt (°)'),
    ('launch_speed_mean',     'Exit Velocity (mph)'),
    ('launch_angle_mean',     'Launch Angle (°)'),
    ('launch_speed_angle_mean','Contact Quality (1–6)'),
    ('peak_std',              'Monthly Peak Timing Std (°)'),
]
NCOLS_F = 4
NROWS_F = int(np.ceil(len(COMPARE_FEATS) / NCOLS_F))
fig4, axes4 = plt.subplots(NROWS_F, NCOLS_F, figsize=(6*NCOLS_F, 5*NROWS_F))
axes4 = axes4.flatten()

print("\nStable vs switcher feature comparison:")
print(f"  {'Variable':<35} {'Stable μ':>9} {'Switch μ':>9} {'Δ':>8} {'p':>8}")
for i, (col, label) in enumerate(COMPARE_FEATS):
    ax = axes4[i]
    s_vals = stable[col].dropna()
    w_vals = switchers[col].dropna()
    if len(s_vals) < 3 or len(w_vals) < 3:
        ax.set_visible(False)
        continue
    t, p = ttest_ind(s_vals, w_vals)
    pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
    delta = s_vals.mean() - w_vals.mean()
    print(f"  {label:<35} {s_vals.mean():>9.2f} {w_vals.mean():>9.2f} {delta:>8.2f} {pstr:>8}")

    data_plot = pd.DataFrame({
        'value': pd.concat([s_vals, w_vals], ignore_index=True),
        'group': ['Stable'] * len(s_vals) + ['Switcher'] * len(w_vals)
    })
    palette = {'Stable': 'steelblue', 'Switcher': 'tomato'}
    sns.violinplot(data=data_plot, x='group', y='value', palette=palette,
                   ax=ax, inner='box', cut=0)
    ax.set_title(f"{label}\nΔ={delta:+.2f}  p={pstr}", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(label, fontsize=8)

for j in range(i + 1, len(axes4)):
    axes4[j].set_visible(False)

fig4.suptitle(
    f"Stable (≥85% in one mode, n={len(stable)}) vs "
    f"Switchers (<85%, n={len(switchers)})\n"
    f"Do mode-switchers look mechanically different?",
    fontsize=13, y=1.01)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'stability_4_switchers.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("\nSaved stability_4_switchers.png")

print("\nDone.")
