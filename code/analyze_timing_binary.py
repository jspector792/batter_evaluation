"""
Timing Binary Hypothesis Test — 2025 MLB Fastballs (In-Play only)

Is timing preference universal (peak near 0 for all hitters) or does it
vary systematically by hitter type, matchup, and pitch location?

Analyses:
  1. Distribution of per-hitter peak timing angles
  2. Same, broken out by batter handedness
  3. Same, paired same-hand vs diff-hand matchup peaks per batter
  4. H1-style LW vs timing: batter handedness × pitcher matchup subsets
  5. H1-style LW vs timing: zone subsets (inside / middle / outside)
  + Extra: bimodality check on peak distribution
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ttest_ind, ttest_rel, mannwhitneyu

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
IN_PLAY  = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
MIN_PA   = 50    # minimum in-play fastballs for per-hitter peak
MIN_PAIR = 30    # minimum per matchup-type for paired matchup analysis
POLY_DEG = 4

COLS = [
    "pitch_type", "batter", "stand", "p_throws", "zone",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "delta_run_exp", "description",
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
print(f"In-play fastballs with all fields: {len(df):,}")

# ── Timing angle ─────────────────────────────────────────────────────────────
df['timing_raw'] = np.degrees(np.arctan2(df[IX], df[IY]))
TIMING_MED = df['timing_raw'].median()
df['timing']  = df['timing_raw'] - TIMING_MED

# ── Derived columns ───────────────────────────────────────────────────────────
df['matchup'] = np.where(df['stand'] == df['p_throws'], 'same-hand', 'diff-hand')

inside_z  = {1, 4, 7, 11, 13}
middle_z  = {2, 5, 8}
outside_z = {3, 6, 9, 12, 14}
def zone_label(z):
    if z in inside_z:  return 'inside'
    if z in middle_z:  return 'middle'
    if z in outside_z: return 'outside'
    return np.nan
df['zone_grp'] = df['zone'].map(zone_label)

print(f"Stand:   {df['stand'].value_counts().to_dict()}")
print(f"Matchup: {df['matchup'].value_counts().to_dict()}")
print(f"Zone:    {df['zone_grp'].value_counts().to_dict()}")

# ── Peak-finding helper ───────────────────────────────────────────────────────
def peak_timing(data, min_n=MIN_PA, degree=POLY_DEG):
    """Fit a degree-`degree` polynomial and return the timing angle of peak LW."""
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
N_BINS = 30
def lw_curve(data):
    data = data.dropna(subset=['timing', 'delta_run_exp'])
    if len(data) < 200:
        return None
    bins = np.linspace(data['timing'].quantile(0.005),
                       data['timing'].quantile(0.995), N_BINS + 1)
    data = data.copy()
    data['bin'] = pd.cut(data['timing'], bins=bins)
    g = (data.groupby('bin', observed=True)
              .agg(mean_lw=('delta_run_exp','mean'),
                   sem_lw =('delta_run_exp','sem'),
                   mid    =('timing','median'),
                   n      =('delta_run_exp','count'))
              .dropna().reset_index())
    return g

def plot_lw_curves(ax, subsets, title, show_ci=True):
    """Overlay multiple LW-vs-timing curves. subsets = {label: DataFrame}"""
    palette = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b']
    for (label, data), color in zip(subsets.items(), palette):
        g = lw_curve(data)
        if g is None:
            continue
        if show_ci:
            ax.fill_between(g['mid'],
                            g['mean_lw'] - 1.96*g['sem_lw'],
                            g['mean_lw'] + 1.96*g['sem_lw'],
                            alpha=0.12, color=color)
        ax.plot(g['mid'], g['mean_lw'], color=color, lw=2,
                marker='o', ms=3, label=f"{label}  (n={len(data):,})")
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(0, color='grey',  lw=0.8, ls=':')
    ax.set_xlabel("Timing Angle (°, centered on population median)", fontsize=9)
    ax.set_ylabel("Mean Δ Run Expectancy", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Per-hitter peak calculations
# ═══════════════════════════════════════════════════════════════════════════════
print("\nCalculating per-hitter peak timing angles...")

# Analysis 1 — all in-play fastballs per batter
all_peaks = (df.groupby('batter')
               .apply(peak_timing)
               .dropna()
               .rename('peak'))
print(f"\nAll batters ≥{MIN_PA} swings: n={len(all_peaks)}")
print(f"  mean={all_peaks.mean():.2f}°  median={all_peaks.median():.2f}°  "
      f"std={all_peaks.std():.2f}°")
print(f"  % peaking early (< 0°): {(all_peaks < 0).mean():.1%}")

# Analysis 2 — per batter, separated by batter handedness
stand_col = df.groupby('batter')['stand'].agg(lambda s: s.mode()[0])
all_peaks_df = pd.DataFrame({'peak': all_peaks, 'stand': stand_col})

l_peaks = all_peaks_df.loc[all_peaks_df['stand'] == 'L', 'peak']
r_peaks = all_peaks_df.loc[all_peaks_df['stand'] == 'R', 'peak']
print(f"\nLHH: n={len(l_peaks)}  mean={l_peaks.mean():.2f}°  median={l_peaks.median():.2f}°")
print(f"RHH: n={len(r_peaks)}  mean={r_peaks.mean():.2f}°  median={r_peaks.median():.2f}°")
t_lr, p_lr = ttest_ind(l_peaks.dropna(), r_peaks.dropna())
print(f"LHH vs RHH t-test: t={t_lr:.3f}  p={p_lr:.4f}")

# Analysis 3 — paired same-hand vs diff-hand peaks per batter
same_peaks = (df[df['matchup']=='same-hand']
              .groupby('batter')
              .apply(lambda g: peak_timing(g, min_n=MIN_PAIR))
              .dropna()
              .rename('same_peak'))
diff_peaks = (df[df['matchup']=='diff-hand']
              .groupby('batter')
              .apply(lambda g: peak_timing(g, min_n=MIN_PAIR))
              .dropna()
              .rename('diff_peak'))

paired = pd.DataFrame({'same_peak': same_peaks, 'diff_peak': diff_peaks}).dropna()
paired['delta_peak'] = paired['diff_peak'] - paired['same_peak']
paired['stand'] = stand_col.reindex(paired.index)
print(f"\nPaired matchup analysis (≥{MIN_PAIR} swings each): n={len(paired)}")
print(f"  Same-hand mean={paired['same_peak'].mean():.2f}°  "
      f"Diff-hand mean={paired['diff_peak'].mean():.2f}°")
print(f"  Mean shift (diff − same) = {paired['delta_peak'].mean():.2f}°")
t_p, p_p = ttest_rel(paired['same_peak'], paired['diff_peak'])
print(f"  Paired t-test: t={t_p:.3f}  p={p_p:.4f}")
for hand in ['L','R']:
    sub = paired[paired['stand'] == hand]
    t_h, p_h = ttest_rel(sub['same_peak'], sub['diff_peak'])
    print(f"  {hand}HH: same mean={sub['same_peak'].mean():.2f}°  "
          f"diff mean={sub['diff_peak'].mean():.2f}°  "
          f"shift={sub['delta_peak'].mean():.2f}°  p={p_h:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: All-batter peak distribution ──────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes1[0]
ax.hist(all_peaks.values, bins=35, edgecolor='k', lw=0.4,
        color='steelblue', alpha=0.75, density=True, label='histogram')
kde = gaussian_kde(all_peaks.values)
xs  = np.linspace(all_peaks.min(), all_peaks.max(), 400)
ax.plot(xs, kde(xs), 'r-', lw=2.5, label='KDE')
ax.axvline(0,                   color='black', lw=1.2, ls='--', label='Neutral (0°)')
ax.axvline(all_peaks.median(),  color='green',  lw=1.5, ls=':',
           label=f"Median peak = {all_peaks.median():.1f}°")
ax.axvline(all_peaks.mean(),    color='orange', lw=1.5, ls=':',
           label=f"Mean peak = {all_peaks.mean():.1f}°")
ax.set_xlabel("Per-Hitter Peak Timing Angle (°, centered on median)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Distribution of Per-Hitter Peak Timing Angles\n"
             f"(n={len(all_peaks)} batters, ≥{MIN_PA} in-play fastballs each)", fontsize=11)
ax.legend(fontsize=8)

ax = axes1[1]
sorted_peaks = np.sort(all_peaks.values)
cdf = np.arange(1, len(sorted_peaks)+1) / len(sorted_peaks)
ax.plot(sorted_peaks, cdf, color='steelblue', lw=2)
ax.axvline(0, color='black', lw=1.2, ls='--', label='Neutral (0°)')
ax.axhline(0.5, color='grey', lw=0.8, ls=':')
pct_early = (all_peaks < 0).mean()
ax.fill_betweenx([0,1], sorted_peaks.min(), 0, alpha=0.08, color='green', label=f'Early (<0): {pct_early:.1%}')
ax.fill_betweenx([0,1], 0, sorted_peaks.max(), alpha=0.08, color='red', label=f'Late (>0): {1-pct_early:.1%}')
ax.set_xlabel("Peak Timing Angle (°)", fontsize=11)
ax.set_ylabel("Cumulative Fraction of Batters", fontsize=11)
ax.set_title("CDF of Per-Hitter Peak Timing Angle", fontsize=11)
ax.legend(fontsize=8)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'timing_binary_1_all.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\nSaved timing_binary_1_all.png")

# ── Fig 2: L vs R batter peak distributions + H1 curve ───────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes2[0]
for sub, label, color, ls in [
    (l_peaks, f'LHH (n={len(l_peaks)})', '#1f77b4', '-'),
    (r_peaks, f'RHH (n={len(r_peaks)})', '#d62728', '--'),
]:
    ax.hist(sub.values, bins=25, color=color, alpha=0.45, density=True,
            edgecolor='k', lw=0.3)
    kde_h = gaussian_kde(sub.dropna().values)
    xs = np.linspace(sub.min(), sub.max(), 300)
    ax.plot(xs, kde_h(xs), color=color, lw=2.5, ls=ls, label=label)
ax.axvline(0, color='black', lw=1.2, ls='--')
ax.set_xlabel("Peak Timing Angle (°)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Per-Hitter Peak: LHH vs RHH\n"
             f"t-test p={p_lr:.4f}  "
             f"LHH mean={l_peaks.mean():.1f}°  RHH mean={r_peaks.mean():.1f}°", fontsize=10)
ax.legend(fontsize=9)

plot_lw_curves(axes2[1], {
    'LHH': df[df['stand']=='L'],
    'RHH': df[df['stand']=='R'],
}, "Mean LW vs Timing Angle\nLHH vs RHH")

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'timing_binary_2_handedness.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("Saved timing_binary_2_handedness.png")

# ── Fig 3: Paired same-hand vs diff-hand matchup peaks ───────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(20, 5))

# Panel 1: paired distribution (all)
ax = axes3[0]
for col, label, color, ls in [
    ('same_peak', f'same-hand (n={len(paired)})', '#ff7f0e', '-'),
    ('diff_peak',  f'diff-hand  (n={len(paired)})', '#9467bd', '--'),
]:
    vals = paired[col].values
    ax.hist(vals, bins=25, color=color, alpha=0.45, density=True, edgecolor='k', lw=0.3)
    kde_h = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(xs, kde_h(xs), color=color, lw=2.5, ls=ls, label=label)
ax.axvline(0, color='black', lw=1.2, ls='--')
ax.set_xlabel("Peak Timing Angle (°)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"Paired Same vs Diff Handedness Peaks\n"
             f"Mean shift (diff−same) = {paired['delta_peak'].mean():.2f}°  p={p_p:.4f}", fontsize=10)
ax.legend(fontsize=8)

# Panel 2: shift distribution (delta_peak = diff - same)
ax = axes3[1]
for hand, color in [('L','#1f77b4'),('R','#d62728'),('all','grey')]:
    if hand == 'all':
        vals = paired['delta_peak'].values
        label = f'All (n={len(vals)})'
    else:
        vals = paired[paired['stand']==hand]['delta_peak'].values
        label = f'{hand}HH (n={len(vals)})'
    if len(vals) > 5:
        kde_h = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xs, kde_h(xs), color=color, lw=2, label=label)
ax.axvline(0, color='black', lw=1.2, ls='--', label='No shift')
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                  0, 50, alpha=0.05, color='red', zorder=0)
ax.set_xlabel("Shift in Peak Timing: Diff-Hand − Same-Hand (°)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Do Batters Shift Timing vs Different-Hand Pitchers?", fontsize=10)
ax.legend(fontsize=8)

# Panel 3: H1 curve by matchup
plot_lw_curves(axes3[2], {
    'same-hand': df[df['matchup']=='same-hand'],
    'diff-hand':  df[df['matchup']=='diff-hand'],
}, "Mean LW vs Timing\nSame vs Diff Handedness Matchup")

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'timing_binary_3_matchup.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved timing_binary_3_matchup.png")

# ── Fig 4: H1-style 2×2 by batter hand × pitcher hand ───────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))

plot_lw_curves(axes4[0,0], {
    'LHH': df[df['stand']=='L'],
    'RHH': df[df['stand']=='R'],
}, "LW vs Timing: All LHH vs All RHH")

plot_lw_curves(axes4[0,1], {
    'same-hand': df[df['matchup']=='same-hand'],
    'diff-hand':  df[df['matchup']=='diff-hand'],
}, "LW vs Timing: Same vs Diff Handedness (all batters)")

plot_lw_curves(axes4[1,0], {
    'LHH vs LHP (same)': df[(df['stand']=='L') & (df['p_throws']=='L')],
    'LHH vs RHP (diff)': df[(df['stand']=='L') & (df['p_throws']=='R')],
}, "LHH: vs LHP (same-hand) vs vs RHP (diff-hand)")

plot_lw_curves(axes4[1,1], {
    'RHH vs RHP (same)': df[(df['stand']=='R') & (df['p_throws']=='R')],
    'RHH vs LHP (diff)': df[(df['stand']=='R') & (df['p_throws']=='L')],
}, "RHH: vs RHP (same-hand) vs vs LHP (diff-hand)")

fig4.suptitle("Analysis 4: H1-Style LW vs Timing — Handedness Breakdown",
              fontsize=13, y=1.01)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'timing_binary_4_h1_matchup.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved timing_binary_4_h1_matchup.png")

# ── Fig 5: H1-style by zone ───────────────────────────────────────────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(16, 10))

plot_lw_curves(axes5[0,0], {
    'inside':  df[df['zone_grp']=='inside'],
    'middle':  df[df['zone_grp']=='middle'],
    'outside': df[df['zone_grp']=='outside'],
}, "LW vs Timing by Pitch Zone (all batters)")

plot_lw_curves(axes5[0,1], {
    'RHH — inside':  df[(df['stand']=='R') & (df['zone_grp']=='inside')],
    'RHH — outside': df[(df['stand']=='R') & (df['zone_grp']=='outside')],
    'LHH — inside':  df[(df['stand']=='L') & (df['zone_grp']=='inside')],
    'LHH — outside': df[(df['stand']=='L') & (df['zone_grp']=='outside')],
}, "LW vs Timing: Zone × Batter Handedness")

plot_lw_curves(axes5[1,0], {
    'same-hand, inside':  df[(df['matchup']=='same-hand') & (df['zone_grp']=='inside')],
    'same-hand, outside': df[(df['matchup']=='same-hand') & (df['zone_grp']=='outside')],
    'diff-hand, inside':  df[(df['matchup']=='diff-hand') & (df['zone_grp']=='inside')],
    'diff-hand, outside': df[(df['matchup']=='diff-hand') & (df['zone_grp']=='outside')],
}, "LW vs Timing: Matchup × Zone")

plot_lw_curves(axes5[1,1], {
    'inside':  df[df['zone_grp']=='inside'],
    'middle':  df[df['zone_grp']=='middle'],
    'outside': df[df['zone_grp']=='outside'],
}, "LW vs Timing by Zone\n(middle panel for context)")

fig5.suptitle("Analysis 5: H1-Style LW vs Timing — Zone Location Breakdown",
              fontsize=13, y=1.01)
plt.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'timing_binary_5_zones.png'), dpi=180, bbox_inches='tight')
plt.close(fig5)
print("Saved timing_binary_5_zones.png")

# ── Fig 6: Bimodality check ───────────────────────────────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(20, 5))

# Multi-bandwidth KDE
ax = axes6[0]
for bw_scale, ls, alpha in [(0.3, '-', 1.0), (1.0, '--', 0.8), (2.5, ':', 0.7)]:
    bw = bw_scale / all_peaks.std()
    kde_bw = gaussian_kde(all_peaks.values, bw_method=bw)
    xs = np.linspace(all_peaks.min(), all_peaks.max(), 400)
    ax.plot(xs, kde_bw(xs), lw=2, ls=ls, label=f'BW factor={bw_scale}°')
ax.axvline(0, color='black', lw=1, ls='--', label='Neutral')
ax.set_xlabel("Peak Timing Angle (°)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Bimodality Check: KDE at Multiple Bandwidths\n"
             "(multimodal = separate humps stable across BW)", fontsize=10)
ax.legend(fontsize=8)

# Per-hitter peak vs sample size (noise check)
ax = axes6[1]
sc = ax.scatter(
    all_peaks_df.dropna()['peak'].reindex(all_peaks_df.dropna().index),
    [df[df['batter']==b].shape[0] for b in all_peaks_df.dropna().index],
    c=all_peaks_df.dropna()['stand'].map({'L':'steelblue','R':'tomato'}),
    alpha=0.4, s=12
)
ax.axvline(0, color='black', lw=1, ls='--')
ax.set_xlabel("Peak Timing Angle (°)", fontsize=10)
ax.set_ylabel("N in-play fastballs", fontsize=10)
ax.set_title("Peak Timing vs Sample Size\n(stability check — real signal ≠ noise-driven)", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='steelblue', label='LHH'), Patch(color='tomato', label='RHH')], fontsize=8)

# Paired scatter: same-hand peak vs diff-hand peak
ax = axes6[2]
for hand, color, marker in [('L','#1f77b4','o'),('R','#d62728','^')]:
    sub = paired[paired['stand']==hand]
    ax.scatter(sub['same_peak'], sub['diff_peak'], color=color, alpha=0.4,
               s=15, marker=marker, label=f'{hand}HH (n={len(sub)})')
lims = [min(paired['same_peak'].min(), paired['diff_peak'].min()),
        max(paired['same_peak'].max(), paired['diff_peak'].max())]
ax.plot(lims, lims, 'k--', lw=1, label='same peak (y=x)')
ax.set_xlabel("Peak Timing — Same-Hand PAs (°)", fontsize=10)
ax.set_ylabel("Peak Timing — Diff-Hand PAs (°)", fontsize=10)
ax.set_title(f"Do Batters Consistently Shift Timing by Matchup?\n"
             f"r = {paired['same_peak'].corr(paired['diff_peak']):.3f}", fontsize=10)
ax.legend(fontsize=8)

fig6.suptitle("Additional: Bimodality Check & Timing Stability", fontsize=12, y=1.01)
plt.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, 'timing_binary_6_bimodality.png'), dpi=180, bbox_inches='tight')
plt.close(fig6)
print("Saved timing_binary_6_bimodality.png")

# ── Fig 7: Mean timing angle per hitter with std error bars ──────────────────
# Test whether hitters with positive peak timing also have positive MEAN timing
# (i.e., all their contact is late, not just their optimal contact).
# If bimodality reflects swing shape, mean and peak should track together.

hitter_stats = (
    df.groupby('batter')['timing']
    .agg(mean_timing='mean', std_timing='std', n='count')
    .dropna()
)
hitter_stats['stand'] = stand_col.reindex(hitter_stats.index)
hitter_stats['peak']  = all_peaks.reindex(hitter_stats.index)
hitter_stats = hitter_stats.dropna(subset=['stand'])

# Require same minimum as per-hitter peak analysis
hitter_stats = hitter_stats[hitter_stats['n'] >= MIN_PA]

print(f"\nFig 7: {len(hitter_stats)} hitters with ≥{MIN_PA} in-play fastballs")
print(f"  Mean timing: mean={hitter_stats['mean_timing'].mean():.2f}°  "
      f"std={hitter_stats['mean_timing'].std():.2f}°")
print(f"  Mean std (within-hitter spread): {hitter_stats['std_timing'].mean():.2f}°")

# Correlation between mean timing and peak timing (when both available)
both = hitter_stats.dropna(subset=['peak', 'mean_timing'])
r_mean_peak = both['mean_timing'].corr(both['peak'])
print(f"  Correlation mean_timing vs peak_timing: r={r_mean_peak:.3f}  (n={len(both)})")

fig7, ax7 = plt.subplots(figsize=(13, 7))

for hand, color, marker, zorder in [('L', 'steelblue', 'o', 3), ('R', 'tomato', '^', 2)]:
    sub = hitter_stats[hitter_stats['stand'] == hand]
    # Sort by mean timing so error bars don't overlap badly
    sub = sub.sort_values('mean_timing')
    ax7.errorbar(
        sub['mean_timing'], sub['n'],
        xerr=sub['std_timing'],
        fmt='none',
        ecolor=color, alpha=0.25, elinewidth=0.8, capsize=0, zorder=zorder,
    )
    ax7.scatter(
        sub['mean_timing'], sub['n'],
        c=color, alpha=0.6, s=18, marker=marker,
        label=f"{hand}HH (n={len(sub)}, mean={sub['mean_timing'].mean():.1f}°)",
        zorder=zorder + 1,
    )

ax7.axvline(0, color='black', lw=1.2, ls='--', label='Neutral timing (0°)')

# Add annotation: if swing-shape hypothesis is true, hitters cluster into two
# horizontal bands; if not, mean timing is similar for all.
ax7.set_xlabel(
    "Mean Timing Angle per Hitter (°, centered on population median)\n"
    "← early (pull-side)                  late (oppo-side) →",
    fontsize=11,
)
ax7.set_ylabel("N In-Play Fastballs", fontsize=11)
ax7.set_title(
    f"Mean Timing Angle per Hitter (all contact, not just peaks)\n"
    f"Error bars = ±1 SD within hitter   "
    f"Corr(mean, peak) r={r_mean_peak:.3f}   n={len(hitter_stats)} hitters",
    fontsize=12,
)
ax7.legend(fontsize=9)

plt.tight_layout()
fig7.savefig(os.path.join(OUT_DIR, 'timing_binary_7_mean_timing.png'), dpi=180, bbox_inches='tight')
plt.close(fig7)
print("Saved timing_binary_7_mean_timing.png")

print("\nDone.")
