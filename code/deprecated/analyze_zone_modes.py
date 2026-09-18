"""
Does the fraction of middle-zone contact predict which GMM timing mode a hitter
falls into month-by-month?

Motivation: the LW-vs-timing curves for inside and outside pitches look strikingly
similar to the oppo-peak hitter curve. If a hitter's monthly mode assignment is
explained by what fraction of their contact was on middle-zone pitches, that
suggests the mode is at least partly a product of pitch location distribution
rather than an intrinsic timing tendency.

Tests:
  1. Population-level: r(frac_middle, monthly peak timing angle) across all hitter-months
  2. Mode-level: violin plots of zone fractions split by monthly mode assignment
  3. Within-hitter: when a hitter switches modes, does their zone distribution shift?
  4. Mode prediction: how well does frac_middle alone predict the monthly mode? (logistic)
  5. Confound check: does overall pitch difficulty (speed) co-vary with zone fraction?
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY, INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw, TIMING_AXIS_LABEL, FASTBALL_TYPES, INSIDE_Z, MIDDLE_Z, OUTSIDE_Z, add_zone_and_matchup)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "deprecated")

COLS = [
    "pitch_type", "batter", "stand", "description", "game_date",
    "delta_run_exp", "zone",
    INTERCEPT_X,
    INTERCEPT_Y,
    "release_speed",
]

MIN_PA_SEASON = 50   # qualify for full-season GMM peak
MIN_PA_MONTH  = 20   # qualify for monthly peak


# ── Load ──────────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
df = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES) & df['description'].isin(ALL_SWINGS)].copy()
df = df.dropna(subset=['delta_run_exp', 'stand', 'zone', INTERCEPT_Y])

TIMING_CENTER = add_timing(df)

df['game_date'] = pd.to_datetime(df['game_date'])
df['month'] = df['game_date'].dt.month

df['zone_grp'] = df['zone'].map(
    lambda z: 'inside'  if z in INSIDE_Z  else
              'middle'  if z in MIDDLE_Z  else
              'outside' if z in OUTSIDE_Z else None
)
df = df[df['zone_grp'].notna()].copy()

print(f"All-swing fastballs with timing + zone: {len(df):,}")
print(f"Zone distribution: {df['zone_grp'].value_counts().to_dict()}")

# ── Full-season per-hitter peaks → GMM mode assignment ────────────────────────
# For the full-season GMM, use ALL spline peaks (significant or not) so the
# overall distribution is preserved.
def peak_timing_all(data, min_n=MIN_PA_SEASON):
    """Return spline peak location regardless of significance (for GMM)."""
    if len(data) < min_n:
        return np.nan
    loc, _, _, _ = smoothed_peak(data['timing'].values,
                                  data['delta_run_exp'].values,
                                  min_n=min_n)
    return loc

def peak_timing_sig(data, min_n=MIN_PA_MONTH):
    """Return spline peak location only if side-significance test passes."""
    if len(data) < min_n:
        return np.nan
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n)
    return loc if sig else np.nan

season_peaks = (df.groupby('batter')
                  .apply(peak_timing_all)
                  .dropna()
                  .rename('peak_season'))
print(f"\nFull-season peaks: {len(season_peaks)} hitters")

# 2-component GMM on full-season peaks
gm = GaussianMixture(n_components=2, random_state=42)
gm.fit(season_peaks.values.reshape(-1, 1))
gm_means = gm.means_.flatten()
oppo_idx = int(np.argmin(gm_means))
pull_idx  = int(np.argmax(gm_means))
season_mode = pd.Series(
    ['oppo' if gm.predict([[p]])[0] == oppo_idx else 'pull'
     for p in season_peaks],
    index=season_peaks.index, name='mode_season'
)
print(f"GMM: oppo-peak μ={gm_means[oppo_idx]:.1f} in  n={( season_mode=='oppo').sum()}")
print(f"     pull-peak μ={gm_means[pull_idx]:.1f} in  n={( season_mode=='pull').sum()}")

# ── Monthly peaks and zone fractions per hitter-month ─────────────────────────
CORE_MONTHS = [4, 5, 6, 7, 8, 9]

rows = []
for (batter, month), grp in df[df['month'].isin(CORE_MONTHS)].groupby(['batter', 'month']):
    if len(grp) < MIN_PA_MONTH:
        continue
    peak_m = peak_timing_sig(grp, min_n=MIN_PA_MONTH)
    if np.isnan(peak_m):
        continue
    n_total  = len(grp)
    n_inside  = (grp['zone_grp'] == 'inside').sum()
    n_middle  = (grp['zone_grp'] == 'middle').sum()
    n_outside = (grp['zone_grp'] == 'outside').sum()
    rows.append({
        'batter': batter,
        'month': month,
        'peak_month': peak_m,
        'frac_inside':  n_inside  / n_total,
        'frac_middle':  n_middle  / n_total,
        'frac_outside': n_outside / n_total,
        'n': n_total,
        'mean_speed': grp['release_speed'].mean() if 'release_speed' in grp else np.nan,
        'stand': grp['stand'].iloc[0],
    })

monthly = pd.DataFrame(rows)
monthly = monthly.join(season_mode, on='batter')
# Monthly mode: assign based on GMM applied to monthly peak
monthly['mode_month'] = monthly['peak_month'].apply(
    lambda p: 'oppo' if gm.predict([[p]])[0] == oppo_idx else 'pull'
)

print(f"\nHitter-months qualifying (≥{MIN_PA_MONTH} swings, significant peak): {len(monthly):,}")
print(f"  oppo-mode months: {(monthly['mode_month']=='oppo').sum()}")
print(f"  pull-mode months: {(monthly['mode_month']=='pull').sum()}")

# ── Correlations ──────────────────────────────────────────────────────────────
print("\n--- Correlations with monthly peak timing ---")
for col in ['frac_middle', 'frac_inside', 'frac_outside']:
    r, p = pearsonr(monthly[col], monthly['peak_month'])
    print(f"  r({col:<15}, peak_month) = {r:+.3f}  p={'<0.0001' if p<0.0001 else f'{p:.4f}'}")

r_speed, p_speed = pearsonr(monthly['mean_speed'].dropna(),
                             monthly.loc[monthly['mean_speed'].notna(), 'peak_month'])
print(f"  r(mean_speed,     peak_month) = {r_speed:+.3f}  p={'<0.0001' if p_speed<0.0001 else f'{p_speed:.4f}'}")

print("\n--- Zone fractions by monthly mode (Mann-Whitney U) ---")
for col in ['frac_middle', 'frac_inside', 'frac_outside']:
    oppo_vals = monthly.loc[monthly['mode_month']=='oppo', col]
    pull_vals = monthly.loc[monthly['mode_month']=='pull', col]
    stat, p = mannwhitneyu(oppo_vals, pull_vals, alternative='two-sided')
    print(f"  {col:<15}  oppo μ={oppo_vals.mean():.3f}  pull μ={pull_vals.mean():.3f}  "
          f"Δ={oppo_vals.mean()-pull_vals.mean():+.3f}  p={'<0.0001' if p<0.0001 else f'{p:.4f}'}")

# ── Logistic regression: does frac_middle predict monthly mode? ───────────────
from_oppo = (monthly['mode_month'] == 'pull').astype(int)  # 1=pull, 0=oppo
X_zone = monthly[['frac_middle', 'frac_inside', 'frac_outside']].values
X_mid  = monthly[['frac_middle']].values

lr_zone = LogisticRegression().fit(X_zone, from_oppo)
lr_mid  = LogisticRegression().fit(X_mid,  from_oppo)
auc_zone = roc_auc_score(from_oppo, lr_zone.predict_proba(X_zone)[:,1])
auc_mid  = roc_auc_score(from_oppo, lr_mid.predict_proba(X_mid)[:,1])
print(f"\nLogistic regression AUC predicting pull-mode month:")
print(f"  All 3 zone fractions: AUC = {auc_zone:.3f}")
print(f"  frac_middle only:     AUC = {auc_mid:.3f}")

# ── Within-hitter analysis: switchers ────────────────────────────────────────
hitters_multi = (monthly.groupby('batter')['mode_month']
                         .nunique()
                         .loc[lambda s: s > 1]
                         .index)
within = monthly[monthly['batter'].isin(hitters_multi)].copy()

# For each switcher: compute mean zone fractions when in oppo vs pull mode
switcher_rows = []
for batter, grp in within.groupby('batter'):
    for mode in ['oppo', 'pull']:
        sub = grp[grp['mode_month'] == mode]
        if len(sub) == 0:
            continue
        switcher_rows.append({
            'batter': batter,
            'mode': mode,
            'frac_middle': sub['frac_middle'].mean(),
            'frac_inside': sub['frac_inside'].mean(),
            'frac_outside': sub['frac_outside'].mean(),
            'n_months': len(sub),
        })
switcher_df = pd.DataFrame(switcher_rows)
if len(switcher_df) and 'mode' in switcher_df.columns:
    sw_oppo = switcher_df[switcher_df['mode']=='oppo']['frac_middle']
    sw_pull = switcher_df[switcher_df['mode']=='pull']['frac_middle']
    if len(sw_oppo) >= 5 and len(sw_pull) >= 5:
        r_w, p_w = mannwhitneyu(sw_oppo, sw_pull, alternative='two-sided')
        print(f"\nWithin-hitter switchers ({len(hitters_multi)} hitters):")
        print(f"  frac_middle when oppo: {sw_oppo.mean():.3f}  "
              f"when pull: {sw_pull.mean():.3f}  "
              f"Δ={sw_oppo.mean()-sw_pull.mean():+.3f}  p={'<0.0001' if p_w<0.0001 else f'{p_w:.4f}'}")
    else:
        print(f"\nToo few switchers (oppo={len(sw_oppo)}, pull={len(sw_pull)}) for within-hitter test")
else:
    sw_oppo = pd.Series(dtype=float)
    sw_pull = pd.Series(dtype=float)
    print(f"\nNo switcher hitters (significance filter eliminates monthly mode changes)")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scatter + binned mean (frac_middle vs monthly peak) — 3 panels
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ZONE_CONFIGS = [
    ('frac_middle',  'Middle-Zone Contact Fraction', '#2ca02c'),
    ('frac_inside',  'Inside-Zone Contact Fraction', '#d62728'),
    ('frac_outside', 'Outside-Zone Contact Fraction', '#1f77b4'),
]

for ax, (col, label, color) in zip(axes, ZONE_CONFIGS):
    x = monthly[col].values
    y = monthly['peak_month'].values
    r, p = pearsonr(x, y)
    pstr = '<0.0001' if p < 0.0001 else f'{p:.4f}'

    sc = ax.scatter(x, y, c=monthly['mode_month'].map({'oppo':'#1f77b4','pull':'#d62728'}),
                    alpha=0.25, s=15, edgecolors='none')

    # binned mean
    bins = pd.qcut(pd.Series(x), q=20, duplicates='drop')
    bm   = pd.DataFrame({'x': x, 'y': y, 'b': bins}).groupby('b', observed=True)
    ax.plot(bm['x'].mean().values, bm['y'].mean().values,
            color=color, lw=2.5, zorder=5, label='Binned mean')

    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Monthly Peak Timing (in)\n← oppo/late      pull/early →", fontsize=9)
    ax.set_title(f"{label}\nr = {r:+.3f}   p = {pstr}   n = {len(monthly):,}", fontsize=10)
    ax.legend(fontsize=9)

    # custom scatter legend for mode color
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4', ms=7, label='oppo-mode month'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#d62728', ms=7, label='pull-mode month'),
        Line2D([0],[0], color=color, lw=2.5, label='Binned mean'),
    ]
    ax.legend(handles=handles, fontsize=8)

fig.suptitle("Do Zone Contact Fractions Predict Monthly Peak Timing?\n"
             "Each point = one hitter-month  ·  Colour = monthly GMM mode assignment",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'zone_modes_1_scatter.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("\nSaved → zone_modes_1_scatter.png")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Violin plots — zone fractions by monthly mode
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 6))

MODE_COLORS = {'oppo': '#1f77b4', 'pull': '#d62728'}
for ax, (col, label, _) in zip(axes, ZONE_CONFIGS):
    for i, mode in enumerate(['oppo', 'pull']):
        vals = monthly.loc[monthly['mode_month']==mode, col].values
        parts = ax.violinplot([vals], positions=[i], widths=0.6, showmedians=True,
                              showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(MODE_COLORS[mode])
            pc.set_alpha(0.55)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)
        ax.scatter([i + np.random.uniform(-0.15, 0.15, len(vals))],
                   vals, c=MODE_COLORS[mode], alpha=0.12, s=8, edgecolors='none')

    oppo_vals = monthly.loc[monthly['mode_month']=='oppo', col]
    pull_vals = monthly.loc[monthly['mode_month']=='pull', col]
    _, p_mw = mannwhitneyu(oppo_vals, pull_vals, alternative='two-sided')
    pstr = '<0.0001' if p_mw < 0.0001 else f'{p_mw:.4f}'

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['oppo-mode\nmonth', 'pull-mode\nmonth'], fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    delta = oppo_vals.mean() - pull_vals.mean()
    ax.set_title(f"{label}\noppo μ={oppo_vals.mean():.3f}  pull μ={pull_vals.mean():.3f}"
                 f"  Δ={delta:+.3f}  p={pstr}", fontsize=10)

fig.suptitle("Zone Contact Fractions by Monthly GMM Mode\n"
             "Each point = one hitter-month",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'zone_modes_2_violin.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → zone_modes_2_violin.png")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Within-hitter — when they switch modes, does zone fraction shift?
# ════════════════════════════════════════════════════════════════════════════════
# For each switcher hitter, pair their oppo-months and pull-months
# Show the distribution of (frac_middle_oppo - frac_middle_pull) per hitter
if len(switcher_df) and 'mode' in switcher_df.columns:
    paired = (switcher_df.pivot_table(index='batter', columns='mode',
                                       values=['frac_middle','frac_inside','frac_outside'],
                                       aggfunc='mean')
                         .dropna())
else:
    paired = pd.DataFrame()
if len(paired) >= 10:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, frac_col, label in zip(axes,
        ['frac_middle', 'frac_inside', 'frac_outside'],
        ['Middle-Zone', 'Inside-Zone', 'Outside-Zone']):

        diff = paired[(frac_col, 'oppo')] - paired[(frac_col, 'pull')]
        r_diff, p_diff = pearsonr(paired[(frac_col, 'oppo')], paired[(frac_col, 'pull')])

        ax.hist(diff, bins=25, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(0,          color='black', lw=1.2, ls='--', alpha=0.7, label='No difference')
        ax.axvline(diff.mean(), color='crimson', lw=2, label=f'Mean = {diff.mean():+.3f}')
        _, p_t = mannwhitneyu(paired[(frac_col, 'oppo')], paired[(frac_col, 'pull')],
                               alternative='two-sided')
        pstr = '<0.0001' if p_t < 0.0001 else f'{p_t:.4f}'
        ax.set_xlabel(f"{label} Fraction: oppo-months − pull-months", fontsize=10)
        ax.set_ylabel("Number of Hitters", fontsize=10)
        ax.set_title(f"Within-Hitter {label} Fraction Shift\n"
                     f"n = {len(diff)} hitters  ·  p (MWU) = {pstr}", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("When Hitters Switch Timing Modes, Does Their Zone Distribution Shift?\n"
                 "Each bar = one switcher hitter (hitters with both oppo and pull months)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'zone_modes_3_within.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print("Saved → zone_modes_3_within.png")
else:
    print(f"Only {len(paired)} paired switchers — skipping Fig 3")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 4: LW curves overlaid — middle vs inside/outside vs oppo/pull-mode hitters
# ════════════════════════════════════════════════════════════════════════════════
# Show directly WHY the curves look similar: overlay the population LW curves for
# middle-only contact alongside the LW curves for oppo-peak-mode hitters
def _lw_curve(data, n=40):
    sub = data.dropna(subset=['timing', 'delta_run_exp'])
    xm, ym, _ = binned_lw(sub['timing'].values, sub['delta_run_exp'].values, n_bins=n)
    return xm, ym

oppo_batters = season_mode[season_mode == 'oppo'].index
pull_batters = season_mode[season_mode == 'pull'].index

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, title, zone_pair in [
    (axes[0], "Middle Zone vs Oppo-Peak Mode Hitters", ('middle', 'inside')),
    (axes[1], "Middle Zone vs Pull-Peak Mode Hitters", ('middle', 'outside')),
]:
    z1, z2 = zone_pair

    # LW curve for middle-zone pitches (all hitters)
    xm, ym = _lw_curve(df[df['zone_grp']=='middle'])
    ax.plot(xm, ym, color='#2ca02c', lw=2.5, label='Middle-zone contact')

    # LW curve for inside/outside pitches
    xi, yi = _lw_curve(df[df['zone_grp']==z2])
    ax.plot(xi, yi, color='#ff7f0e', lw=2, ls='--', label=f'{z2.capitalize()}-zone contact')

    # LW curve for oppo-peak mode hitters (all zones)
    xo, yo = _lw_curve(df[df['batter'].isin(oppo_batters)])
    ax.plot(xo, yo, color='#1f77b4', lw=2, ls=':', label='Oppo-peak mode hitters (all zones)')

    # LW curve for pull-peak mode hitters (all zones)
    xp, yp = _lw_curve(df[df['batter'].isin(pull_batters)])
    ax.plot(xp, yp, color='#d62728', lw=2, ls=':', label='Pull-peak mode hitters (all zones)')

    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(TIMING_AXIS_LABEL, fontsize=10)
    ax.set_ylabel("Mean Delta Run Expectancy", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)

fig.suptitle("Why Do Zone Curves Look Like Mode Curves?\n"
             "Population LW-vs-timing by pitch zone vs by full-season GMM mode",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'zone_modes_4_lw_overlay.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → zone_modes_4_lw_overlay.png")

print("\nDone.")
