"""
Within-hitter analysis: does xwOBA vary significantly with timing mode?

Two levels of aggregation:
  - Month-level  (Apr–Sep, ≥20 contact for mode, ≥15 in-play for xwOBA)
  - Half-season  (H1=Apr–Jun, H2=Jul–Sep, same thresholds × 3)

For each hitter with observations in BOTH modes, compute the within-hitter
difference (pull xwOBA − oppo xwOBA) and test whether the distribution of
differences is centred on zero.

Also shows cross-sectional baseline: always-oppo vs always-pull hitters.
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ttest_ind
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY, INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw, TIMING_AXIS_LABEL, FASTBALL_TYPES, INSIDE_Z, MIDDLE_Z, OUTSIDE_Z, add_zone_and_matchup)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "deprecated")

MIN_SWING_PERIOD  = 20   # all-swing samples needed to compute timing peak
MIN_INPLAY_PERIOD = 15   # in-play balls needed for stable xwOBA
CORE_MONTHS = [4, 5, 6, 7, 8, 9]
H1_MONTHS   = [4, 5, 6]
H2_MONTHS   = [7, 8, 9]

COLS = [
    "pitch_type", "batter", "stand", "description", "game_date",
    "delta_run_exp",
    "estimated_woba_using_speedangle",
    "launch_speed", "launch_speed_angle",
    INTERCEPT_X,
    INTERCEPT_Y,
]

# ── Load ──────────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
raw = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
raw = raw[raw['pitch_type'].isin(FASTBALL_TYPES)].copy()
raw = raw.dropna(subset=[INTERCEPT_Y])

# All swings (in-play + foul + miss) — used for timing-peak / mode assignment.
all_swings = raw[raw['description'].isin(ALL_SWINGS) &
                 raw['delta_run_exp'].notna()].copy()
TIMING_CENTER = add_timing(all_swings)

# In-play swings — used for xwOBA aggregation only. Centre on the same reference.
inplay = raw[raw['description'].isin(IN_PLAY) &
             raw['estimated_woba_using_speedangle'].notna()].copy()
add_timing(inplay, center=TIMING_CENTER)

for sub in (all_swings, inplay):
    sub['game_date'] = pd.to_datetime(sub['game_date'])
    sub['month'] = sub['game_date'].dt.month
    sub['half']  = sub['month'].map(lambda m: 'H1' if m in H1_MONTHS else
                                              ('H2' if m in H2_MONTHS else None))

print(f"All swings (in-play + foul + miss): {len(all_swings):,}")
print(f"In-play swings:                     {len(inplay):,}  "
      f"(xwOBA available: {inplay['estimated_woba_using_speedangle'].notna().sum():,})")

# ── Full-season GMM for mode assignment ───────────────────────────────────────
# All-spline-peaks (significant or not) for full-season GMM so the population
# distribution is preserved.
def peak_timing_all(data, min_n=50):
    if len(data) < min_n:
        return np.nan
    loc, _, _, _ = smoothed_peak(data['timing'].values,
                                  data['delta_run_exp'].values,
                                  min_n=min_n)
    return loc

def peak_timing_sig(data, min_n=MIN_SWING_PERIOD):
    """Return spline peak location only if side-significance test passes."""
    if len(data) < min_n:
        return np.nan
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n)
    return loc if sig else np.nan

season_peaks = (all_swings.groupby('batter')
                          .apply(lambda g: peak_timing_all(g, min_n=50))
                          .dropna()
                          .rename('peak_season'))

gm = GaussianMixture(n_components=2, random_state=42)
gm.fit(season_peaks.values.reshape(-1, 1))
gm_means = gm.means_.flatten()
oppo_comp = int(np.argmin(gm_means))
pull_comp  = int(np.argmax(gm_means))

def assign_mode(peak):
    return 'pull' if gm.predict([[peak]])[0] == pull_comp else 'oppo'

season_mode = season_peaks.map(assign_mode).rename('mode_season')
print(f"\nFull-season GMM  oppo: μ={gm_means[oppo_comp]:.1f} in  "
      f"n={(season_mode=='oppo').sum()}  |  "
      f"pull: μ={gm_means[pull_comp]:.1f} in  n={(season_mode=='pull').sum()}")

# ── Helper: build period-level table ─────────────────────────────────────────
def build_period_table(period_col, period_vals, min_swing, min_ip):
    """
    For each (batter, period) with enough data, compute:
      - timing peak (significant only) → GMM mode    [all-swing population]
      - mean xwOBA (in-play)
      - mean exit velocity, barrel rate (in-play)
    Returns a DataFrame.
    """
    rows = []
    s_grp = all_swings[all_swings[period_col].isin(period_vals)].groupby(['batter', period_col])
    i_grp = inplay[inplay[period_col].isin(period_vals)].groupby(['batter', period_col])

    xwoba_map  = {k: g['estimated_woba_using_speedangle'].mean()
                  for k, g in i_grp if len(g) >= min_ip}
    ev_map     = {k: g['launch_speed'].mean()
                  for k, g in i_grp if len(g) >= min_ip}
    barrel_map = {k: (g['launch_speed_angle'] == 6).mean()
                  for k, g in i_grp if len(g) >= min_ip}
    n_ip_map   = {k: len(g) for k, g in i_grp}

    for (batter, period), grp in s_grp:
        if len(grp) < min_swing:
            continue
        key = (batter, period)
        if key not in xwoba_map:
            continue
        pk = peak_timing_sig(grp, min_n=min_swing)
        if np.isnan(pk):
            continue
        rows.append({
            'batter': batter,
            period_col: period,
            'peak': pk,
            'mode': assign_mode(pk),
            'xwoba': xwoba_map[key],
            'ev': ev_map.get(key, np.nan),
            'barrel_rate': barrel_map.get(key, np.nan),
            'n_swings': len(grp),
            'n_inplay': n_ip_map.get(key, 0),
            'mode_season': season_mode.get(batter, np.nan),
        })
    return pd.DataFrame(rows)

month_df = build_period_table('month', CORE_MONTHS, MIN_SWING_PERIOD, MIN_INPLAY_PERIOD)
half_df  = build_period_table('half',  ['H1','H2'],  MIN_SWING_PERIOD * 3, MIN_INPLAY_PERIOD * 3)

print(f"\nMonth-level records: {len(month_df):,}  "
      f"(oppo={( month_df['mode']=='oppo').sum()}  pull={( month_df['mode']=='pull').sum()})")
print(f"Half-season records: {len(half_df):,}  "
      f"(oppo={( half_df['mode']=='oppo').sum()}  pull={( half_df['mode']=='pull').sum()})")

# ── Within-hitter paired analysis ────────────────────────────────────────────
def within_hitter_diff(df, period_col):
    """
    For hitters with at least one oppo-month and one pull-month,
    compute mean xwOBA in each mode and return the difference pull − oppo.
    """
    rows = []
    for batter, grp in df.groupby('batter'):
        oppo_obs = grp[grp['mode'] == 'oppo']['xwoba']
        pull_obs = grp[grp['mode'] == 'pull']['xwoba']
        if oppo_obs.empty or pull_obs.empty:
            continue
        rows.append({
            'batter': batter,
            'xwoba_oppo': oppo_obs.mean(),
            'xwoba_pull': pull_obs.mean(),
            'diff': pull_obs.mean() - oppo_obs.mean(),   # pull − oppo
            'n_oppo': len(oppo_obs),
            'n_pull': len(pull_obs),
            'mode_season': grp['mode_season'].iloc[0],
        })
    return pd.DataFrame(rows)

within_month = within_hitter_diff(month_df, 'month')
within_half  = within_hitter_diff(half_df,  'half')

def report_within(wdf, label):
    if 'diff' not in wdf.columns or len(wdf) == 0:
        print(f"\n{label}: 0 paired hitters — no switchers under the significance filter")
        return
    d = wdf['diff'].dropna()
    if len(d) < 5:
        print(f"\n{label}: only {len(d)} paired hitters — skipping")
        return
    mean_d = d.mean()
    median_d = d.median()
    try:
        stat, p_wx = wilcoxon(d)
    except Exception:
        p_wx = np.nan
    _, p_tt = ttest_rel(wdf['xwoba_pull'].dropna(), wdf['xwoba_oppo'].dropna())
    n_pos = (d > 0).sum()
    print(f"\n{label}  (n={len(d)} hitters with both modes)")
    print(f"  Mean   pull − oppo xwOBA: {mean_d:+.4f}")
    print(f"  Median pull − oppo xwOBA: {median_d:+.4f}")
    print(f"  Higher in pull mode: {n_pos}/{len(d)} ({n_pos/len(d):.1%})")
    print(f"  Wilcoxon signed-rank p  = {'<0.0001' if p_wx<0.0001 else f'{p_wx:.4f}'}")
    print(f"  Paired t-test p         = {'<0.0001' if p_tt<0.0001 else f'{p_tt:.4f}'}")

report_within(within_month, "Month-level within-hitter")
report_within(within_half,  "Half-season within-hitter")

# Cross-sectional baseline: always-oppo vs always-pull (full-season mode)
always_oppo = month_df[month_df['mode_season']=='oppo'].groupby('batter')['xwoba'].mean()
always_pull = month_df[month_df['mode_season']=='pull'].groupby('batter')['xwoba'].mean()
_, p_cross = mannwhitneyu(always_oppo, always_pull, alternative='two-sided')
print(f"\nCross-sectional (full-season mode):")
print(f"  Always-oppo mean xwOBA: {always_oppo.mean():.4f}  n={len(always_oppo)}")
print(f"  Always-pull mean xwOBA: {always_pull.mean():.4f}  n={len(always_pull)}")
print(f"  Δ (pull − oppo):  {always_pull.mean()-always_oppo.mean():+.4f}  "
      f"p={p_cross:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Month-level — population violin + within-hitter paired scatter
# ════════════════════════════════════════════════════════════════════════════
MODE_C = {'oppo': '#1f77b4', 'pull': '#d62728'}

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Panel A: Violin — all hitter-months, xwOBA by mode
ax = axes[0]
for i, mode in enumerate(['oppo', 'pull']):
    vals = month_df[month_df['mode'] == mode]['xwoba'].dropna().values
    parts = ax.violinplot([vals], positions=[i], widths=0.55,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(MODE_C[mode]); pc.set_alpha(0.5)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2.5)
    ax.scatter(np.random.uniform(i-0.18, i+0.18, len(vals)), vals,
               c=MODE_C[mode], alpha=0.15, s=10, edgecolors='none', zorder=2)
    ax.text(i, np.percentile(vals,97)+0.01,
            f"μ={vals.mean():.3f}\nn={len(vals)}", ha='center', fontsize=8)

oppo_m = month_df[month_df['mode']=='oppo']['xwoba'].dropna()
pull_m = month_df[month_df['mode']=='pull']['xwoba'].dropna()
_, p_pop = mannwhitneyu(oppo_m, pull_m, alternative='two-sided')
ax.set_xticks([0,1]); ax.set_xticklabels(['Oppo-mode\nmonth','Pull-mode\nmonth'], fontsize=11)
ax.set_ylabel("Mean xwOBA (in-play fastballs)", fontsize=10)
ax.set_title(f"All Hitter-Months\nxwOBA by Monthly Mode\n"
             f"Δ = {pull_m.mean()-oppo_m.mean():+.3f}  p={p_pop:.4f}", fontsize=10)

# Panel B: Within-hitter paired scatter (month-level switchers)
ax = axes[1]
if 'diff' in within_month.columns and len(within_month) > 0:
    wm = within_month.dropna(subset=['xwoba_oppo','xwoba_pull'])
else:
    wm = pd.DataFrame(columns=['xwoba_oppo','xwoba_pull','diff','mode_season'])
if len(wm) >= 3:
    ax.scatter(wm['xwoba_oppo'], wm['xwoba_pull'],
               c=[MODE_C[m] for m in wm['mode_season']], alpha=0.55, s=40, edgecolors='k',
               linewidths=0.3)
    lo = min(wm['xwoba_oppo'].min(), wm['xwoba_pull'].min()) - 0.02
    hi = max(wm['xwoba_oppo'].max(), wm['xwoba_pull'].max()) + 0.02
    ax.plot([lo,hi],[lo,hi],'k--',lw=1,alpha=0.5,label='y = x  (no difference)')
    n_above = (wm['diff'] > 0).sum()
    d = wm['diff'].dropna()
    try:
        _, p_w = wilcoxon(d)
        p_str = f"p={'<0.0001' if p_w<0.0001 else f'{p_w:.4f}'}"
    except Exception:
        p_str = "p=n/a"
    ax.set_xlabel("Mean xwOBA in Oppo-Mode Months", fontsize=10)
    ax.set_ylabel("Mean xwOBA in Pull-Mode Months", fontsize=10)
    ax.set_title(f"Within-Hitter: Pull-Mode vs Oppo-Mode xwOBA\n"
                 f"n={len(wm)} switcher hitters  ·  "
                 f"pull higher: {n_above}/{len(wm)} ({n_above/len(wm):.0%})  ·  {p_str}",
                 fontsize=10)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor=MODE_C[m],ms=8,label=f'{m}-season mode')
               for m in ['oppo','pull']]
    ax.legend(handles=handles+[Line2D([0],[0],ls='--',color='k',lw=1,label='y=x')], fontsize=8)
else:
    ax.text(0.5, 0.5, f'Too few switcher hitters\n(n={len(wm)})\nunder the monthly significance filter',
            transform=ax.transAxes, ha='center', va='center', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    d = pd.Series([], dtype=float)

# Panel C: Distribution of within-hitter differences
ax = axes[2]
if len(d) >= 3:
    ax.hist(d, bins=30, color='steelblue', alpha=0.72, edgecolor='white')
    ax.axvline(0,     color='black',  lw=1.5, ls='--', label='No difference')
    ax.axvline(d.mean(), color='crimson', lw=2.2, label=f'Mean = {d.mean():+.3f}')
    ax.axvline(d.median(), color='orange', lw=2, ls=':', label=f'Median = {d.median():+.3f}')
    ax.set_xlabel("xwOBA Difference: pull-mode months − oppo-mode months", fontsize=10)
    ax.set_ylabel("Number of Hitters", fontsize=10)
    ax.set_title(f"Distribution of Within-Hitter xwOBA Differences\n"
                 f"Month-level  ·  {p_str if 'p_str' in dir() else 'p=n/a'}", fontsize=10)
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5, 'Too few switchers for difference distribution',
            transform=ax.transAxes, ha='center', va='center', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Does Timing Mode Predict Within-Hitter xwOBA?  ·  Month Level\n"
             "Colour = full-season GMM mode  ·  In-play fastballs only",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'xwoba_mode_1_month.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("\nSaved → xwoba_mode_1_month.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Half-season — same three panels
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

ax = axes[0]
for i, mode in enumerate(['oppo','pull']):
    vals = half_df[half_df['mode']==mode]['xwoba'].dropna().values
    if len(vals) < 3:
        continue
    parts = ax.violinplot([vals], positions=[i], widths=0.55,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(MODE_C[mode]); pc.set_alpha(0.5)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2.5)
    ax.scatter(np.random.uniform(i-0.18, i+0.18, len(vals)), vals,
               c=MODE_C[mode], alpha=0.25, s=18, edgecolors='none', zorder=2)
    ax.text(i, np.percentile(vals,97)+0.005,
            f"μ={vals.mean():.3f}\nn={len(vals)}", ha='center', fontsize=9)

oppo_h = half_df[half_df['mode']=='oppo']['xwoba'].dropna()
pull_h = half_df[half_df['mode']=='pull']['xwoba'].dropna()
if len(oppo_h) > 1 and len(pull_h) > 1:
    _, p_pop_h = mannwhitneyu(oppo_h, pull_h, alternative='two-sided')
    delta_h = pull_h.mean()-oppo_h.mean()
else:
    p_pop_h, delta_h = np.nan, np.nan
ax.set_xticks([0,1]); ax.set_xticklabels(['Oppo-mode\nhalf','Pull-mode\nhalf'], fontsize=11)
ax.set_ylabel("Mean xwOBA (in-play fastballs)", fontsize=10)
ax.set_title(f"All Hitter-Halves\nxwOBA by Half-Season Mode\n"
             f"Δ = {delta_h:+.3f}  p={p_pop_h:.4f}", fontsize=10)

ax = axes[1]
if 'xwoba_oppo' in within_half.columns and 'xwoba_pull' in within_half.columns:
    wh = within_half.dropna(subset=['xwoba_oppo','xwoba_pull'])
else:
    wh = pd.DataFrame(columns=['xwoba_oppo','xwoba_pull','diff','mode_season'])
if len(wh) >= 5:
    ax.scatter(wh['xwoba_oppo'], wh['xwoba_pull'],
               c=[MODE_C.get(m,'gray') for m in wh['mode_season']],
               alpha=0.65, s=60, edgecolors='k', linewidths=0.4)
    lo2 = min(wh['xwoba_oppo'].min(), wh['xwoba_pull'].min()) - 0.02
    hi2 = max(wh['xwoba_oppo'].max(), wh['xwoba_pull'].max()) + 0.02
    ax.plot([lo2,hi2],[lo2,hi2],'k--',lw=1,alpha=0.5)
    dh = wh['diff'].dropna()
    n_above_h = (dh > 0).sum()
    try:
        _, p_wh = wilcoxon(dh)
        p_str_h = f"p={'<0.0001' if p_wh<0.0001 else f'{p_wh:.4f}'}"
    except Exception:
        p_str_h = "p=n/a"
    ax.set_xlabel("xwOBA — Oppo-Mode Half", fontsize=10)
    ax.set_ylabel("xwOBA — Pull-Mode Half", fontsize=10)
    ax.set_title(f"Within-Hitter: Pull vs Oppo Half-Season xwOBA\n"
                 f"n={len(wh)} switcher hitters  ·  "
                 f"pull higher: {n_above_h}/{len(wh)} ({n_above_h/len(wh):.0%})  ·  {p_str_h}",
                 fontsize=10)
    handles2 = [Line2D([0],[0],marker='o',color='w',markerfacecolor=MODE_C[m],ms=8,
                       label=f'{m}-season mode') for m in ['oppo','pull']]
    ax.legend(handles=handles2+[Line2D([0],[0],ls='--',color='k',lw=1,label='y=x')], fontsize=8)
else:
    ax.text(0.5,0.5,f'Only {len(wh)} paired hitters',transform=ax.transAxes,ha='center')
    dh, p_str_h = pd.Series([], dtype=float), 'n/a'

ax = axes[2]
if len(dh) >= 5:
    ax.hist(dh, bins=20, color='steelblue', alpha=0.72, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, ls='--', label='No difference')
    ax.axvline(dh.mean(), color='crimson', lw=2.2, label=f'Mean = {dh.mean():+.3f}')
    ax.axvline(dh.median(), color='orange', lw=2, ls=':', label=f'Median = {dh.median():+.3f}')
    ax.set_xlabel("xwOBA Difference: pull-mode half − oppo-mode half", fontsize=10)
    ax.set_ylabel("Number of Hitters", fontsize=10)
    ax.set_title(f"Distribution of Within-Hitter xwOBA Differences\n"
                 f"Half-season level  ·  {p_str_h}", fontsize=10)
    ax.legend(fontsize=9)
else:
    ax.text(0.5,0.5,'Insufficient paired data',transform=ax.transAxes,ha='center')

fig.suptitle("Does Timing Mode Predict Within-Hitter xwOBA?  ·  Half-Season Level\n"
             "Colour = full-season GMM mode  ·  In-play fastballs only",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'xwoba_mode_2_half.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → xwoba_mode_2_half.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: xwOBA trajectory — sample hitters who switch modes month-to-month
# ════════════════════════════════════════════════════════════════════════════
# For the clearest illustration, pick hitters with ≥3 months and ≥2 mode switches
switchers = (month_df.groupby('batter').filter(
    lambda g: g['mode'].nunique() > 1 and len(g) >= 4))

# Sort by peak range within the hitter to pick "clean" switchers
hitter_peak_std = switchers.groupby('batter')['peak'].std().sort_values(ascending=False)
sample_h = hitter_peak_std.head(40).index

fig, axes = plt.subplots(5, 8, figsize=(26, 16), sharey=False)
axes = axes.flatten()
MONTH_NAMES = {4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep'}

for i, batter in enumerate(sample_h[:40]):
    ax = axes[i]
    sub = month_df[month_df['batter']==batter].sort_values('month')
    ax.plot(sub['month'], sub['xwoba'], color='gray', lw=1, zorder=1)
    for _, row in sub.iterrows():
        ax.scatter(row['month'], row['xwoba'],
                   c=MODE_C[row['mode']], s=55, zorder=3, edgecolors='k', linewidths=0.4)
    ax.set_xticks(sub['month'].tolist())
    ax.set_xticklabels([MONTH_NAMES.get(m,'') for m in sub['month']], fontsize=6)
    ax.set_title(f"Hitter {i+1}", fontsize=7)
    ax.tick_params(axis='y', labelsize=6)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=MODE_C['oppo'],label='Oppo-mode month'),
                    Patch(color=MODE_C['pull'],label='Pull-mode month')],
           loc='lower right', fontsize=10, framealpha=0.9)
fig.suptitle("xwOBA Trajectory for 40 Mode-Switching Hitters\n"
             "Blue = oppo-mode month  ·  Red = pull-mode month  ·  "
             "Selected for maximum within-hitter mode variation",
             fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'xwoba_mode_3_trajectories.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved → xwoba_mode_3_trajectories.png")

print("\nDone.")
