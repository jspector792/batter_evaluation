"""
Add columns to the existing pitcher_hbp_vs_fastball_strike.csv that restrict
the HBP comparison to fastball HBPs only (FF / SI / FC / FT, any type).

New columns appended:
    n_hbp_fb, mean_hbp_fb, se_hbp_fb,
    diff_fb (mean_strike − mean_hbp_fb), se_diff_fb,
    t_stat_fb, p_value_fb, significant_fb
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "misc")
CSV_PATH = os.path.join(OUT_DIR, 'pitcher_hbp_vs_fastball_strike.csv')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
STRIKE_DESC = {"called_strike", "swinging_strike", "swinging_strike_blocked",
               "foul", "foul_tip"}
MIN_HBP_FB = 2   # fastball-only HBPs are rarer; relax threshold

# ── Load existing CSV ────────────────────────────────────────────────────────
existing = pd.read_csv(CSV_PATH)
print(f"Existing rows: {len(existing)}")

# ── Load pitch data ──────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
df = pd.concat([pd.read_parquet(f, columns=['pitcher', 'pitch_type',
                                              'description', 'release_speed'])
                for f in files], ignore_index=True)
df = df.dropna(subset=['pitcher', 'release_speed', 'description'])

# ── HBP table restricted to fastballs ────────────────────────────────────────
hbp_fb = df[(df['description'] == 'hit_by_pitch') &
            (df['pitch_type'].isin(FASTBALL_TYPES))].copy()
print(f"Fastball HBPs in 2025: {len(hbp_fb):,}")

hbp_agg = (hbp_fb.groupby('pitcher')
                 .agg(n_hbp_fb=('release_speed', 'count'),
                      mean_hbp_fb=('release_speed', 'mean'),
                      std_hbp_fb=('release_speed', 'std')))
hbp_agg['se_hbp_fb'] = hbp_agg['std_hbp_fb'] / np.sqrt(hbp_agg['n_hbp_fb'])
print(f"Pitchers with ≥1 fastball HBP: {len(hbp_agg)}")

# ── Strike table (primary FB, same as the original script) ───────────────────
# We need primary_fb info from the existing CSV (already computed)
primary_fb = existing.set_index('pitcher')['primary_fb']
strike_pool = df[df['pitch_type'].isin(FASTBALL_TYPES) &
                  df['description'].isin(STRIKE_DESC)]
strike_pool = strike_pool.join(primary_fb, on='pitcher')
strike_primary = strike_pool[strike_pool['pitch_type'] == strike_pool['primary_fb']]
# (Re-)compute per-pitcher strike stats so the diff is internally consistent
strike_agg = (strike_primary.groupby('pitcher')
                            .agg(_n_strike=('release_speed', 'count'),
                                 _mean_strike=('release_speed', 'mean'),
                                 _std_strike=('release_speed', 'std')))
strike_agg['_se_strike'] = strike_agg['_std_strike'] / np.sqrt(strike_agg['_n_strike'])

# ── Merge and compute new difference + Welch test ────────────────────────────
joined = hbp_agg.join(strike_agg, how='inner')
joined = joined[joined['n_hbp_fb'] >= MIN_HBP_FB]
joined['diff_fb'] = joined['_mean_strike'] - joined['mean_hbp_fb']
joined['se_diff_fb'] = np.sqrt(joined['_se_strike']**2 + joined['se_hbp_fb']**2)
joined['t_stat_fb']  = joined['diff_fb'] / joined['se_diff_fb']

def welch_p(pid):
    s = strike_primary.loc[strike_primary['pitcher'] == pid, 'release_speed'].values
    h = hbp_fb.loc[hbp_fb['pitcher'] == pid, 'release_speed'].values
    if len(s) < 2 or len(h) < 2:
        return np.nan
    return float(ttest_ind(s, h, equal_var=False).pvalue)

joined['p_value_fb'] = [welch_p(pid) for pid in joined.index]
joined['significant_fb'] = joined['p_value_fb'] < 0.05

# ── Merge back into existing CSV ─────────────────────────────────────────────
new_cols = ['n_hbp_fb', 'mean_hbp_fb', 'se_hbp_fb',
            'diff_fb', 'se_diff_fb', 't_stat_fb', 'p_value_fb', 'significant_fb']
merged = existing.merge(joined[new_cols].reset_index(), on='pitcher', how='left')

# Diagnostics
n_have = merged['n_hbp_fb'].notna().sum()
print(f"\nPitchers with fastball-HBP data: {n_have} / {len(merged)}")
sub = merged.dropna(subset=['diff_fb'])
print(f"  Mean diff_fb (strike − fb-HBP): {sub['diff_fb'].mean():+.3f} mph")
print(f"  Median diff_fb:                 {sub['diff_fb'].median():+.3f} mph")
print(f"  Significant_fb (p<0.05):        {int(sub['significant_fb'].sum())} / {len(sub)} "
      f"({sub['significant_fb'].mean():.1%})")
print(f"  Compared to all-pitch diff: mean was {existing['diff'].mean():+.3f} mph "
      f"(now {sub['diff_fb'].mean():+.3f} mph)")

# ── Save ─────────────────────────────────────────────────────────────────────
merged.to_csv(CSV_PATH, index=False, float_format='%.4f')
print(f"\nSaved {len(merged)} rows → {CSV_PATH}")

print("\nTop 10 by fastball-only diff (strike FB − HBP FB):")
top = merged.dropna(subset=['diff_fb']).sort_values('diff_fb', ascending=False).head(10)
print(top[['name','primary_fb','n_strike','mean_strike','n_hbp_fb',
           'mean_hbp_fb','diff_fb','se_diff_fb','p_value_fb']].to_string(index=False))
print("\nBottom 10:")
bot = merged.dropna(subset=['diff_fb']).sort_values('diff_fb').head(10)
print(bot[['name','primary_fb','n_strike','mean_strike','n_hbp_fb',
           'mean_hbp_fb','diff_fb','se_diff_fb','p_value_fb']].to_string(index=False))

print("\nDone.")
