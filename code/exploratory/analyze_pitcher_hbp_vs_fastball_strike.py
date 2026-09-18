"""
For every 2025 pitcher, compare:
  - HBP velo: mean release_speed across ANY pitch type when description = 'hit_by_pitch'
  - FB-strike velo: mean release_speed of their primary fastball
    (most-thrown of FF/SI/FC/FT) when that fastball was thrown for a strike
    (called_strike, swinging_strike[_blocked], foul, foul_tip).

Output CSV (`out/pitcher_hbp_vs_fastball_strike.csv`):
    pitcher, name, primary_fb,
    n_strike, mean_strike, se_strike,
    n_hbp,    mean_hbp,    se_hbp,
    diff (strike − hbp), se_diff,
    t_stat, p_value, significant (p < 0.05).
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from pybaseball import playerid_reverse_lookup

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "misc")

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
STRIKE_DESC = {"called_strike", "swinging_strike", "swinging_strike_blocked",
               "foul", "foul_tip"}
MIN_HBP   = 3   # pitcher must have ≥3 HBPs to be considered
MIN_STRIKE = 30 # and ≥30 strikes on their primary FB

# ── Load ──────────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
df = pd.concat([pd.read_parquet(f, columns=['pitcher', 'pitch_type',
                                              'description', 'release_speed'])
                for f in files], ignore_index=True)
print(f"Total pitches: {len(df):,}")
df = df.dropna(subset=['pitcher', 'release_speed', 'description'])
print(f"After dropna on essentials: {len(df):,}")

# ── HBP table (any pitch type) ───────────────────────────────────────────────
hbp = df[df['description'] == 'hit_by_pitch'].copy()
print(f"HBPs in dataset: {len(hbp):,}")
hbp_agg = (hbp.groupby('pitcher')
              .agg(n_hbp=('release_speed', 'count'),
                   mean_hbp=('release_speed', 'mean'),
                   std_hbp=('release_speed', 'std')))
hbp_agg = hbp_agg[hbp_agg['n_hbp'] >= MIN_HBP].copy()
hbp_agg['se_hbp'] = hbp_agg['std_hbp'] / np.sqrt(hbp_agg['n_hbp'])
print(f"Pitchers with ≥{MIN_HBP} HBPs: {len(hbp_agg)}")

# ── Identify each pitcher's primary fastball ─────────────────────────────────
fb = df[df['pitch_type'].isin(FASTBALL_TYPES)].copy()
primary_fb = (fb.groupby(['pitcher', 'pitch_type']).size()
                .reset_index(name='n')
                .sort_values(['pitcher', 'n'], ascending=[True, False])
                .drop_duplicates('pitcher', keep='first')
                .set_index('pitcher')['pitch_type']
                .rename('primary_fb'))
print(f"Pitchers throwing any fastball: {len(primary_fb)}")

# ── Strike velo: primary FB only, when thrown for a strike ───────────────────
fb = fb.join(primary_fb, on='pitcher')
fb_primary = fb[fb['pitch_type'] == fb['primary_fb']]
fb_strike = fb_primary[fb_primary['description'].isin(STRIKE_DESC)].copy()
print(f"Primary-FB strikes: {len(fb_strike):,}")

strike_agg = (fb_strike.groupby('pitcher')
                       .agg(n_strike=('release_speed', 'count'),
                            mean_strike=('release_speed', 'mean'),
                            std_strike=('release_speed', 'std')))
strike_agg = strike_agg[strike_agg['n_strike'] >= MIN_STRIKE].copy()
strike_agg['se_strike'] = strike_agg['std_strike'] / np.sqrt(strike_agg['n_strike'])

# ── Combine ──────────────────────────────────────────────────────────────────
combined = hbp_agg.join(strike_agg, how='inner').join(primary_fb, how='left')
combined['diff'] = combined['mean_strike'] - combined['mean_hbp']
combined['se_diff'] = np.sqrt(combined['se_strike']**2 + combined['se_hbp']**2)
combined['t_stat']  = combined['diff'] / combined['se_diff']
print(f"\nPitchers qualifying (≥{MIN_HBP} HBPs AND ≥{MIN_STRIKE} primary-FB strikes): {len(combined)}")

# ── Per-pitcher Welch t-test ─────────────────────────────────────────────────
print("Running per-pitcher t-tests…")
def welch_p(pid):
    s = fb_strike.loc[fb_strike['pitcher'] == pid, 'release_speed'].values
    h = hbp.loc[hbp['pitcher'] == pid, 'release_speed'].values
    if len(s) < 2 or len(h) < 2:
        return np.nan
    return float(ttest_ind(s, h, equal_var=False).pvalue)
combined['p_value'] = [welch_p(pid) for pid in combined.index]
combined['significant'] = combined['p_value'] < 0.05
print(f"  Significant (p < 0.05): {combined['significant'].sum()}  "
      f"({combined['significant'].mean():.1%})")
print(f"  Diff > 0 (strike faster than HBP): {(combined['diff'] > 0).sum()}  "
      f"({(combined['diff'] > 0).mean():.1%})")

# ── Name lookup ──────────────────────────────────────────────────────────────
try:
    ids = combined.index.tolist()
    name_df = playerid_reverse_lookup(ids, key_type='mlbam')[
        ['key_mlbam', 'name_first', 'name_last']
    ].set_index('key_mlbam')
    name_df['name'] = (name_df['name_last'].str.title() + ', ' +
                       name_df['name_first'].str.title())
    combined['name'] = name_df['name']
except Exception as exc:
    print(f"  (name lookup failed: {exc!r})")
    combined['name'] = ''

# ── Save ─────────────────────────────────────────────────────────────────────
out = combined.reset_index()
cols = ['pitcher', 'name', 'primary_fb',
        'n_strike', 'mean_strike', 'se_strike',
        'n_hbp',    'mean_hbp',    'se_hbp',
        'diff', 'se_diff', 't_stat', 'p_value', 'significant']
out = out[cols].sort_values('diff', ascending=False)
out_path = os.path.join(OUT_DIR, 'pitcher_hbp_vs_fastball_strike.csv')
out.to_csv(out_path, index=False, float_format='%.4f')
print(f"\nSaved {len(out):,} pitchers → {out_path}")

print("\nPopulation summary:")
print(f"  Mean diff (strike − hbp):   {out['diff'].mean():+.3f} mph")
print(f"  Median diff:                {out['diff'].median():+.3f} mph")
print(f"  Significantly different:    {out['significant'].sum()} / {len(out)} "
      f"({out['significant'].mean():.1%})")
print("\nTop 10 by largest strike-vs-HBP velo gap:")
print(out.head(10)[['name','primary_fb','n_strike','mean_strike',
                    'n_hbp','mean_hbp','diff','se_diff','p_value']].to_string(index=False))
print("\nBottom 10 (HBP faster than primary-FB strikes):")
print(out.tail(10)[['name','primary_fb','n_strike','mean_strike',
                    'n_hbp','mean_hbp','diff','se_diff','p_value']].to_string(index=False))
print("\nDone.")
