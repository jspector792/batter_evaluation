"""
For every 2025 hitter, compute mean and SD bat speed on swings that made
contact vs swings that missed, and test whether the difference is significant.

Output CSV (`out/hitter_bat_speed_contact_vs_miss.csv`) has, per batter:
    batter, name,
    n_contact, mean_contact, se_contact,
    n_miss,    mean_miss,    se_miss,
    diff (contact − miss), se_diff,
    t_stat, p_value, significant (p < 0.05).
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from pybaseball import playerid_reverse_lookup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_utils import CONTACT, MISS

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "misc")
MIN_PER_GROUP = 10  # require ≥10 contacts AND ≥10 misses

# ── Load ──────────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
df = pd.concat([pd.read_parquet(f, columns=['batter', 'description', 'bat_speed'])
                for f in files], ignore_index=True)
print(f"Total pitches: {len(df):,}")

df = df.dropna(subset=['batter', 'bat_speed'])
df['group'] = df['description'].map(
    lambda d: 'contact' if d in CONTACT else ('miss' if d in MISS else None))
df = df.dropna(subset=['group'])
print(f"After filtering to swings with bat_speed: {len(df):,}")
print(f"  contact: {(df['group']=='contact').sum():,}")
print(f"  miss:    {(df['group']=='miss').sum():,}")

# ── Aggregate per batter × group ─────────────────────────────────────────────
agg = (df.groupby(['batter', 'group'])
         .agg(n=('bat_speed', 'count'),
              mean=('bat_speed', 'mean'),
              std=('bat_speed', 'std'))
         .unstack('group'))
agg.columns = [f"{a}_{b}" for a, b in agg.columns]

req = ['n_contact', 'n_miss', 'mean_contact', 'mean_miss', 'std_contact', 'std_miss']
agg = agg.dropna(subset=req)
agg = agg[(agg['n_contact'] >= MIN_PER_GROUP) & (agg['n_miss'] >= MIN_PER_GROUP)].copy()
print(f"\nHitters qualifying (≥{MIN_PER_GROUP} each): {len(agg)}")

# ── Welch test + standard errors ─────────────────────────────────────────────
agg['se_contact'] = agg['std_contact'] / np.sqrt(agg['n_contact'])
agg['se_miss']    = agg['std_miss']    / np.sqrt(agg['n_miss'])
agg['diff']       = agg['mean_contact'] - agg['mean_miss']
agg['se_diff']    = np.sqrt(agg['se_contact']**2 + agg['se_miss']**2)
agg['t_stat']     = agg['diff'] / agg['se_diff']

# Welch t-test for each hitter (vectorised via individual scipy calls)
print("Running per-hitter t-tests…")
def welch_p(row):
    sub = df[df['batter'] == row.name]
    c = sub.loc[sub['group'] == 'contact', 'bat_speed'].values
    m = sub.loc[sub['group'] == 'miss',    'bat_speed'].values
    if len(c) < 2 or len(m) < 2:
        return np.nan
    return float(ttest_ind(c, m, equal_var=False).pvalue)

agg['p_value'] = [welch_p(row) for _, row in agg.iterrows()]
agg['significant'] = agg['p_value'] < 0.05
print(f"  Significant (p < 0.05): {agg['significant'].sum()}  "
      f"({agg['significant'].mean():.1%})")
print(f"  Diff > 0  (contact faster): {(agg['diff'] > 0).sum()}  "
      f"({(agg['diff'] > 0).mean():.1%})")

# ── Name lookup ──────────────────────────────────────────────────────────────
try:
    ids = agg.index.tolist()
    name_df = playerid_reverse_lookup(ids, key_type='mlbam')[
        ['key_mlbam', 'name_first', 'name_last']
    ].set_index('key_mlbam')
    name_df['name'] = (name_df['name_last'].str.title() + ', ' +
                       name_df['name_first'].str.title())
    agg['name'] = name_df['name']
except Exception as exc:
    print(f"  (name lookup failed: {exc!r})")
    agg['name'] = ''

# ── Save ─────────────────────────────────────────────────────────────────────
out = agg.reset_index()
cols = ['batter', 'name',
        'n_contact', 'mean_contact', 'se_contact',
        'n_miss',    'mean_miss',    'se_miss',
        'diff', 'se_diff', 't_stat', 'p_value', 'significant']
out = out[cols].sort_values('diff', ascending=False)
out_path = os.path.join(OUT_DIR, 'hitter_bat_speed_contact_vs_miss.csv')
out.to_csv(out_path, index=False, float_format='%.4f')
print(f"\nSaved {len(out):,} hitters → {out_path}")

print("\nPopulation summary (all qualifying hitters):")
print(f"  Mean diff (contact − miss): {out['diff'].mean():+.3f} mph")
print(f"  Median diff:                {out['diff'].median():+.3f} mph")
print(f"  Significantly different:    {out['significant'].sum()} / {len(out)} "
      f"({out['significant'].mean():.1%})")
print("\nTop 10 by largest contact-vs-miss gap:")
print(out.head(10)[['name','n_contact','mean_contact','n_miss','mean_miss',
                    'diff','se_diff','p_value']].to_string(index=False))
print("\nBottom 10 (miss faster than contact, if any):")
print(out.tail(10)[['name','n_contact','mean_contact','n_miss','mean_miss',
                    'diff','se_diff','p_value']].to_string(index=False))
print("\nDone.")
