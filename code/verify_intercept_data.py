"""
Verify Statcast intercept_y and intercept_x values match between our 2025 fastball
dataset and a fresh Statcast pull.

Motivation: Statcast added documentation saying:
  intercept_ball_minus_batter_pos_y_inches: Distance, in inches, between the
  intercept point of the bat/ball and the batter's center of mass, in the
  Y (mound-to-plate) direction.

This implies the variable should be centred on the batter's centre of mass
(median ~0). Our existing data has median = +24.73 in for intercept_y, suggesting
either (a) Statcast updated the data and we have a stale snapshot, or (b) the
documentation's "center of mass" wording isn't quite literal.

Approach: redownload a small recent slice via pybaseball.statcast, merge on
(game_pk, at_bat_number, pitch_number), and compare intercept_x / intercept_y
to our stored values.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
from pybaseball import statcast

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")

# Pick a short, mid-season slice to redownload
START_DT = "2025-06-15"
END_DT   = "2025-06-21"

print(f"Redownloading Statcast {START_DT} → {END_DT}…")
fresh = statcast(start_dt=START_DT, end_dt=END_DT, verbose=False)
print(f"Fresh rows: {len(fresh):,}")

KEY = ['game_pk', 'at_bat_number', 'pitch_number']
COMPARE = ['intercept_ball_minus_batter_pos_x_inches',
           'intercept_ball_minus_batter_pos_y_inches']
EXTRA   = ['pitch_type', 'description', 'release_speed', 'batter', 'stand']

# Keep only the columns we need from fresh
fresh = fresh[KEY + COMPARE + EXTRA].copy()
fresh.columns = [c if c not in COMPARE else f"{c}_fresh" for c in fresh.columns]

# Load our stored data and filter to the same date range
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
ours_chunks = []
for f in files:
    df = pd.read_csv(f, usecols=KEY + COMPARE + EXTRA + ['game_date'])
    df['game_date'] = pd.to_datetime(df['game_date'])
    mask = (df['game_date'] >= START_DT) & (df['game_date'] <= END_DT)
    if mask.any():
        ours_chunks.append(df[mask])
ours = pd.concat(ours_chunks, ignore_index=True)
print(f"Stored rows in range:  {len(ours):,}")

# Merge on the pitch-level key
merged = ours.merge(fresh[KEY + [f"{c}_fresh" for c in COMPARE]],
                    on=KEY, how='inner')
print(f"Merged on pitch key:   {len(merged):,}")

# Compare both intercept columns
def compare_col(col):
    fresh_col = f"{col}_fresh"
    sub = merged.dropna(subset=[col, fresh_col])
    diff = sub[fresh_col] - sub[col]
    abs_diff = diff.abs()
    print(f"\n── {col} ──")
    print(f"  n compared:       {len(sub):,}")
    print(f"  stored  median: {sub[col].median():+.3f}  mean: {sub[col].mean():+.3f}")
    print(f"  fresh   median: {sub[fresh_col].median():+.3f}  mean: {sub[fresh_col].mean():+.3f}")
    print(f"  diff (fresh - stored)  median: {diff.median():+.4f}  mean: {diff.mean():+.4f}")
    print(f"  |diff| max:       {abs_diff.max():.4f}")
    print(f"  |diff| p99:       {abs_diff.quantile(0.99):.4f}")
    print(f"  exact match:      {(abs_diff < 1e-6).sum():,}  ({(abs_diff < 1e-6).mean():.1%})")
    print(f"  within 0.01 in:   {(abs_diff < 0.01).sum():,}  ({(abs_diff < 0.01).mean():.1%})")
    print(f"  >1 in different:  {(abs_diff > 1).sum():,}  ({(abs_diff > 1).mean():.2%})")

for col in COMPARE:
    compare_col(col)

# Save diffs > 1 inch for inspection
out_csv = os.path.join(OUT_DIR, '_intercept_redownload_diffs.csv')
big_diffs_rows = []
for col in COMPARE:
    fresh_col = f"{col}_fresh"
    sub = merged.dropna(subset=[col, fresh_col]).copy()
    sub['diff'] = sub[fresh_col] - sub[col]
    sub['col'] = col
    big = sub[sub['diff'].abs() > 1][KEY + EXTRA + [col, fresh_col, 'diff', 'col']]
    big_diffs_rows.append(big)
big_df = pd.concat(big_diffs_rows, ignore_index=True) if big_diffs_rows else pd.DataFrame()
if len(big_df):
    big_df.to_csv(out_csv, index=False)
    print(f"\nLarge-diff (>1 in) rows saved to {out_csv}  (n={len(big_df)})")
else:
    print("\nNo rows with |diff| > 1 in.")

# Coverage check: do new pitches have intercept available where old ones don't (or vice versa)?
print("\n── Coverage check ──")
ours_avail = ours.dropna(subset=COMPARE)
fresh_avail = fresh.dropna(subset=[f"{c}_fresh" for c in COMPARE])
print(f"  Stored: intercept available in {len(ours_avail):,} / {len(ours):,} ({len(ours_avail)/len(ours):.1%})")
print(f"  Fresh:  intercept available in {len(fresh_avail):,} / {len(fresh):,} ({len(fresh_avail)/len(fresh):.1%})")

print("\nDone.")
