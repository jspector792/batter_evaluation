"""
add_barrel_distance.py
======================
Adds a `barrel_distance` column to every parquet file in DATA_DIR.

barrel_distance is a unified barrel-placement metric that puts swing-and-miss
events and contact events on the same continuous scale, in inches.

Geometry
--------
C = radius of bat + radius of ball
  = (2.61 / 2) + (2.90 / 2)
  = 1.305 + 1.450
  = 2.755 inches

For contact events, C * sin(angle) gives the vertical offset between bat
center and ball center at the moment of contact. We use the deviation from
the nearest edge of the MLB barrel launch-angle zone [8°, 32°] so that the
penalty is zero at both boundaries and grows continuously away from them.

barrel_distance definitions
---------------------------
1. Swing and miss:
      barrel_distance = miss_distance + C
   The ball and bat passed each other entirely. Adding C puts the miss on the
   same scale as contact events — a true miss starts where contact ends.

2. Contact, launch angle in barrel zone [8°, 32°]:
      barrel_distance = 0
   Perfect barrel placement by the MLB definition. No penalty.

3. Contact, launch angle below barrel zone (la < 8°):
      barrel_distance = |C * sin(la - 8)|
   Deviation from the lower barrel boundary. At la=8 this is 0 (continuous);
   at la=0 it is C * sin(8°) ≈ 0.38 inches; at la=-90 it is C (maximum).

4. Contact, launch angle above barrel zone (la > 32°):
      barrel_distance = |C * sin(la - 32)|
   Deviation from the upper barrel boundary. At la=32 this is 0 (continuous);
   at la=60 it is C * sin(28°) ≈ 1.29 inches; at la=90 it is C (maximum).

5. All other rows (non-swing pitches, called strikes, balls, etc.):
      barrel_distance = NaN

Usage
-----
  python add_barrel_distance.py [--dry-run] [--verify]

  --dry-run   Print statistics for each file but do not write changes.
  --verify    After writing, reload each file and print a sample + summary.

Outputs
-------
  Parquet files are updated in-place (original columns preserved, 
  barrel_distance appended). A summary CSV is written to OUT_DIR.

Requires: pandas, numpy, pyarrow
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Physical constants ─────────────────────────────────────────────────────────
BAT_DIAMETER  = 2.61   # inches, maximum barrel diameter
BALL_DIAMETER = 2.90   # inches
C = (BAT_DIAMETER / 2) + (BALL_DIAMETER / 2)   # sum of radii = 2.755 inches

# ── MLB barrel launch angle zone ───────────────────────────────────────────────
LA_LOW  =  8.0   # degrees
LA_HIGH = 32.0   # degrees

# ── Event sets ─────────────────────────────────────────────────────────────────
MISS    = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
FOUL    = {'foul', 'foul_tip', 'foul_bunt'}
CONTACT = IN_PLAY | FOUL


def compute_barrel_distance(df: pd.DataFrame) -> pd.Series:
    """
    Compute barrel_distance for every row in df.

    Parameters
    ----------
    df : DataFrame with columns:
           description   — pitch outcome string
           miss_distance — observed miss distance in inches (NaN on contact)
           launch_angle  — launch angle in degrees (NaN on misses)

    Returns
    -------
    pd.Series of float, same index as df, NaN for non-swing rows.
    """
    bd = pd.Series(np.nan, index=df.index, dtype=float)

    # ── Masks ──────────────────────────────────────────────────────────────────
    is_miss    = df['description'].isin(MISS)
    is_contact = df['description'].isin(CONTACT)

    la_rad = np.deg2rad(df['launch_angle'])   # NaN where launch_angle is NaN

    # ── 1. Swing and miss ──────────────────────────────────────────────────────
    # miss_distance + C shifts the miss onto the contact scale.
    # A miss_distance of 0 (just barely missed) maps to C, which equals the
    # penalty for contact right at the barrel zone boundary.
    valid_miss = is_miss & df['miss_distance'].notna()
    bd.loc[valid_miss] = df.loc[valid_miss, 'miss_distance'] + C

    # ── 2. Contact, inside barrel zone ────────────────────────────────────────
    in_zone = is_contact & df['launch_angle'].between(LA_LOW, LA_HIGH)
    bd.loc[in_zone] = 0.0

    # ── 3. Contact, below barrel zone (la < 8°) ───────────────────────────────
    # Penalty = |C * sin(la - 8)|
    # At la=8:  sin(0)   = 0      → continuous with in-zone boundary
    # At la=0:  sin(-8°) ≈ -0.139 → |C * -0.139| ≈ 0.38 inches
    # At la=-90: sin(-98°) ≈ -0.990 → approaches C
    below_zone = is_contact & df['launch_angle'].notna() & (df['launch_angle'] < LA_LOW)
    bd.loc[below_zone] = np.abs(
        C * np.sin(la_rad.loc[below_zone] - np.deg2rad(LA_LOW))
    )

    # ── 4. Contact, above barrel zone (la > 32°) ──────────────────────────────
    # Penalty = |C * sin(la - 32)|
    # At la=32: sin(0)   = 0      → continuous with in-zone boundary
    # At la=60: sin(28°) ≈ 0.469  → C * 0.469 ≈ 1.29 inches
    # At la=90: sin(58°) ≈ 0.848  → C * 0.848 ≈ 2.34 inches
    above_zone = is_contact & df['launch_angle'].notna() & (df['launch_angle'] > LA_HIGH)
    bd.loc[above_zone] = np.abs(
        C * np.sin(la_rad.loc[above_zone] - np.deg2rad(LA_HIGH))
    )

    return bd


def summarise(df: pd.DataFrame, label: str) -> dict:
    """Print and return summary statistics for barrel_distance in df."""
    bd = df['barrel_distance']

    is_miss    = df['description'].isin(MISS)
    is_contact = df['description'].isin(CONTACT)
    in_zone    = is_contact & df['launch_angle'].between(LA_LOW, LA_HIGH)
    below      = is_contact & df['launch_angle'].notna() & (df['launch_angle'] < LA_LOW)
    above      = is_contact & df['launch_angle'].notna() & (df['launch_angle'] > LA_HIGH)

    summary = {
        'file':               label,
        'n_rows':             len(df),
        'n_miss':             is_miss.sum(),
        'n_contact_inzone':   in_zone.sum(),
        'n_contact_below':    below.sum(),
        'n_contact_above':    above.sum(),
        'n_nan':              bd.isna().sum(),
        'bd_mean':            bd.mean(),
        'bd_median':          bd.median(),
        'bd_min':             bd.min(),
        'bd_max':             bd.max(),
        'miss_bd_mean':       bd[is_miss].mean(),
        'contact_bd_mean':    bd[is_contact].mean(),
    }

    print(f'\n  [{label}]')
    print(f'    rows={summary["n_rows"]:,}  '
          f'misses={summary["n_miss"]:,}  '
          f'contact_inzone={summary["n_contact_inzone"]:,}  '
          f'contact_below={summary["n_contact_below"]:,}  '
          f'contact_above={summary["n_contact_above"]:,}')
    print(f'    barrel_distance: '
          f'mean={summary["bd_mean"]:.3f}  '
          f'median={summary["bd_median"]:.3f}  '
          f'range=[{summary["bd_min"]:.3f}, {summary["bd_max"]:.3f}]')
    print(f'    mean by event:  '
          f'miss={summary["miss_bd_mean"]:.3f}  '
          f'contact={summary["contact_bd_mean"]:.3f}')

    return summary


def process_file(path: str, dry_run: bool, verify: bool) -> dict:
    """Load one parquet file, compute barrel_distance, write back, return summary."""
    label = os.path.basename(path)
    df = pd.read_parquet(path)

    # Sanity checks
    for col in ['description', 'miss_distance', 'launch_angle']:
        if col not in df.columns:
            print(f'  WARNING: {label} missing column {col!r} — skipping.')
            return {'file': label, 'skipped': True}

    df['barrel_distance'] = compute_barrel_distance(df)

    summary = summarise(df, label)

    if not dry_run:
        df.to_parquet(path, index=False)
        print(f'    Written: {path}')

        if verify:
            df2 = pd.read_parquet(path)
            assert 'barrel_distance' in df2.columns, 'barrel_distance missing after write'
            print(f'    Verified: barrel_distance present in reloaded file '
                  f'(non-null: {df2["barrel_distance"].notna().sum():,})')
    else:
        print(f'    [dry-run] no file written')

    return summary


def main():
    parser = argparse.ArgumentParser(description='Add barrel_distance to parquet files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and print stats but do not write files')
    parser.add_argument('--verify', action='store_true',
                        help='Reload each file after writing to confirm the column')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

    print(f'Found {len(files)} parquet file(s) in {DATA_DIR}')
    print(f'C = {C:.4f} inches  '
          f'(bat radius {BAT_DIAMETER/2:.4f} + ball radius {BALL_DIAMETER/2:.4f})')
    print(f'Barrel zone: [{LA_LOW}°, {LA_HIGH}°]')
    print(f'Boundary penalties:  '
          f'la=0° → {abs(C * np.sin(np.deg2rad(0 - LA_LOW))):.3f} in  '
          f'la=60° → {abs(C * np.sin(np.deg2rad(60 - LA_HIGH))):.3f} in  '
          f'la=90° → {abs(C * np.sin(np.deg2rad(90 - LA_HIGH))):.3f} in')
    if args.dry_run:
        print('\n[DRY RUN — no files will be modified]\n')

    summaries = []
    for path in files:
        s = process_file(path, dry_run=args.dry_run, verify=args.verify)
        summaries.append(s)

    # Write summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(OUT_DIR, 'barrel_distance_summary.csv')
    if not args.dry_run:
        summary_df.to_csv(summary_path, index=False)
        print(f'\nSummary written to {summary_path}')

    # Aggregate totals across all files
    numeric_cols = ['n_rows', 'n_miss', 'n_contact_inzone',
                    'n_contact_below', 'n_contact_above', 'n_nan']
    totals = summary_df[numeric_cols].sum()
    print('\nAggregate totals across all files:')
    for col in numeric_cols:
        print(f'  {col}: {int(totals[col]):,}')

    print('\nDone.')


if __name__ == '__main__':
    main()