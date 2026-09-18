"""
Test script to fetch pitch-level swing timing and miss distance data
from Baseball Savant's statcast_search endpoint directly.

Findings from JS source:
  - miss_distance_inches     -> pitch-level miss distance (swings & misses only)
  - delta_batball_tiedup_neg_x2    -> Tied Up / Flail (inches, horizontal)
  - delta_batball_late_neg_y2_msec -> Early / Late (milliseconds)
  - ball_pos_above_plane           -> Over / Under (inches, vertical)

These are NOT reliably in the pybaseball statcast() CSV export.
This script hits the statcast_search CSV endpoint directly and requests them.
"""

import requests
import pandas as pd
import io

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE = "2025-04-01"
END_DATE   = "2025-04-07"   # keep narrow for the test; widen once confirmed working
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# These are the extra fields we want on top of the standard statcast columns.
# Savant's CSV endpoint accepts an `extra_stats` param (comma-separated) for
# fields that aren't in the default export.
EXTRA_FIELDS = [
    "miss_distance_inches",
    "delta_batball_tiedup_neg_x2",
    "delta_batball_late_neg_y2_msec",
    "ball_pos_above_plane",
]

PARAMS = {
    "all":             "true",
    "hfPT":            "",          # pitch type filter (blank = all)
    "hfAB":            "",
    "hfGT":            "R|",        # regular season
    "hfPR":            "",
    "hfZ":             "",
    "hfStadium":       "",
    "hfBBL":           "",
    "hfNewZones":      "",
    "hfPull":          "",
    "hfC":             "",
    "hfSea":           "2025|",
    "hfSit":           "",
    "player_type":     "batter",
    "hfOuts":          "",
    "hfOpponent":      "",
    "pitcher_throws":  "",
    "batter_stands":   "",
    "hfSA":            "",
    "game_date_gt":    START_DATE,
    "game_date_lt":    END_DATE,
    "hfMo":            "",
    "hfTeam":          "",
    "home_road":       "",
    "hfRO":            "",
    "position":        "",
    "hfInfield":       "",
    "hfOutfield":      "",
    "hfInn":           "",
    "hfBBT":           "",
    "hfFlag":          "",
    "metric_1":        "",
    "group_by":        "name",
    "min_pitches":     "0",
    "min_results":     "0",
    "type":            "details",
    "extra_stats":     ",".join(EXTRA_FIELDS),   # <-- the key addition
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://baseballsavant.mlb.com/statcast_search",
}


def fetch_statcast(start: str, end: str) -> pd.DataFrame:
    params = {**PARAMS, "game_date_gt": start, "game_date_lt": end}
    print(f"Fetching {start} → {end} ...")
    r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()

    # Savant sometimes returns an HTML error page instead of CSV
    if not r.text.strip().startswith("pitch_type") and "<!DOCTYPE" in r.text[:200]:
        raise ValueError("Got HTML instead of CSV — check params or date range.")

    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    return df


def report(df: pd.DataFrame) -> None:
    print(f"\nRows fetched: {len(df):,}")
    print(f"Columns: {len(df.columns)}\n")

    target_cols = EXTRA_FIELDS + [
        # alternate names Savant sometimes uses
        "miss_distance",
        "timing_x_tiedupflail",
        "timing_y_earlylate",
        "timing_z_overunder",
    ]

    print("─── Target column presence & non-null counts ───")
    found_any = False
    for col in target_cols:
        if col in df.columns:
            n_nonnull = df[col].notna().sum()
            pct = 100 * n_nonnull / len(df) if len(df) else 0
            print(f"  ✓ {col:<45} {n_nonnull:>6,} non-null  ({pct:.1f}%)")
            found_any = True
        else:
            print(f"  ✗ {col:<45} NOT IN RESPONSE")

    if not found_any:
        print("\n  None of the target columns came back.")
        print("  Columns containing 'miss' or 'timing' or 'delta' or 'bat_ball':")
        hits = [c for c in df.columns if any(k in c.lower() for k in
                                              ("miss", "timing", "delta", "bat_ball", "plane"))]
        for c in hits:
            print(f"    {c}")
        if not hits:
            print("    (none)")

    print("\n─── All column names ───")
    for c in sorted(df.columns):
        print(f"  {c}")


if __name__ == "__main__":
    df = fetch_statcast(START_DATE, END_DATE)
    report(df)

    # Save so you can inspect manually
    out = f"swing_timing_raw_{START_DATE}_to_{END_DATE}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")