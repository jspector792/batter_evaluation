"""
Robust downloader for Statcast pitches, for any season.

Season-parameterised generalisation of download_all_pitches_2025.py, which is
kept for provenance of the original 2025 pull. Behaviour is identical; the only
change is that the season (and therefore the output directory and the month
chunks) comes from --season instead of being hard-coded.

- Downloads one day at a time via direct endpoint calls (not pybaseball)
  so that miss_distance_inches is reliably included in the response.
- Retries transient SSL/network failures
- Saves monthly parquet files to data/all_pitches_{season}/
- Keeps only selected columns
- Skips months already on disk, so an interrupted run can be resumed

Why not pybaseball statcast()?
  pybaseball hits the same /statcast_search/csv endpoint but never passes
  the extra_stats parameter, so miss_distance never appears. We replicate
  pybaseball's exact pre-encoded URL and simply append extra_stats at the end.

Why not requests.get(params=dict)?
  requests re-encodes the params dict, turning the already-encoded %7C pipes
  in hfGT (R%7CPO%7CS%7C=) into double-encoded %257C, which breaks Savant's
  filter and causes it to return empty results. We must pass the full URL
  string directly with no params argument.

Usage
-----
  python download_all_pitches.py --season 2025
  python download_all_pitches.py --season 2026
  python download_all_pitches.py --season 2026 --start 2026-07-01 --end 2026-07-31

Days with no games return an empty response and are skipped, so the default
March-through-November window is safe for any season, including one still in
progress (the window is clipped at today's date).
"""

import argparse
import io
import os
import time
import warnings
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from requests.exceptions import SSLError, ConnectionError, Timeout, HTTPError

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Column names as they appear in the CSV response.
# miss_distance_inches is renamed to miss_distance at save time.
KEEP = [
    'game_date', 'game_pk', 'at_bat_number', 'pitch_number',
    # game_type is NOT in download_all_pitches_2025.py's KEEP list. It is kept
    # here because the hfGT filter admits spring training (S) alongside regular
    # (R) and postseason (PO), and the season window opens in mid-March, so
    # spring games are genuinely present in the output. Without this column
    # downstream code has no way to exclude them. Values: S, R, PO (plus F/D/L
    # for the postseason rounds in some seasons).
    'game_type',
    'pitch_type', 'pitch_name',
    'release_speed', 'release_spin_rate', 'spin_axis',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone',
    'description', 'batter', 'pitcher', 'stand', 'p_throws',
    'delta_run_exp', 'estimated_woba_using_speedangle',
    'launch_speed', 'launch_angle', 'launch_speed_angle',
    'intercept_ball_minus_batter_pos_x_inches',
    'intercept_ball_minus_batter_pos_y_inches',
    'attack_angle', 'attack_direction',
    'swing_path_tilt', 'bat_speed', 'swing_length',
    'events',
    'miss_distance',
]

# Season window. March 15 matches download_all_pitches_2025.py's first chunk,
# so --season 2025 reproduces the existing 2025 dataset rather than silently
# extending it. Dates with no games simply come back empty, so the window can
# be wider than any individual season's schedule.
SEASON_START_MONTH_DAY = (3, 15)
SEASON_END_MONTH_DAY   = (11, 15)

MAX_RETRIES = 5

# HTTP statuses worth retrying. Savant intermittently returns 502/504 under
# load; download_all_pitches_2025.py treated those as permanent because
# raise_for_status()'s HTTPError fell through to the generic handler, which
# silently dropped the whole day from the season.
RETRY_STATUSES = {429, 500, 502, 503, 504}

# ------------------------------------------------------------------
# Endpoint config
# ------------------------------------------------------------------

SAVANT_BASE = "https://baseballsavant.mlb.com"

# Pre-encoded URL template copied verbatim from pybaseball's _SC_SMALL_REQUEST,
# with extra_stats=miss_distance_inches appended.
#
# The hfGT value R%7CPO%7CS%7C= must stay pre-encoded. If you pass this
# through requests' params= dict, the %7C pipes get double-encoded to %257C
# and Savant returns empty results.
_URL_TEMPLATE = (
    "/statcast_search/csv?all=true"
    "&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
    "&hfGT=R%7CPO%7CS%7C="
    "&hfSea=&hfSit="
    "&player_type=pitcher"
    "&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA="
    "&game_date_gt={start_dt}"
    "&game_date_lt={end_dt}"
    "&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn="
    "&min_pitches=0&min_results=0&group_by=name"
    "&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc"
    "&min_abs=0&type=details"
    "&extra_stats=miss_distance"
    # Uncomment below if/when MLB exposes swing timing at pitch level:
    # ",delta_batball_tiedup_neg_x2,delta_batball_late_neg_y2_msec,ball_pos_above_plane"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://baseballsavant.mlb.com/statcast_search",
}

# ------------------------------------------------------------------
# Silence noisy pandas FutureWarnings
# ------------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def daterange(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def month_chunks(start_dt: datetime, end_dt: datetime) -> list[tuple[str, str]]:
    """
    Split [start_dt, end_dt] into (start, end) date-string pairs, one per
    calendar month, so each month lands in its own parquet file.
    """
    chunks = []
    cur = start_dt
    while cur <= end_dt:
        if cur.month == 12:
            next_month = datetime(cur.year + 1, 1, 1)
        else:
            next_month = datetime(cur.year, cur.month + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end_dt)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = next_month
    return chunks


def fetch_day(day_str: str) -> pd.DataFrame:
    """
    Fetch all pitches for a single date. Returns empty DataFrame if no games.
    URL is passed as a pre-encoded string — NOT through requests' params= arg.
    """
    url = SAVANT_BASE + _URL_TEMPLATE.format(start_dt=day_str, end_dt=day_str)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    # Strip BOM that Savant prepends, then check we actually have CSV.
    # Column names are quoted ("pitch_type") so check for the string
    # with or without quotes, case-insensitively.
    text = r.text.strip().lstrip("﻿")
    first = text[:60].lower()

    if not text or "<!doctype" in first or "pitch_type" not in first:
        return pd.DataFrame()

    return pd.read_csv(io.StringIO(text), low_memory=False)


def trim_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only KEEP columns, rename miss_distance_inches -> miss_distance."""
    if "miss_distance_inches" in df.columns:
        df = df.rename(columns={"miss_distance_inches": "miss_distance"})
    cols = [c for c in KEEP if c in df.columns]
    return df[cols].copy()


def download_month(start: str, end: str, out_path: str) -> list[str]:
    """
    Download every day in [start, end] and write one parquet file.
    Returns the list of days that could not be fetched, so the caller can
    surface holes instead of writing a quietly-incomplete month.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    daily_dfs = []
    skipped_days: list[str] = []

    for day in daterange(start_dt, end_dt):

        day_str = day.strftime("%Y-%m-%d")
        print(f"  {day_str}", end="", flush=True)

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = fetch_day(day_str)

                if df.empty:
                    print("  (no games)")
                    success = True
                    break

                df = trim_and_rename(df)
                daily_dfs.append(df)

                if "miss_distance" in df.columns:
                    n_whiffs = (df["description"] == "swinging_strike").sum()
                    n_miss   = df["miss_distance"].notna().sum()
                    print(f"  {len(df):,} pitches | "
                          f"miss_distance: {n_miss}/{n_whiffs} whiffs populated")
                else:
                    print(f"  {len(df):,} pitches | miss_distance column absent")

                success = True
                break

            except HTTPError as e:
                status = (e.response.status_code
                          if e.response is not None else None)
                if status not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    print(f"\n    FAILED permanently (HTTP {status}): {e}")
                    break
                wait = 2 ** attempt
                print(f"\n    retry {attempt}/{MAX_RETRIES} "
                      f"(HTTP {status}) waiting {wait}s ...",
                      end="", flush=True)
                time.sleep(wait)

            except (SSLError, ConnectionError, Timeout) as e:
                wait = 2 ** attempt
                print(f"\n    retry {attempt}/{MAX_RETRIES} "
                      f"({type(e).__name__}) waiting {wait}s ...",
                      end="", flush=True)
                time.sleep(wait)

            except Exception as e:
                print(f"\n    FAILED permanently: {e}")
                break

        if not success:
            print(f"    skipped {day_str}")
            skipped_days.append(day_str)

    if not daily_dfs:
        print(f"  No data for {start[:7]}")
        return skipped_days

    month_df = pd.concat(daily_dfs, ignore_index=True)
    month_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(month_df):,} rows -> {out_path}")
    return skipped_days


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Statcast pitch-level data for one season")
    parser.add_argument("--season", type=int, required=True,
                        help="Season year, e.g. 2026")
    parser.add_argument("--start", default=None,
                        help="Override start date YYYY-MM-DD (default Mar 15)")
    parser.add_argument("--end", default=None,
                        help="Override end date YYYY-MM-DD (default Nov 15, "
                             "clipped at today for an in-progress season)")
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory "
                             "(default data/all_pitches_{season})")
    args = parser.parse_args()

    season = args.season

    default_start = datetime(season, *SEASON_START_MONTH_DAY)
    default_end   = datetime(season, *SEASON_END_MONTH_DAY)

    start_dt = (datetime.strptime(args.start, "%Y-%m-%d")
                if args.start else default_start)
    end_dt   = (datetime.strptime(args.end, "%Y-%m-%d")
                if args.end else default_end)

    # An in-progress season has no data past today.
    today = datetime.combine(date.today(), datetime.min.time())
    if end_dt > today:
        print(f"Clipping end date {end_dt:%Y-%m-%d} to today "
              f"({today:%Y-%m-%d}) — season still in progress.")
        end_dt = today

    if end_dt < start_dt:
        raise SystemExit(f"End date {end_dt:%Y-%m-%d} precedes start date "
                         f"{start_dt:%Y-%m-%d} — nothing to download.")

    out_dir = args.out_dir or os.path.join(
        BASE_DIR, "data", f"all_pitches_{season}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Season {season}: {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}")
    print(f"Output dir: {out_dir}")

    all_skipped: list[str] = []

    for start, end in month_chunks(start_dt, end_dt):

        month_str = start[:7]
        out_path = os.path.join(out_dir, f"all_pitches_{month_str}.parquet")

        if os.path.exists(out_path):
            print(f"Skipping {month_str} (already exists)")
            continue

        print(f"\nDownloading month: {month_str}")
        all_skipped.extend(download_month(start, end, out_path))

    if all_skipped:
        # A skipped day is a silent hole in the season: the month parquet is
        # written anyway and looks complete. Fail loudly so the gap is either
        # backfilled (delete that month's parquet and re-run) or knowingly
        # accepted, rather than quietly becoming part of the training data.
        print(f"\nWARNING: {len(all_skipped)} day(s) could not be downloaded "
              f"and are MISSING from the season:")
        for day in all_skipped:
            print(f"  {day}")
        print("\nTo backfill: delete the affected month parquet(s) and re-run.")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
