"""
Robust downloader for 2025 Statcast pitches.

- Downloads one day at a time via direct endpoint calls (not pybaseball)
  so that miss_distance_inches is reliably included in the response.
- Retries transient SSL/network failures
- Saves monthly parquet files
- Keeps only selected columns

Why not pybaseball statcast()?
  pybaseball hits the same /statcast_search/csv endpoint but never passes
  the extra_stats parameter, so miss_distance never appears. We replicate
  pybaseball's exact pre-encoded URL and simply append extra_stats at the end.

Why not requests.get(params=dict)?
  requests re-encodes the params dict, turning the already-encoded %7C pipes
  in hfGT (R%7CPO%7CS%7C=) into double-encoded %257C, which breaks Savant's
  filter and causes it to return empty results. We must pass the full URL
  string directly with no params argument.
"""

import io
import os
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.exceptions import SSLError, ConnectionError

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
os.makedirs(OUT_DIR, exist_ok=True)

# Column names as they appear in the CSV response.
# miss_distance_inches is renamed to miss_distance at save time.
KEEP = [
    'game_date', 'game_pk', 'at_bat_number', 'pitch_number',
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

MONTHS = [
    ('2025-03-15', '2025-03-31'),
    ('2025-04-01', '2025-04-30'),
    ('2025-05-01', '2025-05-31'),
    ('2025-06-01', '2025-06-30'),
    ('2025-07-01', '2025-07-31'),
    ('2025-08-01', '2025-08-31'),
    ('2025-09-01', '2025-09-30'),
]

MAX_RETRIES = 5

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
    text = r.text.strip().lstrip("\ufeff")
    first = text[:60].lower()

    if not text or "<!doctype" in first or "pitch_type" not in first:
        return pd.DataFrame()

    return pd.read_csv(io.StringIO(text), low_memory=False)


def trim_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only KEEP columns, rename miss_distance_inches -> miss_distance."""
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()
    if "miss_distance_inches" in df.columns:
        df = df.rename(columns={"miss_distance_inches": "miss_distance"})
    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

for start, end in MONTHS:

    month_str = start[:7]
    out_path = os.path.join(OUT_DIR, f"all_pitches_{month_str}.parquet")

    if os.path.exists(out_path):
        print(f"Skipping {month_str} (already exists)")
        continue

    print(f"\nDownloading month: {month_str}")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    daily_dfs = []

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

            except (SSLError, ConnectionError) as e:
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

    if not daily_dfs:
        print(f"  No data for {month_str}")
        continue

    month_df = pd.concat(daily_dfs, ignore_index=True)
    month_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(month_df):,} rows -> {out_path}")

print("\nDone.")