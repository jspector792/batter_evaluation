"""
Gather all pitches from an MLB season via pybaseball statcast.
Saves one CSV per hitting team with pitch-level rows and attributes as columns.
pitch_type is included as a column for downstream filtering.

Season-parameterised generalisation of gather_2025_fastballs.py, which is kept
for provenance of the original 2025 pull. Two differences beyond the --season
argument:

  * Every chunk is weekly, not just June. The original split June alone to dodge
    a pybaseball parallel-fetch parse error that hits on full-month queries;
    that failure mode is not June-specific, so all chunks are now small.
  * A failed chunk is reported at the end instead of silently leaving a hole.

Contact depth / barrel location columns:
  intercept_ball_minus_batter_pos_x_inches  (horizontal contact depth)
  intercept_ball_minus_batter_pos_y_inches  (depth into zone)

Usage
-----
  python gather_season_pitches.py --season 2025
  python gather_season_pitches.py --season 2026

Output
------
  data/fastballs_{season}/fastballs_{season}_{TEAM}.csv

The directory name keeps the historical "fastballs" prefix so that the 2025
outputs stay where anything already pointing at them expects; the files hold
all pitch types, as they did before.
"""

import argparse
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
from pybaseball import statcast
from pybaseball import cache

cache.enable()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Widest plausible window for a season; dates with no games come back empty.
SEASON_START_MONTH_DAY = (3, 15)
SEASON_END_MONTH_DAY   = (11, 15)

CHUNK_DAYS = 7

# Columns to keep — batter/pitcher identity, pitch mechanics, swing/contact,
# batted-ball outcome, and game context.
KEEP_COLS = [
    # Identifiers
    "game_pk", "game_date", "game_year", "game_type",
    "at_bat_number", "pitch_number",
    # Teams / game context
    "home_team", "away_team", "inning", "inning_topbot",
    "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "home_score", "away_score", "bat_score", "fld_score",
    # Batter
    "batter", "stand", "age_bat",
    "n_priorpa_thisgame_player_at_bat",
    "batter_days_since_prev_game", "batter_days_until_next_game",
    # Pitcher
    "pitcher", "player_name", "p_throws", "age_pit", "arm_angle",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game", "pitcher_days_until_next_game",
    # Pitch identity & velocity
    "pitch_type", "pitch_name",
    "release_speed", "effective_speed",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension", "release_spin_rate", "spin_axis",
    # Pitch movement & trajectory
    "pfx_x", "pfx_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "api_break_z_with_gravity", "api_break_x_arm", "api_break_x_batter_in",
    # Plate location & zone
    "plate_x", "plate_z", "zone", "sz_top", "sz_bot",
    # Swing mechanics (key contact-depth columns)
    "bat_speed", "swing_length",
    "attack_angle", "attack_direction", "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    # Outcome
    "description", "events", "type", "bb_type",
    "hc_x", "hc_y",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "launch_speed_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    "hyper_speed",
    # Win expectancy / run value
    "delta_run_exp", "delta_pitcher_run_exp",
    "delta_home_win_exp",
    "home_win_exp", "bat_win_exp",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def season_chunks(start_dt: datetime, end_dt: datetime,
                  chunk_days: int = CHUNK_DAYS) -> list[tuple[str, str]]:
    """Split [start_dt, end_dt] into chunk_days-long (start, end) pairs."""
    chunks = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_dt)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


def resolve_window(season: int, start: str | None,
                   end: str | None) -> tuple[datetime, datetime]:
    """Resolve the season window, clipped at today for an in-progress season."""
    start_dt = (datetime.strptime(start, "%Y-%m-%d") if start
                else datetime(season, *SEASON_START_MONTH_DAY))
    end_dt   = (datetime.strptime(end, "%Y-%m-%d") if end
                else datetime(season, *SEASON_END_MONTH_DAY))

    today = datetime.combine(date.today(), datetime.min.time())
    if end_dt > today:
        print(f"Clipping end date {end_dt:%Y-%m-%d} to today "
              f"({today:%Y-%m-%d}) — season still in progress.")
        end_dt = today

    if end_dt < start_dt:
        raise SystemExit(f"End date {end_dt:%Y-%m-%d} precedes start date "
                         f"{start_dt:%Y-%m-%d} — nothing to gather.")
    return start_dt, end_dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gather a season of Statcast pitches into per-team CSVs")
    parser.add_argument("--season", type=int, required=True,
                        help="Season year, e.g. 2026")
    parser.add_argument("--start", default=None,
                        help="Override start date YYYY-MM-DD (default Mar 15)")
    parser.add_argument("--end", default=None,
                        help="Override end date YYYY-MM-DD (default Nov 15, "
                             "clipped at today)")
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory "
                             "(default data/fastballs_{season})")
    args = parser.parse_args()

    season = args.season
    start_dt, end_dt = resolve_window(season, args.start, args.end)

    out_dir = args.out_dir or os.path.join(
        BASE_DIR, "data", f"fastballs_{season}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Season {season}: {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}")
    print(f"Output dir: {out_dir}")

    # ── Pull all pitches ──────────────────────────────────────────────────────
    all_chunks = []
    failed = []
    for start, end in season_chunks(start_dt, end_dt):
        print(f"\n=== Pulling {start} → {end} ===")
        try:
            chunk = statcast(start_dt=start, end_dt=end)
            if chunk is None or chunk.empty:
                print("  No data returned.")
                continue
            print(f"  {len(chunk):,} pitches")
            all_chunks.append(chunk)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((start, end, str(e)))
        time.sleep(2)

    if not all_chunks:
        raise RuntimeError("No data collected — check date range or network.")

    df = pd.concat(all_chunks, ignore_index=True)
    print(f"\nTotal pitches collected: {len(df):,}")
    print(f"Pitch type breakdown:\n{df['pitch_type'].value_counts().to_string()}")

    # ── Derive hitting team ───────────────────────────────────────────────────
    df["hitting_team"] = df["home_team"].where(
        df["inning_topbot"] == "Bot", df["away_team"])

    # ── Trim to requested columns ─────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"\nColumns not found in data (skipped): {missing}")

    df = df[available + ["hitting_team"]]
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

    # ── Save one CSV per hitting team (overwrites existing files) ─────────────
    teams = sorted(df["hitting_team"].dropna().unique())
    print(f"\nSaving {len(teams)} team files to {out_dir}/")

    for team in teams:
        team_df = df[df["hitting_team"] == team].reset_index(drop=True)
        out_path = os.path.join(out_dir, f"fastballs_{season}_{team}.csv")
        team_df.to_csv(out_path, index=False)
        print(f"  {team}: {len(team_df):,} pitches → {out_path}")

    if failed:
        print(f"\nWARNING: {len(failed)} chunk(s) failed and are missing "
              f"from the output:")
        for start, end, err in failed:
            print(f"  {start} → {end}: {err}")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
