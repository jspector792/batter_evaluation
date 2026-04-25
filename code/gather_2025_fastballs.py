"""
Gather all pitches from the 2025 MLB season via pybaseball statcast.
Saves one CSV per hitting team with pitch-level rows and attributes as columns.
pitch_type is included as a column for downstream filtering.

Contact depth / barrel location columns:
  intercept_ball_minus_batter_pos_x_inches  (horizontal contact depth)
  intercept_ball_minus_batter_pos_y_inches  (depth into zone)

June is chunked weekly to avoid a pybaseball parallel-fetch parse error
that occurs when the full month is requested as a single query.
"""

import os
import time
import pandas as pd
from pybaseball import statcast
from pybaseball import cache

cache.enable()

# ---------------------------------------------------------------------------
# Date chunks for the full 2025 season
# June is split weekly to avoid a known parse error on the full-month pull.
# ---------------------------------------------------------------------------

SEASON_CHUNKS = [
    ("2025-03-18", "2025-03-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-07"),   # June split into weekly chunks
    ("2025-06-08", "2025-06-14"),
    ("2025-06-15", "2025-06-21"),
    ("2025-06-22", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-11-01"),   # Postseason
]

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

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "fastballs_2025"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pull all pitches
# ---------------------------------------------------------------------------

all_chunks = []
for start, end in SEASON_CHUNKS:
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
    time.sleep(2)

if not all_chunks:
    raise RuntimeError("No data collected — check date range or network.")

df = pd.concat(all_chunks, ignore_index=True)
print(f"\nTotal pitches collected: {len(df):,}")
print(f"Pitch type breakdown:\n{df['pitch_type'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# Derive hitting team
# ---------------------------------------------------------------------------

df["hitting_team"] = df.apply(
    lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"],
    axis=1
)

# ---------------------------------------------------------------------------
# Trim to requested columns
# ---------------------------------------------------------------------------

available = [c for c in KEEP_COLS if c in df.columns]
missing = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    print(f"\nColumns not found in data (skipped): {missing}")

df = df[available + ["hitting_team"]]
df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

# ---------------------------------------------------------------------------
# Save one CSV per hitting team (overwrites existing files)
# ---------------------------------------------------------------------------

teams = sorted(df["hitting_team"].dropna().unique())
print(f"\nSaving {len(teams)} team files to {OUT_DIR}/")

for team in teams:
    team_df = df[df["hitting_team"] == team].reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, f"fastballs_2025_{team}.csv")
    team_df.to_csv(out_path, index=False)
    print(f"  {team}: {len(team_df):,} pitches → {out_path}")

print("\nDone.")
