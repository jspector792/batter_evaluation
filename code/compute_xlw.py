"""
Compute expected linear weights (xLW) for each unique event type.
xLW = mean(delta_run_exp) across all pitches of that event type, pooled
across all team CSV files.

Output: out/xlw_2025.pkl  — pickled dict {event_str: xLW_float}
"""

import os
import pickle
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "fastballs_2025")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load all team files — only the two columns we need
# ---------------------------------------------------------------------------

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found {len(files)} team files.")

chunks = []
for f in files:
    df = pd.read_csv(f, usecols=["events", "delta_run_exp"])
    chunks.append(df)

data = pd.concat(chunks, ignore_index=True)
print(f"Total pitches loaded: {len(data):,}")

# ---------------------------------------------------------------------------
# Compute xLW
# delta_run_exp is defined for every pitch; we group by the at-bat outcome
# (events column). Pitches mid-at-bat have events == NaN — we keep them
# separate as a "no_event" bucket so the dict is complete.
# ---------------------------------------------------------------------------

data["events"] = data["events"].fillna("no_event")

xlw = (
    data.groupby("events")["delta_run_exp"]
    .mean()
    .sort_values(ascending=False)
)

print("\nxLW by event:")
print(xlw.to_string())

xlw_dict = xlw.to_dict()

out_path = os.path.join(OUT_DIR, "xlw_2025.pkl")
with open(out_path, "wb") as f:
    pickle.dump(xlw_dict, f)

print(f"\nSaved {len(xlw_dict)} event types → {out_path}")
