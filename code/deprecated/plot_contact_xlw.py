"""
Plot 1 — Hex plot: average xLW by contact location (intercept x / y inches).
Plot 2 — Hex plot: xLW predicted from intercept location (x) vs actual
          delta_run_exp (y), coloured by mean launch angle per bin, with y=x line.

Swings are identified by non-null intercept_ball_minus_batter_pos_* columns.
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binned_statistic_2d

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "fastballs_2025")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out")

# ---------------------------------------------------------------------------
# Load xLW dict
# ---------------------------------------------------------------------------

with open(os.path.join(OUT_DIR, "xlw_2025.pkl"), "rb") as f:
    xlw_dict = pickle.load(f)

# ---------------------------------------------------------------------------
# Load all pitches — union of columns needed across both plots
# ---------------------------------------------------------------------------

COLS = [
    "events",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "description",
    "delta_run_exp",
    "launch_angle",
]

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
chunks = []
for fpath in files:
    df = pd.read_csv(fpath, usecols=COLS)
    chunks.append(df)

data = pd.concat(chunks, ignore_index=True)
print(f"Total pitches: {len(data):,}")

# ---------------------------------------------------------------------------
# Filter to swings: intercept columns are only populated when batter swings
# ---------------------------------------------------------------------------

swings = data.dropna(subset=[
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]).copy()
print(f"Swings (non-null intercept): {len(swings):,}")

# ---------------------------------------------------------------------------
# Map event → xLW; mid-at-bat swings (foul, swinging_strike, etc.) map to
# the no_event weight since there's no at-bat-ending event on that pitch.
# ---------------------------------------------------------------------------

swings["events"] = swings["events"].fillna("no_event")
swings["xlw"] = swings["events"].map(xlw_dict)

# Drop any rows where xLW couldn't be assigned (e.g. intent_walk NaN)
swings = swings.dropna(subset=["xlw"])
print(f"Swings after xLW mapping: {len(swings):,}")

# Keep a copy before outlier clipping for plot 2
swings_full = swings.copy()

x = swings["intercept_ball_minus_batter_pos_x_inches"].values
y = swings["intercept_ball_minus_batter_pos_y_inches"].values
c = swings["xlw"].values

# ---------------------------------------------------------------------------
# Clip to reasonable swing-contact range (remove extreme outliers)
# Bounds are stored so plot 2 can apply the same clip.
# ---------------------------------------------------------------------------

x_lo, x_hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
y_lo, y_hi = np.percentile(y, 0.5), np.percentile(y, 99.5)

mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
x, y, c = x[mask], y[mask], c[mask]
print(f"After clipping outliers: {mask.sum():,} swings")

# ---------------------------------------------------------------------------
# Hex plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

# Diverging colormap centred at 0 (neutral run expectancy)
vmin, vmax = c.min(), c.max()
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

hb = ax.hexbin(
    x, y, C=c,
    reduce_C_function=np.mean,
    gridsize=55,
    norm=norm,
    cmap="RdYlGn",
    linewidths=0.2,
    mincnt=5,           # require ≥5 swings per bin
)

cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label("Mean xLW (Δ Run Expectancy per Swing)", fontsize=11)
cbar.ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

ax.set_xlabel("Horizontal Contact Depth\n(intercept_ball_minus_batter_pos_x_inches)\n← arm-side           glove-side →",
              fontsize=11)
ax.set_ylabel("Depth Contact Depth\n(intercept_ball_minus_batter_pos_y_inches)\n← front           back →",
              fontsize=11)
ax.set_title("2025 MLB — Mean xLW by Swing Contact Location\n(all pitch types, all teams)", fontsize=13)

# Faint zero-lines for orientation
ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")

plt.tight_layout()

out_path = os.path.join(OUT_DIR, "contact_xlw_hex_2025.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.show()

# ===========================================================================
# Plot 2 — xLW from intercept location (x) vs actual delta_run_exp (y),
#           coloured by mean launch angle per hex bin.
# Contact only: description must be a ball-in-play event.
# ===========================================================================

IN_PLAY = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}

swings2 = swings_full[
    swings_full["description"].isin(IN_PLAY)
].dropna(subset=[
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "delta_run_exp",
    "launch_angle",
]).copy()
swings2["events"] = swings2["events"].fillna("no_event")
swings2["xlw_event"] = swings2["events"].map(xlw_dict)
swings2 = swings2.dropna(subset=["xlw_event"])
print(f"\nContact-only swings for plot 2: {len(swings2):,}")

ix = swings2["intercept_ball_minus_batter_pos_x_inches"].values
iy = swings2["intercept_ball_minus_batter_pos_y_inches"].values

# Clip to same intercept bounds used in plot 1
mask2 = (ix >= x_lo) & (ix <= x_hi) & (iy >= y_lo) & (iy <= y_hi)
swings2 = swings2[mask2].copy()
ix = swings2["intercept_ball_minus_batter_pos_x_inches"].values
iy = swings2["intercept_ball_minus_batter_pos_y_inches"].values
actual_lw = swings2["delta_run_exp"].values
la = swings2["launch_angle"].values

# --- Predict xLW from intercept location via 2-D binned mean ---------------
# Use the same event-level xLW as the value; bin on (intercept_x, intercept_y)
# and assign each swing the mean xLW of its spatial bin.

NBINS = 55
stat, xedges, yedges, bin_idx = binned_statistic_2d(
    ix, iy,
    swings2["xlw_event"].values,
    statistic="mean",
    bins=NBINS,
    expand_binnumbers=True,
)

# bin_idx is (2, N) with 1-based indices; clamp to valid range
bx = np.clip(bin_idx[0] - 1, 0, NBINS - 1)
by = np.clip(bin_idx[1] - 1, 0, NBINS - 1)
xlw_from_location = stat[bx, by]   # predicted xLW for each swing's bin

valid = np.isfinite(xlw_from_location)
xlw_from_location = xlw_from_location[valid]
actual_lw_v = actual_lw[valid]
la_v = la[valid]

print(f"\nPlot 2: {valid.sum():,} swings with valid location-predicted xLW")

# --- Draw hex plot ----------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(9, 8))

# Launch angle colormap: centre near 0° (line drive ~10-25°), diverging
la_vmin, la_vmax = np.percentile(la_v, 1), np.percentile(la_v, 99)
la_norm = TwoSlopeNorm(vmin=la_vmin, vcenter=15.0, vmax=la_vmax)

hb2 = ax2.hexbin(
    xlw_from_location, actual_lw_v,
    C=la_v,
    reduce_C_function=np.mean,
    gridsize=60,
    norm=la_norm,
    cmap="coolwarm_r",
    linewidths=0.2,
    mincnt=5,
)

cbar2 = fig2.colorbar(hb2, ax=ax2, pad=0.02)
cbar2.set_label("Mean Launch Angle (°)", fontsize=11)
cbar2.ax.axhline(15.0, color="black", linewidth=1.0, linestyle="--")

# y = x reference line
lims = [
    min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
    max(ax2.get_xlim()[1], ax2.get_ylim()[1]),
]
ax2.plot(lims, lims, linestyle="--", color="black", linewidth=1.2,
         label="y = x (perfect prediction)", zorder=3)
ax2.set_xlim(lims)
ax2.set_ylim(lims)

ax2.axhline(0, color="grey", linewidth=0.6, linestyle=":")
ax2.axvline(0, color="grey", linewidth=0.6, linestyle=":")
ax2.legend(fontsize=10)

ax2.set_xlabel("xLW from Intercept Location (predicted, run expectancy)", fontsize=11)
ax2.set_ylabel("Actual LW — delta_run_exp (run expectancy)", fontsize=11)
ax2.set_title("2025 MLB — Predicted vs Actual LW by Contact Location\n"
              "coloured by mean launch angle per bin", fontsize=13)

plt.tight_layout()

out_path2 = os.path.join(OUT_DIR, "xlw_vs_actual_lw_hex_2025.png")
fig2.savefig(out_path2, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path2}")
plt.show()
