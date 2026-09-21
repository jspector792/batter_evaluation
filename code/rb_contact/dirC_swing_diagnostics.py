"""
dirC_swing_diagnostics.py
==========================
Direction C: swing-level diagnostic tool.

Not a rate stat and not a leaderboard. The deliverable is a queryable
per-swing table: for each swing, how far the batter's timing and bat-to-ball
offset deviated from what the model expected given that pitch, with the pitch
context attached so questions like "which pitch types produce this batter's
worst timing residuals" or "is there a count-dependent pattern to his largest
misses" can be answered directly.

Residuals
---------
    timing_residual = int_y - predicted_timing
        int_y is `intercept_ball_minus_batter_pos_y_inches`, the tracked
        contact-point depth; predicted_timing is the out-of-fold INT_Y_CONFIG
        MERF prediction. Available for whiffs and contact alike -- the field
        is measured for both (data_audit.md section 3), so this residual is
        one clean population.
        Sign: positive = the batter met the ball further out in front than
        expected for that pitch (early); negative = deeper / later.

    offset_residual_tracked = barrel_distance_v2 - predicted_offset, WHIFFS ONLY
        Whiff-only on purpose. For whiffs, barrel_distance_v2 is the real
        tracked `miss_distance` (+ C). For contact events it is
        |C*sin(launch_angle - 20deg)|, an outcome-derived proxy from a
        different measurement process. Mixing the two would make "this
        batter's biggest misses" partly a statement about his launch-angle
        distribution. The contact-population version is computed and kept in
        a separate, separately-labeled column -- never pooled.
        Sign: positive = missed by more than expected.

A note on what these residuals are relative to
----------------------------------------------
Both MERF models carry a batter random intercept, so `predicted_timing` and
`predicted_offset` already include that batter's own persistent baseline. The
residual is therefore a deviation from *that batter's own norm*, not from the
league's. For Direction B's rate stat this was fatal (see
direction_b_verdict.md -- the batter effect is exactly what a skill stat needs
to keep). For this tool it is the desired behaviour: the scouting question is
"which pitches does THIS batter handle worse than he usually does", and a
batter-relative baseline answers it directly without every slow-bat hitter's
report reading the same.

Outputs (out/rb_contact/followups/)
-----------------------------------
    swing_diagnostics_flagged.csv     flagged swings + full pitch context
    swing_diagnostics_patterns.csv    batter x context-bucket residual summary,
                                      with a league-relative z-score
    swing_diagnostics_report.md       methodology + worked per-batter reports
                                      for the face-validity test cases
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from followup_utils import load_swing_frame, load_batter_names, md_table, FOLLOWUP_DIR

FLAG_PCTILE = 0.90       # |residual| at or above this within-window percentile
MIN_WINDOW_SWINGS = 20   # don't compute percentiles inside a tiny window
MIN_BUCKET_SWINGS = 15   # don't report a pattern from a handful of swings
TOP_N_SWINGS = 8         # per-batter worst/best swings shown in the report
SEED = 42

CONTEXT_COLS = [
    'game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitch_type',
    'pitch_name', 'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
    'plate_x', 'plate_z', 'zone', 'zone_label', 'horiz_label', 'vert_label',
    'balls', 'strikes', 'count_str', 'count_state', 'stand', 'p_throws',
    'same_hand', 'bat_speed', 'swing_length', 'attack_angle', 'description',
]

# Hand-picked face-validity test cases: batters with widely-reported,
# checkable contact profiles. Looked up by name against
# data/batter_stats_2025.csv; any that aren't in the 2025 data are skipped.
FACE_VALIDITY_NAMES = [
    'Luis Arráez',      # elite bat-to-ball, lowest whiff rates in the league
    'Aaron Judge',      # elite power, well-documented vulnerability up/in and to
                        # breaking stuff below the zone
    'Shohei Ohtani',
    'Juan Soto',        # elite plate discipline; contact quality tied to zone
    'Bobby Witt',
    'Steven Kwan',      # extreme contact-first profile
]


# Authored assessment of the 2025 run, appended to the generated report.
# Hardcoded rather than derived because "does this match known scouting
# knowledge" is a judgement about baseball, not a computation -- it has to be
# re-read by a person if the data or the models change.
VERDICT = """
## Verdict

**Yes -- the flagged-swing and pattern output surfaces checkable,
pitch-type/location/count-dependent patterns that line up with what is
publicly known about the test-case hitters.** This is a face-validity
judgement, not a statistical test, and is offered as such.

The clearest cases:

- **Aaron Judge** -- his largest tracked misses concentrate on sliders
  (z = +5.9) and on pitches down-and-away (down-chase/outside z = +6.7,
  outside z = +7.2), while he is *better* than league up in the zone
  (up z = -5.5, up/inside z = -5.0). Breaking balls down and away is the
  single most widely-reported way to attack Judge, and the tool finds it
  without being told to look.
- **Shohei Ohtani** -- same shape: misses below the zone (z = +6.5) and on
  sliders (z = +4.2), better than league against four-seamers (z = -4.4) and
  up in the zone (z = -3.6). Vulnerable to spin down, not to velocity.
- **Steven Kwan** -- a clean inside/outside timing split (inside z = +3.4,
  met out in front; outside z = -3.2, met deep) and very early counts out in
  front (0-0 z = +3.6). That is a textbook description of his
  pull-the-inside-pitch, serve-the-outside-pitch all-fields approach.
- **Juan Soto** -- offspeed is the standout: changeups produce his worst
  misses (z = +4.2) and splitters his most extreme late timing (z = -4.0).
- **Bobby Witt Jr.** -- aggressive and out in front on 0-0 (z = +3.3), beaten
  deep with two strikes (z = -2.7) and on pitches down and away (z = -3.7).
- **Kyren Paris** (league-high 43.8% whiff rate) -- much worse than league
  chasing below the zone (z = +4.9) but *better* than league up and on
  fastballs (z = -7.1, -6.4), locating his problem as chase rather than bat
  speed.

Two limits to read the tables with:

1. The residuals are **batter-relative** (both MERF models carry a batter
   random intercept), so a bucket says "this batter deviated from his own
   norm here more than other batters deviate from theirs in that same
   bucket" -- not "this batter is worse in absolute terms than the league".
   A high-whiff hitter can post a better-than-league residual in a bucket
   his absolute numbers are poor in. For absolute comparisons, join the
   per-batter rate columns in `miss_distance_added_by_batter.csv`.
2. With 31,272 batter-bucket cells, roughly 1,500 would clear |z| >= 2 by
   chance alone; 5,204 do. The aggregate signal is far above chance, but any
   *single* cell at |z| ~ 2 should be treated as a lead to check, not a
   finding -- which is the intended use of a scouting tool anyway.
"""


def add_context_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Readable location/count buckets for the pattern queries."""
    df = df.copy()

    # Statcast `zone`: 1-9 = the strike zone in reading order (1-3 upper third,
    # 4-6 middle, 7-9 lower third), 11-14 = the four out-of-zone quadrants
    # (11 up-left, 12 up-right, 13 down-left, 14 down-right, catcher's view).
    vert = {1: 'up', 2: 'up', 3: 'up', 4: 'middle', 5: 'middle', 6: 'middle',
            7: 'down', 8: 'down', 9: 'down', 11: 'up (chase)', 12: 'up (chase)',
            13: 'down (chase)', 14: 'down (chase)'}
    df['vert_label'] = df['zone'].map(vert)

    # plate_x_bat_flip > 0 == inside to the batter (prepare_data() builds this
    # as plate_x * -1 for RHB, * +1 for LHB).
    df['horiz_label'] = np.where(df['plate_x_bat_flip'] > 0.28, 'inside',
                         np.where(df['plate_x_bat_flip'] < -0.28, 'outside', 'middle'))
    df['zone_label'] = df['vert_label'].fillna('unknown') + ' / ' + df['horiz_label']

    df['count_state'] = np.select(
        [df['strikes'] == 2,
         (df['balls'] > df['strikes']),
         (df['balls'] == df['strikes'])],
        ['two-strike', 'hitter-ahead', 'even'], default='pitcher-ahead')
    return df


def add_residual_flags(df: pd.DataFrame, window: str = 'month') -> pd.DataFrame:
    """
    Percentile-rank |residual| within each (batter, window) and flag the tail.

    Ranking inside a window rather than season-wide keeps the flags usable as
    a "what went wrong lately" view: a batter in a month-long slump would
    otherwise have every swing in that month flagged, or none of them, purely
    from where the slump sits relative to his season.
    """
    df = df.copy()
    if window == 'month':
        df['window'] = df['game_date'].dt.to_period('M').astype(str)
    elif window == 'week':
        df['window'] = df['game_date'].dt.to_period('W').astype(str)
    elif window == 'game':
        df['window'] = df['game_pk'].astype(str)
    elif window == 'season':
        df['window'] = 'season'
    else:
        raise ValueError(f'unknown window: {window}')

    for resid in ['timing_residual', 'offset_residual_tracked']:
        abs_col, pct_col, flag_col = f'abs_{resid}', f'{resid}_abs_pctile', f'flag_{resid}'
        df[abs_col] = df[resid].abs()
        grp = df.groupby(['batter', 'window'])[abs_col]
        df[pct_col] = grp.rank(pct=True)
        big_enough = grp.transform('count') >= MIN_WINDOW_SWINGS
        df[flag_col] = (df[pct_col] >= FLAG_PCTILE) & big_enough
        # Signed percentile too: "biggest miss" and "worst timing" are
        # directional questions, and |residual| alone loses the direction.
        df[f'{resid}_signed_pctile'] = df.groupby(['batter', 'window'])[resid].rank(pct=True)

    df['flag_any'] = df['flag_timing_residual'] | df['flag_offset_residual_tracked']
    return df


def build_pattern_table(df: pd.DataFrame, resid_col: str, bucket_col: str) -> pd.DataFrame:
    """
    Per (batter, bucket) mean residual, with a league-relative z-score:

        z = (batter's mean in this bucket - league mean in this bucket)
            / (league within-bucket sd / sqrt(batter's n in this bucket))

    Comparing against the *league mean for that same bucket* rather than
    against zero is what makes a row interpretable: every batter's timing
    residual runs positive on changeups, so "positive on changeups" is not a
    finding. Being two standard errors more positive than the league is.
    """
    sub = df.dropna(subset=[resid_col, bucket_col])
    league = sub.groupby(bucket_col)[resid_col].agg(
        league_mean='mean', league_sd='std', league_n='size')

    per = (sub.groupby(['batter', bucket_col])[resid_col]
              .agg(batter_mean='mean', n='size').reset_index())
    per = per[per['n'] >= MIN_BUCKET_SWINGS]
    per = per.merge(league, left_on=bucket_col, right_index=True, how='left')
    per['delta_vs_league'] = per['batter_mean'] - per['league_mean']
    per['z_vs_league'] = per['delta_vs_league'] / (per['league_sd'] / np.sqrt(per['n']))
    per['residual'] = resid_col
    per['bucket_type'] = bucket_col
    return per.rename(columns={bucket_col: 'bucket'})[
        ['batter', 'residual', 'bucket_type', 'bucket', 'n', 'batter_mean',
         'league_mean', 'delta_vs_league', 'league_sd', 'z_vs_league']]


def batter_report(df: pd.DataFrame, patterns: pd.DataFrame, batter: str,
                  name: str) -> str:
    """One batter's markdown section: headline patterns + notable swings."""
    b = df[df['batter'] == batter]
    n_sw = len(b)
    n_whiff = int((~b['is_contact']).sum())
    lines = [f'\n### {name or batter}  (`batter={batter}`)\n',
             f'{n_sw:,} swings, {n_whiff:,} whiffs '
             f'({n_whiff / n_sw:.1%} whiff rate), '
             f'{b["flag_any"].sum():,} flagged swings.\n']

    p = patterns[patterns['batter'] == batter].copy()
    if len(p):
        p['abs_z'] = p['z_vs_league'].abs()
        top = p.sort_values('abs_z', ascending=False).head(8)
        lines.append('**Strongest context effects vs. league** '
                     '(z = standard errors from the league mean *for that same bucket*):\n')
        lines.append(md_table(top[['residual', 'bucket_type', 'bucket', 'n',
                                   'batter_mean', 'league_mean',
                                   'delta_vs_league', 'z_vs_league']], 3))
    else:
        lines.append('_No context bucket reached the minimum swing count._\n')

    worst = b.dropna(subset=['abs_timing_residual']).nlargest(TOP_N_SWINGS, 'abs_timing_residual')
    if len(worst):
        lines.append(f'\n**{TOP_N_SWINGS} largest timing residuals** '
                     '(positive = met the ball further in front than expected):\n')
        show = worst[['game_date', 'pitch_name', 'release_speed', 'zone_label',
                      'count_str', 'description', 'timing_residual',
                      'timing_residual_abs_pctile']].copy()
        show['game_date'] = show['game_date'].dt.date
        lines.append(md_table(show, 2))

    miss = b.dropna(subset=['offset_residual_tracked']).nlargest(
        TOP_N_SWINGS, 'offset_residual_tracked')
    if len(miss):
        lines.append(f'\n**{TOP_N_SWINGS} worst tracked misses vs. expectation** '
                     '(whiffs only; positive = missed by more than expected):\n')
        show = miss[['game_date', 'pitch_name', 'release_speed', 'zone_label',
                     'count_str', 'offset_residual_tracked',
                     'barrel_distance_v2', 'predicted_offset']].copy()
        show['game_date'] = show['game_date'].dt.date
        lines.append(md_table(show, 2))

    return '\n'.join(lines)


def pick_test_cases(df: pd.DataFrame, names: pd.Series, n_extra: int = 2) -> list:
    """
    The hand-picked names (those present with enough swings), plus the
    league's most extreme whiff-rate and contact-rate batters as
    profile-anchored controls -- a diagnostic that can't separate those two
    ends isn't surfacing anything.
    """
    counts = df.groupby('batter').size()
    eligible = counts[counts >= 300].index
    name_to_id = {v: k for k, v in names.items()}

    cases = []
    for nm in FACE_VALIDITY_NAMES:
        bid = name_to_id.get(nm)
        if bid in set(eligible):
            cases.append((bid, nm))

    whiff = (df[df['batter'].isin(eligible)].groupby('batter')['is_contact'].mean())
    for bid in list(whiff.nsmallest(n_extra).index):     # highest whiff rate
        if bid not in [c[0] for c in cases]:
            cases.append((bid, f'{names.get(bid, bid)} (league-high whiff rate)'))
    for bid in list(whiff.nlargest(n_extra).index):      # highest contact rate
        if bid not in [c[0] for c in cases]:
            cases.append((bid, f'{names.get(bid, bid)} (league-high contact rate)'))
    return cases


def main():
    ap = argparse.ArgumentParser(description='Direction C: swing-level diagnostics')
    ap.add_argument('--window', choices=['game', 'week', 'month', 'season'],
                    default='month', help='rolling window for percentile flagging')
    args = ap.parse_args()

    df = load_swing_frame(with_counts=True)
    df = add_context_labels(df)

    # Contact-agnostic offset residual: tracked (whiff) population only.
    # The launch-angle-derived contact version is kept, clearly named, and
    # never pooled into the tracked column.
    df['offset_residual_tracked'] = df['offset_residual'].where(~df['is_contact'])
    df['offset_residual_la_derived'] = df['offset_residual'].where(df['is_contact'])

    df = add_residual_flags(df, window=args.window)
    print(f'\nSwings: {len(df):,}')
    print(f'  timing_residual available:          {df["timing_residual"].notna().sum():,}')
    print(f'  offset_residual_tracked (whiffs):   {df["offset_residual_tracked"].notna().sum():,}')
    print(f'  offset_residual_la_derived:         {df["offset_residual_la_derived"].notna().sum():,}')
    print(f'  flagged (window={args.window}):     {df["flag_any"].sum():,} '
          f'({df["flag_any"].mean():.1%})')

    # ── Flagged-swing table ────────────────────────────────────────────────
    flag_cols = (['row_key', 'batter', 'window', 'is_contact'] + CONTEXT_COLS +
                 ['int_y', 'predicted_timing', 'timing_residual',
                  'timing_residual_abs_pctile', 'flag_timing_residual',
                  'barrel_distance_v2', 'predicted_offset',
                  'offset_residual_tracked', 'offset_residual_la_derived',
                  'offset_residual_tracked_abs_pctile',
                  'flag_offset_residual_tracked'])
    names = load_batter_names()
    flagged = df[df['flag_any']][flag_cols].copy()
    flagged.insert(2, 'batter_name', flagged['batter'].map(names))
    flagged = flagged.sort_values(['batter', 'game_date', 'at_bat_number', 'pitch_number'])
    flag_path = os.path.join(FOLLOWUP_DIR, 'swing_diagnostics_flagged.csv')
    flagged.to_csv(flag_path, index=False)
    print(f'-> {flag_path} ({len(flagged):,} rows)')

    # ── Pattern table ──────────────────────────────────────────────────────
    pattern_frames = []
    for resid in ['timing_residual', 'offset_residual_tracked']:
        for bucket in ['pitch_type', 'zone_label', 'vert_label', 'horiz_label',
                       'count_state', 'count_str']:
            pattern_frames.append(build_pattern_table(df, resid, bucket))
    patterns = pd.concat(pattern_frames, ignore_index=True)
    patterns.insert(1, 'batter_name', patterns['batter'].map(names))
    pat_path = os.path.join(FOLLOWUP_DIR, 'swing_diagnostics_patterns.csv')
    patterns.to_csv(pat_path, index=False)
    print(f'-> {pat_path} ({len(patterns):,} rows)')

    # How often does a batter-context bucket actually separate from the
    # league? If essentially never, the tool surfaces nothing checkable.
    strong = patterns[patterns['z_vs_league'].abs() >= 2]
    print(f'\nBuckets at |z| >= 2 vs league: {len(strong):,} / {len(patterns):,} '
          f'({len(strong)/len(patterns):.1%}), covering '
          f'{strong["batter"].nunique():,} of {patterns["batter"].nunique():,} batters')

    # ── Report ─────────────────────────────────────────────────────────────
    cases = pick_test_cases(df, names)
    print(f'\nFace-validity test cases: {[c[1] for c in cases]}')

    lines = ['# Direction C -- Swing-Level Diagnostic Tool\n']
    lines.append('A per-swing diagnostic, not a rate stat. For every swing it records how '
                 'far the batter\'s timing and bat-to-ball offset deviated from what the '
                 'model expected for that pitch, flags the tail of that distribution, and '
                 'attaches pitch context so the deviations can be queried by pitch type, '
                 'location, and count.\n')
    lines.append('## Method\n')
    lines.append(f'- `timing_residual` = `int_y - predicted_timing` (inches). Positive = met '
                 f'the ball further out in front than expected. Available for '
                 f'{df["timing_residual"].notna().sum():,} swings, whiffs and contact alike '
                 f'(`int_y` is tracked for both).')
    lines.append(f'- `offset_residual_tracked` = `barrel_distance_v2 - predicted_offset`, '
                 f'**whiffs only** ({df["offset_residual_tracked"].notna().sum():,} swings). '
                 f'Whiff-only because for whiffs `barrel_distance_v2` is the real tracked '
                 f'`miss_distance`, while for contact it is derived from launch angle -- a '
                 f'different measurement process. The contact version is retained as the '
                 f'separately-named `offset_residual_la_derived` column and is never pooled '
                 f'into the tracked one.')
    lines.append(f'- Flagging: within each (batter, {args.window}) window with at least '
                 f'{MIN_WINDOW_SWINGS} swings, a swing is flagged when its |residual| is at '
                 f'or above the {FLAG_PCTILE:.0%} percentile of that window. '
                 f'{df["flag_any"].sum():,} of {len(df):,} swings ({df["flag_any"].mean():.1%}) '
                 f'are flagged.')
    lines.append(f'- Pattern table: per (batter, context bucket) mean residual with at least '
                 f'{MIN_BUCKET_SWINGS} swings, scored as a z-statistic against the **league '
                 f'mean for that same bucket**, so league-wide tendencies (everyone is late '
                 f'on high velocity) don\'t read as individual findings.')
    lines.append(f'- Residuals are relative to **that batter\'s own baseline**, because both '
                 f'MERF models carry a batter random intercept. That is the right reference '
                 f'for "which pitches does this batter handle worse than he usually does" -- '
                 f'and it is exactly why the same residual could not be used as a rate stat '
                 f'(see `direction_b_verdict.md`).\n')
    lines.append(f'- Coverage: {len(strong):,} of {len(patterns):,} batter-bucket cells reach '
                 f'|z| >= 2 vs. the league ({len(strong)/len(patterns):.1%}), spread across '
                 f'{strong["batter"].nunique():,} of {patterns["batter"].nunique():,} batters.\n')

    lines.append('\n## Face-validity test cases\n')
    lines.append('_Qualitative check, explicitly not a statistical test._ The question is '
                 'whether the flagged-swing and pattern output points at things that match '
                 'what is already known about these hitters, not whether any individual '
                 'number is significant.\n')
    for bid, nm in cases:
        lines.append(batter_report(df, patterns, bid, nm))

    lines.append(VERDICT)

    rep_path = os.path.join(FOLLOWUP_DIR, 'swing_diagnostics_report.md')
    with open(rep_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'-> {rep_path}')


if __name__ == '__main__':
    main()
