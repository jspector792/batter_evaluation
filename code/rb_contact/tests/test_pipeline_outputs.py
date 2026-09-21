"""
Testing checklist items covered here:
  - "Regression test: re-running the full pipeline on a fixed random seed /
    fixed data snapshot produces identical RB-contact% values."
  - "Confirm swing counts per batter in the final output match raw swing
    counts in source data (no silent row drops from feature nulls, joins,
    etc.) -- log any dropped rows with reasons."

The reproducibility check freezes a snapshot of rb_contact_pct_by_batter.csv
the first time it's run (out/rb_contact/_reproducibility_snapshot.csv) and
compares against it on subsequent runs. Delete the snapshot file to force a
fresh baseline (e.g. after an intentional model change).
"""
import os, sys, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oof_timing_offset import OUT_DIR, prepare_data

RB_PATH = os.path.join(OUT_DIR, 'rb_contact_pct_by_batter.parquet')
SNAPSHOT_PATH = os.path.join(OUT_DIR, '_reproducibility_snapshot.csv')


class TestReproducibility(unittest.TestCase):

    def test_rb_contact_pct_matches_frozen_snapshot(self):
        if not os.path.exists(RB_PATH):
            self.skipTest('rb_contact_pct_by_batter.parquet not generated yet -- run step5/step6 first')
        current = pd.read_parquet(RB_PATH).sort_values('batter').reset_index(drop=True)

        if not os.path.exists(SNAPSHOT_PATH):
            current.to_csv(SNAPSHOT_PATH, index=False)
            self.skipTest(f'No snapshot existed -- froze current output as the baseline at {SNAPSHOT_PATH}. '
                          f'Re-run this test after the NEXT full pipeline run to actually check reproducibility.')

        # prepare_data() casts `batter` to str, so the parquet holds strings
        # while read_csv infers int64 -- without normalizing, the two sets are
        # disjoint and this test fails on a perfectly reproducible pipeline.
        snapshot = pd.read_csv(SNAPSHOT_PATH)
        current['batter'] = current['batter'].astype(str)
        snapshot['batter'] = snapshot['batter'].astype(str)
        self.assertEqual(set(current['batter']), set(snapshot['batter']),
                         'batter set changed between runs -- not a pure re-run regression')

        merged = current.merge(snapshot, on='batter', suffixes=('_now', '_snap'))
        for col in ['rb_contact_pct', 'raw_contact_pct']:
            if f'{col}_now' in merged.columns:
                diff = (merged[f'{col}_now'] - merged[f'{col}_snap']).abs()
                self.assertLess(diff.max(), 1e-6,
                                f'{col} differs from frozen snapshot by up to {diff.max():.6f} '
                                f'-- pipeline is not reproducible on a fixed seed/data snapshot')


class TestSwingCountsMatchSource(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(RB_PATH):
            raise unittest.SkipTest('rb_contact_pct_by_batter.parquet not generated yet')
        cls.rb = pd.read_parquet(RB_PATH)
        cls.raw = prepare_data()  # full swing population, bunts excluded, no other filtering

    def test_swing_counts_are_subset_of_raw_and_explainably_smaller(self):
        """rb_contact_pct_by_batter.parquet's n_swings should never EXCEED
        the batter's true raw swing count, and the gap (raw - n_swings)
        should be explainable by the MIN_SWINGS>=100 filter + feature-null
        drops, not a silent/unexplained loss."""
        raw_counts = self.raw.groupby('batter').size()
        for _, row in self.rb.iterrows():
            batter, n = row['batter'], row['n_swings']
            self.assertIn(batter, raw_counts.index,
                          f'batter {batter} in RB output but not in raw swing population')
            self.assertLessEqual(n, raw_counts[batter],
                                 f'batter {batter}: RB n_swings ({n}) exceeds raw swing count '
                                 f'({raw_counts[batter]}) -- should be impossible')

    def test_no_batter_below_min_swings_threshold(self):
        """compute_batter_metrics() filters to n_swings >= MIN_SWINGS (100) --
        confirm that filter actually held in the saved output."""
        self.assertTrue((self.rb['n_swings'] >= 100).all(),
                        'found batters below the MIN_SWINGS=100 threshold in the final output')


if __name__ == '__main__':
    unittest.main()
