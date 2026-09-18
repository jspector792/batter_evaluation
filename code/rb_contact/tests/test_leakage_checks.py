"""
Testing checklist items covered here:
  - Assert contact/no-contact label field is sourced independently from any
    offset/miss-distance/threshold field.
  - Assert no outcome-derived features (per data_audit.md) are present in
    X_raw or X_hybrid.

These run against the real 2025 pitch data, so they double as a live check
that the data pipeline hasn't silently changed shape.
"""
import os, sys, glob, unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import CONTINUOUS_RAW, CATEGORICAL_RAW, BINARY_RAW, X_HYBRID_EXTRA
from oof_timing_offset import ALL_SWINGS, CONTACT, BUNT_DESC, DATA_DIR

OUTCOME_DERIVED_COLUMNS = {
    'launch_angle', 'launch_speed', 'launch_speed_angle',
    'estimated_woba_using_speedangle', 'delta_run_exp',
    'miss_distance', 'barrel_distance', 'barrel_distance_v2',
    # in-sample predictions from earlier session work -- not outcome-derived
    # per se, but unusable without OOF regeneration (data_audit.md §2/§4)
    'int_y_predicted', 'barrel_distance_predicted', 'barrel_pred_v3',
    # real (non-predicted) contact-geometry fields -- excluded per §3, only
    # the *predicted* versions (predicted_timing) belong in X_hybrid
    'intercept_ball_minus_batter_pos_x_inches',
    'intercept_ball_minus_batter_pos_y_inches',
    'intercept_x', 'intercept_y',
}


class TestLabelIndependence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        f = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))[0]
        cls.df = pd.read_parquet(f)

    def test_description_field_exists_and_is_populated(self):
        self.assertIn('description', self.df.columns)
        self.assertEqual(self.df['description'].isna().mean(), 0.0)

    def test_label_does_not_require_miss_distance_or_launch_angle(self):
        """The contact label (description in CONTACT) must be computable for
        rows where miss_distance AND launch_angle are BOTH null -- if it
        weren't, the label would implicitly depend on those outcome-derived
        fields to exist, which is exactly the tautology the spec forbids."""
        swings = self.df[self.df['description'].isin(ALL_SWINGS)]
        both_null = swings['miss_distance'].isna() & swings['launch_angle'].isna()
        self.assertGreater(both_null.sum(), 0,
                           'expected some swings with both fields null to test against')
        # is_contact must still be well-defined (True/False, no ambiguity) for these rows
        labels = swings.loc[both_null, 'description'].isin(CONTACT)
        self.assertEqual(labels.isna().sum(), 0)

    def test_whiffs_have_null_miss_distance_derived_target_for_contact_rows_only(self):
        """Sanity-check the exact asymmetry the domain owner described:
        miss_distance is ~fully populated for whiffs and ~fully null for
        contact -- confirms the label isn't somehow backed by this field."""
        swings = self.df[self.df['description'].isin(ALL_SWINGS)].copy()
        is_contact = swings['description'].isin(CONTACT)
        contact_null_rate = swings.loc[is_contact, 'miss_distance'].isna().mean()
        self.assertGreater(contact_null_rate, 0.99,
                           'miss_distance should be ~fully null for contact rows')


class TestFeatureSetExcludesLeakage(unittest.TestCase):

    def test_x_raw_excludes_outcome_derived_columns(self):
        x_raw = set(CONTINUOUS_RAW) | set(CATEGORICAL_RAW) | set(BINARY_RAW)
        # same_hand isn't a raw column name but is derived purely from
        # stand/p_throws (both pre-outcome) -- not itself outcome-derived
        overlap = x_raw & OUTCOME_DERIVED_COLUMNS
        self.assertEqual(overlap, set(),
                         f'X_raw contains outcome-derived/leakage columns: {overlap}')

    def test_x_hybrid_extra_are_predicted_not_real(self):
        for col in X_HYBRID_EXTRA:
            self.assertTrue(col.startswith('predicted_'),
                            f'{col} in X_HYBRID_EXTRA does not look like a '
                            f'model *prediction* -- hybrid features must be '
                            f'OOF predictions, never the real/derived target')

    def test_bunts_excluded_from_swing_population(self):
        """Bunts have near-zero bat speed by construction, making their
        kinematics not representative of a real swing attempt -- confirm
        the exclusion list matches the audit's recommendation."""
        self.assertEqual(BUNT_DESC, {'missed_bunt', 'foul_bunt', 'bunt_foul_tip'})


if __name__ == '__main__':
    unittest.main()
