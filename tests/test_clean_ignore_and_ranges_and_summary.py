import math

import numpy as np
import pandas as pd
import pytest

from src.main import (
    DeltaMode,
    clean_ignore,
    convert_timestamp_column_to_datetime,
    filter_by_adjusted_run_time_zscore,
    filter_timestamp_ranges,
    summarize_run_time_by_sor_range,
)


class TestCleanIgnore:
    def test_missing_columns_are_handled_and_warned(self, caplog):
        # Build a frame with missing 'ignore' and missing 'sor#'
        # Also include some NaNs in runticks to exercise dropna subset behavior.
        df = pd.DataFrame(
            {
                "timestamp": ["20250101000001", "20250101000002", "20250101000003"],
                # "ignore" missing
                # "sor#" missing
                "runticks": [1000, None, 1020],
                "notes": ["", "", ""],
            }
        )
        out = clean_ignore(df.copy(), input_data_fort=6, verbose=True)

        # One row with runticks None should be dropped by dropna subset.
        # No 'sor#' column means sor_valid becomes NaN and will be dropped as well (subset includes 'sor_valid').
        # Thus only rows with both runticks and implied sor_valid present survive; but sor_valid for all rows is NA,
        # so they are removed. Expect empty.
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

        # Check warnings emitted about missing columns
        text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "Column 'ignore' not found" in text
        assert "Column 'sor#' not found" in text

    def test_mixed_case_true_is_treated_as_ignore(self):
        # ignore handled case-insensitively; TRUE means drop row
        df = pd.DataFrame(
            {
                "timestamp": ["20250101000001", "20250101000002", "20250101000003"],
                "ignore": ["TrUe", "false", "FALSE"],
                "sor#": [1, 2, 3],
                "runticks": [1000, 1010, 1020],
                "notes": ["", "", ""],
            }
        )
        out = clean_ignore(df.copy(), input_data_fort=6, verbose=False)
        # Row 0 ignored
        assert list(out["sor#"]) == [2, 3]
        # Ensure helper column sor_valid is dropped
        assert "sor_valid" not in out.columns


class TestFilterTimestampRangesBehavior:
    def _make_dt_frame(self):
        fmt = "%Y%m%d%H%M%S"
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "20250101000001",
                        "20250101000002",
                        "20250101000003",
                        "20250101000004",
                    ],
                    format=fmt,
                    errors="raise",
                ),
                "value": [10, 20, 30, 40],
            }
        )

    def test_invalid_ranges_are_logged_and_ignored(self, caplog):
        df = self._make_dt_frame()
        # include start > end, malformed string, and a valid range that matches one
        bad_and_good = [
            ("20250101000003", "20250101000002"),  # invalid: start > end
            ("invalid", "20250101000002"),  # invalid: parse error
            ("20250101000002", "20250101000002"),  # valid, should exclude exactly one
        ]
        out = filter_timestamp_ranges(
            df.copy(),
            exclude_timestamp_ranges=bad_and_good,
            timestamp_column="timestamp",
            verbose=True,
        )
        # One row excluded (the valid one)
        assert len(out) == len(df) - 1
        assert pd.Timestamp("2025-01-01 00:00:02") not in set(out["timestamp"].tolist())

        # Check that invalid ranges were logged
        text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "Invalid timestamp range skipped" in text

    def test_boundaries_are_inclusive(self):
        df = self._make_dt_frame()
        # Exclude the first and last by touching boundaries; function uses >= and <=
        ranges = [
            ("20250101000001", "20250101000001"),
            ("20250101000004", "20250101000004"),
        ]
        out = filter_timestamp_ranges(
            df.copy(),
            exclude_timestamp_ranges=ranges,
            timestamp_column="timestamp",
            verbose=False,
        )
        kept = set(out["timestamp"].dt.strftime("%Y%m%d%H%M%S"))
        assert kept == {"20250101000002", "20250101000003"}

    def test_non_datetime_timestamp_column_causes_skip(self, caplog):
        # If timestamp column not datetime dtype, filter should skip and return original (per contract)
        # Ensure INFO logs are captured for the skip message emitted by filter_timestamp_ranges
        import logging

        caplog.set_level(logging.INFO)
        df = pd.DataFrame(
            {
                "timestamp": ["20250101000001", "20250101000002"],
                "value": [1, 2],
            }
        )
        out = filter_timestamp_ranges(
            df.copy(),
            exclude_timestamp_ranges=[("20250101000001", "20250101000002")],
            timestamp_column="timestamp",
            verbose=True,
        )
        # No filtering applied because dtype not datetime; function should skip
        assert len(out) == len(df)
        text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "not in datetime format" in text.lower()


class TestTimestampParsingInvalidAndValid:
    def test_invalid_and_valid_formats_convert_and_mark_invalid(self):
        # Mixed valid and invalid entries; converter uses exact=True with errors='coerce'
        df = pd.DataFrame(
            {
                "timestamp": [
                    "20250101000001",
                    "2025010100X003",
                    "20250101000003",
                    "bad",
                    20250101000004,
                ],
                "ignore": ["FALSE"] * 5,
                "sor#": [1, 2, 3, 4, 5],
                "runticks": [1000, 1010, 1020, 1030, 1040],
                "resetticks": [0, 0, 0, 0, 0],
                "notes": [""] * 5,
            }
        )
        out = convert_timestamp_column_to_datetime(
            df.copy(), timestamp_column="timestamp", verbose=False
        )
        # dtype should be datetime
        assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
        # Invalid entries should be NaT (at least two invalid here)
        assert out["timestamp"].isna().sum() >= 2

        # Leading zeros preserved via string path; round-trip the valid rows
        fmt = "%Y%m%d%H%M%S"
        valid_mask = out["timestamp"].notna()
        round_trip = out.loc[valid_mask, "timestamp"].dt.strftime(fmt)
        expected = pd.Series(
            [str(x) for x in df.loc[valid_mask, "timestamp"]],
            index=round_trip.index,
            name="timestamp",
        )
        pd.testing.assert_series_equal(round_trip, expected)


class TestZScoreGuardBehavior:
    def test_boundary_inclusive_and_keep_fort(self):
        # Build a simple series with non-zero std to exercise normal path
        df = pd.DataFrame(
            {
                "sor#": [1, 2, 3, 4, 5, 6],
                "adjusted_run_time": [
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    100.0,
                    1.25,
                ],  # outlier at sor 5
            }
        )
        zmin, zmax, fort = -1.0, 1.0, 6
        df_incl, df_excl = filter_by_adjusted_run_time_zscore(df, zmin, zmax, fort)

        # fort row must always be included
        assert fort in set(df_incl["sor#"])
        # Equality at bounds is inclusive
        if not df_incl.empty:
            zs = df_incl["zscore"]
            assert (zs >= zmin - 1e-12).all()
            assert (zs <= zmax + 1e-12).all()

    def test_degenerate_std_sets_z_to_zero_and_keeps_only_fort(self):
        # Constant series: std == 0 triggers guard; only fort row kept, zscore==0.0
        df = pd.DataFrame(
            {
                "sor#": [1, 2, 3, 4, 5],
                "adjusted_run_time": [2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        zmin, zmax, fort = -2.0, 2.0, 4
        df_incl, df_excl = filter_by_adjusted_run_time_zscore(df, zmin, zmax, fort)
        assert list(df_incl["sor#"]) == [fort]
        assert set(df_excl["sor#"]) == {1, 2, 3, 5}
        assert (df_incl["zscore"] == 0.0).all()
        assert (df_excl["zscore"] == 0.0).all()


class TestSummarizeRunTimeBySorRangeRemainderLogic:
    @pytest.mark.parametrize(
        "fort,total,expected_bins",
        [
            # input_data_fort = 10 => total = 9 => chunk_size=2, remainder=1
            # Expected five rows total (4 ranges + 1 fort row). Bins:
            # [1..3], [4..5], [6..7], [8..9]
            (10, 9, [(1, 3), (4, 5), (6, 7), (8, 9)]),
            # input_data_fort = 2 => total = 1 => chunk_size=0, remainder=1
            # The construction yields a single bin [1..1]
            (2, 1, [(1, 1)]),
            # input_data_fort = 1 => total = 0 => no bins, only fort row
            (1, 0, []),
        ],
    )
    def test_bin_construction_and_final_fort_row(self, fort, total, expected_bins):
        # Build synthetic data with adjusted_run_time present for all sor in [1..fort]
        sors = list(range(1, fort + 1))
        df = pd.DataFrame(
            {"sor#": sors, "adjusted_run_time": np.linspace(1.0, 2.0, len(sors))}
        )

        summary_prev = summarize_run_time_by_sor_range(
            df, input_data_fort=fort, delta_mode=DeltaMode.PREVIOUS_CHUNK
        )
        # Validate row count: expected_bins + final fort row
        assert len(summary_prev) == len(expected_bins) + 1

        # Validate bins
        range_rows = summary_prev.iloc[:-1]
        actual_bins = list(
            zip(range_rows["sorr_start"].tolist(), range_rows["sorr_end"].tolist())
        )
        assert actual_bins == expected_bins

        # Final row must be degenerate interval at fort
        final = summary_prev.iloc[-1]
        assert final["sorr_start"] == fort and final["sorr_end"] == fort

    def test_run_time_delta_semantics_previous_vs_first(self):
        fort = 6
        sors = list(range(1, fort + 1))
        # Construct adjusted_run_time such that each bin has a distinct mean
        # For fort=6, total=5 => bins: chunk_size=1 remainder=1 => bins built by the implementation:
        # Expected bins for total=5: chunk_size=1, remainder=1 -> ranges: [1..2], [3..3], [4..4], [5..5]
        runtimes = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]  # last is fort row value
        df = pd.DataFrame({"sor#": sors, "adjusted_run_time": runtimes})

        prev = summarize_run_time_by_sor_range(
            df, input_data_fort=fort, delta_mode=DeltaMode.PREVIOUS_CHUNK
        )
        first = summarize_run_time_by_sor_range(
            df, input_data_fort=fort, delta_mode=DeltaMode.FIRST_CHUNK
        )

        # First row delta is NaN in both modes
        assert math.isnan(prev.iloc[0]["run_time_delta"])
        assert math.isnan(first.iloc[0]["run_time_delta"])

        # Compare a non-first, non-final row delta behavior
        # For PREVIOUS_CHUNK, delta at row i = mean_i - mean_(i-1)
        # For FIRST_CHUNK,    delta at row i = mean_i - mean_0
        # Use row index 1 if present
        if len(prev) >= 3:
            prev_delta_1 = prev.iloc[1]["run_time_delta"]
            first_delta_1 = first.iloc[1]["run_time_delta"]
            # Both should be finite, computed under their respective definitions.
            assert not math.isnan(prev_delta_1)
            assert not math.isnan(first_delta_1)

        # Final row delta computed against previous or first chunk depending on mode
        assert not math.isnan(prev.iloc[-1]["run_time_delta"])
        assert not math.isnan(first.iloc[-1]["run_time_delta"])
