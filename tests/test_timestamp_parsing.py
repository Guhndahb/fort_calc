import pandas as pd
import pytest

from src.main import (
    DeltaMode,
    TransformParams,
    convert_timestamp_column_to_datetime,
    transform_pipeline,
)


def make_base_params():
    return TransformParams(
        zscore_min=-2.0,
        zscore_max=2.0,
        input_data_fort=6,
        ignore_mrt=True,
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
        exclude_timestamp_ranges=None,
        verbose_filtering=False,
        fail_on_any_invalid_timestamps=True,
    )


def test_parsing_uses_exact_and_errors_coerce_and_preserves_leading_zeros_string_path():
    # Valid timestamps as strings including one with leading zero in month/day/hour
    df = pd.DataFrame(
        {
            "timestamp": ["20250101000001", "20250101000002", "20240102030405"],
            "ignore": ["FALSE", "FALSE", "FALSE"],
            "sor#": [
                1,
                2,
                6,
            ],  # include 'input_data_fort' row to satisfy later zscore step if used
            "runticks": [1000, 1010, 1020],
            "resetticks": [0, 0, 0],
            "notes": ["", "", ""],
        }
    )

    # Convert only; do not run entire pipeline to isolate behavior
    out = convert_timestamp_column_to_datetime(
        df.copy(), timestamp_column="timestamp", verbose=False
    )
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    # exact=True would reject malformed length or pattern; these should be converted
    assert out["timestamp"].isna().sum() == 0

    # Round-trip to verify exact parsing (format is precise)
    fmt = "%Y%m%d%H%M%S"
    round_trip = out["timestamp"].dt.strftime(fmt)
    pd.testing.assert_series_equal(
        round_trip.reset_index(drop=True),
        pd.Series(df["timestamp"]).reset_index(drop=True),
    )


def test_numeric_input_is_parsed_via_string_path_to_preserve_leading_zeros_and_exact_matching():
    # Mix of numeric-looking inputs; if parsed via numeric dtype then cast to str
    # All valid; length 14 exact
    df = pd.DataFrame(
        {
            "timestamp": [20250101000001, 20250101000002, 20250101000003],
            "ignore": ["FALSE", "FALSE", "FALSE"],
            "sor#": [1, 2, 6],
            "runticks": [1000, 1010, 1020],
            "resetticks": [0, 0, 0],
            "notes": ["", "", ""],
        }
    )

    out = convert_timestamp_column_to_datetime(
        df.copy(), timestamp_column="timestamp", verbose=False
    )
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    assert out["timestamp"].isna().sum() == 0

    fmt = "%Y%m%d%H%M%S"
    # Ensure string conversion used exact format; round-trip equals numeric->string of original
    # Note: numeric had no leading zeros at head, but exact formatting is still enforced.
    round_trip = out["timestamp"].dt.strftime(fmt)
    expected = pd.Series([str(x) for x in df["timestamp"]], name="timestamp")
    pd.testing.assert_series_equal(
        round_trip.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_errors_coerce_marks_invalid_and_pipeline_fail_fast_raises():
    # Pandas with a format may parse shorter strings by padding, so to guarantee an invalid,
    # use a non-numeric character which must coerce to NaT with errors='coerce'.
    df = pd.DataFrame(
        {
            "timestamp": [
                "20250101000001",
                "2025x101000000",  # invalid due to non-numeric 'x'
                "20250101000003",
                "20250101000004",
                "20250101000005",
                "20250101000006",
            ],
            "ignore": ["FALSE"] * 6,
            "sor#": [1, 2, 3, 4, 5, 6],
            "runticks": [1000, 1010, 1020, 1030, 1040, 1050],
            "resetticks": [0, 0, 0, 0, 0, 0],
            "notes": [""] * 6,
        }
    )

    params = make_base_params()

    with pytest.raises(ValueError) as ei:
        _ = transform_pipeline(df.copy(), params)

    msg = str(ei.value)
    assert "Invalid timestamps detected" in msg or "invalid timestamps" in msg.lower()


def test_pipeline_succeeds_when_all_valid_and_creates_datetime_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": [
                "20250101000001",
                "20250101000002",
                "20250101000003",
                "20250101000004",
                "20250101000005",
                "20250101000006",
            ],
            "ignore": ["FALSE"] * 6,
            "sor#": [1, 2, 3, 4, 5, 6],
            "runticks": [1000, 1010, 1020, 1030, 1040, 1050],
            "resetticks": [0, 0, 0, 0, 0, 0],
            "notes": [""] * 6,
        }
    )

    params = make_base_params()
    out = transform_pipeline(df.copy(), params)
    # transform_pipeline returns TransformOutputs; df_range should carry parsed timestamp
    assert pd.api.types.is_datetime64_any_dtype(out.df_range["timestamp"])
    assert out.df_range["timestamp"].isna().sum() == 0
