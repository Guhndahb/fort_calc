import pandas as pd

from src.main import (
    build_manifest_dict,
    utc_timestamp_seconds,
)


def _build_test_data():
    """Build minimal test data for manifest testing."""
    # Small df_range
    df_range = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.15, 1.05],
        }
    )

    return df_range


def test_build_manifest_dict_single_plot():
    """Test manifest generation with single plot."""
    abs_input_posix = "/test/path/log.csv"
    counts = {
        "total_input_rows": 100,
        "processed_row_count": 90,
        "excluded_row_count": 10,
        "pre_zscore_excluded": 0,
    }
    effective_params = {
        "load": {
            "log_path": "/test/path/log.csv",
            "start_line": None,
            "end_line": None,
            "include_header": True,
        },
        "transform": {
            "zscore_min": -2.0,
            "zscore_max": 2.0,
            "input_data_fort": 5,
            "ignore_resetticks": True,
            "delta_mode": "PREVIOUS_CHUNK",
            "exclude_timestamp_ranges": None,
            "verbose_filtering": False,
            "fail_on_any_invalid_timestamps": True,
        },
    }
    hashes = ("testhash", "fulltesthash")
    artifact_paths = ["plot-testhash-DEFAULT.svg"]

    manifest = build_manifest_dict(
        abs_input_posix, counts, effective_params, hashes, artifact_paths
    )

    # Check manifest structure
    assert manifest["version"] == "1"
    assert "timestamp_utc" in manifest
    assert manifest["absolute_input_path"] == abs_input_posix
    assert manifest["total_input_rows"] == 100
    assert manifest["processed_row_count"] == 90
    assert manifest["excluded_row_count"] == 10
    assert "exclusion_reasons" in manifest
    assert manifest["effective_parameters"] == effective_params
    assert manifest["canonical_hash"] == "fulltesthash"
    assert manifest["canonical_hash_short"] == "testhash"
    assert manifest["artifacts"]["plot_svgs"] == artifact_paths


def test_build_manifest_dict_multi_plot():
    """Test manifest generation with multiple plots."""
    abs_input_posix = "/test/path/log.csv"
    counts = {
        "total_input_rows": 100,
        "processed_row_count": 90,
        "excluded_row_count": 10,
        "pre_zscore_excluded": 0,
    }
    effective_params = {
        "load": {
            "log_path": "/test/path/log.csv",
            "start_line": None,
            "end_line": None,
            "include_header": True,
        },
        "transform": {
            "zscore_min": -2.0,
            "zscore_max": 2.0,
            "input_data_fort": 5,
            "ignore_resetticks": True,
            "delta_mode": "PREVIOUS_CHUNK",
            "exclude_timestamp_ranges": None,
            "verbose_filtering": False,
            "fail_on_any_invalid_timestamps": True,
        },
    }
    hashes = ("testhash", "fulltesthash")
    artifact_paths = [
        "plot-testhash-00-DEFAULT.svg",
        "plot-testhash-01-ALL_COST.svg",
        "plot-testhash-02-ALL_WLS.svg",
    ]

    manifest = build_manifest_dict(
        abs_input_posix, counts, effective_params, hashes, artifact_paths
    )

    # Check manifest structure
    assert manifest["version"] == "1"
    assert "timestamp_utc" in manifest
    assert manifest["absolute_input_path"] == abs_input_posix
    assert manifest["total_input_rows"] == 100
    assert manifest["processed_row_count"] == 90
    assert manifest["excluded_row_count"] == 10
    assert "exclusion_reasons" in manifest
    assert manifest["effective_parameters"] == effective_params
    assert manifest["canonical_hash"] == "fulltesthash"
    assert manifest["canonical_hash_short"] == "testhash"
    assert manifest["artifacts"]["plot_svgs"] == artifact_paths


def test_manifest_timestamp_format():
    """Test that manifest timestamp is in correct format."""
    timestamp = utc_timestamp_seconds()
    # Should be ISO-8601 UTC timestamp with seconds precision and Z suffix
    assert timestamp.endswith("Z")
    assert "T" in timestamp
    # Should be parseable
    import datetime

    datetime.datetime.fromisoformat(timestamp[:-1])  # Remove Z for parsing
