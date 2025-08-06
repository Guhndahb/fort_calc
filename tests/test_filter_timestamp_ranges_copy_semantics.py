import warnings

import pandas as pd

from src.main import filter_timestamp_ranges


def test_filter_timestamp_ranges_returns_independent_copy_without_settingwithcopy_warning():
    # Construct a frame with datetime timestamps to satisfy the filter's dtype check
    ts_fmt = "%Y%m%d%H%M%S"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "20250101000001",
                    "20250101000002",
                    "20250102000003",
                    "20250103000004",
                ],
                format=ts_fmt,
                errors="raise",
            ),
            "value": [10, 20, 30, 40],
        }
    )

    # Exclude a specific range that matches exactly one row
    # e.g., exclude the second timestamp only
    exclude_ranges = [("20250101000002", "20250101000002")]

    # Perform filtering
    filtered_df = filter_timestamp_ranges(
        df.copy(),
        exclude_timestamp_ranges=exclude_ranges,
        timestamp_column="timestamp",
        verbose=False,
    )

    # Ensure the excluded row is removed
    assert len(filtered_df) == len(df) - 1
    assert pd.Timestamp("2025-01-01 00:00:02") not in set(
        filtered_df["timestamp"].tolist()
    )

    # Now perform a downstream column assignment that would raise a SettingWithCopy warning
    # if filtered_df were a view rather than an independent copy.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        filtered_df["new_col"] = 1  # this should not emit SettingWithCopyWarning

        # Assert no SettingWithCopy warnings were emitted
        setting_with_copy_warnings = [
            warn
            for warn in w
            if issubclass(warn.category, pd.errors.SettingWithCopyWarning)
        ]
        assert len(setting_with_copy_warnings) == 0, (
            "SettingWithCopyWarning was emitted on assignment to filtered_df"
        )

    # Also verify that mutation on filtered_df did not affect the original df
    assert "new_col" not in df.columns
