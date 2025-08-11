import numpy as np
import pandas as pd

from src.main import (
    compute_adjusted_run_time,
    filter_by_adjusted_run_time_iqr,
    filter_by_adjusted_run_time_zscore,
)


def test_compute_adjusted_run_time_sanitizes_and_drops():
    """
    Verify compute_adjusted_run_time converts non-finite adjusted_run_time
    values (±inf) to NaN and drops rows with invalid adjusted_run_time
    per the chosen pipeline policy.
    """
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3],
            # Row 2 has an infinite runticks => adjusted_run_time will be ±inf
            "runticks": [1000, np.inf, 3000],
            "resetticks": [0, 0, 0],
        }
    )

    out = compute_adjusted_run_time(df, ignore_resetticks=True)

    # All adjusted_run_time values in the returned frame must be finite and numeric
    assert "adjusted_run_time" in out.columns
    assert np.isfinite(out["adjusted_run_time"]).all()

    # The row that produced ±inf should have been removed
    assert 2 not in out["sor#"].values
    # Remaining rows count should be 2
    assert len(out) == 2


def test_filters_ignore_nonfinite_in_stats_and_preserve_fort():
    """
    Verify filter functions compute statistics ignoring non-finite values
    in the non-fort set, and that the fort row is preserved.
    This constructs a frame with an infinite non-fort value and calls the
    filters directly to ensure they handle it robustly.
    """
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4],
            "adjusted_run_time": [1.0, np.inf, 2.0, 100.0],
        }
    )
    fort = 4

    # IQR filter should not crash and should preserve the fort row
    df_incl_iqr, df_excl_iqr = filter_by_adjusted_run_time_iqr(
        df, iqr_k_low=1.5, iqr_k_high=1.5, input_data_fort=fort
    )
    assert fort in df_incl_iqr["sor#"].values
    # Filters should not leave infinities in returned frames (they should be handled/skipped)
    assert not np.isinf(df_incl_iqr["adjusted_run_time"]).any()
    assert not np.isinf(df_excl_iqr["adjusted_run_time"]).any()

    # Z-score filter should behave similarly
    df_incl_z, df_excl_z = filter_by_adjusted_run_time_zscore(
        df, zscore_min=-3.0, zscore_max=3.0, input_data_fort=fort
    )
    assert fort in df_incl_z["sor#"].values
    assert not np.isinf(df_incl_z["adjusted_run_time"]).any()
    assert not np.isinf(df_excl_z["adjusted_run_time"]).any()
