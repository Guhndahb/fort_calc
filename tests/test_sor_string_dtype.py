import pandas as pd

from src.main import (
    filter_by_adjusted_run_time_iqr,
    filter_by_adjusted_run_time_zscore,
)


def test_zscore_preserves_fort_with_string_sor():
    # sor# column is string-typed; fort is numeric 100
    df = pd.DataFrame(
        {
            "sor#": ["1", "2", "3", "100"],
            "adjusted_run_time": [1.0, 1.1, 1.2, 10.0],
        }
    )
    zmin, zmax, fort = -2.0, 2.0, 100

    df_incl, df_excl = filter_by_adjusted_run_time_zscore(df, zmin, zmax, fort)

    # Fort row must be preserved even when 'sor#' is a string
    assert fort in df_incl["sor#"].astype(int).values


def test_iqr_preserves_fort_with_string_sor():
    # sor# column is string-typed; fort is numeric 100
    df = pd.DataFrame(
        {
            "sor#": ["1", "2", "3", "100"],
            "adjusted_run_time": [1.0, 1.1, 1.2, 10.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 100

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row must be preserved even when 'sor#' is a string
    assert fort in df_incl["sor#"].astype(int).values
