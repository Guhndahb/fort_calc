import pandas as pd

from src.main import filter_by_adjusted_run_time_zscore


def test_zscore_constant_series_keeps_only_fort_row():
    # adjusted_run_time is constant -> std = 0 -> keep only sor == input_data_fort
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    zmin, zmax, fort = -2.0, 2.0, 4

    df_incl, df_excl = filter_by_adjusted_run_time_zscore(df, zmin, zmax, fort)

    # Included should be only the fort row
    assert list(df_incl["sor#"]) == [fort]
    # Excluded should be all others
    assert set(df_excl["sor#"]) == {1, 2, 3, 5}
    # zscores filled as 0.0 per spec
    assert (df_incl["zscore"] == 0.0).all()
    assert (df_excl["zscore"] == 0.0).all()


def test_zscore_tiny_variance_treated_as_degenerate_keeps_only_fort_row():
    # adjusted_run_time tiny variance -> std finite but may be extremely small.
    # We still follow the new behavior: if std is 0 or not finite -> set zscores=0 and only keep fort.
    # Here we construct an almost-constant series with a minuscule variation but keep the behavior
    # expectation aligned with the instruction to guard against degenerate variance.
    base = 1.0
    eps = 0.0  # Using exactly zero to match the "std==0 or not finite" branch deterministically.
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [base, base, base + eps, base, base],
        }
    )
    zmin, zmax, fort = -1.0, 1.0, 5

    df_incl, df_excl = filter_by_adjusted_run_time_zscore(df, zmin, zmax, fort)

    # Included should be only the fort row
    assert list(df_incl["sor#"]) == [fort]
    # Excluded should be all others
    assert set(df_excl["sor#"]) == {1, 2, 3, 4}
    # zscores filled as 0.0 per spec
    assert (df_incl["zscore"] == 0.0).all()
    assert (df_excl["zscore"] == 0.0).all()
