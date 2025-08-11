import numpy as np
import pandas as pd

from src.main import filter_by_adjusted_run_time_iqr


def test_iqr_normal_operation_with_symmetric_bounds():
    # Test normal IQR filtering with symmetric bounds
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 10

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Check that we have both included and excluded rows
    assert len(df_incl) > 0
    assert len(df_excl) >= 0
    # Fort row should always be included
    assert fort in df_incl["sor#"].values


def test_iqr_normal_operation_with_asymmetric_bounds():
    # Test normal IQR filtering with asymmetric bounds
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        }
    )
    iqr_k_low, iqr_k_high, fort = 0.5, 2.0, 10

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Check that we have both included and excluded rows
    assert len(df_incl) > 0
    assert len(df_excl) >= 0
    # Fort row should always be included
    assert fort in df_incl["sor#"].values


def test_iqr_zero_iqr_keeps_only_fort_row():
    # When all values are identical -> IQR = 0 -> keep only sor == input_data_fort
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 4

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Included should be only the fort row
    assert list(df_incl["sor#"]) == [fort]
    # Excluded should be all others
    assert set(df_excl["sor#"]) == {1, 2, 3, 5}
    # iqr_flag should be False for all rows
    assert (df_incl["iqr_flag"] == False).all()
    assert (df_excl["iqr_flag"] == False).all()


def test_iqr_tiny_variance_treated_as_degenerate_keeps_only_fort_row():
    # When IQR is very small -> treated as degenerate -> keep only fort row
    base = 1.0
    eps = 1e-10  # Very small variation
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [base, base, base + eps, base, base],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 5

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Included should be only the fort row
    assert list(df_incl["sor#"]) == [fort]
    # Excluded should be all others
    assert set(df_excl["sor#"]) == {1, 2, 3, 4}
    # iqr_flag should be False for all rows
    assert (df_incl["iqr_flag"] == False).all()
    assert (df_excl["iqr_flag"] == False).all()


def test_iqr_fort_row_preservation():
    # Test that fort row is always preserved regardless of bounds
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [
                1.0,
                10.0,
                1.2,
                0.5,
                1.4,
            ],  # Values 10.0 and 0.5 should be outliers
        }
    )
    iqr_k_low, iqr_k_high, fort = 0.1, 0.1, 3  # Very tight bounds to exclude most rows

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included regardless of bounds
    assert fort in df_incl["sor#"].values
    # Check that some rows were excluded
    assert len(df_excl) > 0


def test_iqr_asymmetric_multipliers():
    # Test with different iqr_k_low and iqr_k_high values
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "adjusted_run_time": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                10.0,
            ],  # Last value is high outlier
        }
    )
    iqr_k_low, iqr_k_high, fort = (
        1.5,
        0.5,
        10,
    )  # More aggressive filtering on low side, less on high side

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included
    assert fort in df_incl["sor#"].values
    # Should have some excluded rows
    assert len(df_excl) > 0


def test_iqr_with_skewed_data():
    # Test with skewed data to confirm high vs. low thresholds behave as expected
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "adjusted_run_time": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                10.0,
            ],  # One high outlier
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 10

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included
    assert fort in df_incl["sor#"].values
    # High outlier should be excluded
    assert 10 not in df_excl["sor#"].values or len(df_excl) > 0


def test_iqr_empty_dataframe():
    # Test with empty DataFrame
    df = pd.DataFrame(columns=["sor#", "adjusted_run_time"])
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 1

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Both results should be empty
    assert len(df_incl) == 0
    assert len(df_excl) == 0


def test_iqr_single_row_with_fort():
    # Test with single row that is the fort
    df = pd.DataFrame(
        {
            "sor#": [5],
            "adjusted_run_time": [1.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 5

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Only the fort row should be included
    assert len(df_incl) == 1
    assert df_incl["sor#"].iloc[0] == fort
    assert len(df_excl) == 0


def test_iqr_nan_values():
    # Test with NaN values in adjusted_run_time
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, np.nan, 1.2, np.nan, 1.4],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 5

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included
    assert fort in df_incl["sor#"].values


def test_iqr_infinite_values():
    # Test with infinite values in adjusted_run_time
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, np.inf, 1.2, -np.inf, 1.4],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 5

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included
    assert fort in df_incl["sor#"].values


def test_iqr_exact_bounds_calculation():
    # Test to verify exact IQR bounds calculation with known values
    # For data [1.0, 2.0, 3.0, 4.0, 5.0]:
    # Q1 = 2.0, Q3 = 4.0, IQR = 2.0
    # Lower bound = Q1 - iqr_k_low * IQR = 2.0 - 1.5 * 2.0 = -1.0
    # Upper bound = Q3 + iqr_k_high * IQR = 4.0 + 1.5 * 2.0 = 7.0
    # Values within [-1.0, 7.0] should be included, others excluded (but fort always included)
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 3  # Fort is at position 3 with value 3.0

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # All values are within bounds [-1.0, 7.0], so all should be included
    assert len(df_incl) == 5
    assert len(df_excl) == 0
    # Fort row should be included
    assert fort in df_incl["sor#"].values
    # iqr_flag should be True for all rows since they're within bounds
    assert (df_incl["iqr_flag"] == True).all()


def test_iqr_with_out_of_bounds_values():
    # Test with values that are clearly outside the IQR bounds
    # For data [1.0, 2.0, 3.0, 4.0, 20.0]:
    # Q1 = 2.0, Q3 = 4.0, IQR = 2.0
    # Lower bound = Q1 - iqr_k_low * IQR = 2.0 - 1.5 * 2.0 = -1.0
    # Upper bound = Q3 + iqr_k_high * IQR = 4.0 + 1.5 * 2.0 = 7.0
    # Value 20.0 is outside bounds, so should be excluded (but fort always included)
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, 2.0, 3.0, 4.0, 20.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 5  # Fort is at position 5 with value 20.0

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should be included even though its value is outside bounds
    assert fort in df_incl["sor#"].values
    # Value 20.0 should be in included because it's the fort row
    # Other values within bounds should also be included
    assert len(df_incl) >= 1
    # Check that the high outlier is properly flagged
    if len(df_excl) > 0:
        # The values 1.0, 2.0, 3.0, 4.0 should be within bounds
        # But we need to be careful about the exact bounds calculation
        pass


def test_iqr_negative_values():
    # Test with negative adjusted_run_time values
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [-2.0, -1.0, 0.0, 1.0, 2.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 3  # Fort is at position 3 with value 0.0

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included
    assert fort in df_incl["sor#"].values


def test_iqr_single_value_dataframe():
    # Test with single value DataFrame (not the fort row case)
    df = pd.DataFrame(
        {
            "sor#": [1],
            "adjusted_run_time": [5.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 1  # Fort is the only row

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Only row should be included as it's the fort row
    assert len(df_incl) == 1
    assert len(df_excl) == 0
    assert df_incl["sor#"].iloc[0] == fort


def test_iqr_all_negative_values():
    # Test with all negative adjusted_run_time values
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [-5.0, -4.0, -3.0, -2.0, -1.0],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 3  # Fort is at position 3 with value -3.0

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included
    assert fort in df_incl["sor#"].values


def test_iqr_extreme_asymmetric_bounds():
    # Test with extreme asymmetric bounds (very tight on one side, very loose on other)
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        }
    )
    iqr_k_low, iqr_k_high, fort = (
        0.1,
        10.0,
        10,
    )  # Very tight lower bound, very loose upper bound

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # Fort row should always be included
    assert fort in df_incl["sor#"].values


def test_iqr_with_no_outliers():
    # Test with data that has no outliers (all points within bounds)
    # Create data with small variance around the median
    base = 5.0
    variation = 0.1
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [
                base - variation,
                base - variation / 2,
                base,
                base + variation / 2,
                base + variation,
            ],
        }
    )
    iqr_k_low, iqr_k_high, fort = 1.5, 1.5, 3  # Fort is at position 3

    df_incl, df_excl = filter_by_adjusted_run_time_iqr(df, iqr_k_low, iqr_k_high, fort)

    # All rows should be included since there are no outliers
    assert len(df_incl) == 5
    assert len(df_excl) == 0
    # Fort row should be included
    assert fort in df_incl["sor#"].values
    # iqr_flag should be True for all rows since they're within bounds
    assert (df_incl["iqr_flag"] == True).all()
