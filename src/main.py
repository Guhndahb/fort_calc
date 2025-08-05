#!/usr/bin/env python3
"""
FORT Calculator
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def read_csv_range(
    file_path: Path, start_line: int, end_line: int = None
) -> pd.DataFrame:
    """
    Read a specific range of lines from a CSV file.

    Args:
        file_path (Path): Path to the CSV file
        start_line (int): Starting line number (1-indexed, including header)
        end_line (int, optional): Ending line number (1-indexed). If None, reads to end

    Returns:
        pd.DataFrame: DataFrame containing the specified range of rows
    """
    try:
        # Read the CSV file with specified range
        df = pd.read_csv(
            file_path,
            skiprows=range(1, start_line),
            nrows=end_line - start_line if end_line else None,
        )

        return df

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()  # Return an empty DataFrame on error
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV range: {e}")
        return pd.DataFrame()


def clean_ignore(df: pd.DataFrame, input_data_fort: int) -> pd.DataFrame:
    return (
        df.assign(
            ignore=lambda d: d["ignore"].astype(str).str.strip().str.upper().eq("TRUE"),
            sor_valid=lambda d: pd.to_numeric(
                d["sor#"], errors="coerce"
            ),  # convert safely
        )
        .dropna(subset=["runticks", "sor_valid"])
        .loc[lambda d: ~d["ignore"]]
        .loc[lambda d: (d["sor_valid"] >= 1) & (d["sor_valid"] <= input_data_fort)]
        .drop(columns=["sor_valid"])  # clean up helper column
    )


def fill_first_note_if_empty(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if pd.isna(df.iloc[0]["notes"]) or df.iloc[0]["notes"] == "":
        df.at[df.index[0], "notes"] = "<DATA START>"
    return df


def compute_adjusted_run_time(df, ignore_mrt):
    if ignore_mrt:
        # No need to touch resetticks at all
        df["adjusted_run_time"] = df["runticks"] / 1000.0
    else:
        # Clean resetticks only if we're going to use it
        df["resetticks"] = pd.to_numeric(df["resetticks"], errors="coerce").fillna(0)
        df["adjusted_run_time"] = (df["runticks"] - df["resetticks"]) / 1000.0
    return df


# def validate_sor_column(df: pd.DataFrame, max_sor: int) -> None:
#     if not pd.api.types.is_integer_dtype(df["sor#"]):
#         if not df["sor#"].dropna().apply(lambda x: isinstance(x, int)).all():
#             raise ValueError("Column 'sor#' contains non-integer values.")

#     if (df["sor#"] < 1).any():
#         raise ValueError("Column 'sor#' contains values less than 1.")

#     if (df["sor#"] > max_sor).any():
#         raise ValueError(f"Column 'sor#' contains values greater than {max_sor}.")


def filter_by_adjusted_run_time_zscore(
    df: pd.DataFrame, zscore_min: float, zscore_max: float, input_data_fort: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    zscores = (df["adjusted_run_time"] - df["adjusted_run_time"].mean()) / df[
        "adjusted_run_time"
    ].std(ddof=1)
    df["zscore"] = zscores

    # Mask for rows within z-score bounds
    mask_zscore = (zscores >= zscore_min) & (zscores <= zscore_max)
    # Mask for rows where sor# == input_data_fort (always keep)
    mask_keep_sor = df["sor#"] == input_data_fort

    # Combine masks: keep rows that pass z-score OR have sor# == input_data_fort
    mask = mask_zscore | mask_keep_sor

    df_filtered = df[mask].copy()
    df_excluded = df[~mask].copy()
    return df_filtered, df_excluded


# Recalculate output assuming df_range and input_data_fort are still in memory
def summarize_run_time_by_sor_range(
    df_range: pd.DataFrame, input_data_fort: int
) -> pd.DataFrame:
    ranges = []
    chunk_size = (input_data_fort - 1) // 4
    remainder = (input_data_fort - 1) % 4

    start = 1
    for i in range(4):
        end = start + chunk_size - 1
        if i < remainder:
            end += 1
        ranges.append((start, end))
        start = end + 1

    # Final row for input_data_fort
    ranges.append((input_data_fort, None))

    output_rows = []
    for i, (start, end) in enumerate(ranges):
        if end is None:
            mask = df_range["sor#"] == start
        else:
            mask = df_range["sor#"].between(start, end)

        mean_runtime = df_range.loc[mask, "adjusted_run_time"].mean()
        if i == 0:
            delta = np.nan
        else:
            # delta = mean_runtime - output_rows[-1][2]
            delta = mean_runtime - output_rows[0][2]

        output_rows.append([start, end, mean_runtime, delta])

    return pd.DataFrame(
        output_rows,
        columns=["sorr_start", "sorr_end", "run_time_mean", "run_time_delta"],
    )


def main():
    # Use pathlib for cross-platform path handling
    log_path = Path("./samples/log-reset-01.csv").resolve()

    # Check if file exists before reading
    if not log_path.exists():
        print(f"Error: CSV file not found at {log_path}")
        return

    try:
        # Read the CSV file once
        df_full = pd.read_csv(log_path)

        # Get total lines directly from the DataFrame
        total_lines = len(df_full)
        print(f"Total lines in CSV: {total_lines}")

        start_line = 0
        end_line = None
        zscore_min = -100.5
        zscore_max = 100
        input_data_fort = 100
        ignore_mrt = True

        if start_line > total_lines:
            raise ValueError(
                f"Start line {start_line} exceeds total lines {total_lines}"
            )

        df_range = read_csv_range(log_path, start_line, end_line)

        if df_range.empty:
            raise ValueError("No data found in the specified range")

        print(f"\nSuccessfully loaded lines {start_line} to {total_lines}:")
        print(f"Number of rows loaded: {len(df_range)}")
        print("\nFirst few rows of the range:")
        print(df_range.head())
        print("\nLast few rows of the range:")
        print(df_range.tail())

        df_range = (
            df_range.pipe(clean_ignore, input_data_fort=input_data_fort)
            .pipe(fill_first_note_if_empty)
            .pipe(compute_adjusted_run_time, ignore_mrt=ignore_mrt)
        )

        # validate_sor_column(df_range, input_data_fort)

        df_range, df_excluded = filter_by_adjusted_run_time_zscore(
            df_range, zscore_min, zscore_max, input_data_fort
        )

        print("\nIncluded range:")
        print(df_range)

        print("\nExcluded range:")
        print(df_excluded)

        if len(df_range) < 5:
            raise ValueError(
                f"Too few rows remaining after filtering (found {len(df_range)}, need at least 5)"
            )

        df_summary = summarize_run_time_by_sor_range(df_range, input_data_fort)

        # Get offline_cost from the last row of df_output
        offline_cost = df_summary.iloc[-1]["run_time_delta"]
        print("Run Time Summary by sor# Range (Updated):")
        print(df_summary)

        # Enhanced regression analysis using statsmodels
        df_results, regression_diagnostics = enhanced_regression_analysis(
            df_range, input_data_fort
        )

        # Add cumulative sum columns
        df_results["sum_lin"] = df_results["linear_model_output"].cumsum()
        df_results["sum_quad"] = df_results["quadratic_model_output"].cumsum()

        # Add cost per run columns
        df_results["cost_per_run_at_fort_lin"] = (
            df_results["sum_lin"] + offline_cost
        ) / df_results["sor#"]
        df_results["cost_per_run_at_fort_quad"] = (
            df_results["sum_quad"] + offline_cost
        ) / df_results["sor#"]

        print(df_results)

        # Calculate the sor# with the lowest cost per run
        sor_min_cost_lin = df_results.loc[
            df_results["cost_per_run_at_fort_lin"].idxmin(), "sor#"
        ]
        sor_min_cost_quad = df_results.loc[
            df_results["cost_per_run_at_fort_quad"].idxmin(), "sor#"
        ]

        print(regression_diagnostics["linear"]["Summary"])
        print(regression_diagnostics["quadratic"]["Summary"])

        print(f"Minimum cost per run at fort (linear): sor# {sor_min_cost_lin}")
        print(f"Minimum cost per run at fort (quadratic): sor# {sor_min_cost_quad}")

    except FileNotFoundError:
        print(f"Error: The file {log_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"Error processing CSV file: {e}")


def enhanced_regression_analysis(df_range, input_data_fort):
    # Generate sequence from 1 to input_data_fort
    sor_sequence = np.arange(1, input_data_fort + 1)

    # Linear regression
    X_linear = sm.add_constant(df_range["sor#"])
    linear_model = sm.OLS(df_range["adjusted_run_time"], X_linear).fit()
    linear_model_output = linear_model.predict(sm.add_constant(sor_sequence))

    # Quadratic regression
    df_range["sor#_squared"] = df_range["sor#"] ** 2
    X_quadratic = sm.add_constant(df_range[["sor#", "sor#_squared"]])
    quadratic_model = sm.OLS(df_range["adjusted_run_time"], X_quadratic).fit()
    quadratic_model_output = quadratic_model.predict(
        sm.add_constant(np.column_stack((sor_sequence, sor_sequence**2)))
    )

    # Create a new DataFrame with the required columns
    result_df = (
        pd.DataFrame(
            {
                "sor#": sor_sequence,
                "linear_model_output": linear_model_output,
                "quadratic_model_output": quadratic_model_output,
            }
        )
        .sort_values(by="sor#")
        .reset_index(drop=True)
    )

    # Diagnostics
    diagnostics = {
        "linear": {
            "R-squared": linear_model.rsquared,
            "Adj. R-squared": linear_model.rsquared_adj,
            "F-statistic": linear_model.fvalue,
            "p-value": linear_model.f_pvalue,
            "Coefficients": linear_model.params,
            "Standard Errors": linear_model.bse,
            "Confidence Intervals": linear_model.conf_int(),
            "Residuals": linear_model.resid,
            "Summary": linear_model.summary(),
        },
        "quadratic": {
            "R-squared": quadratic_model.rsquared,
            "Adj. R-squared": quadratic_model.rsquared_adj,
            "F-statistic": quadratic_model.fvalue,
            "p-value": quadratic_model.f_pvalue,
            "Coefficients": quadratic_model.params,
            "Standard Errors": quadratic_model.bse,
            "Confidence Intervals": quadratic_model.conf_int(),
            "Residuals": quadratic_model.resid,
            "Summary": quadratic_model.summary(),
        },
    }

    return result_df, diagnostics


if __name__ == "__main__":
    main()
