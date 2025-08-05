#!/usr/bin/env python3
"""
FORT Calculator
"""

import logging
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from csv_processor import CSVRangeProcessor

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeltaMode(Enum):
    PREVIOUS_CHUNK = auto()  # delta vs. the immediately-preceding chunk
    FIRST_CHUNK = auto()  # delta vs. the very first chunk


class FilterResult:
    """Container for filter operation results and diagnostics."""

    def __init__(self, label: Optional[str] = None):
        # Identification
        self.label = label

        # Row counters
        self.original_rows: int = 0
        self.filtered_rows: int = 0
        # Keep existing field for backwards compatibility with timestamp filter
        self.excluded_ranges: int = 0
        # New: specifically for row-level filters
        self.excluded_rows: int = 0

        # Diagnostics
        self.invalid_ranges: list[tuple[str, str, str]] = []
        self.warnings: list[str] = []
        self.events: list[str] = []
        self.metrics: dict[str, int | float | str] = {}
        self.skipped_reason: Optional[str] = None

        # Timing
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    # Timing helpers
    def start(self):
        import time

        self.started_at = time.perf_counter()

    def stop(self):
        import time

        self.finished_at = time.perf_counter()
        if self.started_at is not None:
            self.elapsed_ms = (self.finished_at - self.started_at) * 1000.0

    # Logging helpers
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)

    def add_event(self, message: str):
        """Add an info-level event message."""
        self.events.append(message)
        logger.info(message)

    def add_metric(self, name: str, value):
        """Attach a named metric."""
        self.metrics[name] = value

    def log_invalid_range(self, start: str, end: str, reason: str):
        """Log an invalid timestamp range."""
        self.invalid_ranges.append((start, end, reason))
        logger.warning(f"Invalid timestamp range skipped: {start}-{end} - {reason}")

    def set_skipped(self, reason: str):
        """Mark the step as skipped with a reason."""
        self.skipped_reason = reason
        self.add_warning(f"Step skipped: {reason}")

    def summarize(self) -> str:
        """Produce a concise summary string for diagnostics."""
        lbl = f"{self.label} " if self.label else ""
        parts = [f"{lbl}result: {self.original_rows} → {self.filtered_rows}"]
        if self.excluded_rows:
            parts.append(f"excluded_rows={self.excluded_rows}")
        if self.excluded_ranges:
            parts.append(f"excluded_ranges={self.excluded_ranges}")
        if self.invalid_ranges:
            parts.append(f"invalid_ranges={len(self.invalid_ranges)}")
        if self.skipped_reason:
            parts.append(f"skipped={self.skipped_reason}")
        if self.elapsed_ms is not None:
            parts.append(f"elapsed_ms={self.elapsed_ms:.1f}")
        if self.metrics:
            parts.append(f"metrics={self.metrics}")
        return " | ".join(parts)


def clean_ignore(
    df: pd.DataFrame, input_data_fort: int, verbose: bool = False
) -> pd.DataFrame:
    """
    Filter out rows marked as ignore and rows with out-of-bounds or invalid sor#.
    Uses FilterResult for standardized diagnostics while returning a DataFrame
    to preserve pandas piping.
    """
    result = FilterResult(label="clean_ignore")
    result.start()
    result.original_rows = len(df)

    if df.empty:
        result.set_skipped("empty dataframe")
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    # Assign helper columns and drop impossible rows early
    df_work = df.assign(
        ignore=lambda d: d["ignore"].astype(str).str.strip().str.upper().eq("TRUE")
        if "ignore" in d.columns
        else False,
        sor_valid=lambda d: pd.to_numeric(d["sor#"], errors="coerce")
        if "sor#" in d.columns
        else pd.Series([pd.NA] * len(d), index=d.index),
    ).dropna(subset=["runticks", "sor_valid"], how="any")

    # Metrics prior to boolean filtering
    rows_after_dropna = len(df_work)
    result.add_metric("rows_after_dropna", rows_after_dropna)
    result.add_metric("input_data_fort", input_data_fort)

    # Warn if required columns were missing
    if "ignore" not in df.columns:
        result.add_warning("Column 'ignore' not found - assuming all rows not ignored")
    if "sor#" not in df.columns:
        result.add_warning("Column 'sor#' not found - sor bounds cannot be validated")

    # Apply filtering masks
    mask_not_ignored = ~df_work["ignore"]
    mask_sor_bounds = (df_work["sor_valid"] >= 1) & (
        df_work["sor_valid"] <= input_data_fort
    )
    df_work = df_work.loc[mask_not_ignored & mask_sor_bounds]

    result.filtered_rows = len(df_work)
    result.excluded_rows = result.original_rows - result.filtered_rows

    # Drop helper column to keep schema tidy
    if "sor_valid" in df_work.columns:
        df_work = df_work.drop(columns=["sor_valid"])

    result.stop()

    if verbose:
        logger.info(result.summarize())

    return df_work


def fill_first_note_if_empty(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Ensure the first row's 'notes' column contains a value.
    Uses FilterResult diagnostics while preserving DataFrame return for piping.
    Avoids copying; mutates in place only when needed.
    """
    result = FilterResult(label="fill_first_note_if_empty")
    result.start()
    result.original_rows = len(df)

    # Early returns without copying when no work is needed
    if df.empty:
        result.set_skipped("empty dataframe - no notes to fill")
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    if "notes" not in df.columns:
        result.set_skipped("column 'notes' not found - skipping fill")
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    # Determine if first note is empty
    first_idx = df.index[0]
    first_val = df.at[first_idx, "notes"] if first_idx in df.index else None
    is_empty = pd.isna(first_val) or (
        isinstance(first_val, str) and first_val.strip() == ""
    )
    result.add_metric("first_note_was_empty", int(bool(is_empty)))

    if not is_empty:
        # Nothing to change; keep original reference
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            result.add_event("First note already present - no changes made")
            logger.info(result.summarize())
        return df

    # Mutate in place only when needed
    df.at[first_idx, "notes"] = "<DATA START>"
    result.filtered_rows = len(df)
    result.stop()
    if verbose:
        result.add_event("Filled first empty note with '<DATA START>'")
        logger.info(result.summarize())

    return df


def compute_adjusted_run_time(
    df: pd.DataFrame, ignore_mrt: bool, verbose: bool = False
) -> pd.DataFrame:
    """
    Compute adjusted_run_time with diagnostics via FilterResult.
    Preserves DataFrame -> DataFrame contract for pandas piping.
    NOTE: Avoids df.copy() to respect in-place performance preference.

    Behavior:
    - If ignore_mrt is True: adjusted_run_time = runticks / 1000.0
    - Else: adjusted_run_time = (runticks - resetticks) / 1000.0 with resetticks coerced to numeric and NaN→0
    - Emits warnings/metrics for NaN/negative adjusted values and missing columns.
    """
    result = FilterResult(label="compute_adjusted_run_time")
    result.start()
    result.original_rows = len(df)

    # Validate required columns
    required = ["runticks"] + ([] if ignore_mrt else ["resetticks"])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        result.set_skipped(f"missing required columns: {missing_cols}")
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df  # return original if we cannot compute safely

    # Ensure runticks numeric (in-place)
    df["runticks"] = pd.to_numeric(df["runticks"], errors="coerce")

    if ignore_mrt:
        # No need to touch resetticks at all
        df["adjusted_run_time"] = df["runticks"] / 1000.0
    else:
        # Clean resetticks only if we're going to use it (in-place)
        df["resetticks"] = pd.to_numeric(df["resetticks"], errors="coerce").fillna(0)
        df["adjusted_run_time"] = (df["runticks"] - df["resetticks"]) / 1000.0

    # Diagnostics on results
    nan_count = df["adjusted_run_time"].isna().sum()
    neg_count = (df["adjusted_run_time"] < 0).sum()

    if nan_count > 0:
        result.add_warning(
            f"{nan_count} NaN adjusted_run_time values after computation"
        )
    if neg_count > 0:
        result.add_warning(
            f"{neg_count} negative adjusted_run_time values after computation"
        )

    # Track metrics
    result.add_metric("ignore_mrt", ignore_mrt)
    result.add_metric("nan_adjusted_values", int(nan_count))
    result.add_metric("negative_adjusted_values", int(neg_count))

    result.filtered_rows = len(df)
    result.stop()

    if verbose:
        logger.info(result.summarize())

    return df


def convert_timestamp_column_to_datetime(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    timestamp_format: str = "%Y%m%d%H%M%S",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert timestamp column to datetime format consistently regardless of filtering.
    Uses FilterResult diagnostics and avoids unnecessary copying by mutating in place.
    """
    result = FilterResult(label="convert_timestamp_column_to_datetime")
    result.start()
    result.original_rows = len(df)

    if timestamp_column not in df.columns:
        result.set_skipped(
            f"timestamp column '{timestamp_column}' not found - skipping conversion"
        )
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    if verbose:
        logger.info(f"Converting timestamp column '{timestamp_column}' to datetime")
        logger.info(f"Original dtype: {df[timestamp_column].dtype}")
        logger.info(f"Sample values: {df[timestamp_column].iloc[:3].tolist()}")

    # Convert timestamp column to datetime (in place)
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            # Handle both string and integer timestamps
            if df[timestamp_column].dtype in ["int64", "float64"]:
                df[timestamp_column] = pd.to_datetime(
                    df[timestamp_column].astype(str),
                    format=timestamp_format,
                    errors="coerce",
                )
            else:
                df[timestamp_column] = pd.to_datetime(
                    df[timestamp_column], format=timestamp_format, errors="coerce"
                )

        # Check for parsing failures
        invalid_timestamps = int(df[timestamp_column].isna().sum())
        if invalid_timestamps > 0:
            result.add_warning(
                f"Found {invalid_timestamps} invalid timestamps in column '{timestamp_column}'"
            )
        result.add_metric("invalid_timestamps", invalid_timestamps)
        result.add_metric("converted_dtype", str(df[timestamp_column].dtype))

        if verbose:
            logger.info(f"Converted dtype: {df[timestamp_column].dtype}")
            logger.info(
                f"Sample converted values: {df[timestamp_column].iloc[:3].tolist()}"
            )

    except Exception as e:
        result.set_skipped(f"conversion failed: {str(e)}")
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.error(
                f"Failed to convert timestamp column '{timestamp_column}' to datetime: {str(e)}"
            )
            logger.info(result.summarize())
        return df

    result.filtered_rows = len(df)
    result.stop()
    if verbose:
        logger.info(result.summarize())

    return df


def filter_timestamp_ranges(
    df: pd.DataFrame,
    exclude_timestamp_ranges: Optional[List[Tuple[str, str]]] = None,
    timestamp_column: str = "timestamp",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter out rows that fall within specified timestamp ranges with comprehensive debugging.

    Enhanced version with logging for invalid ranges and observability features.

    Args:
        df: Input DataFrame
        exclude_timestamp_ranges: List of (start, end) timestamp tuples as strings in YYYYMMDDHHMMSS format
        timestamp_column: Name of the timestamp column to filter on
        verbose: Enable detailed logging for debugging

    Returns:
        DataFrame with rows outside the specified timestamp ranges
    """
    result = FilterResult()
    result.original_rows = len(df)

    # Early return for empty DataFrame
    if df.empty:
        result.add_warning("Empty DataFrame provided - no filtering performed")
        if verbose:
            logger.info(f"Filter result: {result.__dict__}")
        return df

    # Early return for empty ranges
    if not exclude_timestamp_ranges:
        return df

    # Check if timestamp column exists
    if timestamp_column not in df.columns:
        result.add_warning(
            f"Timestamp column '{timestamp_column}' not found - no filtering performed"
        )
        if verbose:
            logger.info(f"Filter result: {result.__dict__}")
        return df

    # Process timestamp column - now expects datetime format
    df_work = df.copy()

    # Add diagnostic logging for timestamp column
    if verbose:
        logger.info(f"Timestamp column dtype: {df_work[timestamp_column].dtype}")
        logger.info(
            f"First few timestamp values: {df_work[timestamp_column].iloc[:3].tolist()}"
        )
        logger.info(f"Sample exclude ranges: {exclude_timestamp_ranges}")

    # Verify timestamp column is properly parsed as datetime
    if not pd.api.types.is_datetime64_any_dtype(df_work[timestamp_column]):
        result.add_warning(
            f"Timestamp column '{timestamp_column}' is not in datetime format - skipping filtering"
        )
        if verbose:
            logger.info(f"Filter result: {result.__dict__}")
        return df

    # Process valid timestamp ranges
    valid_ranges = []
    for start_str, end_str in exclude_timestamp_ranges:
        try:
            start_ts = pd.to_datetime(start_str, format="%Y%m%d%H%M%S")
            end_ts = pd.to_datetime(end_str, format="%Y%m%d%H%M%S")

            if verbose:
                logger.info(
                    f"Processing range: {start_str} -> {start_ts}, {end_str} -> {end_ts}"
                )

            if start_ts > end_ts:
                result.log_invalid_range(start_str, end_str, "start > end")
                continue

            valid_ranges.append((start_ts, end_ts))

        except (ValueError, TypeError) as e:
            result.log_invalid_range(start_str, end_str, str(e))
            continue

    # Early return if no valid ranges
    if not valid_ranges:
        result.add_warning("No valid timestamp ranges found - no filtering performed")
        if verbose:
            logger.info(
                f"DataFrame timestamp column sample: {df_work[timestamp_column].iloc[:5]}"
            )
            logger.info(
                f"Parsed timestamps sample: {df_work[timestamp_column].iloc[:5].dt.strftime('%Y%m%d%H%M%S')}"
            )
        return df

    # Build exclusion mask for valid ranges
    exclude_mask = pd.Series([False] * len(df_work), index=df_work.index)

    if verbose:
        logger.info(f"Valid ranges to exclude: {valid_ranges}")
        logger.info(
            f"Timestamp range in data: {df_work[timestamp_column].min()} to {df_work[timestamp_column].max()}"
        )

    for start_ts, end_ts in valid_ranges:
        range_mask = (df_work[timestamp_column] >= start_ts) & (
            df_work[timestamp_column] <= end_ts
        )
        if verbose:
            matched_count = range_mask.sum()
            logger.info(
                f"Range {start_ts} to {end_ts}: {matched_count} rows would be excluded"
            )
            if matched_count > 0:
                logger.info(
                    f"Sample excluded timestamps: {df_work[range_mask][timestamp_column].iloc[:3].tolist()}"
                )
        exclude_mask |= range_mask

    # Apply filtering
    filtered_df = df_work[~exclude_mask]
    result.filtered_rows = len(filtered_df)
    result.excluded_ranges = exclude_mask.sum()

    # Log summary
    if verbose or result.warnings:
        logger.info(
            f"Timestamp filtering completed: "
            f"{result.original_rows} → {result.filtered_rows} rows "
            f"({result.excluded_ranges} rows excluded)"
        )

        if result.invalid_ranges:
            logger.info(f"Invalid ranges skipped: {len(result.invalid_ranges)}")

        if result.warnings:
            logger.info(f"Warnings: {len(result.warnings)}")

    return filtered_df


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
    df_range: pd.DataFrame,
    input_data_fort: int,
    delta_mode: DeltaMode = DeltaMode.PREVIOUS_CHUNK,
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
            if delta_mode is DeltaMode.PREVIOUS_CHUNK:
                baseline = output_rows[-1][2]  # previous chunk
            else:  # DeltaMode.FIRST_CHUNK
                baseline = output_rows[0][2]  # first chunk
            delta = mean_runtime - baseline

        output_rows.append([start, end, mean_runtime, delta])

    return pd.DataFrame(
        output_rows,
        columns=["sorr_start", "sorr_end", "run_time_mean", "run_time_delta"],
    )


def regression_analysis(df_range, input_data_fort):
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


def calculate_fort(
    log_path,
    start_line=None,
    end_line=None,
    include_header=True,
    zscore_min=-1.5,
    zscore_max=3,
    input_data_fort=100,
    ignore_mrt=True,
    delta_mode=DeltaMode.PREVIOUS_CHUNK,
    exclude_timestamp_ranges=None,
    verbose_filtering=False,
):
    # Check if file exists before reading
    if not log_path.exists():
        print(f"Error: CSV file not found at {log_path}")
        return

    try:
        # Read the CSV file once
        # df_range = pd.read_csv(log_path)

        with CSVRangeProcessor(log_path) as processor:
            processor.print_range_info(start_line, end_line)

            # Read the range
            df_range = processor.read_range(
                start_line=start_line, end_line=end_line, include_header=include_header
            )

            if df_range.empty:
                raise ValueError("No data found in the specified range")

            print(
                f"\nSuccessfully loaded lines {start_line if start_line else 'START'} to {end_line if end_line else 'END'}:"
            )
            print(f"Number of rows loaded: {len(df_range)}")
            print("\nLoaded range:")
            print(df_range)

            df_range = (
                df_range.pipe(
                    clean_ignore,
                    input_data_fort=input_data_fort,
                    verbose=verbose_filtering,
                )
                .pipe(compute_adjusted_run_time, ignore_mrt=ignore_mrt)
                .pipe(
                    convert_timestamp_column_to_datetime,
                    timestamp_column="timestamp",
                    verbose=verbose_filtering,
                )
                .pipe(
                    filter_timestamp_ranges,
                    exclude_timestamp_ranges=exclude_timestamp_ranges,
                    verbose=verbose_filtering,
                )
                .pipe(fill_first_note_if_empty, verbose=verbose_filtering)
            )

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

            df_summary = summarize_run_time_by_sor_range(
                df_range, input_data_fort, delta_mode
            )

            # Validate that all rows have valid run_time_mean values (not NaN)
            if df_summary["run_time_mean"].isna().any():
                invalid_rows = df_summary[df_summary["run_time_mean"].isna()]
                raise ValueError(
                    f"Insufficient input data: Found {len(invalid_rows)} summary rows with no mean run time values. "
                )

            # Get offline_cost from the last row of df_output
            offline_cost = df_summary.iloc[-1]["run_time_delta"]
            print("Run Time Summary by sor# Range (Updated):")
            print(df_summary)

            # Enhanced regression analysis using statsmodels
            df_results, regression_diagnostics = regression_analysis(
                df_range,
                input_data_fort - 1,  # we do not want to include offline run
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

            # Plotting
            # Create a scatter plot
            plt.style.use("dark_background")
            plt.figure(figsize=(10, 6))
            plt.scatter(
                df_range["sor#"],
                df_range["adjusted_run_time"],
                s=20,  # Adjusted size to half the original
                color="cyan",
                label="Data Points",
            )

            # Overlay linear and quadratic models
            plt.plot(
                df_results["sor#"],
                df_results["linear_model_output"],
                color="yellow",
                label="Linear Model",
            )
            plt.plot(
                df_results["sor#"],
                df_results["quadratic_model_output"],
                color="magenta",
                label="Quadratic Model",
            )

            # Plot cost per run at fort for linear and quadratic models
            plt.plot(
                df_results["sor#"],
                df_results["cost_per_run_at_fort_lin"],
                color="green",
                linestyle="--",
                label="Cost/Run @ FORT (Linear)",
            )
            plt.plot(
                df_results["sor#"],
                df_results["cost_per_run_at_fort_quad"],
                color="blue",
                linestyle="--",
                label="Cost/Run @ FORT (Quadratic)",
            )

            # Add vertical lines for minimum cost points
            plt.axvline(
                x=sor_min_cost_lin,
                color="green",
                linestyle="--",
                label="Min Cost (Linear)",
            )
            plt.axvline(
                x=sor_min_cost_quad,
                color="blue",
                linestyle="--",
                label="Min Cost (Quadratic)",
            )

            # Add labels and legend
            plt.xlabel("Sequential Online Run #")
            plt.ylabel("Adjusted Run Time")
            plt.title("Scatterplot with Linear and Quadratic Models")
            plt.legend()

            # Save the plot as an SVG file
            plt.savefig("scatterplot.svg", format="svg")
            plt.close()

            print("Scatterplot with models saved as 'scatterplot.svg'.")

    except FileNotFoundError:
        print(f"Error: The file {log_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"Error processing CSV file: {e}")


def main():
    start_line = None
    end_line = None
    zscore_min = -1.5
    zscore_max = 3
    input_data_fort = 100
    ignore_mrt = True
    delta_mode = DeltaMode.PREVIOUS_CHUNK
    exclude_timestamp_ranges = [("20250801124409", "20250805165454")]
    # Use pathlib for cross-platform path handling
    # log_path = Path("./samples/log-reset-01.csv").resolve()
    # log_path = Path("./samples/log-reset-02.csv").resolve()

    log_path = Path("C:/Games/Utility/ICScriptHub/log-reset.csv").resolve()
    start_line = 4883

    # log_path = Path("./samples/log-reset-extraversion-2025-08-05.csv").resolve()
    calculate_fort(
        log_path=log_path,
        start_line=start_line,
        end_line=end_line,
        include_header=True,
        zscore_min=zscore_min,
        zscore_max=zscore_max,
        input_data_fort=input_data_fort,
        ignore_mrt=ignore_mrt,
        delta_mode=delta_mode,
        exclude_timestamp_ranges=exclude_timestamp_ranges,
        verbose_filtering=True,
    )


if __name__ == "__main__":
    main()
