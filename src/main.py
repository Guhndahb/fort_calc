#!/usr/bin/env python3
"""
FORT Calculator - Refactored into pure functional units.

This module exposes four pure functions:
- load_and_slice_csv()
- transform_pipeline()
- summarize_and_model()
- render_outputs()

Each function takes explicit inputs and returns explicit outputs, avoiding prints and
global state mutation. Logging kept for internal diagnostics but functions are pure by contract.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure a non-interactive Matplotlib backend is selected early to prevent GUI-backend
# selection/hangs in headless environments. We set the backend here before importing
# pyplot so that any later imports of matplotlib.pyplot will pick up the enforced backend.
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D

# Support both package and script execution modes
try:
    # When run as a package: python -m src.main
    from .csv_processor import CSVRangeProcessor
    from .utils import (
        build_effective_parameters,
        canonical_json_hash,
        normalize_abs_posix,
        utc_timestamp_seconds,
        write_manifest,
    )
except ImportError:
    # When run directly: python src/main.py
    from csv_processor import CSVRangeProcessor
    from utils import (
        build_effective_parameters,
        canonical_json_hash,
        normalize_abs_posix,
        utc_timestamp_seconds,
        write_manifest,
    )

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Module-level sentinel for plots that should omit the final FORT point.
# Use a named constant instead of a string literal to avoid typos and to make
# intent explicit: compare against OMIT_FORT rather than the raw string.
OMIT_FORT: str = "OMIT_FORT"


class DeltaMode(Enum):
    """
    Controls how run_time_delta is computed in summarize_run_time_by_sor_range().

    Semantics:
    - PREVIOUS_CHUNK: run_time_delta = run_time_mean(current_chunk) - run_time_mean(previous_chunk)
                      The first chunk has no previous baseline and thus yields NaN.
    - FIRST_CHUNK:    run_time_delta = run_time_mean(current_chunk) - run_time_mean(first_chunk)
                      The first chunk uses itself as baseline and thus yields NaN.

    See also: summarize_run_time_by_sor_range() for details on how deltas are populated,
    including the degenerate final 'fort' row.
    """

    PREVIOUS_CHUNK = auto()  # delta vs. the immediately-preceding chunk
    FIRST_CHUNK = auto()  # delta vs. the very first chunk


class FilterResult:
    """Container for filter operation results and diagnostics."""

    def __init__(self, label: Optional[str] = None) -> None:
        # Identification
        self.label: Optional[str] = label

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
    def start(self) -> None:
        import time

        self.started_at = time.perf_counter()

    def stop(self) -> None:
        import time

        self.finished_at = time.perf_counter()
        if self.started_at is not None:
            self.elapsed_ms = (self.finished_at - self.started_at) * 1000.0

    # Logging helpers
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)

    def add_event(self, message: str) -> None:
        """Add an info-level event message."""
        self.events.append(message)
        logger.info(message)

    def add_metric(self, name: str, value: int | float | str) -> None:
        """Attach a named metric."""
        self.metrics[name] = value

    def log_invalid_range(self, start: str, end: str, reason: str) -> None:
        """Log an invalid timestamp range."""
        self.invalid_ranges.append((start, end, reason))
        logger.warning(f"Invalid timestamp range skipped: {start}-{end} - {reason}")

    def set_skipped(self, reason: str, verbose: bool = False) -> None:
        """Mark the step as skipped with a reason."""
        self.skipped_reason = reason
        if verbose:
            self.add_event(f"Step skipped: {reason}")

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

    # Ensure the 'notes' column is string-like to avoid pandas dtype-conflict warnings
    # when assigning a Python string into a column that may currently be numeric.
    # Perform the cast only if the column is not already string-like.
    if not pd.api.types.is_string_dtype(df["notes"]):
        # Use pandas' nullable StringDtype for consistent text/missing semantics.
        df["notes"] = df["notes"].astype("string")

    # Mutate in place only when needed
    df.at[first_idx, "notes"] = "<DATA START>"
    result.filtered_rows = len(df)
    result.stop()
    if verbose:
        result.add_event("Filled first empty note with '<DATA START>'")
        logger.info(result.summarize())

    return df


def compute_adjusted_run_time(
    df: pd.DataFrame, ignore_resetticks: bool, verbose: bool = False
) -> pd.DataFrame:
    """
    Compute adjusted_run_time with diagnostics via FilterResult.
    Preserves DataFrame -> DataFrame contract for pandas piping.
    NOTE: Avoids df.copy() to respect in-place performance preference.

    Column/parameter semantics:
    - runticks:     Run duration in ticks (milliseconds).
    - resetticks:   The duration in ticks (milliseconds) that it takes for the Modron reset
                    to occur. This column may be omitted from the data.
    - ignore_resetticks:   When True, ignore resetticks entirely. When False, subtract resetticks
                    from runticks to eliminate Modron reset time from the regression input.

    Behavior:
    - If ignore_resetticks is True:
        adjusted_run_time = runticks / 1000.0
      (resetticks is not accessed)
    - Else:
        adjusted_run_time = (runticks - resetticks) / 1000.0
        where resetticks is coerced to numeric and NaN → 0.

    Units:
    - adjusted_run_time is expressed in seconds.

    Diagnostics:
    - Emits warnings/metrics for NaN/negative adjusted values and missing columns.
    """
    result = FilterResult(label="compute_adjusted_run_time")
    result.start()
    result.original_rows = len(df)

    # Validate required columns
    required = ["runticks"] + ([] if ignore_resetticks else ["resetticks"])
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

    if ignore_resetticks:
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
    result.add_metric("ignore_resetticks", ignore_resetticks)
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
            # Always parse using a string path to preserve leading zeros when dtype is numeric.
            # Use exact=True per spec; errors='coerce' to mark invalids as NaT.
            series_to_parse = df[timestamp_column].astype(str)
            df[timestamp_column] = pd.to_datetime(
                series_to_parse,
                format=timestamp_format,
                errors="coerce",
                exact=True,
            )

        # Check for parsing failures and always persist metrics on the frame
        # even if the column was already datetime dtype.
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            invalid_timestamps = int(df[timestamp_column].isna().sum())
            converted_dtype = str(df[timestamp_column].dtype)
        else:
            # Not datetime -> treat as full failure
            invalid_timestamps = int(len(df))
            converted_dtype = str(df[timestamp_column].dtype)
        total_rows = int(len(df))
        if invalid_timestamps > 0:
            result.add_warning(
                f"Found {invalid_timestamps} invalid timestamps in column '{timestamp_column}' out of {total_rows} rows"
            )
        result.add_metric("invalid_timestamps", invalid_timestamps)
        result.add_metric("invalid_timestamps_total_rows", total_rows)
        result.add_metric("converted_dtype", converted_dtype)
        # Persist invalid count on the frame for downstream checks
        df.attrs["invalid_timestamps"] = invalid_timestamps
        df.attrs["invalid_timestamps_total_rows"] = total_rows

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
    Refactored to use enhanced FilterResult diagnostics (label, timing, metrics/events)
    while preserving DataFrame -> DataFrame piping. Avoids unnecessary .copy().
    """
    result = FilterResult(label="filter_timestamp_ranges")
    result.start()
    result.original_rows = len(df)

    # Early return for empty DataFrame
    if df.empty:
        result.set_skipped("empty dataframe - no filtering performed")
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    # Early return for empty ranges
    if not exclude_timestamp_ranges:
        result.set_skipped("no exclude ranges provided", verbose=verbose)
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    # Check if timestamp column exists
    if timestamp_column not in df.columns:
        result.set_skipped(f"timestamp column '{timestamp_column}' not found")
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    # Add diagnostic logging for timestamp column
    if verbose:
        logger.info(f"Timestamp column dtype: {df[timestamp_column].dtype}")
        logger.info(
            f"First few timestamp values: {df[timestamp_column].iloc[:3].tolist()}"
        )
        logger.info(f"Sample exclude ranges: {exclude_timestamp_ranges}")

    result.add_metric("timestamp_dtype", str(df[timestamp_column].dtype))

    # Verify timestamp column is properly parsed as datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        # Preserve explicit diagnostic message for tests that assert on it
        logger.info(f"timestamp column '{timestamp_column}' is not in datetime format")
        result.set_skipped(
            f"timestamp column '{timestamp_column}' is not in datetime format"
        )
        result.filtered_rows = len(df)
        result.stop()
        if verbose:
            logger.info(result.summarize())
        return df

    # Process valid timestamp ranges
    valid_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
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

    result.add_metric("valid_range_count", len(valid_ranges))

    # Early return if no valid ranges
    if not valid_ranges:
        result.set_skipped("no valid timestamp ranges found")
        if verbose:
            # keep previous helpful samples
            logger.info(
                f"DataFrame timestamp column sample: {df[timestamp_column].iloc[:5]}"
            )
            logger.info(
                f"Parsed timestamps sample: {df[timestamp_column].iloc[:5].dt.strftime('%Y%m%d%H%M%S')}"
            )
            logger.info(result.summarize())
        result.filtered_rows = len(df)
        result.stop()
        return df

    # Build exclusion mask for valid ranges (avoid copy; operate on df directly)
    exclude_mask = pd.Series(False, index=df.index)

    if verbose:
        logger.info(f"Valid ranges to exclude: {valid_ranges}")
        logger.info(
            f"Timestamp range in data: {df[timestamp_column].min()} to {df[timestamp_column].max()}"
        )

    for start_ts, end_ts in valid_ranges:
        range_mask = (df[timestamp_column] >= start_ts) & (
            df[timestamp_column] <= end_ts
        )
        if verbose:
            matched_count = int(range_mask.sum())
            logger.info(
                f"Range {start_ts} to {end_ts}: {matched_count} rows would be excluded"
            )
            if matched_count > 0:
                logger.info(
                    f"Sample excluded timestamps: {df.loc[range_mask, timestamp_column].iloc[:3].tolist()}"
                )
        exclude_mask |= range_mask

    # Apply filtering and explicitly create an independent copy to avoid SettingWithCopy warnings downstream
    filtered_df = df.loc[~exclude_mask].copy()

    # Populate diagnostics
    excluded_rows = int(exclude_mask.sum())
    result.filtered_rows = len(filtered_df)
    result.excluded_rows = excluded_rows
    result.excluded_ranges = excluded_rows  # keep legacy counter aligned to rows here
    result.add_metric("excluded_rows", excluded_rows)

    # Log summary
    result.stop()
    if verbose or result.warnings:
        logger.info(result.summarize())
        if result.invalid_ranges:
            logger.info(f"Invalid ranges skipped: {len(result.invalid_ranges)}")

    return filtered_df


def filter_by_adjusted_run_time_zscore(
    df: pd.DataFrame, zscore_min: float, zscore_max: float, input_data_fort: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter rows by z-score bounds with robust handling for degenerate variance.

    Behavior:
    - Compute std = df['adjusted_run_time'].std(ddof=1)
    - If std is 0 or not finite (NaN/inf), set zscores = 0 for all rows and
      effectively skip z-score filtering, keeping only rows where sor# == input_data_fort.
      That is, mask_zscore becomes all True (zscores==0 within any reasonable bounds),
      but we explicitly set mask_zscore to False to force the "keep only fort row" behavior.
    - Otherwise compute standard zscores and filter by [zscore_min, zscore_max], always
      keeping sor# == input_data_fort.
    """
    df = df.copy()

    mean_val = df["adjusted_run_time"].mean()
    std_val = df["adjusted_run_time"].std(ddof=1)

    if not np.isfinite(std_val) or std_val == 0:
        # Degenerate variance case: set zscores to 0 but skip z filtering,
        # keep only the fort row.
        df["zscore"] = 0.0
        mask_zscore = pd.Series(False, index=df.index)
    else:
        zscores = (df["adjusted_run_time"] - mean_val) / std_val
        df["zscore"] = zscores
        mask_zscore = (zscores >= zscore_min) & (zscores <= zscore_max)

    mask_keep_sor = df["sor#"] == input_data_fort
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
    """
    Summarize adjusted_run_time by SOR ranges using vectorized binning.

    Contract precondition:
    - Upstream summarize_and_model enforces that the input contains at least one row
      with sor# == input_data_fort so the final degenerate 'fort' row mean is computed
      from actual data (not inferred). If the dataset does not include the fort row,
      summarize_and_model will raise before calling this function.

    Behavior:
    - Partition the inclusive range [1..input_data_fort-1] into 4 contiguous bins
      as evenly as possible, preserving the original chunking logic.
    - Compute run_time_mean per bin via groupby aggregation.
    - Append a final row for the exact fort (sor == input_data_fort) as a degenerate interval,
      setting both sorr_start and sorr_end to input_data_fort.
    - Compute run_time_delta according to delta_mode using the bin means; the first
      row's delta is NaN.

    Delta semantics (delta_mode):
    - PREVIOUS_CHUNK: run_time_delta = run_time_mean(current_chunk) - run_time_mean(previous_chunk)
    - FIRST_CHUNK:    run_time_delta = run_time_mean(current_chunk) - run_time_mean(first_chunk)

    Offline cost interpretation:
    - The final degenerate 'fort' row’s run_time_delta is used downstream as offline_cost,
      representing the estimated extra time game restarts (usually offline stacks) take
      versus an online stack.

    Returns:
    - DataFrame with columns: ['sorr_start', 'sorr_end', 'run_time_mean', 'run_time_delta'].
    """
    # Compute boundaries for the 4 bins covering [1..input_data_fort-1]
    total = max(input_data_fort - 1, 0)
    chunk_size = total // 4
    remainder = total % 4

    starts: list[int] = []
    ends: list[int] = []
    start = 1
    for i in range(4):
        end = start + chunk_size - 1
        if i < remainder:
            end += 1
        # Clamp for empty prefixes when input_data_fort <= 1
        if start <= end:
            starts.append(start)
            ends.append(end)
        start = end + 1

    # Build cut edges for pandas.cut (right-closed bins to match between(start,end))
    # Edges must be strictly increasing; if no bins, edges will be empty.
    edges: list[float] = []
    if starts and ends:
        # Combine starts and ends into edges: [s1, e1, e2, e3, e4]
        edges = [float(starts[0])]
        for e in ends:
            edges.append(float(e))

    # Prepare DataFrame slice for values in [1..input_data_fort-1]
    # Coerce sor# to numeric to be robust
    sor_numeric = pd.to_numeric(df_range["sor#"], errors="coerce")
    df_vals = df_range.loc[
        (sor_numeric >= 1) & (sor_numeric <= input_data_fort - 1)
    ].copy()
    df_vals["sor_numeric"] = sor_numeric.loc[df_vals.index]

    # If we have valid bin edges, assign bins; otherwise create an empty grouping
    if len(edges) >= 2:
        # pandas.cut expects edges for right-closed bins: (edges[i-1], edges[i]]
        # We supply labels as 0..len(ends)-1 to map back to (start,end)
        labels = list(range(len(ends)))
        df_vals["bin"] = pd.cut(
            df_vals["sor_numeric"],
            bins=edges,
            right=True,
            include_lowest=True,
            labels=labels,
        )
        # Group by bin and aggregate mean
        grouped = (
            df_vals.groupby("bin", observed=True)["adjusted_run_time"]
            .mean()
            .reindex(labels)  # ensure all bins present in order
        )
        run_time_means = grouped.to_numpy()
    else:
        # No bins (e.g., input_data_fort <= 1)
        run_time_means = np.array([], dtype=float)

    # Build summary rows for the 4 range bins actually created
    rows: list[list[float | int | None]] = []
    for i in range(len(run_time_means)):
        start_i = int(starts[i])
        end_i = int(ends[i])
        mean_runtime = (
            float(run_time_means[i]) if pd.notna(run_time_means[i]) else np.nan
        )
        if i == 0:
            delta = np.nan
        else:
            if delta_mode is DeltaMode.PREVIOUS_CHUNK:
                baseline = rows[-1][2]  # previous chunk mean
            else:  # DeltaMode.FIRST_CHUNK
                baseline = rows[0][2]  # first chunk mean
            # baseline may be NaN if previous chunk had no data; subtraction yields NaN
            delta = (
                mean_runtime - float(baseline)  # type: ignore[arg-type]
                if pd.notna(baseline)
                else np.nan
            )
        rows.append([start_i, end_i, mean_runtime, delta])

    # Append the final exact fort row (degenerate interval: start == end == input_data_fort)
    mask_fort = sor_numeric == input_data_fort
    mean_fort = df_range.loc[mask_fort, "adjusted_run_time"].mean()
    if len(rows) == 0:
        # If there were no range rows, the "first" baseline for FIRST_CHUNK mode
        # should remain NaN; previous-chunk also yields NaN.
        delta_final = np.nan
    else:
        if delta_mode is DeltaMode.PREVIOUS_CHUNK:
            baseline_final = rows[-1][2]
        else:
            baseline_final = rows[0][2]
        # baseline_final element can be float | int | None due to list storage; guard with pd.notna
        delta_final = (
            mean_fort - float(baseline_final)  # type: ignore[arg-type]
            if pd.notna(baseline_final)
            else np.nan
        )

    rows.append(
        [
            int(input_data_fort),
            int(input_data_fort),  # set end equal to start (degenerate interval)
            float(mean_fort) if pd.notna(mean_fort) else np.nan,
            delta_final,
        ]
    )

    # Build DataFrame
    df_summary = pd.DataFrame(
        rows,
        columns=["sorr_start", "sorr_end", "run_time_mean", "run_time_delta"],
    )

    # Enforce domain invariants:
    # 1) sorr_start and sorr_end must be positive integers
    # 2) Non-overlapping, ordered bins
    # 3) Cast to plain int64 to avoid float/NaN upcasts and enforce strictness
    #    (degenerate final row ensures no NaN in sorr_end)
    df_summary["sorr_start"] = df_summary["sorr_start"].astype("int64")
    df_summary["sorr_end"] = df_summary["sorr_end"].astype("int64")

    assert (df_summary["sorr_start"] > 0).all() and (df_summary["sorr_end"] > 0).all()
    # end >= start for each row
    assert (df_summary["sorr_end"] >= df_summary["sorr_start"]).all()
    # strictly increasing without overlaps for range rows vs next start
    if len(df_summary) > 1:
        # For all transitions up to the penultimate row, ensure next start > current end
        assert (
            df_summary["sorr_start"].iloc[1:].values
            > df_summary["sorr_end"].iloc[:-1].values
        ).all()

    return df_summary


@dataclass
class LoadSliceParams:
    """
    Parameters used when loading a slice of the CSV.

    Attributes:
        log_path: Path to the CSV file to read.
        start_line: 1-based inclusive start line (data rows, header excluded) or None to start at first row.
        end_line: 1-based inclusive end line (data rows) or None to read to the end.
        include_header: Whether to preserve the header row when reading.
        header_map: Optional mapping of input header name -> canonical target name.
            - Populated from the CLI via repeatable --header-map OLD:NEW flags.
            - Matching is case-insensitive and applied after the CSV is loaded.
            - Collision checks are performed before rename (conflicting targets or
              rename-induced duplicate column names raise ValueError).
    """

    log_path: Path
    start_line: Optional[int]
    end_line: Optional[int]
    include_header: bool = True
    # Mapping of input header name -> canonical target name.
    # Populated from the CLI via --header-map OLD:NEW (repeatable).
    header_map: dict[str, str] = field(default_factory=dict)


class PlotLayer(IntFlag):
    # Data
    DATA_SCATTER = 1 << 0
    # New: Excluded-by-zscore data points
    DATA_SCATTER_EXCLUDED = 1 << 14

    # OLS predictions
    OLS_PRED_LINEAR = 1 << 1
    OLS_PRED_QUAD = 1 << 2

    # OLS cost-per-run curves
    OLS_COST_LINEAR = 1 << 3
    OLS_COST_QUAD = 1 << 4

    # OLS min-cost markers
    OLS_MIN_LINEAR = 1 << 5
    OLS_MIN_QUAD = 1 << 6

    # WLS predictions
    WLS_PRED_LINEAR = 1 << 7
    WLS_PRED_QUAD = 1 << 8

    # WLS cost-per-run curves
    WLS_COST_LINEAR = 1 << 9
    WLS_COST_QUAD = 1 << 10

    # WLS min-cost markers
    WLS_MIN_LINEAR = 1 << 11
    WLS_MIN_QUAD = 1 << 12

    # Legend visibility
    LEGEND = 1 << 13

    # Presets
    NONE = 0
    ALL_DATA = DATA_SCATTER
    ALL_SCATTER = DATA_SCATTER | DATA_SCATTER_EXCLUDED
    ALL_OLS = (
        OLS_PRED_LINEAR
        | OLS_PRED_QUAD
        | OLS_COST_LINEAR
        | OLS_COST_QUAD
        | OLS_MIN_LINEAR
        | OLS_MIN_QUAD
        | LEGEND
    )
    ALL_WLS = (
        WLS_PRED_LINEAR
        | WLS_PRED_QUAD
        | WLS_COST_LINEAR
        | WLS_COST_QUAD
        | WLS_MIN_LINEAR
        | WLS_MIN_QUAD
        | LEGEND
    )
    SCATTER_PREDICTION = (
        ALL_SCATTER
        | OLS_PRED_LINEAR
        | OLS_PRED_QUAD
        | WLS_PRED_LINEAR
        | WLS_PRED_QUAD
        | LEGEND
    )
    ALL_PREDICTION = (
        OLS_PRED_LINEAR | OLS_PRED_QUAD | WLS_PRED_LINEAR | WLS_PRED_QUAD | LEGEND
    )
    ALL_COST = (
        OLS_COST_LINEAR | OLS_COST_QUAD | WLS_COST_LINEAR | WLS_COST_QUAD | LEGEND
    )
    MIN_MARKERS_ONLY = (
        OLS_MIN_LINEAR | OLS_MIN_QUAD | WLS_MIN_LINEAR | WLS_MIN_QUAD | LEGEND
    )
    DEFAULT = (
        DATA_SCATTER
        | OLS_PRED_LINEAR
        | OLS_PRED_QUAD
        | OLS_COST_LINEAR
        | OLS_COST_QUAD
        | OLS_MIN_LINEAR
        | OLS_MIN_QUAD
        | LEGEND
    )
    EVERYTHING = DEFAULT | ALL_WLS | DATA_SCATTER_EXCLUDED


@dataclass
class TransformParams:
    zscore_min: float
    zscore_max: float
    input_data_fort: int
    ignore_resetticks: bool
    delta_mode: DeltaMode
    exclude_timestamp_ranges: Optional[List[Tuple[str, str]]]
    verbose_filtering: bool = False
    # Fail fast if any timestamps fail to parse (simple and strict by default)
    fail_on_any_invalid_timestamps: bool = True


@dataclass
class TransformOutputs:
    df_range: pd.DataFrame
    df_excluded: pd.DataFrame


@dataclass
class SummaryModelOutputs:
    df_summary: pd.DataFrame
    df_results: pd.DataFrame
    regression_diagnostics: dict
    offline_cost: float
    sor_min_cost_lin: int
    sor_min_cost_quad: int
    # New: WLS min-cost markers (optional presence based on include_wls_overlay)
    sor_min_cost_lin_wls: Optional[int] = None
    sor_min_cost_quad_wls: Optional[int] = None


def regression_analysis(
    df_range: pd.DataFrame, input_data_fort: int
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Perform regression analysis for linear and quadratic forms with statsmodels.

    Behavior:
    - Trains baseline OLS models for both linear and quadratic specifications.
    - Replaces the previous fixed WLS (1/(sor#^2)) with empirical WLS that estimates a variance-power p̂
      from OLS residuals via a log–log regression of residual^2 on sor#.
    - Builds prediction matrices with named columns and explicit constant handling
      via statsmodels add_constant(has_constant='add') to avoid column-order issues.

    Variants included in diagnostics:
      - 'ols':        Baseline OLS
      - 'ols_hc1':    OLS with heteroskedasticity-robust covariance (HC1)
      - 'wls_emp':    Empirical WLS with weights 1 / (sor#^p̂)
      - 'wls_emp_hc1':Empirical WLS with HC1 robust covariance

    Returns:
      tuple[pd.DataFrame, dict]
        - result_df: DataFrame with columns:
            ["sor#", "linear_model_output", "quadratic_model_output",
             "linear_model_output_wls", "quadratic_model_output_wls"]
        - diagnostics: Nested dict containing model statistics for all variants.

    Notes:
    - Empirical WLS:
        p̂ is estimated by regressing log(e_i^2 + eps) on log(sor_i) using OLS residuals e_i and a small eps to
        avoid -inf. Weights are set to 1/(sor#^p̂). Non-finite weights are replaced by the median finite weight;
        if none are finite, fall back to uniform weights. If estimation fails, we skip WLS_emp and record an error.
    - Robust covariance is requested via fit(cov_type="HC1").
    """

    # Helper: ensure constant named 'const'
    def _ensure_const_named_const(X: pd.DataFrame) -> pd.DataFrame:
        if "const" not in X.columns:
            const_col = [c for c in X.columns if c.lower() in ("const", "intercept")]
            if const_col:
                X = X.rename(columns={const_col[0]: "const"})
        return X

    # Helper: prepare design matrices
    def _make_designs(
        df: pd.DataFrame, seq: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Xlin_tr = pd.DataFrame({"sor#": df["sor#"].to_numpy()})
        Xlin_tr = sm.add_constant(Xlin_tr, has_constant="add")
        Xlin_tr = _ensure_const_named_const(Xlin_tr)

        Xquad_tr = pd.DataFrame(
            {"sor#": df["sor#"].to_numpy(), "sor2": df["sor#"].to_numpy() ** 2}
        )
        Xquad_tr = sm.add_constant(Xquad_tr, has_constant="add")
        Xquad_tr = _ensure_const_named_const(Xquad_tr)

        Xlin_pr = pd.DataFrame({"sor#": seq})
        Xlin_pr = sm.add_constant(Xlin_pr, has_constant="add")
        Xlin_pr = _ensure_const_named_const(Xlin_pr)

        Xquad_pr = pd.DataFrame({"sor#": seq, "sor2": seq**2})
        Xquad_pr = sm.add_constant(Xquad_pr, has_constant="add")
        Xquad_pr = _ensure_const_named_const(Xquad_pr)

        return Xlin_tr, Xquad_tr, Xlin_pr, Xquad_pr

    # Helper: pack stats into dict
    def _stats_dict(res) -> Dict[str, Any]:
        return {
            "R-squared": res.rsquared,
            "Adj. R-squared": res.rsquared_adj,
            "F-statistic": res.fvalue,
            "p-value": res.f_pvalue,
            "Coefficients": res.params,
            "Standard Errors": res.bse,
            "Confidence Intervals": res.conf_int(),
            "Residuals": res.resid,
            "Summary": res.summary(),
        }

    # Sequence and response
    sor_sequence = np.arange(1, input_data_fort + 1)
    y = df_range["adjusted_run_time"].to_numpy()

    # Base designs
    X_linear_train, X_quadratic_train, X_linear_pred, X_quadratic_pred = _make_designs(
        df_range, sor_sequence
    )

    # Baseline OLS fits
    lin_ols = sm.OLS(y, X_linear_train).fit()
    quad_ols = sm.OLS(y, X_quadratic_train).fit()

    # Align prediction matrices to trained exog names (order only)
    X_linear_pred = X_linear_pred.reindex(columns=lin_ols.model.exog_names)
    X_quadratic_pred = X_quadratic_pred.reindex(columns=quad_ols.model.exog_names)

    # Baseline predictions used in result_df
    linear_model_output = lin_ols.predict(X_linear_pred)
    quadratic_model_output = quad_ols.predict(X_quadratic_pred)

    # Diagnostics container with baseline 'ols'
    diagnostics: Dict[str, Dict[str, Any]] = {
        "linear": {"ols": _stats_dict(lin_ols)},
        "quadratic": {"ols": _stats_dict(quad_ols)},
    }

    # Attach AIC/BIC for baseline OLS fits
    try:
        diagnostics["linear"]["ols"]["aic"] = float(lin_ols.aic)
        diagnostics["linear"]["ols"]["bic"] = float(lin_ols.bic)
    except Exception as _e_aicbic_lin_ols:
        diagnostics["linear"]["ols"]["aicbic_exception"] = str(_e_aicbic_lin_ols)
    try:
        diagnostics["quadratic"]["ols"]["aic"] = float(quad_ols.aic)
        diagnostics["quadratic"]["ols"]["bic"] = float(quad_ols.bic)
    except Exception as _e_aicbic_quad_ols:
        diagnostics["quadratic"]["ols"]["aicbic_exception"] = str(_e_aicbic_quad_ols)

    # Compute in-sample RMSEs for baseline OLS fits (common, unweighted)
    try:
        rmse_lin_ols = float(np.sqrt(np.mean(np.square(lin_ols.resid))))
        rmse_quad_ols = float(np.sqrt(np.mean(np.square(quad_ols.resid))))
        diagnostics["linear"]["ols"]["RMSE"] = rmse_lin_ols
        diagnostics["quadratic"]["ols"]["RMSE"] = rmse_quad_ols
    except Exception as _e_rmse_ols:
        diagnostics["linear"]["ols"]["rmse_exception"] = str(_e_rmse_ols)
        diagnostics["quadratic"]["ols"]["rmse_exception"] = str(_e_rmse_ols)

    # Variant: OLS with robust SEs (HC1)
    try:
        lin_ols_hc1 = sm.OLS(y, X_linear_train).fit(cov_type="HC1")
        quad_ols_hc1 = sm.OLS(y, X_quadratic_train).fit(cov_type="HC1")
        diagnostics["linear"]["ols_hc1"] = _stats_dict(lin_ols_hc1)
        diagnostics["quadratic"]["ols_hc1"] = _stats_dict(quad_ols_hc1)
    except Exception as e:
        diagnostics.setdefault("linear", {})["ols_hc1_error"] = str(e)
        diagnostics.setdefault("quadratic", {})["ols_hc1_error"] = str(e)

    # Variant: Empirical WLS (replace fixed 1/(sor#^2))
    # Estimate variance-power p via log(resid^2 + eps) ~ log(sor#), then weights = 1 / (sor#^p)
    linear_model_output_wls = None
    quadratic_model_output_wls = None
    try:
        sor_vals = pd.to_numeric(df_range["sor#"], errors="coerce").to_numpy(
            dtype=float
        )

        # Choose residuals from the model whose mean structure we'll weight.
        # Use quadratic residuals to allow curvature; fallback to linear if needed.
        resid_source = quad_ols.resid if hasattr(quad_ols, "resid") else lin_ols.resid
        resid = np.asarray(resid_source, dtype=float)

        # Build design for variance-power regression: z = log(resid^2 + eps), x = log(sor)
        eps = 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.log(np.square(resid) + eps)
            x = np.log(sor_vals)

        # Keep only finite pairs and sor_vals > 0
        mask = np.isfinite(z) & np.isfinite(x) & (sor_vals > 0)
        if mask.sum() < 3:
            raise ValueError("Insufficient finite points to estimate variance power")

        X_var = sm.add_constant(x[mask], has_constant="add")
        var_fit = sm.OLS(z[mask], X_var).fit()
        p_hat = (
            float(var_fit.params[1])
            if len(var_fit.params) >= 2
            else float(var_fit.params[0])
        )

        # Compute empirical weights: w = 1 / (sor#^p_hat)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / np.where(
                np.isfinite(sor_vals) & (sor_vals > 0),
                np.power(sor_vals, p_hat),
                np.nan,
            )

        # Replace NaN/Inf weights with median finite weight to keep WLS stable
        if not np.isfinite(w).any():
            w = np.ones_like(sor_vals, dtype=float)
        else:
            finite_mask = np.isfinite(w)
            median_w = float(np.nanmedian(w[finite_mask]))
            w = np.where(
                np.isfinite(w),
                w,
                median_w if np.isfinite(median_w) and median_w > 0 else 1.0,
            )

        # Fit empirical WLS
        lin_wls = sm.WLS(y, X_linear_train, weights=w).fit()
        quad_wls = sm.WLS(y, X_quadratic_train, weights=w).fit()
        # Backward-compatible keys expected by tests: expose empirical WLS under 'wls' and 'wls_hc1'
        diagnostics["linear"]["wls"] = {
            **_stats_dict(lin_wls),
            "weights_spec": f"1/(sor#^{p_hat:.6g})",
            "p_hat": p_hat,
        }
        diagnostics["quadratic"]["wls"] = {
            **_stats_dict(quad_wls),
            "weights_spec": f"1/(sor#^{p_hat:.6g})",
            "p_hat": p_hat,
        }

        # Attach AIC/BIC for empirical WLS fits
        try:
            diagnostics["linear"]["wls"]["aic"] = float(lin_wls.aic)
            diagnostics["linear"]["wls"]["bic"] = float(lin_wls.bic)
        except Exception as _e_aicbic_lin_wls:
            diagnostics["linear"]["wls"]["aicbic_exception"] = str(_e_aicbic_lin_wls)
        try:
            diagnostics["quadratic"]["wls"]["aic"] = float(quad_wls.aic)
            diagnostics["quadratic"]["wls"]["bic"] = float(quad_wls.bic)
        except Exception as _e_aicbic_quad_wls:
            diagnostics["quadratic"]["wls"]["aicbic_exception"] = str(
                _e_aicbic_quad_wls
            )

        # In-sample RMSEs using unweighted residuals for comparability
        try:
            resid_lin_wls = y - lin_wls.fittedvalues
            resid_quad_wls = y - quad_wls.fittedvalues
            diagnostics["linear"]["wls"]["RMSE"] = float(
                np.sqrt(np.mean(np.square(resid_lin_wls)))
            )
            diagnostics["quadratic"]["wls"]["RMSE"] = float(
                np.sqrt(np.mean(np.square(resid_quad_wls)))
            )
        except Exception as _e_rmse_wls:
            diagnostics["linear"]["wls"]["rmse_exception"] = str(_e_rmse_wls)
            diagnostics["quadratic"]["wls"]["rmse_exception"] = str(_e_rmse_wls)

        # Predictions (align exog)
        X_linear_pred_wls = X_linear_pred.reindex(columns=lin_wls.model.exog_names)
        X_quadratic_pred_wls = X_quadratic_pred.reindex(
            columns=quad_wls.model.exog_names
        )
        linear_model_output_wls = lin_wls.predict(X_linear_pred_wls)
        quadratic_model_output_wls = quad_wls.predict(X_quadratic_pred_wls)

        # Empirical WLS + robust SEs (HC1)
        try:
            lin_wls_hc1 = sm.WLS(y, X_linear_train, weights=w).fit(cov_type="HC1")
            quad_wls_hc1 = sm.WLS(y, X_quadratic_train, weights=w).fit(cov_type="HC1")
            diagnostics["linear"]["wls_hc1"] = {
                **_stats_dict(lin_wls_hc1),
                "weights_spec": f"1/(sor#^{p_hat:.6g})",
                "p_hat": p_hat,
            }
            diagnostics["quadratic"]["wls_hc1"] = {
                **_stats_dict(quad_wls_hc1),
                "weights_spec": f"1/(sor#^{p_hat:.6g})",
                "p_hat": p_hat,
            }
        except Exception as e:
            diagnostics["linear"]["wls_hc1_error"] = str(e)
            diagnostics["quadratic"]["wls_hc1_error"] = str(e)
    except Exception as e:
        diagnostics.setdefault("linear", {})["wls_error"] = str(e)
        diagnostics.setdefault("quadratic", {})["wls_error"] = str(e)

    # Assemble result_df with both OLS and WLS predictions (WLS columns may be None)
    df_dict = {
        "sor#": sor_sequence,
        "linear_model_output": linear_model_output,
        "quadratic_model_output": quadratic_model_output,
    }
    if linear_model_output_wls is not None:
        df_dict["linear_model_output_wls"] = linear_model_output_wls
    else:
        df_dict["linear_model_output_wls"] = pd.Series([np.nan] * len(sor_sequence))
    if quadratic_model_output_wls is not None:
        df_dict["quadratic_model_output_wls"] = quadratic_model_output_wls
    else:
        df_dict["quadratic_model_output_wls"] = pd.Series([np.nan] * len(sor_sequence))

    result_df = pd.DataFrame(df_dict).sort_values(by="sor#").reset_index(drop=True)

    return result_df, diagnostics


def load_and_slice_csv(params: LoadSliceParams) -> pd.DataFrame:
    """
    Pure function to load a CSV line range as a DataFrame.
    No prints; raises exceptions on error.
    """
    if not params.log_path.exists():
        raise FileNotFoundError(f"CSV file not found at {params.log_path}")
    with CSVRangeProcessor(params.log_path) as processor:
        df_range = processor.read_range(
            start_line=params.start_line,
            end_line=params.end_line,
            include_header=params.include_header,
        )

    # Apply optional header mapping (case-insensitive best-effort) with collision detection.
    # We only rename columns that are present in the loaded frame; we warn about keys not found.
    if getattr(params, "header_map", None):
        original_columns = list(df_range.columns)

        # 1) Detect conflicting targets among provided mappings (two OLD -> same NEW).
        provided_value_to_keys: dict[str, list[str]] = {}
        for k, v in params.header_map.items():
            provided_value_to_keys.setdefault(v, []).append(k)
        duplicate_provided_targets = {
            tgt: keys for tgt, keys in provided_value_to_keys.items() if len(keys) > 1
        }
        if duplicate_provided_targets:
            parts = [
                f"target '{tgt}' specified by keys {keys}"
                for tgt, keys in duplicate_provided_targets.items()
            ]
            raise ValueError(
                "Conflicting --header-map targets specified (multiple OLD map to same NEW): "
                + "; ".join(parts)
            )

        # Build normalized lookup for provided keys (strip + lower) and normalized targets (strip)
        lower_map = {k.strip().lower(): v.strip() for k, v in params.header_map.items()}

        # 2) Build remap for columns actually present (case-insensitive + trim match)
        remap: dict[str, str] = {}
        for col in original_columns:
            normalized_col = col.strip().lower()
            mapped = lower_map.get(normalized_col)
            if mapped:
                remap[col] = mapped

        # 3) Predict resulting column names after rename and detect duplicates
        new_names = [remap.get(col, col) for col in original_columns]
        # Find any names that would be duplicated
        seen: set[str] = set()
        dup_targets: set[str] = set()
        for name in new_names:
            if name in seen:
                dup_targets.add(name)
            else:
                seen.add(name)
        if dup_targets:
            # Build conflict details: which original columns would produce each duplicate target
            conflicts: dict[str, list[str]] = {}
            for col in original_columns:
                target = remap.get(col, col)
                if target in dup_targets:
                    conflicts.setdefault(target, []).append(col)
            msg_parts = [
                f"'{tgt}' <= columns {cols}" for tgt, cols in conflicts.items()
            ]
            raise ValueError(
                "Header mapping would produce duplicate column names after rename: "
                + "; ".join(msg_parts)
            )

        # Safe to rename
        if remap:
            df_range = df_range.rename(columns=remap)
            logger.info(f"Applied header mappings: {remap}")
        else:
            logger.info("No header mappings matched CSV columns; nothing renamed.")

        # Warn about any user-provided keys that did not match any column (helpful warning)
        provided_keys = list(params.header_map.keys())
        found_lower = {c.strip().lower() for c in original_columns}
        missing = [k for k in provided_keys if k.strip().lower() not in found_lower]
        if missing:
            logger.warning(f"Header map keys not found in CSV columns: {missing}")

    if df_range.empty:
        raise ValueError("No data found in the specified range")
    return df_range


def transform_pipeline(
    df_range: pd.DataFrame, params: TransformParams
) -> TransformOutputs:
    """
    Pure transformation pipeline from raw range to filtered frames.
    Returns included and excluded DataFrames.

    Example (typical pipeline):
        >>> # Load a CSV slice (see load_and_slice_csv)
        >>> df = pd.DataFrame({
        ...     "timestamp": [20250101000001, 20250101000002, 20250101000003,
        ...                    20250101000004, 20250101000005, 20250101000006],
        ...     "ignore": ["FALSE"]*6,
        ...     "sor#": [1, 2, 3, 4, 5, 6],
        ...     "runticks": [1000, 1010, 1020, 1030, 1040, 1050],
        ...     "resetticks": [0, 0, 0, 0, 0, 0],
        ...     "notes": [""]*6,
        ... })
        >>> params = TransformParams(
        ...     zscore_min=-2.0,
        ...     zscore_max=2.0,
        ...     input_data_fort=6,
        ...     ignore_resetticks=True,  # if False, resetticks will be subtracted from runticks
        ...     delta_mode=DeltaMode.PREVIOUS_CHUNK,
        ...     exclude_timestamp_ranges=None,
        ... )
        >>> out = transform_pipeline(df, params)
        >>> out.df_range.columns  # doctest: +ELLIPSIS
        Index([... 'adjusted_run_time' ...], dtype='object')

    Related definitions:
    - resetticks: duration in ticks (ms) that Modron reset takes; may be absent in input.
    - ignore_resetticks (param): when True, the resetticks column is ignored;
      when False, resetticks is subtracted from runticks before converting to seconds.
    - adjusted_run_time: computed seconds value, see compute_adjusted_run_time() for units and rules.
    - delta_mode: controls how run_time_delta is computed, see DeltaMode and summarize_run_time_by_sor_range().
    - offline_cost: derived later in summarize_and_model() from the final row’s run_time_delta.
    """
    # Fail fast on required schema: must have both 'sor#' and 'runticks'
    required_cols = ["sor#", "runticks"]
    missing = [c for c in required_cols if c not in df_range.columns]
    if missing:
        present = list(df_range.columns)
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. Found columns: {present}"
        )

    # Build the pipeline step-by-step so we can fail-fast immediately after timestamp parsing
    df_range = df_range.pipe(
        clean_ignore,
        input_data_fort=params.input_data_fort,
        verbose=params.verbose_filtering,
    )
    df_range = compute_adjusted_run_time(
        df_range, ignore_resetticks=params.ignore_resetticks
    )
    # Convert timestamps early, once — only when timestamp column exists.
    # Timestamp is optional unless exclude_timestamp_ranges is provided.
    if "timestamp" in df_range.columns:
        df_range = convert_timestamp_column_to_datetime(
            df_range, timestamp_column="timestamp", verbose=params.verbose_filtering
        )
        # Fail fast on invalid timestamps only when:
        #  - caller provided exclude ranges (we need timestamps to filter), or
        #  - the policy explicitly asks to fail and the column exists.
        persisted_invalid = df_range.attrs.get("invalid_timestamps", None)
        persisted_total = df_range.attrs.get("invalid_timestamps_total_rows", None)
        if persisted_invalid is None:
            # Fallback: derive from the current column
            if pd.api.types.is_datetime64_any_dtype(df_range["timestamp"]):
                invalid_ts_count = int(df_range["timestamp"].isna().sum())
            else:
                invalid_ts_count = int(len(df_range))
            total_rows = int(len(df_range))
        else:
            invalid_ts_count = int(persisted_invalid)
            total_rows = int(
                persisted_total if persisted_total is not None else len(df_range)
            )
        # If exclude ranges are provided, invalid timestamps are a hard error (needed to filter).
        if params.exclude_timestamp_ranges and invalid_ts_count > 0:
            raise ValueError(
                f"Invalid timestamps detected after parsing: {invalid_ts_count} of {total_rows} rows are NaT (required for exclude ranges)"
            )
        # Otherwise apply fail-fast policy only if explicitly requested (default True), but only matters if column exists.
        if (
            (not params.exclude_timestamp_ranges)
            and params.fail_on_any_invalid_timestamps
            and invalid_ts_count > 0
        ):
            raise ValueError(
                f"Invalid timestamps detected after parsing: {invalid_ts_count} of {total_rows} rows are NaT"
            )
    # Continue with the rest of the pipeline. filter_timestamp_ranges will itself no-op
    # if no ranges are provided or if timestamp column is missing/not datetime.
    df_range = df_range.pipe(
        filter_timestamp_ranges,
        exclude_timestamp_ranges=params.exclude_timestamp_ranges,
        verbose=params.verbose_filtering,
    ).pipe(fill_first_note_if_empty, verbose=params.verbose_filtering)

    df_included, df_excluded = filter_by_adjusted_run_time_zscore(
        df_range, params.zscore_min, params.zscore_max, params.input_data_fort
    )

    if len(df_included) < 5:
        raise ValueError(
            f"Too few rows remaining after filtering (found {len(df_included)}, need at least 5)"
        )

    return TransformOutputs(df_range=df_included, df_excluded=df_excluded)


def summarize_and_model(
    df_range: pd.DataFrame, params: TransformParams
) -> SummaryModelOutputs:
    """
    Produce summary table, regression outputs, and cost metrics.

    Contract:
    - Input data MUST contain at least one row where sor# == input_data_fort. This ensures the
      degenerate final 'fort' row has a valid (non-NaN) run_time_mean computed from actual data.
    - Any NaN in run_time_mean anywhere in the summary is an error.

    Definitions:
    - offline_cost: The estimated extra time game restarts (usually offline stacks) take
      versus an online stack. Computed as the run_time_delta of the final degenerate 'fort'
      row produced by summarize_run_time_by_sor_range() under the chosen delta_mode.

    Notes:
    - The final "offline" row created by summarize_run_time_by_sor_range has start == end == input_data_fort
      and must have a valid (non-NaN) run_time_mean. Any NaN in run_time_mean anywhere is an error.
    """
    # Enforce program contract: there must be at least one row with sor# == input_data_fort
    # so that the degenerate 'fort' row mean is computed from actual data.
    if not (
        pd.to_numeric(df_range["sor#"], errors="coerce") == params.input_data_fort
    ).any():
        raise ValueError(
            f"Input data must contain at least one row where sor# == input_data_fort "
            f"({params.input_data_fort}); none found."
        )

    df_summary = summarize_run_time_by_sor_range(
        df_range, params.input_data_fort, params.delta_mode
    )

    # Disallow any NaNs in run_time_mean (including the final 'fort' row).
    run_time_mean_isna = df_summary["run_time_mean"].isna()
    nan_count = int(run_time_mean_isna.sum())
    if nan_count > 0:
        raise ValueError(
            f"Insufficient input data: Found {nan_count} summary rows with no mean run time values."
        )

    # Offline cost comes from the final row's delta; must be finite and not NaN
    final_delta = df_summary.iloc[-1]["run_time_delta"]
    if not np.isfinite(final_delta):
        raise ValueError(
            "Offline cost could not be computed: final run_time_delta is NaN or not finite."
        )
    offline_cost = float(final_delta)

    # Extend prediction sequence to include k == input_data_fort in evaluated curve
    # while leaving training behavior unchanged inside regression_analysis.
    df_results, regression_diagnostics = regression_analysis(
        df_range, params.input_data_fort
    )

    # Cumulative sums (OLS)
    df_results["sum_lin"] = df_results["linear_model_output"].cumsum()
    df_results["sum_quad"] = df_results["quadratic_model_output"].cumsum()

    # If WLS predictions exist, compute their sums too
    has_wls_cols = all(
        c in df_results.columns
        for c in ["linear_model_output_wls", "quadratic_model_output_wls"]
    )
    if has_wls_cols:
        df_results["sum_lin_wls"] = df_results["linear_model_output_wls"].cumsum()
        df_results["sum_quad_wls"] = df_results["quadratic_model_output_wls"].cumsum()

    # Cost per run columns (OLS)
    df_results["cost_per_run_at_fort_lin"] = (
        df_results["sum_lin"] + offline_cost
    ) / df_results["sor#"]
    df_results["cost_per_run_at_fort_quad"] = (
        df_results["sum_quad"] + offline_cost
    ) / df_results["sor#"]

    # If WLS available, compute WLS cost-per-run
    if has_wls_cols:
        df_results["cost_per_run_at_fort_lin_wls"] = (
            df_results["sum_lin_wls"] + offline_cost
        ) / df_results["sor#"]
        df_results["cost_per_run_at_fort_quad_wls"] = (
            df_results["sum_quad_wls"] + offline_cost
        ) / df_results["sor#"]

    # Handle potential all-NA series robustly for small synthetic inputs
    def _safe_idxmin(series: pd.Series) -> int:
        # Prefer to drop NaNs; if all NaN, fall back to first index
        if series.dropna().empty:
            return int(series.index[0])
        return int(series.dropna().idxmin())

    sor_min_cost_lin = int(
        df_results.loc[_safe_idxmin(df_results["cost_per_run_at_fort_lin"]), "sor#"]
    )
    sor_min_cost_quad = int(
        df_results.loc[_safe_idxmin(df_results["cost_per_run_at_fort_quad"]), "sor#"]
    )

    sor_min_cost_lin_wls: Optional[int] = None
    sor_min_cost_quad_wls: Optional[int] = None
    if has_wls_cols:
        sor_min_cost_lin_wls = int(
            df_results.loc[
                _safe_idxmin(df_results["cost_per_run_at_fort_lin_wls"]), "sor#"
            ]
        )
        sor_min_cost_quad_wls = int(
            df_results.loc[
                _safe_idxmin(df_results["cost_per_run_at_fort_quad_wls"]), "sor#"
            ]
        )

    return SummaryModelOutputs(
        df_summary=df_summary,
        df_results=df_results,
        regression_diagnostics=regression_diagnostics,
        offline_cost=offline_cost,
        sor_min_cost_lin=sor_min_cost_lin,
        sor_min_cost_quad=sor_min_cost_quad,
        sor_min_cost_lin_wls=sor_min_cost_lin_wls,
        sor_min_cost_quad_wls=sor_min_cost_quad_wls,
    )


def _plot_layers_suffix(flags: PlotLayer) -> str:
    """
    Build a stable, human-readable suffix for filenames describing selected plot layers.

    Rules:
    - If flags exactly match one of the named presets, return that preset name.
    - Otherwise, return a compact '+'-joined list of the set atomic flags in canonical order.

    Example:
      PlotLayer.DEFAULT -> "DEFAULT"
      PlotLayer.WLS_PRED_LINEAR | PlotLayer.WLS_PRED_QUAD | PlotLayer.LEGEND
        -> "WLS_PRED_LINEAR+WLS_PRED_QUAD+LEGEND"
    """
    # Ordered presets to check for exact equality
    preset_order = [
        "DEFAULT",
        "EVERYTHING",
        "NONE",
        "ALL_OLS",
        "ALL_WLS",
        "ALL_DATA",
        "ALL_PREDICTION",
        "ALL_COST",
        "MIN_MARKERS_ONLY",
    ]
    for name in preset_order:
        if hasattr(PlotLayer, name):
            preset_val = getattr(PlotLayer, name)
            if flags == preset_val:
                return name

    # Canonical order of atomic flags
    atomic_order = [
        "DATA_SCATTER",
        "DATA_SCATTER_EXCLUDED",
        "OLS_PRED_LINEAR",
        "OLS_PRED_QUAD",
        "OLS_COST_LINEAR",
        "OLS_COST_QUAD",
        "OLS_MIN_LINEAR",
        "OLS_MIN_QUAD",
        "WLS_PRED_LINEAR",
        "WLS_PRED_QUAD",
        "WLS_COST_LINEAR",
        "WLS_COST_QUAD",
        "WLS_MIN_LINEAR",
        "WLS_MIN_QUAD",
        "LEGEND",
    ]
    tokens: list[str] = []
    for name in atomic_order:
        bit = getattr(PlotLayer, name)
        if flags & bit:
            tokens.append(name)
    return "+".join(tokens) if tokens else "NONE"


@dataclass
class PlotParams:
    """
    Plotting controls including layer selection and optional axis bounds.

    x_min/x_max/y_min/y_max:
      - Optional float bounds applied via plt.xlim/plt.ylim if provided.
      - One-sided limits are allowed by passing only one side (the other remains automatic).
      - x_min/x_max are floats to align with matplotlib float limits and allow half-step framing.
    """

    plot_layers: PlotLayer = PlotLayer.DEFAULT
    x_min: Optional[float] = None
    x_max: Optional[Union[float, str]] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


def add_legend_extra_line(text, color=None, **legend_kwargs):
    """
    Add an extra text line at the end of the current Matplotlib legend without a marker.

    This function appends a new, invisible legend handle with the specified text label,
    allowing you to add custom notes or information below existing legend entries.
    The extra line's text color can be customized.

    Parameters:
        text (str): The string to display as an additional legend entry.
        color (str or tuple, optional): Color specification for the extra legend text.
            If None (default), uses the default legend text color.
        **legend_kwargs: Additional keyword arguments passed to plt.legend().
            Useful for controlling legend appearance (e.g., handlelength, handletextpad).

    Returns:
        matplotlib.legend.Legend: The updated legend instance.
    """
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # Create an invisible Line2D handle (no line, no marker)
    invisible_handle = Line2D([], [], linestyle="None", marker=None, label=text)

    handles.append(invisible_handle)
    labels.append(text)

    legend = ax.legend(handles=handles, labels=labels, **legend_kwargs)

    if color:
        legend.get_texts()[-1].set_color(color)

    return legend


def filter_omit_fort(df):
    """
    Return a filtered view of df excluding max 'sor#'
    """
    if df is None or df.empty or "sor#" not in df.columns:
        return df
    return df.loc[df["sor#"] != df["sor#"].max()]


def render_outputs(
    df_range: pd.DataFrame,
    summary: SummaryModelOutputs,
    output_svg: str = "plot.svg",
    df_excluded: Optional[pd.DataFrame] = None,
    plot_params: Optional[PlotParams] = None,
) -> str:
    """
    Pure renderer: builds the plot and returns the output path.
    No prints; deterministic given inputs.

    Parameters:
      - df_range: DataFrame of included rows after filtering (i.e., transformed.df_range).
      - summary: SummaryModelOutputs computed from df_range.
      - output_svg: Target path for the generated SVG.
      - df_excluded: Optional DataFrame of rows excluded by z-score filtering
                     (i.e., transformed.df_excluded). Only used when the
                     DATA_SCATTER_EXCLUDED flag is set; otherwise ignored.
      - plot_params: Optional PlotParams controlling plot layers and axis limits.
                     When provided, plot layers come from plot_params.plot_layers and
                     axis limits are applied using its x/y bounds. If not provided,
                     defaults to PlotLayer.DEFAULT with automatic axis limits.

    Axis limit behavior:
      - If plot_params is provided:
          plt.xlim(left=plot_params.x_min if plot_params.x_min is not None else None,
                   right=plot_params.x_max if plot_params.x_max is not None else None)
          plt.ylim(bottom=plot_params.y_min if plot_params.y_min is not None else None,
                   top=plot_params.y_max if plot_params.y_max is not None else None)
        One-sided limits are supported by leaving the other side None (automatic).
      - If no plot_params: use automatic axis scaling (default matplotlib behavior).

    Notes:
      - DATA_SCATTER_EXCLUDED is not part of any preset and will render only when
        explicitly requested via the effective plot_layers and when df_excluded is provided.
    """
    # Determine effective plot layer flags (PlotParams is the single source of truth)
    # Callers are expected to always pass PlotParams; it already carries a default PlotLayer.DEFAULT.
    if plot_params is None:
        raise TypeError("render_outputs requires plot_params (PlotParams)")
    effective_flags = plot_params.plot_layers

    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))

    if plot_params.x_max == OMIT_FORT:
        omit_fort = True
        plot_params.x_max = None
        df_range_filtered = filter_omit_fort(df_range)
        df_excluded_filtered = filter_omit_fort(df_excluded)
        df_summary_filtered = filter_omit_fort(summary.df_results)
    else:
        omit_fort = False
        df_range_filtered = df_range
        df_excluded_filtered = df_excluded
        df_summary_filtered = summary.df_results

    # Data points
    if effective_flags & PlotLayer.DATA_SCATTER:
        plt.scatter(
            df_range_filtered["sor#"],
            df_range_filtered["adjusted_run_time"],
            s=20,
            color="#00FFFF",  # cyan (bright) for dark bg
            edgecolors="#003A3A",  # subtle teal edge for separation
            linewidths=0.3,
            label="Data Points",
            alpha=0.2 if omit_fort else 0.5,
        )

    # Excluded-by-zscore points (optional)
    if (
        (effective_flags & PlotLayer.DATA_SCATTER_EXCLUDED)
        and (df_excluded_filtered is not None)
        and (not df_excluded_filtered.empty)
    ):
        plt.scatter(
            df_excluded_filtered["sor#"],
            df_excluded_filtered["adjusted_run_time"],
            s=24,
            marker="x",
            color="#FF3B30",  # bright red
            label="Excluded by Z-score",
            alpha=0.70 if omit_fort else 0.90,
        )

    # OLS predictions
    if effective_flags & PlotLayer.OLS_PRED_LINEAR:
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["linear_model_output"],
            color="#FFD60A",  # bright yellow (prediction)
            linewidth=2.2,
            label="Linear Model (OLS)",
        )
    if effective_flags & PlotLayer.OLS_PRED_QUAD:
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["quadratic_model_output"],
            color="#FF2DFF",  # fuchsia/magenta (prediction)
            linewidth=2.2,
            label="Quadratic Model (OLS)",
        )

    # WLS predictions
    if (effective_flags & PlotLayer.WLS_PRED_LINEAR) and (
        "linear_model_output_wls" in summary.df_results.columns
    ):
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["linear_model_output_wls"],
            color="#FF9F0A",  # orange/amber (prediction)
            linewidth=2.0,
            label="Linear Model (WLS)",
        )
    if (effective_flags & PlotLayer.WLS_PRED_QUAD) and (
        "quadratic_model_output_wls" in summary.df_results.columns
    ):
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["quadratic_model_output_wls"],
            color="#BF5AF2",  # violet (prediction)
            linewidth=2.0,
            label="Quadratic Model (WLS)",
        )

    # Cost per run curves (solid; distinct palette from prediction lines)
    # Define dedicated colors for cost curves to avoid overlap with prediction hues.
    _c_cost_lin_ols = "#00FFA2"  # neon mint (distinct from yellow/orange/blue/violet)
    _c_cost_quad_ols = "#00B3FF"  # azure
    _c_cost_lin_wls = "#F6FF00"  # neon yellow-green
    _c_cost_quad_wls = "#FF6BD6"  # pink

    if effective_flags & PlotLayer.OLS_COST_LINEAR:
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["cost_per_run_at_fort_lin"],
            color=_c_cost_lin_ols,
            linestyle="-",  # solid per request
            linewidth=2.2,
            label="Cost/Run @ FORT (Linear, OLS)",
        )
    if effective_flags & PlotLayer.OLS_COST_QUAD:
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["cost_per_run_at_fort_quad"],
            color=_c_cost_quad_ols,
            linestyle="-",  # solid per request
            linewidth=2.2,
            label="Cost/Run @ FORT (Quadratic, OLS)",
        )

    # WLS cost-per-run curves (solid)
    if (effective_flags & PlotLayer.WLS_COST_LINEAR) and (
        "cost_per_run_at_fort_lin_wls" in summary.df_results.columns
    ):
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["cost_per_run_at_fort_lin_wls"],
            color=_c_cost_lin_wls,
            linestyle="-",  # solid per request
            linewidth=2.2,
            label="Cost/Run @ FORT (Linear, WLS)",
        )
    if (effective_flags & PlotLayer.WLS_COST_QUAD) and (
        "cost_per_run_at_fort_quad_wls" in summary.df_results.columns
    ):
        plt.plot(
            df_summary_filtered["sor#"],
            df_summary_filtered["cost_per_run_at_fort_quad_wls"],
            color=_c_cost_quad_wls,
            linestyle="-",  # solid per request
            linewidth=2.2,
            label="Cost/Run @ FORT (Quadratic, WLS)",
        )

    # Min cost verticals: dotted lines using the SAME colors as their corresponding cost curves
    if effective_flags & PlotLayer.OLS_MIN_LINEAR:
        plt.axvline(
            x=summary.sor_min_cost_lin,
            color=_c_cost_lin_ols,
            linestyle=":",  # dotted per request
            linewidth=2.0,
            label="Min Cost (Linear, OLS)",
        )
    if effective_flags & PlotLayer.OLS_MIN_QUAD:
        plt.axvline(
            x=summary.sor_min_cost_quad,
            color=_c_cost_quad_ols,
            linestyle=":",  # dotted per request
            linewidth=2.0,
            label="Min Cost (Quadratic, OLS)",
        )

    if (effective_flags & PlotLayer.WLS_MIN_LINEAR) and (
        summary.sor_min_cost_lin_wls is not None
    ):
        plt.axvline(
            x=summary.sor_min_cost_lin_wls,
            color=_c_cost_lin_wls,
            linestyle=":",  # dotted per request
            linewidth=2.0,
            label="Min Cost (Linear, WLS)",
        )
    if (effective_flags & PlotLayer.WLS_MIN_QUAD) and (
        summary.sor_min_cost_quad_wls is not None
    ):
        plt.axvline(
            x=summary.sor_min_cost_quad_wls,
            color=_c_cost_quad_wls,
            linestyle=":",  # dotted per request
            linewidth=2.0,
            label="Min Cost (Quadratic, WLS)",
        )

    plt.xlabel("Sequential Online Run #")
    plt.ylabel("Run Time")
    # plt.title("FORT Regression Models")

    if effective_flags & PlotLayer.LEGEND:
        plt.legend()
        if omit_fort:
            add_legend_extra_line("SOR# = FORT omitted", color="red")

    # Apply axis limits from PlotParams; support one-sided bounds
    plt.xlim(
        left=plot_params.x_min if plot_params.x_min is not None else None,
        right=plot_params.x_max if plot_params.x_max is not None else None,
    )
    plt.ylim(
        bottom=plot_params.y_min if plot_params.y_min is not None else None,
        top=plot_params.y_max if plot_params.y_max is not None else None,
    )

    plt.savefig(output_svg, format="svg")
    plt.close()
    return output_svg


# --- Helper extractions to simplify main() ---


def render_plots(
    list_plot_params: List[PlotParams],
    df_included: pd.DataFrame,
    summary: SummaryModelOutputs,
    short_hash: str,
    df_excluded: Optional[pd.DataFrame] = None,
) -> list[str]:
    """
    Render one or more plots based on a list of PlotParams. Returns artifact paths.

    Filenames:
      - Always include zero-based index: plot-{short_hash}-{ii}-{suffix}.svg
        where ii is 0-based, zero-padded to the width of len(list_plot_params)-1 (e.g., 00, 01).

    DATA_SCATTER_EXCLUDED points are rendered only if requested by each PlotParams.
    """
    n = len(list_plot_params)
    # Always calculate padding to ensure zero-based indexing, even for single plots
    pad = max(2, len(str(max(0, n - 1)))) if n > 0 else 2
    artifact_paths: list[str] = []
    for idx, pp in enumerate(list_plot_params):
        flags = pp.plot_layers
        suffix = _plot_layers_suffix(flags)
        # Always include index in filename, even for single plots
        out_svg = f"plot-{short_hash}-{str(idx).zfill(pad)}-{suffix}.svg"
        render_outputs(
            df_included,
            summary,
            output_svg=out_svg,
            df_excluded=df_excluded
            if (flags & PlotLayer.DATA_SCATTER_EXCLUDED)
            else None,
            plot_params=pp,
        )
        artifact_paths.append(out_svg)
    return artifact_paths


def get_default_params() -> tuple[LoadSliceParams, TransformParams, PlotParams]:
    """
    Build default LoadSliceParams, TransformParams, and PlotParams.

    This function returns a single default PlotParams, which is used as the base
    for inheritance when parsing --plot-spec flags. The CLI can now accept multiple
    plot specifications that inherit from this default, allowing for flexible
    customization of plot parameters in a multi-plot system.
    """
    # Example defaults (preserve existing behavior)
    # log_path = Path("C:/Games/Utility/ICScriptHub/log-reset.csv").resolve()
    load = LoadSliceParams(
        log_path=None,
        start_line=None,
        end_line=None,
        include_header=True,
    )
    trans = TransformParams(
        zscore_min=-1.5,
        zscore_max=3,
        input_data_fort=100,
        ignore_resetticks=True,
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
        exclude_timestamp_ranges=None,  # [("20250801124409", "20250805165454")],
        verbose_filtering=False,
    )
    plotp = PlotParams(
        plot_layers=PlotLayer.DEFAULT,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )
    return load, trans, plotp


def build_run_identity(
    load: LoadSliceParams, trans: TransformParams
) -> tuple[str, str, str, dict]:
    """
    Returns (abs_input_posix, short_hash, full_hash, effective_params)
    """
    abs_input_posix = normalize_abs_posix(load.log_path)
    effective_params = build_effective_parameters(load, trans)
    canonical_payload = {
        "absolute_input_path": abs_input_posix,
        "effective_parameters": effective_params,
    }
    short_hash, full_hash = canonical_json_hash(canonical_payload)
    return abs_input_posix, short_hash, full_hash, effective_params


def build_model_comparison(diag: dict) -> tuple[str, str]:
    """
    Build the model-comparison table and return (best_label, table_text).

    Formatting:
      - Headers: [Model, BIC, AIC, RMSE_in, Adj R²]
      - Fixed notation only
      - Widths/decimals:
          BIC:  width=12, decimals=1
          AIC:  width=12, decimals=1
          RMSE: width=12, decimals=5
          AdjR²:width=10, decimals=5
      - Right-aligned numeric columns with trailing zero padding for decimals
      - Missing/non-finite rendered as "-" centered in the field
    """

    def _safe_get(d: dict, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    import math
    from typing import NamedTuple
    from typing import Optional as _Optional

    # Fixed-width format helpers
    def _fmt_fixed(x: float | None, width: int, decimals: int) -> str:
        """
        Format a number in fixed notation with specified width and decimals.
        Returns '-' centered in the field if x is None or not finite.
        """
        if x is None:
            s = "-"
        else:
            try:
                xf = float(x)
                if not math.isfinite(xf):
                    s = "-"
                else:
                    s = f"{xf:.{decimals}f}"
            except Exception:
                s = "-"
        # Right-align numbers; center '-' for missing
        if s == "-":
            return s.center(width)
        return s.rjust(width)

    class ModelRow(NamedTuple):
        label: str
        rmse: _Optional[float]
        adj_r2: _Optional[float]
        aic: _Optional[float]
        bic: _Optional[float]
        complexity_rank: int

    def _complexity_rank_for(label: str) -> int:
        is_linear = "Linear" in label and "Quadratic" not in label
        is_quadratic = "Quadratic" in label
        is_wls = "WLS" in label
        form_rank = 0 if is_linear else (1 if is_quadratic else 2)
        est_rank = 1 if is_wls else 0
        return form_rank * 10 + est_rank

    rows_spec = [
        ("OLS Linear", ("linear", "ols")),
        ("OLS Quadratic", ("quadratic", "ols")),
        ("WLS Linear (empirical)", ("linear", "wls")),
        ("WLS Quadratic (empirical)", ("quadratic", "wls")),
    ]

    raw_rows: list[ModelRow] = []
    for label, path in rows_spec:
        node = _safe_get(diag, *path, default={}) or {}
        rmse = _safe_get(diag, path[0], path[1], "RMSE", default=None)
        adj_r2 = _safe_get(node, "Adj. R-squared", default=None)
        aic = _safe_get(node, "aic", default=None)
        bic = _safe_get(node, "bic", default=None)
        raw_rows.append(
            ModelRow(
                label=label,
                rmse=float(rmse) if rmse is not None else None,
                adj_r2=float(adj_r2) if adj_r2 is not None else None,
                aic=float(aic) if aic is not None else None,
                bic=float(bic) if bic is not None else None,
                complexity_rank=_complexity_rank_for(label),
            )
        )

    def select_best_model(rows: list[ModelRow]) -> ModelRow:
        def pos_inf_if_none(x: _Optional[float]) -> float:
            return float("inf") if x is None or not np.isfinite(x) else float(x)

        def neg_inf_if_none(x: _Optional[float]) -> float:
            if x is None:
                return float("-inf")
            try:
                xv = float(x)
                return xv if np.isfinite(xv) else float("-inf")
            except Exception:
                return float("-inf")

        return sorted(
            rows,
            key=lambda r: (
                pos_inf_if_none(r.bic),
                pos_inf_if_none(r.aic),
                pos_inf_if_none(r.rmse),
                -neg_inf_if_none(r.adj_r2),
                r.complexity_rank,
                r.label,
            ),
        )[0]

    best_row = select_best_model(raw_rows)

    # Column layout and formats
    headers = ("Model", "BIC", "AIC", "RMSE_in", "Adj R²")
    width_bic, dec_bic = 12, 1
    width_aic, dec_aic = 12, 1
    width_rmse, dec_rmse = 12, 5
    width_adj, dec_adj = 10, 5

    # Format all rows according to fixed widths/decimals
    table_rows = [
        (
            r.label,
            _fmt_fixed(r.bic, width_bic, dec_bic),
            _fmt_fixed(r.aic, width_aic, dec_aic),
            _fmt_fixed(r.rmse, width_rmse, dec_rmse),
            _fmt_fixed(r.adj_r2, width_adj, dec_adj),
        )
        for r in raw_rows
    ]

    # Compute Model column width based on label/header; numeric are fixed
    col0_width = max(len(headers[0]), max(len(r[0]) for r in table_rows))
    header_line = (
        f"{headers[0]:<{col0_width}}  "
        f"{headers[1]:>{width_bic}}  "
        f"{headers[2]:>{width_aic}}  "
        f"{headers[3]:>{width_rmse}}  "
        f"{headers[4]:>{width_adj}}"
    )
    sep_line = "-" * len(header_line)

    lines = []
    lines.append("Model Comparison (OLS/WLS x Linear/Quadratic)")
    lines.append(header_line)
    lines.append(sep_line)
    for r in table_rows:
        lines.append(f"{r[0]:<{col0_width}}  {r[1]}  {r[2]}  {r[3]}  {r[4]}")
    lines.append("")
    lines.append(f"Selected model (by policy): {best_row.label}")
    lines.append("")

    return best_row.label, "\n".join(lines)


def build_manifest_dict(
    abs_input_posix: str,
    counts: dict,
    effective_params: dict,
    hashes: tuple[str, str],
    artifact_paths: list[str],
) -> dict:
    short_hash, full_hash = hashes
    exclusion_reasons = (
        f"timestamp_range_excluded_rows={counts.get('pre_zscore_excluded', 0)}; "
        f"zscore_excluded_rows={counts.get('excluded_row_count', 0)}"
    )
    return {
        "version": "1",
        "timestamp_utc": utc_timestamp_seconds(),
        "absolute_input_path": abs_input_posix,
        "total_input_rows": int(counts.get("total_input_rows", 0)),
        "processed_row_count": int(counts.get("processed_row_count", 0)),
        "excluded_row_count": int(counts.get("excluded_row_count", 0)),
        "exclusion_reasons": exclusion_reasons,
        "effective_parameters": effective_params,
        "canonical_hash": full_hash,
        "canonical_hash_short": short_hash,
        "artifacts": {"plot_svgs": artifact_paths},
    }


def assemble_text_report(
    input_df: pd.DataFrame,
    transformed: TransformOutputs,
    summary: SummaryModelOutputs,
    table_text: str,
    best_label: str,
) -> str:
    """
    Create a concise, readable report. Keeps current information content but in one place.
    """
    # 1) Prepare base sections (input head/tail, filtered head/tail)
    parts: list[str] = []
    parts.append("\n")

    def _fmt_head_tail(df: pd.DataFrame, n: int = 10) -> str:
        """
        Render head and tail with original headers and no index.
        If rows <= 2n, show only head to avoid duplication.
        """
        if df.empty:
            return "(no rows)"
        head_txt = df.head(n).to_string(index=False)
        tail_txt = df.tail(n).to_string(index=False)
        if len(df) <= 2 * n:
            return head_txt
        return f"{head_txt}\n...\n{tail_txt}"

    parts.append(f"Input data (head/tail):\n{_fmt_head_tail(input_df, n=5)}")
    parts.append("\n")
    parts.append(
        f"Filtered data (head/tail):\n{_fmt_head_tail(transformed.df_range, n=5)}"
    )
    parts.append("\n")
    parts.append(
        f"Excluded data (head/tail):\n{_fmt_head_tail(transformed.df_excluded, n=5)}"
    )
    parts.append("\n")

    # 1.5) Insert model results preview (head and tail) with shortened headers
    def _shorten_headers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a shallow header-renamed view for display only.
        This does not mutate the original DataFrame.
        """
        rename_map = {
            "sor#": "sor",
            "linear_model_output": "lin",
            "quadratic_model_output": "quad",
            "linear_model_output_wls": "lin_wls",
            "quadratic_model_output_wls": "quad_wls",
            "sum_lin": "Σlin",
            "sum_quad": "Σquad",
            "sum_lin_wls": "Σlin_wls",
            "sum_quad_wls": "Σquad_wls",
            "cost_per_run_at_fort_lin": "cpr_lin",
            "cost_per_run_at_fort_quad": "cpr_quad",
            "cost_per_run_at_fort_lin_wls": "cpr_lin_wls",
            "cost_per_run_at_fort_quad_wls": "cpr_quad_wls",
        }
        # Only rename columns that exist to avoid KeyError
        present = {k: v for k, v in rename_map.items() if k in df.columns}
        return df.rename(columns=present)

    def _fmt_table(df: pd.DataFrame, n: int = 10) -> str:
        """
        Render head and tail with no index for compactness.
        """
        if df.empty:
            return "(no rows)"
        df_disp = _shorten_headers(df)
        head_txt = df_disp.head(n).to_string(index=False)
        tail_txt = df_disp.tail(n).to_string(index=False)
        # If total rows <= 2n, head and tail overlap; show only head
        if len(df) <= 2 * n:
            return head_txt
        return f"{head_txt}\n...\n{tail_txt}"

    parts.append("Model results (head/tail):")
    parts.append(_fmt_table(summary.df_results, n=5))
    parts.append("\n")

    # 1.6) Insert summary of run-time by SOR range (df_summary) right after results
    def _shorten_summary_headers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Display-only header shortening for df_summary.
        """
        rename_map = {
            "sorr_start": "start",
            "sorr_end": "end",
            "run_time_mean": "rt_mean",
            "run_time_delta": "rt_delta",
        }
        present = {k: v for k, v in rename_map.items() if k in df.columns}
        return df.rename(columns=present)

    def _fmt_summary(df: pd.DataFrame) -> str:
        if df.empty:
            return "(no summary rows)"
        df_disp = _shorten_summary_headers(df)
        # Show full summary; it's only a handful of rows (<=5)
        return df_disp.to_string(index=False)

    parts.append("Run-time summary by SOR range:")
    parts.append(_fmt_summary(summary.df_summary))
    parts.append("\n")

    # 2) Trim the selected-model tail from the model comparison table
    #    The original builder appends: "", "Selected model (by policy): …", ""
    #    We remove the final two trailing lines if present.
    tbl_lines = table_text.splitlines()

    def _trim_selected_tail(lines: list[str]) -> list[str]:
        if not lines:
            return lines
        # Remove trailing empty lines first
        while lines and lines[-1].strip() == "":
            lines.pop()
        # If last line starts with Selected model..., drop it
        if lines and lines[-1].lstrip().startswith("Selected model (by policy):"):
            lines.pop()
        # Remove any trailing blank again to keep consistent spacing
        while lines and lines[-1].strip() == "":
            lines.pop()
        # Add exactly one terminating blank line to separate from next section
        lines.append("")
        return lines

    trimmed_table_text = "\n".join(_trim_selected_tail(tbl_lines))
    parts.append(trimmed_table_text)

    # 3) Build a local ranking mirroring build_model_comparison policy
    #    Tie-breakers: BIC asc, AIC asc, RMSE asc, Adj R² desc, complexity_rank asc, label asc.
    from typing import NamedTuple
    from typing import Optional as _Optional

    def _safe_get(d: dict, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    class ModelRow(NamedTuple):
        label: str
        rmse: _Optional[float]
        adj_r2: _Optional[float]
        aic: _Optional[float]
        bic: _Optional[float]
        complexity_rank: int
        sor_value: _Optional[int]

    def _complexity_rank_for(label: str) -> int:
        is_linear = "Linear" in label and "Quadratic" not in label
        is_quadratic = "Quadratic" in label
        is_wls = "WLS" in label
        form_rank = 0 if is_linear else (1 if is_quadratic else 2)
        est_rank = 1 if is_wls else 0
        return form_rank * 10 + est_rank

    diag = summary.regression_diagnostics

    # Rows spec holds canonical labels and value sources; display labels will be normalized below
    rows_spec = [
        ("OLS Linear", ("linear", "ols"), summary.sor_min_cost_lin),
        ("OLS Quadratic", ("quadratic", "ols"), summary.sor_min_cost_quad),
        ("WLS Linear (empirical)", ("linear", "wls"), summary.sor_min_cost_lin_wls),
        (
            "WLS Quadratic (empirical)",
            ("quadratic", "wls"),
            summary.sor_min_cost_quad_wls,
        ),
    ]

    # Label normalization to short forms per requirement
    def _normalize_label(lbl: str) -> str:
        mapping = {
            "OLS Linear": "OLS linear",
            "OLS Quadratic": "OLS quadratic",
            "WLS Linear (empirical)": "WLS linear",
            "WLS Quadratic (empirical)": "WLS quadratic",
        }
        return mapping.get(lbl, lbl)

    raw_rows: list[ModelRow] = []
    for label, path, sor_val in rows_spec:
        node = _safe_get(diag, *path, default={}) or {}
        rmse = _safe_get(diag, path[0], path[1], "RMSE", default=None)
        adj_r2 = _safe_get(node, "Adj. R-squared", default=None)
        aic = _safe_get(node, "aic", default=None)
        bic = _safe_get(node, "bic", default=None)
        raw_rows.append(
            ModelRow(
                label=label,
                rmse=float(rmse) if rmse is not None else None,
                adj_r2=float(adj_r2) if adj_r2 is not None else None,
                aic=float(aic) if aic is not None else None,
                bic=float(bic) if bic is not None else None,
                complexity_rank=_complexity_rank_for(label),
                sor_value=int(sor_val) if sor_val is not None else None,
            )
        )

    def _pos_inf_if_none(x: _Optional[float]) -> float:
        return float("inf") if x is None or not np.isfinite(x) else float(x)

    def _neg_inf_if_none(x: _Optional[float]) -> float:
        if x is None:
            return float("-inf")
        try:
            xv = float(x)
            return xv if np.isfinite(xv) else float("-inf")
        except Exception:
            return float("-inf")

    ranked = sorted(
        raw_rows,
        key=lambda r: (
            _pos_inf_if_none(r.bic),
            _pos_inf_if_none(r.aic),
            _pos_inf_if_none(r.rmse),
            -_neg_inf_if_none(r.adj_r2),
            r.complexity_rank,
            r.label,
        ),
    )

    # 4) Render only the four FORT lines in ranked order with dynamic right-edge alignment.
    # Robust alignment strategy:
    # - Build the exact left segment that will be printed per line: f"  ({idx}) {label}: "
    # - Compute dynamic max widths for label and value strings.
    # - Pad so that the right-most digit of each value aligns to the same column.
    def _value_str(val: _Optional[int]) -> str:
        return "-" if val is None else str(int(val))

    # Normalize labels and precompute value strings
    rows_disp: list[tuple[str, str]] = []
    for row in ranked:
        rows_disp.append((_normalize_label(row.label), _value_str(row.sor_value)))

    max_label_len = max((len(lbl) for lbl, _ in rows_disp), default=0)
    max_val_width = max((len(v) for _, v in rows_disp), default=1)

    parts.append("\nFORTs for lowest cost/run (models ordered by fit quality)")

    # Render using exact left segment and computed padding
    for idx, (disp_label, val_str) in enumerate(rows_disp, start=1):
        left = f"  ({idx}) {disp_label}: "
        # Base one space is included in 'left' after the colon; we only need to
        # add extra padding so that the value's right edge aligns.
        extra_pad = (max_label_len - len(disp_label)) + (max_val_width - len(val_str))
        if extra_pad < 0:
            extra_pad = 0
        parts.append(f"{left}{' ' * extra_pad}{val_str}")

    return "\n".join(parts)


def _orchestrate(
    params_load: LoadSliceParams,
    params_transform: TransformParams,
    list_plot_params: List[PlotParams],
) -> None:
    """
    Orchestrate the full pipeline given explicit parameter objects.
    Split from main() so the CLI can remain thin and tests can call this directly.
    """
    abs_input_posix, short_hash, full_hash, effective_params = build_run_identity(
        params_load, params_transform
    )

    df_range = load_and_slice_csv(params_load)
    transformed = transform_pipeline(df_range, params_transform)
    summary = summarize_and_model(transformed.df_range, params_transform)

    best_label, table_text = build_model_comparison(summary.regression_diagnostics)

    artifact_paths = render_plots(
        list_plot_params=list_plot_params,
        df_included=transformed.df_range,
        summary=summary,
        short_hash=short_hash,
        df_excluded=transformed.df_excluded,
    )

    total_input_rows = int(len(df_range))
    processed_row_count = int(len(transformed.df_range))
    excluded_row_count = int(len(transformed.df_excluded))
    pre_zscore_excluded = max(
        total_input_rows - processed_row_count - excluded_row_count, 0
    )
    counts = {
        "total_input_rows": total_input_rows,
        "processed_row_count": processed_row_count,
        "excluded_row_count": excluded_row_count,
        "pre_zscore_excluded": pre_zscore_excluded,
    }
    manifest = build_manifest_dict(
        abs_input_posix=abs_input_posix,
        counts=counts,
        effective_params=effective_params,
        hashes=(short_hash, full_hash),
        artifact_paths=artifact_paths,
    )
    write_manifest(f"manifest-{short_hash}.json", manifest)

    report = assemble_text_report(
        df_range, transformed, summary, table_text, best_label
    )
    print(report)


def _parse_plot_spec_kv(spec: str, default: PlotParams) -> PlotParams:
    """
    Parse a plot specification in key=value[,key=value...] format.
    """
    # Create a copy of the default params to avoid modifying the original
    params = PlotParams(
        plot_layers=default.plot_layers,
        x_min=default.x_min,
        x_max=default.x_max,
        y_min=default.y_min,
        y_max=default.y_max,
    )
    for kv in spec.split(","):
        key, value = kv.split("=")
        key = key.strip()
        value = value.strip()
        if key == "layers":
            params.plot_layers = _parse_plot_layers(value)
        elif key == "x_min":
            params.x_min = float(value)
        elif key == "x_max":
            params.x_max = float(value)
        elif key == "y_min":
            params.y_min = float(value)
        elif key == "y_max":
            params.y_max = float(value)
        else:
            raise ValueError(f"Unknown key in plot spec: {key}")
    return params


def _parse_plot_spec_json(spec: str, default: PlotParams) -> PlotParams:
    """
    Parse a plot specification as a JSON object.
    """
    import json

    # Create a copy of the default params to avoid modifying the original
    params = PlotParams(
        plot_layers=default.plot_layers,
        x_min=default.x_min,
        x_max=default.x_max,
        y_min=default.y_min,
        y_max=default.y_max,
    )
    spec_dict = json.loads(spec)
    if "layers" in spec_dict:
        params.plot_layers = _parse_plot_layers(spec_dict["layers"])
    # Only coerce numeric bounds when JSON value is not null. Preserve None to mean "no limit".
    if "x_min" in spec_dict and spec_dict["x_min"] is not None:
        params.x_min = float(spec_dict["x_min"])
    if "x_max" in spec_dict and spec_dict["x_max"] is not None:
        params.x_max = float(spec_dict["x_max"])
    if "y_min" in spec_dict and spec_dict["y_min"] is not None:
        params.y_min = float(spec_dict["y_min"])
    if "y_max" in spec_dict and spec_dict["y_max"] is not None:
        params.y_max = float(spec_dict["y_max"])
    return params


def _parse_plot_layers(spec: str) -> PlotLayer:
    """
    Parse plot layer specification.
    Accepts preset names (e.g., 'DEFAULT') or '+'-joined atomic names
    (e.g., 'DATA_SCATTER+OLS_PRED_LINEAR+LEGEND'), case-insensitive.
    """
    s = spec.strip().upper()
    if hasattr(PlotLayer, s):
        return getattr(PlotLayer, s)
    flags = PlotLayer(0)
    for token in s.split("+"):
        token = token.strip()
        if not token:
            continue
        if not hasattr(PlotLayer, token):
            raise ValueError(f"Unknown plot layer token: {token}")
        flags |= getattr(PlotLayer, token)
    return flags


def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="fort-calc",
        description="FORT Calculator pipeline (load -> transform -> model -> plot).",
        # Keep formatter; defaults we inject will be shown when building a help-only parser.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--print-defaults",
        action="store_true",
        help="Print default parameter values and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full tracebacks for debugging (also FORT_CALC_DEBUG=1).",
    )

    # LoadSliceParams
    g_load = parser.add_argument_group("LoadSliceParams")
    g_load.add_argument(
        "--log-path", type=str, required=True, help="Path to CSV log file (required)."
    )
    # Do not specify default at definition-time; we inject for help rendering only.
    g_load.add_argument("--start-line", type=int, help="1-based inclusive start line.")
    g_load.add_argument("--end-line", type=int, help="1-based inclusive end line.")
    g_load.add_argument(
        "--header-map",
        action="append",
        metavar="OLD:NEW",
        help="Map input header OLD to canonical NEW. Repeatable; format OLD:NEW.",
    )
    # Headers are required, let's not give users a parameter that just breaks the entire program
    # g_load.add_argument(
    #     "--no-header",
    #     action="store_true",
    #     help="Slice has no header (include_header=False).",
    # )

    # TransformParams
    g_tr = parser.add_argument_group("TransformParams")
    # Do not set defaults here; we will inject for help display.
    g_tr.add_argument("--zscore-min", type=float, help="Minimum z-score bound.")
    g_tr.add_argument("--zscore-max", type=float, help="Maximum z-score bound.")
    g_tr.add_argument(
        "--fort",
        type=int,
        dest="input_data_fort",
        help="Target SOR run (input_data_fort).",
    )
    # Boolean pair to allow explicit choice; default comes from get_default_params
    g_tr.add_argument(
        "--ignore-resetticks",
        dest="ignore_resetticks",
        action="store_true",
        help="Ignore resetticks when computing adjusted_run_time.",
    )
    g_tr.add_argument(
        "--use-resetticks",
        dest="ignore_resetticks",
        action="store_false",
        help="Subtract resetticks from runticks when computing adjusted_run_time.",
    )
    g_tr.add_argument(
        "--delta-mode",
        choices=["PREVIOUS_CHUNK", "FIRST_CHUNK"],
        help="Delta mode for summarize_run_time_by_sor_range.",
    )
    g_tr.add_argument(
        "--exclude-range",
        action="append",
        metavar="START,END",
        help="Exclude timestamp range(s) YYYYMMDDHHMMSS,YYYYMMDDHHMMSS. May be repeated.",
    )
    g_tr.add_argument(
        "--verbose-filtering",
        action="store_true",
        help="Enable verbose diagnostics during filtering.",
    )
    g_tr.add_argument(
        "--no-fail-on-invalid-ts",
        dest="fail_on_any_invalid_timestamps",
        action="store_false",
        help="Do not fail when any timestamps fail to parse.",
    )

    # PlotParams
    g_plot = parser.add_argument_group("PlotParams")
    g_plot.add_argument(
        "--plot-spec",
        action="append",
        help="Plot specification in key=value[,key=value...] format. Repeatable.",
    )
    g_plot.add_argument(
        "--plot-spec-json",
        action="append",
        help="Plot specification as a JSON object. Repeatable.",
    )
    g_plot.add_argument(
        "--plot-layers",
        type=str,
        help="(Deprecated) Preset or '+'-joined atomic flags (e.g., DEFAULT or DATA_SCATTER+OLS_PRED_LINEAR+LEGEND).",
    )
    g_plot.add_argument("--x-min", type=float, default=None, help="X axis minimum.")
    g_plot.add_argument("--x-max", type=float, default=None, help="X axis maximum.")
    g_plot.add_argument("--y-min", type=float, default=None, help="Y axis minimum.")
    g_plot.add_argument("--y-max", type=float, default=None, help="Y axis maximum.")

    return parser


def _args_to_params(args) -> tuple[LoadSliceParams, TransformParams, List[PlotParams]]:
    """
    Merge CLI args over defaults to build parameter objects.
    Only override values explicitly provided by user; otherwise keep defaults.
    """
    d_load, d_trans, d_plot = get_default_params()

    # LoadSliceParams
    log_path = (
        Path(args.log_path).resolve()
        if getattr(args, "log_path", None)
        else d_load.log_path
    )
    include_header = d_load.include_header
    if hasattr(args, "no_header") and args.no_header:
        include_header = False

    # Parse header_map if provided as repeatable --header-map OLD:NEW
    header_map: dict[str, str] = {}
    if getattr(args, "header_map", None):
        for item in args.header_map:
            try:
                old, new = item.split(":", 1)
            except ValueError:
                raise ValueError(
                    f"Invalid --header-map value: '{item}'. Expected OLD:NEW"
                )
            old = old.strip()
            new = new.strip()
            if not old or not new:
                raise ValueError(
                    f"Invalid --header-map value: '{item}'. OLD and NEW must be non-empty"
                )
            header_map[old] = new

    load = LoadSliceParams(
        log_path=log_path,
        start_line=args.start_line
        if args.start_line is not None
        else d_load.start_line,
        end_line=args.end_line if args.end_line is not None else d_load.end_line,
        include_header=include_header,
        header_map=header_map,
    )

    # TransformParams
    delta_mode = d_trans.delta_mode
    if getattr(args, "delta_mode", None):
        delta_mode = DeltaMode[args.delta_mode]

    exclude_ranges = d_trans.exclude_timestamp_ranges
    if getattr(args, "exclude_range", None):
        parsed: list[tuple[str, str]] = []
        for item in args.exclude_range:
            try:
                start_s, end_s = [p.strip() for p in item.split(",", 1)]
            except ValueError:
                raise ValueError(
                    f"Invalid --exclude-range value: '{item}'. Expected START,END"
                )
            parsed.append((start_s, end_s))
        exclude_ranges = parsed

    ignore_resetticks = d_trans.ignore_resetticks
    if hasattr(args, "ignore_resetticks") and args.ignore_resetticks is not None:
        ignore_resetticks = args.ignore_resetticks

    fail_on_invalid = d_trans.fail_on_any_invalid_timestamps
    if (
        hasattr(args, "fail_on_any_invalid_timestamps")
        and args.fail_on_any_invalid_timestamps is not None
    ):
        fail_on_invalid = args.fail_on_any_invalid_timestamps

    transform = TransformParams(
        zscore_min=args.zscore_min
        if args.zscore_min is not None
        else d_trans.zscore_min,
        zscore_max=args.zscore_max
        if args.zscore_max is not None
        else d_trans.zscore_max,
        input_data_fort=args.input_data_fort
        if args.input_data_fort is not None
        else d_trans.input_data_fort,
        ignore_resetticks=ignore_resetticks,
        delta_mode=delta_mode,
        exclude_timestamp_ranges=exclude_ranges,
        verbose_filtering=bool(
            getattr(args, "verbose_filtering", d_trans.verbose_filtering)
        ),
        fail_on_any_invalid_timestamps=fail_on_invalid,
    )

    # PlotParams
    plot_params_list = []
    if getattr(args, "plot_spec", None):
        for spec in args.plot_spec:
            plot_params_list.append(_parse_plot_spec_kv(spec, d_plot))
    if getattr(args, "plot_spec_json", None):
        for spec in args.plot_spec_json:
            plot_params_list.append(_parse_plot_spec_json(spec, d_plot))
    if not plot_params_list:
        # No plot-specs provided; use default configurations
        if getattr(args, "plot_layers", None):
            # Use deprecated plot-layers if provided
            plot = PlotParams(
                plot_layers=_parse_plot_layers(args.plot_layers),
                x_min=args.x_min if args.x_min is not None else d_plot.x_min,
                x_max=args.x_max if args.x_max is not None else d_plot.x_max,
                y_min=args.y_min if args.y_min is not None else d_plot.y_min,
                y_max=args.y_max if args.y_max is not None else d_plot.y_max,
            )
            logger.warning(
                "--plot-layers is deprecated; use --plot-spec or --plot-spec-json instead."
            )
            plot_params_list.append(plot)
        else:
            # Use three default configurations when no specific plots are requested
            default_plot_0 = PlotParams(
                plot_layers=PlotLayer.DATA_SCATTER | PlotLayer.ALL_PREDICTION,
                x_min=args.x_min if args.x_min is not None else d_plot.x_min,
                x_max=OMIT_FORT,
                y_min=args.y_min if args.y_min is not None else d_plot.y_min,
                y_max=args.y_max if args.y_max is not None else d_plot.y_max,
            )

            default_plot_1 = PlotParams(
                plot_layers=PlotLayer.ALL_COST | PlotLayer.MIN_MARKERS_ONLY,
                x_min=args.x_min if args.x_min is not None else d_plot.x_min,
                x_max=args.x_max if args.x_max is not None else d_plot.x_max,
                y_min=args.y_min if args.y_min is not None else d_plot.y_min,
                y_max=args.y_max if args.y_max is not None else d_plot.y_max,
            )

            default_plot_2 = PlotParams(
                plot_layers=PlotLayer.ALL_SCATTER,
                x_min=args.x_min if args.x_min is not None else d_plot.x_min,
                x_max=args.x_max if args.x_max is not None else d_plot.x_max,
                y_min=args.y_min if args.y_min is not None else d_plot.y_min,
                y_max=args.y_max if args.y_max is not None else d_plot.y_max,
            )

            plot_params_list.extend([default_plot_0, default_plot_1, default_plot_2])
    else:
        if getattr(args, "plot_layers", None):
            logger.warning(
                "--plot-layers is ignored when --plot-spec or --plot-spec-json is provided."
            )
        if args.x_min or args.x_max or args.y_min or args.y_max:
            logger.warning(
                "Top-level x/y min/max are ignored when --plot-spec or --plot-spec-json is provided."
            )

    return load, transform, plot_params_list


def _build_cli_parser_with_policy_defaults():
    """
    Build a display-only parser whose defaults are injected from get_default_params()
    so that -h/--help shows values synchronized with policy defaults. This parser is
    used ONLY for help rendering; normal execution uses the undecorated parser
    combined with _args_to_params() merging over get_default_params().
    """
    # Start from a fresh parser
    parser = _build_cli_parser()

    # Pull policy defaults
    d_load, d_trans, d_plot = get_default_params()

    # Map: CLI dest name -> default value to display
    injected_defaults = {
        # LoadSliceParams
        "log_path": str(d_load.log_path),
        "start_line": d_load.start_line,
        "end_line": d_load.end_line,
        "no_header": (not d_load.include_header),
        # TransformParams
        "zscore_min": d_trans.zscore_min,
        "zscore_max": d_trans.zscore_max,
        "input_data_fort": d_trans.input_data_fort,
        "ignore_resetticks": d_trans.ignore_resetticks,
        "delta_mode": d_trans.delta_mode.name,
        "exclude_range": None,  # remains unset by default
        "verbose_filtering": d_trans.verbose_filtering,
        "fail_on_any_invalid_timestamps": d_trans.fail_on_any_invalid_timestamps,
        # PlotParams
        "plot_layers": _plot_layers_suffix(d_plot.plot_layers),
        "x_min": d_plot.x_min,
        "x_max": d_plot.x_max,
        "y_min": d_plot.y_min,
        "y_max": d_plot.y_max,
        # Utility
        "print_defaults": False,
    }

    # Apply to each action so ArgumentDefaultsHelpFormatter displays them
    for action in parser._actions:
        if action.dest in injected_defaults:
            action.default = injected_defaults[action.dest]

    return parser


def main() -> None:
    """
    CLI entry point. Parses arguments, builds parameter objects, then orchestrates.
    With no CLI args, defaults from get_default_params() are used.
    """
    import sys

    argv = sys.argv[1:]

    # Help path: render help with a parser whose defaults mirror policy
    # Accept common mistyped long forms as help, but do NOT partially match other options.
    help_aliases = {"-h", "--help", "--he", "--hel", "--h"}
    if any(x in help_aliases for x in argv):
        parser_for_help = _build_cli_parser_with_policy_defaults()
        # Print help directly to ensure ArgumentDefaultsHelpFormatter uses injected defaults
        parser_for_help.print_help()
        return

    # Build the execution parser (no injected defaults for behavior)
    parser = _build_cli_parser()

    # If user requested --print-defaults, honor it without requiring --log-path.
    if "--print-defaults" in argv:
        import json

        d_load, d_trans, d_plot = get_default_params()

        # Preserve JSON types: only stringify Path when present; keep None as None (-> null)
        log_path_json = None if d_load.log_path is None else str(d_load.log_path)

        payload = {
            "LoadSliceParams": {
                "log_path": log_path_json,
                "start_line": d_load.start_line,
                "end_line": d_load.end_line,
                "include_header": d_load.include_header,
            },
            "TransformParams": {
                "zscore_min": d_trans.zscore_min,
                "zscore_max": d_trans.zscore_max,
                "input_data_fort": d_trans.input_data_fort,
                "ignore_resetticks": d_trans.ignore_resetticks,
                "delta_mode": d_trans.delta_mode.name,
                "exclude_timestamp_ranges": d_trans.exclude_timestamp_ranges,
                "verbose_filtering": d_trans.verbose_filtering,
                "fail_on_any_invalid_timestamps": d_trans.fail_on_any_invalid_timestamps,
            },
            "PlotParams": {
                "plot_layers": _plot_layers_suffix(d_plot.plot_layers),
                "x_min": d_plot.x_min,
                "x_max": d_plot.x_max,
                "y_min": d_plot.y_min,
                "y_max": d_plot.y_max,
            },
        }
        print(json.dumps(payload, indent=2))
        return

    # Regular flow: parse args (this will require --log-path) and run pipeline.
    args = parser.parse_args(argv)
    # Enable debug mode via --debug flag or environment variable FORT_CALC_DEBUG=1
    debug_mode = bool(
        getattr(args, "debug", False) or os.getenv("FORT_CALC_DEBUG", "") == "1"
    )
    params_load, params_transform, plot_params_list = _args_to_params(args)

    if debug_mode:
        logger.setLevel(logging.DEBUG)

    try:
        _orchestrate(params_load, params_transform, plot_params_list)
    except (FileNotFoundError, ValueError, TypeError) as e:
        # Concise, user-facing errors for common/user-correctable problems.
        logger.info("User-facing error: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        # Unexpected/internal errors: log full exception. Show traceback only when debugging.
        logger.exception("Unhandled exception during execution")
        if debug_mode:
            # Local import to avoid top-level unused import while fixing the undefined name
            # reported by linters (Ruff / Pylance). Local import keeps the scope tight and
            # only incurs cost when debug output is requested.
            import traceback

            traceback.print_exc()
        else:
            print(f"Unexpected error: {e}", file=sys.stderr)
            print(
                "Run with --debug or set FORT_CALC_DEBUG=1 to see the full traceback.",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
