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
from dataclasses import dataclass
from enum import Enum, IntFlag, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Support both package and script execution modes
try:
    # When run as a package: python -m src.main
    from .csv_processor import CSVRangeProcessor
    from .utils import (
        build_effective_parameters,
        canonical_json_hash,
        normalize_abs_posix,
        utc_timestamp_seconds,
        with_hash_suffix,
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
        with_hash_suffix,
        write_manifest,
    )

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    def set_skipped(self, reason: str) -> None:
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
        result.set_skipped("no exclude ranges provided")
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
    log_path: Path
    start_line: Optional[int]
    end_line: Optional[int]
    include_header: bool = True


class PlotLayer(IntFlag):
    # Data
    DATA_SCATTER = 1 << 0

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
    EVERYTHING = DEFAULT | ALL_WLS


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
    # Plot layer configuration via flags
    plot_layers: PlotLayer = PlotLayer.DEFAULT


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
    - Additionally trains WLS variants and returns their predictions alongside OLS.
    - Builds prediction matrices with named columns and explicit constant handling
      via statsmodels add_constant(has_constant='add') to avoid column-order issues.

    Variants included in diagnostics:
      - 'ols':     Baseline OLS
      - 'ols_hc1': OLS with heteroskedasticity-robust covariance (HC1)
      - 'wls':     Weighted Least Squares with weights 1 / (sor#^2)
      - 'wls_hc1': WLS with HC1 robust covariance

    Returns:
      tuple[pd.DataFrame, dict]
        - result_df: DataFrame with columns:
            ["sor#", "linear_model_output", "quadratic_model_output",
             "linear_model_output_wls", "quadratic_model_output_wls"]
        - diagnostics: Nested dict containing model statistics for all variants.

    Notes:
    - WLS weights default to 1/(sor#^2) with robust handling of zero/NaN/inf values.
      Non-finite weights are replaced by the median finite weight; if none are finite,
      the code falls back to uniform weights.
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

    # Variant: WLS and WLS with robust SEs
    # Default weights strategy: 1 / (sor#^2) guarding zero/NaN
    linear_model_output_wls = None
    quadratic_model_output_wls = None
    try:
        sor_vals = pd.to_numeric(df_range["sor#"], errors="coerce").to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / np.where(
                np.isfinite(sor_vals) & (sor_vals > 0), sor_vals**2, np.nan
            )
        # Replace NaN/Inf weights with median of finite weights to keep WLS stable
        if not np.isfinite(w).any():
            # Fallback: all invalid -> uniform weights
            w = np.ones_like(sor_vals, dtype=float)
        else:
            finite_mask = np.isfinite(w)
            median_w = float(np.nanmedian(w[finite_mask]))
            w = np.where(
                np.isfinite(w),
                w,
                median_w if np.isfinite(median_w) and median_w > 0 else 1.0,
            )

        lin_wls = sm.WLS(y, X_linear_train, weights=w).fit()
        quad_wls = sm.WLS(y, X_quadratic_train, weights=w).fit()
        diagnostics["linear"]["wls"] = {
            **_stats_dict(lin_wls),
            "weights_spec": "1/(sor#^2)",
        }
        diagnostics["quadratic"]["wls"] = {
            **_stats_dict(quad_wls),
            "weights_spec": "1/(sor#^2)",
        }

        # Attach AIC/BIC for WLS fits
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

        # Compute in-sample RMSEs for WLS fits using unweighted residuals for comparability
        try:
            resid_lin_wls = y - lin_wls.fittedvalues
            resid_quad_wls = y - quad_wls.fittedvalues
            rmse_lin_wls = float(np.sqrt(np.mean(np.square(resid_lin_wls))))
            rmse_quad_wls = float(np.sqrt(np.mean(np.square(resid_quad_wls))))
            diagnostics["linear"]["wls"]["RMSE"] = rmse_lin_wls
            diagnostics["quadratic"]["wls"]["RMSE"] = rmse_quad_wls
        except Exception as _e_rmse_wls:
            diagnostics["linear"]["wls"]["rmse_exception"] = str(_e_rmse_wls)
            diagnostics["quadratic"]["wls"]["rmse_exception"] = str(_e_rmse_wls)

        # Align prediction matrices for WLS models (same exog names)
        X_linear_pred_wls = X_linear_pred.reindex(columns=lin_wls.model.exog_names)
        X_quadratic_pred_wls = X_quadratic_pred.reindex(
            columns=quad_wls.model.exog_names
        )
        linear_model_output_wls = lin_wls.predict(X_linear_pred_wls)
        quadratic_model_output_wls = quad_wls.predict(X_quadratic_pred_wls)

        # WLS + robust SEs (HC1)
        try:
            lin_wls_hc1 = sm.WLS(y, X_linear_train, weights=w).fit(cov_type="HC1")
            quad_wls_hc1 = sm.WLS(y, X_quadratic_train, weights=w).fit(cov_type="HC1")
            diagnostics["linear"]["wls_hc1"] = {
                **_stats_dict(lin_wls_hc1),
                "weights_spec": "1/(sor#^2)",
            }
            diagnostics["quadratic"]["wls_hc1"] = {
                **_stats_dict(quad_wls_hc1),
                "weights_spec": "1/(sor#^2)",
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
    # Build the pipeline step-by-step so we can fail-fast immediately after timestamp parsing
    df_range = df_range.pipe(
        clean_ignore,
        input_data_fort=params.input_data_fort,
        verbose=params.verbose_filtering,
    )
    df_range = compute_adjusted_run_time(
        df_range, ignore_resetticks=params.ignore_resetticks
    )
    # Convert timestamps early, once
    df_range = convert_timestamp_column_to_datetime(
        df_range, timestamp_column="timestamp", verbose=params.verbose_filtering
    )
    # Fail fast on invalid timestamps immediately after parsing (before any downstream filtering)
    if "timestamp" in df_range.columns:
        # Prefer metrics persisted by the converter for robustness against view/copy behavior
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
        if params.fail_on_any_invalid_timestamps and invalid_ts_count > 0:
            raise ValueError(
                f"Invalid timestamps detected after parsing: {invalid_ts_count} of {total_rows} rows are NaT"
            )
    # Continue with the rest of the pipeline
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

    df_results, regression_diagnostics = regression_analysis(
        df_range, params.input_data_fort - 1
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


def render_outputs(
    df_range: pd.DataFrame,
    summary: SummaryModelOutputs,
    output_svg: str = "plot.svg",
    plot_layers: Optional[PlotLayer] = None,
) -> str:
    """
    Pure renderer: builds the plot and returns the output path.
    No prints; deterministic given inputs.

    plot_layers:
      - Bitflag (PlotLayer) selecting which visual elements to draw.
      - If None, defaults to PlotLayer.DEFAULT.
    """
    if plot_layers is None:
        plot_layers = PlotLayer.DEFAULT

    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))

    # Data points
    if plot_layers & PlotLayer.DATA_SCATTER:
        plt.scatter(
            df_range["sor#"],
            df_range["adjusted_run_time"],
            s=20,
            color="cyan",
            label="Data Points",
        )

    # OLS predictions
    if plot_layers & PlotLayer.OLS_PRED_LINEAR:
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["linear_model_output"],
            color="yellow",
            label="Linear Model (OLS)",
        )
    if plot_layers & PlotLayer.OLS_PRED_QUAD:
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["quadratic_model_output"],
            color="magenta",
            label="Quadratic Model (OLS)",
        )

    # WLS predictions
    if (plot_layers & PlotLayer.WLS_PRED_LINEAR) and (
        "linear_model_output_wls" in summary.df_results.columns
    ):
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["linear_model_output_wls"],
            color="orange",
            linestyle="-.",
            label="Linear Model (WLS)",
        )
    if (plot_layers & PlotLayer.WLS_PRED_QUAD) and (
        "quadratic_model_output_wls" in summary.df_results.columns
    ):
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["quadratic_model_output_wls"],
            color="violet",
            linestyle="-.",
            label="Quadratic Model (WLS)",
        )

    # OLS cost per run curves
    if plot_layers & PlotLayer.OLS_COST_LINEAR:
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["cost_per_run_at_fort_lin"],
            color="green",
            linestyle="--",
            label="Cost/Run @ FORT (Linear, OLS)",
        )
    if plot_layers & PlotLayer.OLS_COST_QUAD:
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["cost_per_run_at_fort_quad"],
            color="blue",
            linestyle="--",
            label="Cost/Run @ FORT (Quadratic, OLS)",
        )

    # WLS cost-per-run curves
    if (plot_layers & PlotLayer.WLS_COST_LINEAR) and (
        "cost_per_run_at_fort_lin_wls" in summary.df_results.columns
    ):
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["cost_per_run_at_fort_lin_wls"],
            color="lime",
            linestyle=":",
            label="Cost/Run @ FORT (Linear, WLS)",
        )
    if (plot_layers & PlotLayer.WLS_COST_QUAD) and (
        "cost_per_run_at_fort_quad_wls" in summary.df_results.columns
    ):
        plt.plot(
            summary.df_results["sor#"],
            summary.df_results["cost_per_run_at_fort_quad_wls"],
            color="cyan",
            linestyle=":",
            label="Cost/Run @ FORT (Quadratic, WLS)",
        )

    # Min cost verticals (OLS)
    if plot_layers & PlotLayer.OLS_MIN_LINEAR:
        plt.axvline(
            x=summary.sor_min_cost_lin,
            color="green",
            linestyle="--",
            label="Min Cost (Linear, OLS)",
        )
    if plot_layers & PlotLayer.OLS_MIN_QUAD:
        plt.axvline(
            x=summary.sor_min_cost_quad,
            color="blue",
            linestyle="--",
            label="Min Cost (Quadratic, OLS)",
        )

    # Min cost verticals (WLS) if present
    if (plot_layers & PlotLayer.WLS_MIN_LINEAR) and (
        summary.sor_min_cost_lin_wls is not None
    ):
        plt.axvline(
            x=summary.sor_min_cost_lin_wls,
            color="lime",
            linestyle=":",
            label="Min Cost (Linear, WLS)",
        )
    if (plot_layers & PlotLayer.WLS_MIN_QUAD) and (
        summary.sor_min_cost_quad_wls is not None
    ):
        plt.axvline(
            x=summary.sor_min_cost_quad_wls,
            color="cyan",
            linestyle=":",
            label="Min Cost (Quadratic, WLS)",
        )

    plt.xlabel("Sequential Online Run #")
    plt.ylabel("Adjusted Run Time")
    plt.title("FORT Regression Models")

    if plot_layers & PlotLayer.LEGEND:
        plt.legend()

    plt.savefig(output_svg, format="svg")
    plt.close()
    return output_svg


def main() -> None:
    # Example configuration (was previously hardcoded)
    log_path = Path("C:/Games/Utility/ICScriptHub/log-reset.csv").resolve()
    params_load = LoadSliceParams(
        log_path=log_path,
        start_line=4883,
        end_line=None,
        include_header=True,
    )
    params_transform = TransformParams(
        zscore_min=-1.5,
        zscore_max=3,
        input_data_fort=100,
        ignore_resetticks=True,
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
        exclude_timestamp_ranges=None,  # [("20250801124409", "20250805165454")],
        verbose_filtering=True,
        plot_layers=PlotLayer.DEFAULT,
    )

    # Build hashing payload (simplified: path + effective parameters)

    abs_input_posix = normalize_abs_posix(params_load.log_path)
    effective_params = build_effective_parameters(params_load, params_transform)
    canonical_payload = {
        "absolute_input_path": abs_input_posix,
        "effective_parameters": effective_params,
    }
    short_hash, full_hash = canonical_json_hash(canonical_payload)

    # Orchestration (side-effect free besides plot file write in render)
    df_range = load_and_slice_csv(params_load)
    print("\n\n")
    print(f"Input data:\n{df_range}")
    print("\n\n")
    transformed = transform_pipeline(df_range, params_transform)
    print("\n\n")
    print(f"Filtered data:\n{transformed.df_range}")
    print("\n\n")
    summary = summarize_and_model(transformed.df_range, params_transform)
    print(f"Regression Diagnostics:\n{summary.regression_diagnostics}")
    print("\n\n")
    print(f"Summary:\n{summary.df_summary}")
    print("\n\n")
    print(f"Predictions:\n{summary.df_results}")
    print("\n\n")

    # Pretty comparison table for four model forms (OLS/WLS x Linear/Quadratic)
    diag = summary.regression_diagnostics

    def _safe_get(d: dict, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # Mapping of display rows to diagnostic nodes
    rows_spec = [
        ("OLS Linear", ("linear", "ols")),
        ("OLS Quadratic", ("quadratic", "ols")),
        ("WLS Linear", ("linear", "wls")),
        ("WLS Quadratic", ("quadratic", "wls")),
    ]

    # Build rows with requested metrics
    table_rows = []
    for label, path in rows_spec:
        node = _safe_get(diag, *path, default={}) or {}
        # Extract metrics; many statsmodels results are pandas/ndarray — cast to float when possible
        rmse = _safe_get(diag, path[0], path[1], "RMSE", default=None)
        adj_r2 = _safe_get(node, "Adj. R-squared", default=None)
        # AIC/BIC if available on the fitted result; if not, leave blank
        aic = None
        bic = None
        # Try to fetch from the stored "Summary" or parameters. We stored full result objects' summary()
        # but not direct attributes. However, AIC/BIC can be reconstructed if res is available.
        # Our diagnostics store "Summary" text, so for reliability we prefer direct attributes if present.
        # The _stats_dict included Coefficients/SE/etc. but not 'aic'/'bic' scalar; we can compute from res if available
        # but since we didn't keep res, fall back to checking if node has 'aic'/'bic' already (future-proof).
        aic = _safe_get(node, "aic", default=None)
        bic = _safe_get(node, "bic", default=None)

        # If not present, attempt to derive from the Summary object if it has .aic/.bic attributes (statsmodels Summary2 doesn't),
        # so we just leave as None if absent.
        def fmt(x):
            try:
                return (
                    f"{float(x):.6g}"
                    if x is not None and np.isfinite(float(x))
                    else "-"
                )
            except Exception:
                return "-"

        table_rows.append((label, fmt(rmse), fmt(adj_r2), fmt(aic), fmt(bic)))

    # Column widths
    headers = ("Model", "RMSE (in-sample)", "Adj R^2", "AIC", "BIC")
    col_widths = [
        max(len(headers[0]), max(len(r[0]) for r in table_rows)),
        len(headers[1]),
        len(headers[2]),
        len(headers[3]),
        len(headers[4]),
    ]

    # Render header
    header_line = f"{headers[0]:<{col_widths[0]}}  {headers[1]:>{col_widths[1]}}  {headers[2]:>{col_widths[2]}}  {headers[3]:>{col_widths[3]}}  {headers[4]:>{col_widths[4]}}"
    sep_line = "-" * len(header_line)
    print("Model Comparison (OLS/WLS x Linear/Quadratic)")
    print(header_line)
    print(sep_line)
    for r in table_rows:
        print(
            f"{r[0]:<{col_widths[0]}}  {r[1]:>{col_widths[1]}}  {r[2]:>{col_widths[2]}}  {r[3]:>{col_widths[3]}}  {r[4]:>{col_widths[4]}}"
        )
    print("\n")

    print(
        "FORT for minimum cost/run:\n"
        f"  (OLS linear): {summary.sor_min_cost_lin}\n"
        f"  (OLS quadratic): {summary.sor_min_cost_quad}\n"
        f"  (WLS linear): {summary.sor_min_cost_lin_wls}\n"
        f"  (WLS quadratic): {summary.sor_min_cost_quad_wls}\n"
    )
    print("\n")

    # Render multiple plots with different flag presets
    presets_to_render = [
        PlotLayer.ALL_OLS,
        PlotLayer.ALL_WLS,
        (PlotLayer.ALL_OLS | PlotLayer.ALL_WLS) & ~PlotLayer.MIN_MARKERS_ONLY,
        PlotLayer.ALL_PREDICTION | PlotLayer.LEGEND,
        PlotLayer.ALL_PREDICTION | PlotLayer.LEGEND | PlotLayer.DATA_SCATTER,
    ]
    artifact_paths: list[str] = []
    for flags in presets_to_render:
        layer_suffix = _plot_layers_suffix(flags)
        out_svg = with_hash_suffix(f"plot-{layer_suffix}", short_hash, ".svg")
        _ = render_outputs(
            transformed.df_range,
            summary,
            output_svg=out_svg,
            plot_layers=flags,
        )
        artifact_paths.append(out_svg)

    # Build and write manifest next to artifact (current working directory)
    total_input_rows = int(len(df_range))
    processed_row_count = int(len(transformed.df_range))
    excluded_row_count = int(len(transformed.df_excluded))
    # Derive pre-zscore exclusions (timestamp range and earlier filters)
    pre_zscore_excluded = max(
        total_input_rows - processed_row_count - excluded_row_count, 0
    )
    exclusion_reasons = (
        f"timestamp_range_excluded_rows={pre_zscore_excluded}; "
        f"zscore_excluded_rows={excluded_row_count}"
    )

    manifest = {
        "version": "1",
        "timestamp_utc": utc_timestamp_seconds(),
        "absolute_input_path": abs_input_posix,
        "total_input_rows": total_input_rows,
        "processed_row_count": processed_row_count,
        "excluded_row_count": excluded_row_count,
        "exclusion_reasons": exclusion_reasons,
        "effective_parameters": effective_params,
        "canonical_hash": full_hash,
        "canonical_hash_short": short_hash,
        "artifacts": {"plot_svgs": artifact_paths},
    }
    manifest_name = f"manifest-{short_hash}.json"
    write_manifest(manifest_name, manifest)


if __name__ == "__main__":
    main()
