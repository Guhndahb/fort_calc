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
from enum import Enum, auto
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


def regression_analysis(
    df_range: pd.DataFrame, input_data_fort: int
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    OLS regression for linear and quadratic models using pandas DataFrames
    for design matrices. Training and prediction frames use matching column
    names and consistent constant handling (has_constant='add') to avoid
    column-order issues during predict.
    """
    # Generate sequence from 1 to input_data_fort
    sor_sequence = np.arange(1, input_data_fort + 1)

    # Prepare response as 1D ndarray to align with statsmodels OLS expectations
    y = df_range["adjusted_run_time"].to_numpy()

    # Training design matrices as DataFrames with explicit column names
    X_linear_train = pd.DataFrame({"sor#": df_range["sor#"].to_numpy()})
    X_linear_train = sm.add_constant(X_linear_train, has_constant="add")
    # Ensure constant column is named 'const' as per statsmodels default
    if "const" not in X_linear_train.columns:
        # Statsmodels may sometimes name it differently; standardize
        const_col = [
            c for c in X_linear_train.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_linear_train = X_linear_train.rename(columns={const_col[0]: "const"})

    linear_model = sm.OLS(y, X_linear_train).fit()

    # Quadratic training design matrix with named columns
    X_quadratic_train = pd.DataFrame(
        {
            "sor#": df_range["sor#"].to_numpy(),
            "sor2": df_range["sor#"].to_numpy() ** 2,
        }
    )
    X_quadratic_train = sm.add_constant(X_quadratic_train, has_constant="add")
    if "const" not in X_quadratic_train.columns:
        const_col = [
            c for c in X_quadratic_train.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_quadratic_train = X_quadratic_train.rename(
                columns={const_col[0]: "const"}
            )

    quadratic_model = sm.OLS(y, X_quadratic_train).fit()

    # Prediction frames as DataFrames with the same column names
    X_linear_pred = pd.DataFrame({"sor#": sor_sequence})
    X_linear_pred = sm.add_constant(X_linear_pred, has_constant="add")
    if "const" not in X_linear_pred.columns:
        const_col = [
            c for c in X_linear_pred.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_linear_pred = X_linear_pred.rename(columns={const_col[0]: "const"})
    # Ensure column order matches trained model exog names to avoid any positional misalignment
    X_linear_pred = X_linear_pred.reindex(columns=linear_model.model.exog_names)
    # Ensure column order matches trained model exog names to avoid any positional misalignment
    X_linear_pred = X_linear_pred.reindex(columns=linear_model.model.exog_names)

    X_quadratic_pred = pd.DataFrame({"sor#": sor_sequence, "sor2": sor_sequence**2})
    X_quadratic_pred = sm.add_constant(X_quadratic_pred, has_constant="add")
    if "const" not in X_quadratic_pred.columns:
        const_col = [
            c for c in X_quadratic_pred.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_quadratic_pred = X_quadratic_pred.rename(columns={const_col[0]: "const"})
    # Ensure column order matches trained model exog names
    X_quadratic_pred = X_quadratic_pred.reindex(
        columns=quadratic_model.model.exog_names
    )
    # Ensure column order matches trained model exog names
    X_quadratic_pred = X_quadratic_pred.reindex(
        columns=quadratic_model.model.exog_names
    )

    # Align prediction columns to trained model exog_names to avoid positional mismatch
    X_linear_pred = X_linear_pred.reindex(columns=linear_model.model.exog_names)
    X_quadratic_pred = X_quadratic_pred.reindex(
        columns=quadratic_model.model.exog_names
    )

    # Predict using DataFrames after explicit column alignment
    linear_model_output = linear_model.predict(X_linear_pred)
    quadratic_model_output = quadratic_model.predict(X_quadratic_pred)

    # Create results DataFrame
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

    Definitions:
    - offline_cost: The estimated extra time game restarts (usually offline stacks) take
      versus an online stack. Computed as the run_time_delta of the final degenerate 'fort'
      row produced by summarize_run_time_by_sor_range() under the chosen delta_mode.

    Notes:
    - The final "offline" row created by summarize_run_time_by_sor_range has no range
      and can legitimately have NaN for run_time_mean. We treat exactly one NaN in the
      final row as acceptable and do not raise in that case.
    """
    df_summary = summarize_run_time_by_sor_range(
        df_range, params.input_data_fort, params.delta_mode
    )

    # Allow exactly one NaN in the final row (offline cost row). If more NaNs exist
    # or NaNs occur outside the final row, raise.
    run_time_mean_isna = df_summary["run_time_mean"].isna()
    nan_count = int(run_time_mean_isna.sum())
    if nan_count > 0:
        bad_positions = df_summary.index[run_time_mean_isna].tolist()
        only_final_row_nan = nan_count == 1 and bad_positions == [df_summary.index[-1]]
        if not only_final_row_nan:
            raise ValueError(
                f"Insufficient input data: Found {nan_count} summary rows with no mean run time values."
            )

    # Offline cost comes from the final row's delta
    offline_cost = float(df_summary.iloc[-1]["run_time_delta"])

    df_results, regression_diagnostics = regression_analysis(
        df_range, params.input_data_fort - 1
    )

    # Cumulative sums
    df_results["sum_lin"] = df_results["linear_model_output"].cumsum()
    df_results["sum_quad"] = df_results["quadratic_model_output"].cumsum()

    # Cost per run columns
    df_results["cost_per_run_at_fort_lin"] = (
        df_results["sum_lin"] + offline_cost
    ) / df_results["sor#"]
    df_results["cost_per_run_at_fort_quad"] = (
        df_results["sum_quad"] + offline_cost
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

    return SummaryModelOutputs(
        df_summary=df_summary,
        df_results=df_results,
        regression_diagnostics=regression_diagnostics,
        offline_cost=offline_cost,
        sor_min_cost_lin=sor_min_cost_lin,
        sor_min_cost_quad=sor_min_cost_quad,
    )


def render_outputs(
    df_range: pd.DataFrame,
    summary: SummaryModelOutputs,
    output_svg: str = "plot.svg",
) -> str:
    """
    Pure renderer: builds the plot and returns the output path.
    No prints; deterministic given inputs.
    """
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df_range["sor#"],
        df_range["adjusted_run_time"],
        s=20,
        color="cyan",
        label="Data Points",
    )

    # Overlay models
    plt.plot(
        summary.df_results["sor#"],
        summary.df_results["linear_model_output"],
        color="yellow",
        label="Linear Model",
    )
    plt.plot(
        summary.df_results["sor#"],
        summary.df_results["quadratic_model_output"],
        color="magenta",
        label="Quadratic Model",
    )

    # Cost per run curves
    plt.plot(
        summary.df_results["sor#"],
        summary.df_results["cost_per_run_at_fort_lin"],
        color="green",
        linestyle="--",
        label="Cost/Run @ FORT (Linear)",
    )
    plt.plot(
        summary.df_results["sor#"],
        summary.df_results["cost_per_run_at_fort_quad"],
        color="blue",
        linestyle="--",
        label="Cost/Run @ FORT (Quadratic)",
    )

    # Min cost verticals
    plt.axvline(
        x=summary.sor_min_cost_lin,
        color="green",
        linestyle="--",
        label="Min Cost (Linear)",
    )
    plt.axvline(
        x=summary.sor_min_cost_quad,
        color="blue",
        linestyle="--",
        label="Min Cost (Quadratic)",
    )

    plt.xlabel("Sequential Online Run #")
    plt.ylabel("Adjusted Run Time")
    plt.title("Plot with Linear and Quadratic Models")
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
        exclude_timestamp_ranges=[("20250801124409", "20250805165454")],
        verbose_filtering=True,
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
    print(f"Summary\n{summary.df_summary}")
    print("\n\n")
    print(f"Minimum cost per run at fort (linear): sor# {summary.sor_min_cost_lin}")
    print(f"Minimum cost per run at fort (quadratic): sor# {summary.sor_min_cost_quad}")
    print("\n")

    # Suffix artifact filename with short hash
    out_svg = with_hash_suffix("plot", short_hash, ".svg")
    _ = render_outputs(transformed.df_range, summary, output_svg=out_svg)

    # Build and write manifest next to artifact (current working directory)
    total_input_rows = int(len(df_range))
    processed_row_count = int(len(transformed.df_range))
    excluded_row_count = int(len(transformed.df_excluded))
    exclusion_reasons = f"timestamp_range_excluded_rows={total_input_rows - len(df_range)}; zscore_excluded_rows={excluded_row_count}"
    # Note: timestamp exclusion count cannot be derived post-hoc without deeper plumbing; keep simple summary.

    manifest = {
        "version": "1",
        "timestamp_utc": utc_timestamp_seconds(),
        "absolute_input_path": abs_input_posix,
        "total_input_rows": total_input_rows,
        "processed_row_count": processed_row_count,
        "excluded_row_count": excluded_row_count,
        "exclusion_reasons": f"zscore_excluded_rows={excluded_row_count}",
        "effective_parameters": effective_params,
        "canonical_hash": full_hash,
        "canonical_hash_short": short_hash,
        "artifacts": {"plot_svg": out_svg},
    }
    manifest_name = f"manifest-{short_hash}.json"
    write_manifest(manifest_name, manifest)


if __name__ == "__main__":
    main()
