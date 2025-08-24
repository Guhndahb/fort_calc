"""
Validation test runner for iterative excision + re-summary scenarios.

Provides:
- validate_fort_ladder_down(params_load, params_transform, list_plot_params, plot_each_iteration=True, output_base_dir="output_validation")

This module intentionally performs local (lazy) imports from .main inside the runner function to
avoid circular import / CLI side-effects at module import time.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

# Numerical / tabular helpers used by the cached-model runner
import pandas as pd

logger = logging.getLogger(__name__)


def validate_fort_ladder_down(
    params_load,
    params_transform,
    list_plot_params: List,
    plot_each_iteration: bool = True,
    output_base_dir: str | Path = "output_validation",
):
    """
    Run a series of validation iterations using the existing pipeline functions from src.main.

    Behavior (summary):
    - Creates an output root: output_base_dir / f"{timestamp}"
    - Loads CSV via load_and_slice_csv(params_load)
    - Runs initial transform_pipeline and summarize_and_model to compute master_offline_cost
      and writes artifacts into iter-00-original
    - Iteratively excises rows and adjusts the runtimes then re-runs summarize_and_model for
      each decremented fort value (cur_fort -> cur_fort-10) until input_data_fort <= 10.
    - Each iteration (including original) is saved into its own subdirectory
      iter-00-original, iter-01-fort-<N>, ...
    - Exceptions from summarize_and_model are captured and written as error.txt for that
      iteration; the run continues.

    Notes:
    - All imports from src.main are performed lazily inside this function to avoid CLI/circular imports.
    - Uses logging.getLogger(__name__) for auditable run logs.
    - Does not call sys.exit(); returns after completion.
    """
    # Local imports to avoid circular import / CLI parsing side-effects at import time.
    # Import only what we need from main.py with robust fallbacks so the runner works
    # whether src is executed as a package (python -m src.main) or as a script.
    import importlib
    import sys

    def _resolve_main_module():
        # Try absolute import first to avoid package-relative failures when main is run as a script.
        for candidate in ("src.main", "main"):
            try:
                return importlib.import_module(candidate)
            except Exception:
                pass

        # Next, check sys.modules for a previously-loaded module under common names.
        for candidate in ("src.main", "main", "__main__"):
            if candidate in sys.modules:
                return sys.modules[candidate]

        # As a last attempt, try package-relative import if a package name is available.
        try:
            pkg = __package__ if __package__ else None
            if pkg:
                return importlib.import_module(".main", package=pkg)
        except Exception:
            pass

        raise ImportError(
            "Could not import main module for validation tests (tried absolute imports and sys.modules)"
        )

    _main = _resolve_main_module()

    # Bind required attributes from the resolved main module
    TransformOutputs = _main.TransformOutputs
    TransformParams = _main.TransformParams
    assemble_text_report = _main.assemble_text_report
    build_manifest_dict = _main.build_manifest_dict
    build_model_comparison = _main.build_model_comparison
    build_run_identity = _main.build_run_identity
    load_and_slice_csv = _main.load_and_slice_csv
    render_plots = _main.render_plots
    # New: master plot renderer (overlays original + iterations)
    render_master_plots = _main.render_master_plots
    summarize_and_model = _main.summarize_and_model
    transform_pipeline = _main.transform_pipeline
    write_manifest = _main.write_manifest
    # New: model helpers and canonical model ordering (dynamic)
    MODEL_PRIORITY = _main.MODEL_PRIORITY
    model_output_column = _main.model_output_column
    model_sum_column = _main.model_sum_column
    model_cost_column = _main.model_cost_column

    out_root = Path(output_base_dir)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = out_root / ts
    run_root.mkdir(parents=True, exist_ok=True)
    logger.info("Validation run root directory: %s", str(run_root))

    # Load original CSV slice
    logger.info("Loading CSV using provided LoadSliceParams")
    df_loaded = load_and_slice_csv(params_load)

    # Initial transform
    logger.info("Running initial transform_pipeline")
    initial_transformed = transform_pipeline(df_loaded, params_transform)

    # Initial summary / master offline cost
    logger.info("Running initial summarize_and_model")
    try:
        summary_initial = summarize_and_model(
            initial_transformed.df_range, params_transform
        )
        master_offline_cost = float(summary_initial.offline_cost)
    except Exception as e:
        # If initial summary fails, write an error and abort the validation run (nothing to iterate over)
        err_dir = run_root / "iter-00-original"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / "error.txt").write_text(
            f"Initial summarize_and_model failed: {type(e).__name__}: {e}\n",
            encoding="utf-8",
        )
        logger.exception(
            "Initial summarize_and_model failed; aborting validation suite"
        )
        print(f"Validation suite aborted: initial summarize_and_model failed: {e}")
        return

    # Save artifacts for the original run (iter-00-original)
    iter_dirs: list[Path] = []
    iter0_dir = run_root / "iter-00-original"
    iter0_dir.mkdir(parents=True, exist_ok=True)
    iter_dirs.append(iter0_dir)
    logger.info("Writing artifacts for original run to %s", str(iter0_dir))

    # Build run identity (use the original params for canonical naming)
    abs_input_posix, short_hash, full_hash, effective_params = build_run_identity(
        params_load, params_transform
    )

    # Render plots (may be empty if plot list is empty)
    artifact_paths = []
    # Accumulator used to build master overlay plots after iterations complete
    master_plot_inputs: list[dict] = []
    # Collect the original iteration inputs
    master_plot_inputs.append(
        {
            "label": "iter-00-original",
            "df_included": initial_transformed.df_range.copy(),
            "summary": summary_initial,
            "df_excluded": initial_transformed.df_excluded,
        }
    )
    if plot_each_iteration and list_plot_params:
        try:
            artifact_paths = render_plots(
                list_plot_params=list_plot_params,
                df_included=initial_transformed.df_range,
                summary=summary_initial,
                short_hash=short_hash,
                df_excluded=initial_transformed.df_excluded,
                output_dir=str(iter0_dir),
            )
        except Exception as e:
            logger.exception("Failed to render plots for original run: %s", e)

    # Build manifest counts analogous to _orchestrate
    total_input_rows = int(len(df_loaded))
    processed_row_count = int(len(initial_transformed.df_range))
    excluded_row_count = (
        int(len(initial_transformed.df_excluded))
        if initial_transformed.df_excluded is not None
        else 0
    )
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
    try:
        write_manifest(str(iter0_dir / f"manifest-{short_hash}.json"), manifest)
    except Exception:
        logger.exception("Failed to write manifest for initial run")

    # Textual report
    try:
        best_label, table_text = build_model_comparison(
            summary_initial.regression_diagnostics
        )
        report_text = assemble_text_report(
            df_loaded,
            TransformOutputs(
                df_range=initial_transformed.df_range,
                df_excluded=initial_transformed.df_excluded,
            ),
            summary_initial,
            table_text,
            best_label,
            params_transform.verbose_filtering,
        )
        (iter0_dir / f"report-{short_hash}.txt").write_text(
            report_text, encoding="utf-8"
        )
    except Exception:
        logger.exception("Failed to build/write textual report for initial run")

    # Iterative excision loop
    logger.info(
        "Starting iterative excision loop (step size 10) from input_data_fort=%s",
        params_transform.input_data_fort,
    )
    cur_fort = int(params_transform.input_data_fort)
    iter_index = 1

    # Use the original included frame as the base for each iteration
    original_included = initial_transformed.df_range
    original_excluded = initial_transformed.df_excluded

    while cur_fort > 50:
        new_fort = cur_fort - 10
        iter_name = f"iter-{iter_index:02d}-fort-{new_fort}"
        iter_dir = run_root / iter_name
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_dirs.append(iter_dir)
        logger.info(
            "Iteration %d: creating %s (new_fort=%d)",
            iter_index,
            str(iter_dir),
            new_fort,
        )

        # Start from the original included frame copy
        df_iter = original_included.copy()

        # Excise rows where (sor# > (cur_fort - 10)) & (sor# < cur_fort)
        lower_excl = cur_fort - 10
        mask_excl = df_iter["sor#"] > lower_excl
        if mask_excl.any():
            df_iter = df_iter.loc[~mask_excl].copy()
            logger.info(
                "Iteration %d: excised %d rows based on sor# window",
                iter_index,
                int(mask_excl.sum()),
            )
        else:
            logger.info(
                "Iteration %d: no rows to excise for window (%d..%d)",
                iter_index,
                lower_excl + 1,
                cur_fort - 1,
            )

        # For all rows where sor# == new_fort, increment adjusted_run_time by master_offline_cost
        try:
            mask_eq = df_iter["sor#"] == new_fort
            if mask_eq.any():
                df_iter.loc[mask_eq, "adjusted_run_time"] = (
                    df_iter.loc[mask_eq, "adjusted_run_time"] + master_offline_cost
                )
                logger.info(
                    "Iteration %d: incremented adjusted_run_time for %d rows at sor#=%d by %f seconds",
                    iter_index,
                    int(mask_eq.sum()),
                    new_fort,
                    master_offline_cost,
                )
            else:
                logger.info(
                    "Iteration %d: no rows with sor# == %d to increment",
                    iter_index,
                    new_fort,
                )
        except Exception:
            logger.exception(
                "Iteration %d: failed to increment adjusted_run_time", iter_index
            )

        # Build shallow copy of params_transform with updated input_data_fort
        try:
            params_for_iter = replace(params_transform, input_data_fort=new_fort)
        except Exception:
            # Fallback: construct a new TransformParams by copying attributes (defensive)
            try:
                params_for_iter = TransformParams(
                    zscore_min=params_transform.zscore_min,
                    zscore_max=params_transform.zscore_max,
                    input_data_fort=new_fort,
                    ignore_resetticks=params_transform.ignore_resetticks,
                    delta_mode=params_transform.delta_mode,
                    exclude_timestamp_ranges=params_transform.exclude_timestamp_ranges,
                    verbose_filtering=params_transform.verbose_filtering,
                    fail_on_any_invalid_timestamps=params_transform.fail_on_any_invalid_timestamps,
                    iqr_k_low=params_transform.iqr_k_low,
                    iqr_k_high=params_transform.iqr_k_high,
                    use_iqr_filtering=params_transform.use_iqr_filtering,
                )
            except Exception:
                logger.exception("Failed to build params_for_iter; skipping iteration")
                cur_fort = new_fort
                iter_index += 1
                continue

        # Run summarize_and_model for this iteration and save artifacts; capture exceptions
        try:
            summary_iter = summarize_and_model(df_iter, params_for_iter)

            # Early-stop: if any model's recommended sor_min_cost equals the current cur_fort,
            # omit this iteration (do not render plots/manifests/reports) and stop iterating.
            try:
                # New: check the canonical sor_min_costs for any enabled model token.
                reached = False
                sor_min_dict = getattr(summary_iter, "sor_min_costs", {}) or {}
                for token in MODEL_PRIORITY:
                    v = sor_min_dict.get(token)
                    if v is not None and np.isfinite(v) and int(v) == int(cur_fort):
                        reached = True
                        break
                if reached:
                    logger.info(
                        "Iteration %d: a model recommended sor_min_cost == current cur_fort (%d); omitting this run and stopping further iterations",
                        iter_index,
                        cur_fort,
                    )
                    # Remove the iter_dir we created for this iteration from the list of created dirs
                    try:
                        if iter_dirs and iter_dirs[-1] == iter_dir:
                            iter_dirs.pop()
                    except Exception:
                        pass
                    break
            except Exception:
                logger.exception(
                    "Iteration %d: failed during early-stop check; continuing",
                    iter_index,
                )

            # Collect inputs for master overlay plots
            try:
                master_plot_inputs.append(
                    {
                        "label": iter_name,
                        "df_included": df_iter.copy(),
                        "summary": summary_iter,
                        "df_excluded": original_excluded,
                    }
                )
            except Exception:
                logger.exception(
                    "Iteration %d: failed to append to master_plot_inputs", iter_index
                )
            # Build run identity for naming
            abs_input_posix_i, short_hash_i, full_hash_i, effective_params_i = (
                build_run_identity(params_load, params_for_iter)
            )

            artifact_paths_i = []
            if plot_each_iteration and list_plot_params:
                try:
                    artifact_paths_i = render_plots(
                        list_plot_params=list_plot_params,
                        df_included=df_iter,
                        summary=summary_iter,
                        short_hash=short_hash_i,
                        df_excluded=original_excluded,
                        output_dir=str(iter_dir),
                    )
                except Exception:
                    logger.exception("Iteration %d: failed to render plots", iter_index)

            # Counts and manifest similar to _orchestrate
            total_input_rows_i = int(len(df_loaded))
            processed_row_count_i = int(len(df_iter))
            excluded_row_count_i = (
                int(len(original_excluded)) if original_excluded is not None else 0
            )
            pre_zscore_excluded_i = max(
                total_input_rows_i - processed_row_count_i - excluded_row_count_i, 0
            )
            counts_i = {
                "total_input_rows": total_input_rows_i,
                "processed_row_count": processed_row_count_i,
                "excluded_row_count": excluded_row_count_i,
                "pre_zscore_excluded": pre_zscore_excluded_i,
            }

            manifest_i = build_manifest_dict(
                abs_input_posix=abs_input_posix_i,
                counts=counts_i,
                effective_params=effective_params_i,
                hashes=(short_hash_i, full_hash_i),
                artifact_paths=artifact_paths_i,
            )
            try:
                write_manifest(
                    str(iter_dir / f"manifest-{short_hash_i}.json"), manifest_i
                )
            except Exception:
                logger.exception("Iteration %d: failed to write manifest", iter_index)

            # Textual report using assemble_text_report and build_model_comparison
            try:
                best_label_i, table_text_i = build_model_comparison(
                    summary_iter.regression_diagnostics
                )
                report_i = assemble_text_report(
                    df_loaded,
                    TransformOutputs(df_range=df_iter, df_excluded=original_excluded),
                    summary_iter,
                    table_text_i,
                    best_label_i,
                    params_for_iter.verbose_filtering,
                )
                (iter_dir / f"report-{short_hash_i}.txt").write_text(
                    report_i, encoding="utf-8"
                )
            except Exception:
                logger.exception(
                    "Iteration %d: failed to build/write textual report", iter_index
                )

        except Exception as e_iter:
            # Write a small error.txt describing the failure and continue
            logger.exception(
                "Iteration %d: summarize_and_model failed: %s", iter_index, e_iter
            )
            (iter_dir / "error.txt").write_text(
                f"{type(e_iter).__name__}: {e_iter}\n", encoding="utf-8"
            )

        # Prepare next iteration
        cur_fort = new_fort
        iter_index += 1

    # After all iterations, attempt to render combined master plots that overlay all iterations
    if plot_each_iteration and list_plot_params and master_plot_inputs:
        try:
            master_artifacts = render_master_plots(
                list_plot_params=list_plot_params,
                per_iteration_inputs=master_plot_inputs,
                output_dir=str(run_root),
            )
            if master_artifacts:
                print("Created master plot artifacts:")
                for p in master_artifacts:
                    print(str(p))
        except Exception:
            logger.exception("Failed to render master overlay plots")

    # Final concise summary to stdout
    print("Validation run complete.")
    print(f"Master offline cost (seconds): {master_offline_cost:.6f}")
    print("Created iteration directories:")
    for d in iter_dirs:
        print(str(d))

    logger.info(
        "Validation run finished; created %d iteration directories", len(iter_dirs)
    )


# Helper used by the fixed-model validation runner to either produce the initial
# model-based SummaryModelOutputs (and populate the provided cache dict) or to
# construct an iteration SummaryModelOutputs using cached model artifacts.
def _summarize_with_cached_models(df_range, params, cache: dict | None):
    """
    Lightweight, MODEL_PRIORITY-driven cached summarizer.

    On first (empty cache) call:
      - Runs summarize_and_model with DeltaMode.MODEL_BASED and stores canonical
        artifacts into the cache under well-known keys:
          - 'df_results_full' (DataFrame copy)
          - 'regression_diagnostics' (dict)
          - 'offline_cost' (float)
          - 'per_model_offline_costs' (dict[token->float])
          - 'sor_min_costs_full' (dict[token->int])
      - Returns the SummaryModelOutputs returned by summarize_and_model (no mutation).

    On subsequent calls (cache populated):
      - Computes df_summary via summarize_run_time_by_sor_range(..., DeltaMode.PREVIOUS_CHUNK)
      - Builds df_results_iter by slicing cached['df_results_full'] to sor# <= params.input_data_fort
      - For each token in MODEL_PRIORITY that has an output column in df_results_iter:
          - recomputes cumulative sum column via model_sum_column(token)
          - computes cost-per-run via model_cost_column(token) using per-model offline when finite,
            else falling back to cache['offline_cost']
      - Computes sor_min_costs by choosing idxmin over available model cost columns.
      - Returns a SummaryModelOutputs instance compatible with src.main.SummaryModelOutputs.
    """
    import importlib
    import sys

    def _resolve_main_module():
        for candidate in ("src.main", "main"):
            try:
                return importlib.import_module(candidate)
            except Exception:
                pass
        for candidate in ("src.main", "main", "__main__"):
            if candidate in sys.modules:
                return sys.modules[candidate]
        try:
            pkg = __package__ if __package__ else None
            if pkg:
                return importlib.import_module(".main", package=pkg)
        except Exception:
            pass
        raise ImportError(
            "Could not import main module for validation tests (tried absolute imports and sys.modules)"
        )

    _main = _resolve_main_module()

    # Bind required symbols from main (including canonical model helpers)
    summarize_and_model = _main.summarize_and_model
    summarize_run_time_by_sor_range = _main.summarize_run_time_by_sor_range
    SummaryModelOutputs = _main.SummaryModelOutputs
    DeltaMode = _main.DeltaMode
    MODEL_PRIORITY = _main.MODEL_PRIORITY
    model_output_column = _main.model_output_column
    model_sum_column = _main.model_sum_column
    model_cost_column = _main.model_cost_column

    # Ensure cache is a dict we can mutate
    if cache is None:
        cache_local: dict = {}
    else:
        cache_local = cache

    # Initial model-fit path: run summarize_and_model with MODEL_BASED and populate cache
    if not cache_local:
        try:
            params_for_model = replace(params, delta_mode=DeltaMode.MODEL_BASED)
        except Exception:
            params_for_model = params
            # type: ignore[attr-defined]
            params_for_model.delta_mode = DeltaMode.MODEL_BASED

        summary_initial = summarize_and_model(df_range, params_for_model)

        # Populate canonical cache entries
        cache_local["df_results_full"] = summary_initial.df_results.copy()
        cache_local["regression_diagnostics"] = summary_initial.regression_diagnostics
        cache_local["offline_cost"] = float(summary_initial.offline_cost)
        cache_local["per_model_offline_costs"] = (
            getattr(summary_initial, "per_model_offline_costs", {}) or {}
        )
        cache_local["sor_min_costs_full"] = (
            getattr(summary_initial, "sor_min_costs", {}) or {}
        )

        # If caller provided a dict, update it in-place
        if cache is not None:
            cache.update(cache_local)

        return summary_initial

    # Cached path: compute df_summary and synthesize df_results from cached df_results_full
    df_summary = summarize_run_time_by_sor_range(
        df_range, params.input_data_fort, DeltaMode.PREVIOUS_CHUNK
    )

    # Disallow NaNs in run_time_mean (same contract as summarize_and_model)
    run_time_mean_isna = df_summary["run_time_mean"].isna()
    nan_count = int(run_time_mean_isna.sum())
    if nan_count > 0:
        raise ValueError(
            f"Insufficient input data: Found {nan_count} summary rows with no mean run time values."
        )

    # Slice cached full results to current fort
    df_full = cache_local["df_results_full"]
    df_results_iter = df_full.loc[df_full["sor#"] <= params.input_data_fort].copy()

    # Helper: choose offline cost for a model column (prefer per-model finite value)
    def _offline_for_model(token: str) -> float:
        per_model = cache_local.get("per_model_offline_costs", {}) or {}
        model_offline = per_model.get(token)
        if model_offline is not None and np.isfinite(model_offline):
            return float(model_offline)
        return float(cache_local["offline_cost"])

    # For each enabled model token, if its output column exists, recompute sum & cost columns
    for token in MODEL_PRIORITY:
        out_col = model_output_column(token)
        if out_col in df_results_iter.columns:
            sum_col = model_sum_column(token)
            cost_col = model_cost_column(token)
            # Cumulative sum of model outputs
            try:
                df_results_iter[sum_col] = df_results_iter[out_col].cumsum()
            except Exception:
                # Best-effort: if cumsum fails, leave column absent and continue
                continue
            # Compute cost-per-run using per-model offline when available
            try:
                offline_for_col = _offline_for_model(token)
                df_results_iter[cost_col] = (
                    df_results_iter[sum_col] + float(offline_for_col)
                ) / df_results_iter["sor#"]
            except Exception:
                # If division fails (e.g., missing sor#), skip cost computation for this model
                if cost_col in df_results_iter.columns:
                    df_results_iter.drop(columns=[cost_col], inplace=True, errors=True)

    # Safe idxmin helper (mirror main._safe_idxmin behavior)
    def _safe_idxmin(series: pd.Series) -> int:
        if series.dropna().empty:
            return int(series.index[0])
        return int(series.dropna().idxmin())

    # Build sor_min_costs by inspecting canonical cost columns for enabled models
    sor_min_costs: dict[str, int] = {}
    for token in MODEL_PRIORITY:
        cost_col = model_cost_column(token)
        if cost_col in df_results_iter.columns:
            try:
                idx = _safe_idxmin(df_results_iter[cost_col])
                sor_val = int(df_results_iter.loc[idx, "sor#"])
                sor_min_costs[token] = sor_val
            except Exception:
                # best-effort; skip this token
                continue

    # Build and return SummaryModelOutputs consistent with main.SummaryModelOutputs
    return SummaryModelOutputs(
        df_summary=df_summary,
        df_results=df_results_iter,
        regression_diagnostics=cache_local.get("regression_diagnostics", {}),
        offline_cost=float(cache_local.get("offline_cost", float("nan"))),
        per_model_offline_costs=cache_local.get("per_model_offline_costs", {}) or {},
        sor_min_costs=sor_min_costs,
    )


def validate_fort_ladder_down_fixed_model(
    params_load,
    params_transform,
    list_plot_params: List,
    plot_each_iteration: bool = True,
    output_base_dir: str | Path = "output_validation",
):
    """
    Validation runner that reuses a master model fit from the original dataset (MODEL_BASED).
    Behavior mirrors validate_fort_ladder_down but avoids re-fitting models on subsequent
    iterations by slicing cached model predictions and recomputing cumulative sums/costs.

    Important:
      - This function performs the same lazy main-module resolution as the original runner.
      - On the initial run it calls summarize_and_model with DeltaMode.MODEL_BASED to
        obtain and cache master models/output.
      - For subsequent iterations it uses the cached df_results_full and diagnostics to
        synthesize SummaryModelOutputs for the truncated input (no re-fit).
    """
    # Local imports & robust main resolution (copied from validate_fort_ladder_down)
    import importlib
    import sys

    def _resolve_main_module():
        for candidate in ("src.main", "main"):
            try:
                return importlib.import_module(candidate)
            except Exception:
                pass
        for candidate in ("src.main", "main", "__main__"):
            if candidate in sys.modules:
                return sys.modules[candidate]
        try:
            pkg = __package__ if __package__ else None
            if pkg:
                return importlib.import_module(".main", package=pkg)
        except Exception:
            pass
        raise ImportError(
            "Could not import main module for validation tests (tried absolute imports and sys.modules)"
        )

    _main = _resolve_main_module()

    # Bind required attributes from main
    TransformOutputs = _main.TransformOutputs
    TransformParams = _main.TransformParams
    assemble_text_report = _main.assemble_text_report
    build_manifest_dict = _main.build_manifest_dict
    build_model_comparison = _main.build_model_comparison
    build_run_identity = _main.build_run_identity
    load_and_slice_csv = _main.load_and_slice_csv
    render_plots = _main.render_plots
    summarize_and_model = _main.summarize_and_model
    transform_pipeline = _main.transform_pipeline
    write_manifest = _main.write_manifest
    # New: model helpers and canonical model ordering (dynamic)
    MODEL_PRIORITY = _main.MODEL_PRIORITY
    model_output_column = _main.model_output_column
    model_sum_column = _main.model_sum_column
    model_cost_column = _main.model_cost_column

    out_root = Path(output_base_dir)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = out_root / ts
    run_root.mkdir(parents=True, exist_ok=True)
    logger.info("Validation run root directory: %s", str(run_root))

    # Load original CSV slice
    logger.info("Loading CSV using provided LoadSliceParams")
    df_loaded = load_and_slice_csv(params_load)

    # Initial transform
    logger.info("Running initial transform_pipeline")
    initial_transformed = transform_pipeline(df_loaded, params_transform)

    # Prepare cache dict (will be populated by helper on initial call)
    cache: dict = {}

    # Initial summary / master offline cost (MODEL_BASED)
    logger.info("Running initial summarize_and_model (MODEL_BASED)")
    try:
        # Ensure helper will perform MODEL_BASED fitting by passing a params instance
        params_for_model = replace(
            params_transform, delta_mode=_main.DeltaMode.MODEL_BASED
        )
        summary_initial = _summarize_with_cached_models(
            initial_transformed.df_range, params_for_model, cache
        )
        master_offline_cost = float(cache["offline_cost"])
    except Exception as e:
        # If initial summary fails, write an error and abort the validation run (nothing to iterate over)
        err_dir = run_root / "iter-00-original"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / "error.txt").write_text(
            f"Initial summarize_and_model (MODEL_BASED) failed: {type(e).__name__}: {e}\n",
            encoding="utf-8",
        )
        logger.exception(
            "Initial summarize_and_model failed; aborting validation suite"
        )
        print(f"Validation suite aborted: initial summarize_and_model failed: {e}")
        return

    # Save artifacts for the original run (iter-00-original)
    iter_dirs: list[Path] = []
    iter0_dir = run_root / "iter-00-original"
    iter0_dir.mkdir(parents=True, exist_ok=True)
    iter_dirs.append(iter0_dir)
    logger.info("Writing artifacts for original run to %s", str(iter0_dir))

    # Build run identity (use the original params for canonical naming)
    abs_input_posix, short_hash, full_hash, effective_params = build_run_identity(
        params_load, params_transform
    )

    # Render plots (may be empty if plot list is empty)
    artifact_paths = []
    # Accumulator used to build master overlay plots after iterations complete (fixed-model runner)
    master_plot_inputs: list[dict] = []
    # Collect the original iteration inputs
    master_plot_inputs.append(
        {
            "label": "iter-00-original",
            "df_included": initial_transformed.df_range.copy(),
            "summary": summary_initial,
            "df_excluded": initial_transformed.df_excluded,
        }
    )
    if plot_each_iteration and list_plot_params:
        try:
            artifact_paths = render_plots(
                list_plot_params=list_plot_params,
                df_included=initial_transformed.df_range,
                summary=summary_initial,
                short_hash=short_hash,
                df_excluded=initial_transformed.df_excluded,
                output_dir=str(iter0_dir),
            )
        except Exception as e:
            logger.exception("Failed to render plots for original run: %s", e)

    # Build manifest counts analogous to _orchestrate
    total_input_rows = int(len(df_loaded))
    processed_row_count = int(len(initial_transformed.df_range))
    excluded_row_count = (
        int(len(initial_transformed.df_excluded))
        if initial_transformed.df_excluded is not None
        else 0
    )
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
    try:
        write_manifest(str(iter0_dir / f"manifest-{short_hash}.json"), manifest)
    except Exception:
        logger.exception("Failed to write manifest for initial run")

    # Textual report
    try:
        best_label, table_text = build_model_comparison(
            summary_initial.regression_diagnostics
        )
        report_text = assemble_text_report(
            df_loaded,
            TransformOutputs(
                df_range=initial_transformed.df_range,
                df_excluded=initial_transformed.df_excluded,
            ),
            summary_initial,
            table_text,
            best_label,
            params_transform.verbose_filtering,
        )
        (iter0_dir / f"report-{short_hash}.txt").write_text(
            report_text, encoding="utf-8"
        )
    except Exception:
        logger.exception("Failed to build/write textual report for initial run")

    # Iterative excision loop (reuse master models from cache)
    logger.info(
        "Starting iterative excision loop (step size 10) from input_data_fort=%s (MODEL_BASED reuse)",
        params_transform.input_data_fort,
    )
    cur_fort = int(params_transform.input_data_fort)
    iter_index = 1

    # Use the original included frame as the base for each iteration
    original_included = initial_transformed.df_range
    original_excluded = initial_transformed.df_excluded

    while cur_fort > 10:
        new_fort = cur_fort - 10
        iter_name = f"iter-{iter_index:02d}-fort-{new_fort}"
        iter_dir = run_root / iter_name
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_dirs.append(iter_dir)
        logger.info(
            "Iteration %d: creating %s (new_fort=%d)",
            iter_index,
            str(iter_dir),
            new_fort,
        )

        # Start from the original included frame copy
        df_iter = original_included.copy()

        # Excise rows where (sor# > (cur_fort - 10)) & (sor# < cur_fort)
        lower_excl = cur_fort - 10
        mask_excl = (df_iter["sor#"] > lower_excl) & (df_iter["sor#"] < cur_fort)
        if mask_excl.any():
            df_iter = df_iter.loc[~mask_excl].copy()
            logger.info(
                "Iteration %d: excised %d rows based on sor# window",
                iter_index,
                int(mask_excl.sum()),
            )
        else:
            logger.info(
                "Iteration %d: no rows to excise for window (%d..%d)",
                iter_index,
                lower_excl + 1,
                cur_fort - 1,
            )

        # For all rows where sor# == new_fort, increment adjusted_run_time by master_offline_cost from cache
        try:
            mask_eq = df_iter["sor#"] == new_fort
            if mask_eq.any():
                df_iter.loc[mask_eq, "adjusted_run_time"] = df_iter.loc[
                    mask_eq, "adjusted_run_time"
                ] + float(cache["offline_cost"])
                logger.info(
                    "Iteration %d: incremented adjusted_run_time for %d rows at sor#=%d by %f seconds",
                    iter_index,
                    int(mask_eq.sum()),
                    new_fort,
                    float(cache["offline_cost"]),
                )
            else:
                logger.info(
                    "Iteration %d: no rows with sor# == %d to increment",
                    iter_index,
                    new_fort,
                )
        except Exception:
            logger.exception(
                "Iteration %d: failed to increment adjusted_run_time", iter_index
            )

        # Build shallow copy of params_transform with updated input_data_fort
        try:
            params_for_iter = replace(params_transform, input_data_fort=new_fort)
        except Exception:
            try:
                params_for_iter = TransformParams(
                    zscore_min=params_transform.zscore_min,
                    zscore_max=params_transform.zscore_max,
                    input_data_fort=new_fort,
                    ignore_resetticks=params_transform.ignore_resetticks,
                    delta_mode=params_transform.delta_mode,
                    exclude_timestamp_ranges=params_transform.exclude_timestamp_ranges,
                    verbose_filtering=params_transform.verbose_filtering,
                    fail_on_any_invalid_timestamps=params_transform.fail_on_any_invalid_timestamps,
                    iqr_k_low=params_transform.iqr_k_low,
                    iqr_k_high=params_transform.iqr_k_high,
                    use_iqr_filtering=params_transform.use_iqr_filtering,
                )
            except Exception:
                logger.exception("Failed to build params_for_iter; skipping iteration")
                cur_fort = new_fort
                iter_index += 1
                continue

        # Build SummaryModelOutputs for this iteration using cached master models (no re-fit)
        try:
            summary_iter = _summarize_with_cached_models(
                df_iter, params_for_iter, cache
            )

            # Early-stop: if any model's recommended sor_min_cost equals the current cur_fort,
            # omit this iteration (do not render plots/manifests/reports) and stop iterating.
            try:
                # New: check canonical sor_min_costs for any enabled model token.
                reached = False
                sor_min_dict = getattr(summary_iter, "sor_min_costs", {}) or {}
                for token in MODEL_PRIORITY:
                    v = sor_min_dict.get(token)
                    if v is not None and np.isfinite(v) and int(v) == int(cur_fort):
                        reached = True
                        break
                if reached:
                    logger.info(
                        "Iteration %d (fixed-model): a model recommended sor_min_cost == current cur_fort (%d); omitting this run and stopping further iterations",
                        iter_index,
                        cur_fort,
                    )
                    # Remove the iter_dir we created for this iteration from the list of created dirs
                    try:
                        if iter_dirs and iter_dirs[-1] == iter_dir:
                            iter_dirs.pop()
                    except Exception:
                        pass
                    break
            except Exception:
                logger.exception(
                    "Iteration %d: failed during early-stop check (fixed-model); continuing",
                    iter_index,
                )

            # Collect inputs for master overlay plots (keep parity with non-fixed runner)
            try:
                master_plot_inputs.append(
                    {
                        "label": iter_name,
                        "df_included": df_iter.copy(),
                        "summary": summary_iter,
                        "df_excluded": original_excluded,
                    }
                )
            except Exception:
                logger.exception(
                    "Iteration %d: failed to append to master_plot_inputs (fixed-model)",
                    iter_index,
                )

            # Build run identity for naming
            abs_input_posix_i, short_hash_i, full_hash_i, effective_params_i = (
                build_run_identity(params_load, params_for_iter)
            )

            artifact_paths_i = []
            if plot_each_iteration and list_plot_params:
                try:
                    artifact_paths_i = render_plots(
                        list_plot_params=list_plot_params,
                        df_included=df_iter,
                        summary=summary_iter,
                        short_hash=short_hash_i,
                        df_excluded=original_excluded,
                        output_dir=str(iter_dir),
                    )
                except Exception:
                    logger.exception("Iteration %d: failed to render plots", iter_index)

            # Counts and manifest similar to _orchestrate
            total_input_rows_i = int(len(df_loaded))
            processed_row_count_i = int(len(df_iter))
            excluded_row_count_i = (
                int(len(original_excluded)) if original_excluded is not None else 0
            )
            pre_zscore_excluded_i = max(
                total_input_rows_i - processed_row_count_i - excluded_row_count_i, 0
            )
            counts_i = {
                "total_input_rows": total_input_rows_i,
                "processed_row_count": processed_row_count_i,
                "excluded_row_count": excluded_row_count_i,
                "pre_zscore_excluded": pre_zscore_excluded_i,
            }

            manifest_i = build_manifest_dict(
                abs_input_posix=abs_input_posix_i,
                counts=counts_i,
                effective_params=effective_params_i,
                hashes=(short_hash_i, full_hash_i),
                artifact_paths=artifact_paths_i,
            )
            try:
                write_manifest(
                    str(iter_dir / f"manifest-{short_hash_i}.json"), manifest_i
                )
            except Exception:
                logger.exception("Iteration %d: failed to write manifest", iter_index)

            # Textual report using assemble_text_report and build_model_comparison
            try:
                best_label_i, table_text_i = build_model_comparison(
                    summary_iter.regression_diagnostics
                )
                report_i = assemble_text_report(
                    df_loaded,
                    TransformOutputs(df_range=df_iter, df_excluded=original_excluded),
                    summary_iter,
                    table_text_i,
                    best_label_i,
                    params_for_iter.verbose_filtering,
                )
                (iter_dir / f"report-{short_hash_i}.txt").write_text(
                    report_i, encoding="utf-8"
                )
            except Exception:
                logger.exception(
                    "Iteration %d: failed to build/write textual report", iter_index
                )

        except Exception as e_iter:
            # Write a small error.txt describing the failure and continue
            logger.exception(
                "Iteration %d: cached summarize failed: %s", iter_index, e_iter
            )
            (iter_dir / "error.txt").write_text(
                f"{type(e_iter).__name__}: {e_iter}\n", encoding="utf-8"
            )

        # Prepare next iteration
        cur_fort = new_fort
        iter_index += 1

    # Final concise summary to stdout
    print("Validation run complete (fixed-model reuse).")
    print(
        f"Master offline cost (seconds): {cache.get('offline_cost', float('nan')):.6f}"
    )
    print("Created iteration directories:")
    for d in iter_dirs:
        print(str(d))

    logger.info(
        "Validation run finished; created %d iteration directories", len(iter_dirs)
    )
