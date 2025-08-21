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
    summarize_and_model = _main.summarize_and_model
    transform_pipeline = _main.transform_pipeline
    write_manifest = _main.write_manifest

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

    # Final concise summary to stdout
    print("Validation run complete.")
    print(f"Master offline cost (seconds): {master_offline_cost:.6f}")
    print("Created iteration directories:")
    for d in iter_dirs:
        print(str(d))

    logger.info(
        "Validation run finished; created %d iteration directories", len(iter_dirs)
    )
