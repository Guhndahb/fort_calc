"""Gradio UI wrapper for fort_calc pipeline.

Provides a simple interface to upload CSV and run pipeline producing SVGs and ZIP.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

# Backend selection is enforced centrally in src.main at import-time to avoid races
# and duplicated backend-setting. Do NOT set or override MPLBACKEND here.
# (Previous attempts to set the backend in multiple places caused regressions.)

try:
    from .main import (
        OMIT_FORT,
        LoadSliceParams,
        PlotParams,
        TransformParams,
        _parse_plot_layers,
        _plot_layers_suffix,
        assemble_text_report,
        build_model_comparison,
        build_run_identity,
        get_default_params,
        load_and_slice_csv,
        render_plots,
        summarize_and_model,
        transform_pipeline,
    )
except Exception:
    from main import (  # type: ignore
        OMIT_FORT,
        LoadSliceParams,
        PlotParams,
        TransformParams,
        _parse_plot_layers,
        _plot_layers_suffix,
        assemble_text_report,
        build_model_comparison,
        build_run_identity,
        get_default_params,
        load_and_slice_csv,
        render_plots,
        summarize_and_model,
        transform_pipeline,
    )

import logging
import time
import traceback

import gradio as gr

logger = logging.getLogger(__name__)


def _parse_optional_int(val: Optional[float]) -> Optional[int]:
    if val is None:
        return None
    try:
        # Accept 0 as a valid index. Treat empty strings or unparseable values as None.
        s = val
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return None
        # Convert via float to accept numeric inputs from Gradio (which may provide floats)
        return int(float(s))
    except Exception:
        return None


def parse_plot_specs(raw: Optional[str], default_plot: PlotParams) -> List[PlotParams]:
    """
    Parse multiline plot spec input. Each non-empty line is either a JSON object
    (starts with '{' or '[') or a key=value[,key=value...] style spec.
    Returns a list of PlotParams instances. If raw is None/blank returns [].
    On parse failure raises ValueError mentioning the offending line.

    This implementation removes the temporary file-based debug logging and refactors
    parsing into small helpers for readability while preserving original semantics.
    """
    import json

    if raw is None or str(raw).strip() == "":
        return []

    def _baseline() -> PlotParams:
        return PlotParams(
            plot_layers=default_plot.plot_layers,
            x_min=default_plot.x_min,
            x_max=default_plot.x_max,
            y_min=default_plot.y_min,
            y_max=default_plot.y_max,
        )

    def _float_or_none(v, field_name: str):
        if v is None:
            return None
        try:
            return float(v)
        except Exception as e:
            raise ValueError(
                f"Invalid numeric value for {field_name}: {v!r} -> {e}"
            ) from e

    def _parse_json_line(ln: str) -> PlotParams:
        try:
            spec = json.loads(ln)
        except Exception as e:
            raise ValueError(f"Invalid JSON in plot spec: {e}") from e

        if not isinstance(spec, dict):
            raise ValueError(f"JSON plot spec must be an object/dict, got {type(spec)}")

        params = _baseline()

        if "layers" in spec:
            try:
                params.plot_layers = _parse_plot_layers(spec["layers"])
            except Exception as e:
                raise ValueError(f"Failed to parse layers: {e}") from e

        if "x_min" in spec:
            params.x_min = None if spec["x_min"] is None else float(spec["x_min"])
        if "x_max" in spec:
            v = spec["x_max"]
            if v is None:
                params.x_max = None
            elif isinstance(v, str) and str(v).strip().upper() == OMIT_FORT:
                params.x_max = OMIT_FORT
            else:
                params.x_max = float(v)

        if "y_min" in spec:
            params.y_min = None if spec["y_min"] is None else float(spec["y_min"])
        if "y_max" in spec:
            params.y_max = None if spec["y_max"] is None else float(spec["y_max"])

        return params

    def _parse_kv_line(ln: str) -> PlotParams:
        params = _baseline()
        # Split on commas but allow '=' inside values after first '='
        parts = [p.strip() for p in ln.split(",") if p.strip()]
        for kv in parts:
            if "=" not in kv:
                raise ValueError(f"Invalid key=value pair: {kv!r}")
            key, value = kv.split("=", 1)
            key = key.strip()
            val = value.strip()
            # Strip surrounding quotes if present
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val_unq = val[1:-1].strip()
            else:
                val_unq = val

            if key == "layers":
                try:
                    params.plot_layers = _parse_plot_layers(val_unq)
                except Exception as e:
                    raise ValueError(f"Failed to parse layers: {e}") from e
            elif key == "x_min":
                params.x_min = _float_or_none(val_unq, "x_min")
            elif key == "x_max":
                if isinstance(val_unq, str) and val_unq.strip().upper() == OMIT_FORT:
                    params.x_max = OMIT_FORT
                else:
                    params.x_max = _float_or_none(val_unq, "x_max")
            elif key == "y_min":
                params.y_min = _float_or_none(val_unq, "y_min")
            elif key == "y_max":
                params.y_max = _float_or_none(val_unq, "y_max")
            else:
                raise ValueError(f"Unknown key in plot spec: {key}")
        return params

    lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    parsed: List[PlotParams] = []
    for ln in lines:
        try:
            if ln and ln[0] in ("{", "["):
                params = _parse_json_line(ln)
            else:
                params = _parse_kv_line(ln)
            parsed.append(params)
        except Exception as e:
            # Preserve helpful error text and mention offending line like before
            raise ValueError(f"Failed to parse plot spec line: {ln!r} -> {e}") from e

    return parsed


def _prune_old_runs(run_root: Path, keep: Optional[int] = None) -> None:
    """
    Retention helper: keep only the newest `keep` subdirectories under `run_root`.

    Behavior changes:
      - Prefer deterministic name-based ordering for timestamped run directories (YYYYmmddTHHMMSS).
      - Fall back to mtime ordering when names are not parseable.
      - Log deletion failures at WARNING so they are visible; no retries — failures are retried on future runs.
    """
    try:
        # Resolve keep from env if not explicitly provided
        if keep is None:
            try:
                keep = int(os.getenv("FORT_GRADIO_RETENTION_KEEP", "10"))
            except Exception:
                keep = 10
        if keep <= 0:
            logger.debug(f"retention keep <=0 ({keep}) -> skipping prune")
            return

        if not run_root.exists() or not run_root.is_dir():
            return

        # Collect subdirectories only
        subdirs = [p for p in run_root.iterdir() if p.is_dir()]
        if not subdirs:
            return

        # Helper: detect names matching the run_ts format used elsewhere: YYYYmmddTHHMMSS...
        def _looks_like_run_ts(name: str) -> bool:
            try:
                return (
                    len(name) >= 15
                    and name[0:4].isdigit()
                    and name[4:8].isdigit()
                    and name[8] == "T"
                    and name[9:15].isdigit()
                )
            except Exception:
                return False

        # If all directory names look like the timestamp format, prefer lexicographic (name) sort
        # which is monotonic for the timestamp format. Newest first.
        if all(_looks_like_run_ts(p.name) for p in subdirs):
            subdirs_sorted = sorted(subdirs, key=lambda p: p.name, reverse=True)
        else:
            # Fallback to mtime sort (newest first) for mixed or non-standard names
            subdirs_sorted = sorted(
                subdirs, key=lambda p: p.stat().st_mtime, reverse=True
            )

        to_delete = subdirs_sorted[keep:]

        # Resolve run_root once for repeated checks
        try:
            run_root_resolved = run_root.resolve()
        except Exception:
            # If we cannot resolve the run_root, avoid deleting anything as a safety measure.
            logger.warning(f"Could not resolve run_root {run_root} - skipping prune")
            return

        def _safe_is_subpath(child: Path, parent: Path) -> bool:
            """
            Return True if `child.resolve()` is located inside `parent.resolve()`.
            Conservative: on any error return False to avoid accidental deletes.
            """
            try:
                # Resolve both paths (non-strict to allow deletion of directories even if mount points change)
                parent_r = parent.resolve()
                child_r = child.resolve()
            except Exception:
                return False
            try:
                import os

                return os.path.commonpath([str(parent_r), str(child_r)]) == str(
                    parent_r
                )
            except Exception:
                return False

        for d in to_delete:
            try:
                # Skip symbolic links/junctions to avoid following targets outside run_root.
                if d.is_symlink():
                    logger.warning(f"Skipping symlink during prune: {d}")
                    continue

                # Ensure the resolved target is inside the intended run_root to defend against
                # junctions / symlinks pointing outside the app-owned directory (platform dependent).
                if not _safe_is_subpath(d, run_root_resolved):
                    logger.warning(f"Skipping prune of {d} - resolved outside run_root")
                    continue

                shutil.rmtree(d)
                logger.info(f"Pruned old run dir: {d}")
            except Exception as e:
                # Surface deletion failures so they're visible in production logs;
                # do not retry here — a future run will attempt again.
                logger.warning(f"Failed to prune {d}: {e}")

    except Exception as e:
        logger.exception(f"Prune old runs unexpected error: {e}")


def _run_pipeline(
    uploaded_file_path: Optional[str],
    start_line: Optional[float],
    end_line: Optional[float],
    zscore_min: float,
    zscore_max: float,
    input_data_fort: int,
    ignore_resetticks: bool,
    verbose_filtering: bool,
    delta_mode: str,
    iqr_k_low: Optional[float],
    iqr_k_high: Optional[float],
    use_iqr_filtering: bool,
    col_sor: Optional[int] = None,
    col_ticks: Optional[int] = None,
    plot_specs_raw: Optional[str] = None,
):
    """
    Execute pipeline and return (html_embed, zip_path, report_text) or (error_html, None, error_html) on failure.
    Added debug prints to trace progress when running via the Gradio UI.
    """
    t0 = time.time()
    logger.info(f"_run_pipeline START - uploaded_file_path={uploaded_file_path!r}")

    if not uploaded_file_path:
        msg = "Error: No CSV file uploaded. Please upload a CSV file."
        logger.debug("_run_pipeline EARLY_EXIT no file")
        return msg, None, msg

    # Load defaults and overlay UI params
    _, d_trans, d_plots = get_default_params()
    logger.debug("Loaded default params")

    # Build LoadSliceParams
    lp = LoadSliceParams(
        log_path=Path(uploaded_file_path).resolve(),
        start_line=_parse_optional_int(start_line),
        end_line=_parse_optional_int(end_line),
        include_header=True,
        col_sor=_parse_optional_int(col_sor),
        col_ticks=_parse_optional_int(col_ticks),
    )
    logger.debug(f"Built LoadSliceParams -> {lp}")

    # Map delta_mode string to enum based on the default TransformParams instance
    try:
        DeltaModeEnum = d_trans.delta_mode.__class__  # type: ignore[attr-defined]
        delta_mode_enum = DeltaModeEnum[delta_mode]
    except Exception:
        # Fallback to whatever default is present
        delta_mode_enum = d_trans.delta_mode

    tp = TransformParams(
        zscore_min=zscore_min if zscore_min is not None else d_trans.zscore_min,
        zscore_max=zscore_max if zscore_max is not None else d_trans.zscore_max,
        input_data_fort=input_data_fort
        if input_data_fort is not None
        else d_trans.input_data_fort,
        ignore_resetticks=bool(ignore_resetticks),
        delta_mode=delta_mode_enum,
        exclude_timestamp_ranges=d_trans.exclude_timestamp_ranges,
        verbose_filtering=bool(verbose_filtering),
        fail_on_any_invalid_timestamps=d_trans.fail_on_any_invalid_timestamps,
        iqr_k_low=iqr_k_low if iqr_k_low is not None else d_trans.iqr_k_low,
        iqr_k_high=iqr_k_high if iqr_k_high is not None else d_trans.iqr_k_high,
        use_iqr_filtering=use_iqr_filtering,
    )
    logger.debug(f"Built TransformParams -> {tp}")

    try:
        logger.debug("Calling build_run_identity...")
        abs_input_posix, short_hash, full_hash, effective_params = build_run_identity(
            lp, tp
        )
        logger.debug(f"build_run_identity returned short_hash={short_hash}")

        # Ensure run directory (use per-run output_gradio/<timestamp> like main)
        run_ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        run_dir = Path("output_gradio") / run_ts
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured run_dir={run_dir}")

        # Execute pipeline
        logger.debug("Calling load_and_slice_csv...")
        df = load_and_slice_csv(lp)
        logger.debug(f"load_and_slice_csv returned {len(df)} rows")

        logger.debug("Calling transform_pipeline...")
        transformed = transform_pipeline(df, tp)
        logger.debug(
            f"transform_pipeline complete; df_range len={len(transformed.df_range)}"
        )

        logger.debug("Calling summarize_and_model...")
        summary = summarize_and_model(transformed.df_range, tp)
        logger.debug("summarize_and_model complete")

        # Build textual report from pipeline outputs
        try:
            best_label, table_text = build_model_comparison(
                summary.regression_diagnostics
            )
            report_text = assemble_text_report(
                df, transformed, summary, table_text, best_label, tp.verbose_filtering
            )
            logger.debug("Assembled textual report")

            # Save textual report into the run directory so it can be archived with the plots.
            # Use UTF-8 encoding and best-effort logging on failure (do not fail the pipeline).
            try:
                report_path = run_dir / f"report-{short_hash}.txt"
                report_path.write_text(report_text, encoding="utf-8")
                logger.debug(f"Saved textual report to {report_path}")
            except Exception:
                logger.debug(f"Failed to write textual report to {report_path}")
        except Exception:
            rpt_tb = traceback.format_exc()
            report_text = f"Failed to assemble report: {rpt_tb}"
            logger.debug(f"Report assembly FAILED: {rpt_tb}")

        # Decide plot parameter list: parse UI-provided multiline specs or fallback to defaults
        try:
            logger.debug(f"Parsing plot specs (raw={plot_specs_raw!r})")
            parsed_list = parse_plot_specs(plot_specs_raw, d_plots[0])
            logger.debug(f"parse_plot_specs returned {len(parsed_list)} entries")
        except ValueError as ve:
            tb = traceback.format_exc()
            msg = f"Plot specs parse error\n{str(ve)}\n{tb}"
            logger.debug(f"parse_plot_specs ERROR: {ve}")
            return msg, None, msg

        if parsed_list:
            list_plot_params = parsed_list
        else:
            # Use canonical default plot list
            list_plot_params = d_plots
        logger.debug(f"Will render {len(list_plot_params)} plot(s)")

        # Render plots (will create SVG files in run output directory)
        logger.debug("Calling render_plots...")
        artifact_paths = render_plots(
            list_plot_params,
            transformed.df_range,
            summary,
            short_hash,
            transformed.df_excluded,
            output_dir=str(run_dir),
        )
        logger.debug(f"render_plots returned {artifact_paths}")

        # Move generated svgs into run_dir (skip move if already in run_dir)
        saved_svgs: List[Path] = []
        for p in artifact_paths:
            src = Path(p)
            if src.exists():
                try:
                    if src.parent.resolve() != run_dir.resolve():
                        dst = run_dir / src.name
                        shutil.move(str(src), str(dst))
                        saved_svgs.append(dst)
                    else:
                        saved_svgs.append(src)
                except Exception:
                    # Fallback: attempt move if resolve fails; best-effort only
                    dst = run_dir / src.name
                    try:
                        shutil.move(str(src), str(dst))
                        saved_svgs.append(dst)
                    except Exception:
                        pass
        logger.debug(f"Saved svgs: {saved_svgs}")

        # If we successfully wrote a textual report above, include it in the artifacts to be zipped.
        # Keep a separate list for zip artifacts so we inline only SVGs into the HTML.
        artifacts_for_zip = list(saved_svgs)
        try:
            if "report_path" in locals() and report_path.exists():
                artifacts_for_zip.append(report_path)
                logger.debug(f"Included report in zip artifacts: {report_path}")
        except Exception as _e_inc:
            logger.debug(f"Failed to include report in artifacts: {_e_inc}")

        # Prune older run directories according to retention policy so public-facing UI
        # doesn't accumulate unlimited artifacts. The helper reads FORT_GRADIO_RETENTION_KEEP
        # (default 10) to determine how many most-recent run dirs to keep.
        try:
            _prune_old_runs(Path("output_gradio"))
        except Exception as _e_prune:
            # Never fail the request because pruning had an issue; log lightweight debug info.
            logger.debug(f"Prune old runs error: {_e_prune}")

        # Create zip archive inside run_dir: plots-{short_hash}.zip
        # To avoid blocking the Gradio worker (which appeared to hang for some users),
        # create the zip archive asynchronously in a daemon thread and return immediately.
        # The UI will still display generated SVGs; the ZIP will be produced shortly.
        import threading

        zip_base = run_dir / f"plots-{short_hash}"
        zip_path = str(zip_base) + ".zip"

        def _create_zip_async(zip_path_local: str, svgs: list[Path]):
            try:
                with zipfile.ZipFile(
                    zip_path_local, "w", compression=zipfile.ZIP_DEFLATED
                ) as zf:
                    for svg in svgs:
                        zf.write(svg, arcname=svg.name)
                logger.debug(f"Async zip created at {zip_path_local}")
            except Exception as e_zip:
                logger.debug(f"Async zip failed: {e_zip}")

        try:
            thread = threading.Thread(
                target=_create_zip_async,
                args=(zip_path, list(artifacts_for_zip)),
                daemon=True,
            )
            thread.start()
            logger.debug("Zip creation started in background thread")
        except Exception as _e_thread:
            logger.debug(f"Failed to start async zip thread: {_e_thread}")
            zip_path = ""

        # Read SVGs and embed raw content
        parts = []
        for svg_path in sorted(saved_svgs):
            try:
                txt = svg_path.read_text(encoding="utf-8")
            except Exception:
                txt = f"<!-- Failed to read {svg_path} -->"
            parts.append(f"<div>{txt}</div>")
        html = "\n".join(parts)

        logger.info(
            f"_run_pipeline COMPLETE (duration_ms={(time.time() - t0) * 1000:.1f})"
        )
        return html, str(zip_path), report_text

    except Exception as e:
        tb = traceback.format_exc()
        msg = f"Error running pipeline\n{str(e)}\n{tb}"
        logger.debug(f"_run_pipeline EXCEPTION: {e}\n{tb}")
        return msg, None, msg


def _build_ui():
    with gr.Blocks() as demo:
        d_load, d_trans, d_plots = get_default_params()
        gr.Markdown(
            "### FORT Calculator GUI — [GitHub repository](https://github.com/Guhndahb/fort_calc)"
        )
        gr.HTML("""
<style>
  /* Style the report textbox to use a popular monospace stack and allow vertical resize */
  #report_box textarea {
    font-family: "SF Mono", "Menlo", "Monaco", "Consolas", "Liberation Mono", "Courier New", monospace;
    font-size: 13px;
    line-height: 1.3;
    resize: vertical;
    min-height: 200px;
    max-height: 800px;
  }
</style>
""")
        with gr.Row():
            file_input = gr.File(label="Upload CSV file", file_types=[".csv"])
        with gr.Row():
            start_line = gr.Number(
                label="start_line (optional)",
                value=d_load.start_line,
                precision=0,
                placeholder="leave blank to read from start of file",
            )
            end_line = gr.Number(
                label="end_line (optional)",
                value=d_load.end_line,
                precision=0,
                placeholder="leave blank to read to end of file",
            )
            col_sor = gr.Number(
                label="col_sor (optional, 0-based column for sor#)",
                value=d_load.col_sor,
                precision=0,
                placeholder="leave blank to disable - sor# column will be required in original data",
            )
            col_ticks = gr.Number(
                label="col_ticks (optional, 0-based column for runticks)",
                value=d_load.col_ticks,
                precision=0,
                placeholder="leave blank to disable - runticks column will be required in original data",
            )
        with gr.Row():
            fort = gr.Number(
                label="input_data_fort (critical)",
                value=d_trans.input_data_fort,
                precision=0,
                placeholder="this must match the FORT used in data collection - all data processed must have the same FORT",
            )
            ignore = gr.Checkbox(
                label="ignore_resetticks",
                value=d_trans.ignore_resetticks,
            )
            delta = gr.Radio(
                label="delta_mode",
                choices=["PREVIOUS_CHUNK", "FIRST_CHUNK"],
                value=d_trans.delta_mode.name,
            )
            verbose = gr.Checkbox(label="verbose_filtering", value=False)
        with gr.Row():
            use_iqr = gr.Checkbox(
                label="use_iqr_filtering",
                value=d_trans.use_iqr_filtering,
            )
            iqr_k_low = gr.Number(
                label="iqr_k_low",
                value=(
                    None if d_trans.iqr_k_low is None else round(d_trans.iqr_k_low, 2)
                ),
                precision=2,
                step=0.25,
                visible=d_trans.use_iqr_filtering,
            )
            iqr_k_high = gr.Number(
                label="iqr_k_high",
                value=(
                    None if d_trans.iqr_k_high is None else round(d_trans.iqr_k_high, 2)
                ),
                precision=2,
                step=0.25,
                visible=d_trans.use_iqr_filtering,
            )
            zmin = gr.Number(
                label="zscore_min",
                value=(
                    None if d_trans.zscore_min is None else round(d_trans.zscore_min, 2)
                ),
                precision=2,
                step=0.25,
                visible=not d_trans.use_iqr_filtering,
            )
            zmax = gr.Number(
                label="zscore_max",
                value=(
                    None if d_trans.zscore_max is None else round(d_trans.zscore_max, 2)
                ),
                precision=2,
                step=0.25,
                visible=not d_trans.use_iqr_filtering,
            )

        # Callback to toggle visibility of IQR vs zscore controls based on checkbox.
        def _toggle_iqr_visibility(use_iqr_val):
            # When use_iqr_val is True: show iqr_k_low/iqr_k_high, hide zmin/zmax
            # When False: hide iqr_k_low/iqr_k_high, show zmin/zmax
            return (
                gr.update(visible=use_iqr_val),
                gr.update(visible=use_iqr_val),
                gr.update(visible=not use_iqr_val),
                gr.update(visible=not use_iqr_val),
            )

        # Wire checkbox change to visibility toggle (placed before run_button click)
        use_iqr.change(
            _toggle_iqr_visibility,
            inputs=[use_iqr],
            outputs=[iqr_k_low, iqr_k_high, zmin, zmax],
        )

        # Build a default multiline plot-spec value from the canonical default plot list
        plot_default_lines: list[str] = []
        for pp in d_plots:
            layers_token = _plot_layers_suffix(pp.plot_layers)
            parts: list[str] = [f"layers={layers_token}"]
            if pp.x_min is not None:
                parts.append(f"x_min={pp.x_min}")
            if pp.x_max is not None:
                parts.append(
                    f"x_max={OMIT_FORT}"
                    if pp.x_max == OMIT_FORT
                    else f"x_max={pp.x_max}"
                )
            if pp.y_min is not None:
                parts.append(f"y_min={pp.y_min}")
            if pp.y_max is not None:
                parts.append(f"y_max={pp.y_max}")
            plot_default_lines.append(",".join(parts))
        plot_defaults_value = "\n".join(plot_default_lines)

        plot_specs = gr.Textbox(
            label="Plot specs (optional) - one per line (key=value,... or JSON)",
            placeholder='layers=DEFAULT\nlayers=DATA_SCATTER+OLS_PRED_LINEAR,x_max=200\n{"layers":"ALL_COST","x_max":null}',
            value=plot_defaults_value,
            lines=6,
        )
        run_button = gr.Button("Run")
        report_code = gr.Textbox(
            value="", lines=20, interactive=False, elem_id="report_box", label="Report"
        )
        output_html = gr.HTML(label="Plots")
        output_zip = gr.File(label="Download ZIP")

        def _click(
            file_obj,
            s_line,
            e_line,
            col_sor_v,
            col_ticks_v,
            zmin_v,
            zmax_v,
            fort_v,
            ignore_v,
            verbose_v,
            iqr_k_low_v,
            iqr_k_high_v,
            use_iqr_v,
            delta_v,
            plot_specs_raw,
        ):
            # gr.File returns a dict with "name" and "tmp_path" in some versions; accept both
            path = None
            if file_obj is None:
                path = None
            elif isinstance(file_obj, dict):
                path = file_obj.get("name") or file_obj.get("tmp_path")
            elif isinstance(file_obj, str):
                path = file_obj
            else:
                try:
                    path = file_obj.name
                except Exception:
                    path = None
            html, zip_p, report_p = _run_pipeline(
                path,
                s_line,
                e_line,
                zmin_v,
                zmax_v,
                int(fort_v) if fort_v is not None else None,
                ignore_v,
                verbose_v,
                delta_v,
                iqr_k_low_v,
                iqr_k_high_v,
                use_iqr_v,
                col_sor_v,
                col_ticks_v,
                plot_specs_raw,
            )
            # gr.File accepts None to indicate no file available
            file_out = zip_p if zip_p is not None else None
            # report_p may be HTML or plain text; present as text block
            return html, file_out, report_p

        run_button.click(
            _click,
            inputs=[
                file_input,
                start_line,
                end_line,
                col_sor,
                col_ticks,
                zmin,
                zmax,
                fort,
                ignore,
                verbose,
                iqr_k_low,
                iqr_k_high,
                use_iqr,
                delta,
                plot_specs,
            ],
            outputs=[output_html, output_zip, report_code],
        )

    return demo


if __name__ == "__main__":
    demo = _build_ui()
    demo.launch()
