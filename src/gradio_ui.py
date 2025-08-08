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
        assemble_text_report,
        build_model_comparison,
        build_run_identity,
        get_default_params,
        load_and_slice_csv,
        render_plots,
        summarize_and_model,
        transform_pipeline,
    )

import time
import traceback

import gradio as gr


def _short_ts(t: float) -> str:
    """Return a short timestamp for debug prints (seconds with millisecond resolution)."""
    return f"{t:.3f}"


def _parse_optional_int(val: Optional[float]) -> Optional[int]:
    if val is None:
        return None
    try:
        # Treat 0 as "no value"
        if int(val) == 0:
            return None
        return int(val)
    except Exception:
        return None


def parse_plot_specs(raw: Optional[str], default_plot: PlotParams) -> List[PlotParams]:
    """
    Parse multiline plot spec input. Each non-empty line is either a JSON object
    (starts with '{' or '[') or a key=value[,key=value...] style spec.
    Returns a list of PlotParams instances. If raw is None/blank returns [].
    On parse failure raises ValueError mentioning the offending line.

    NOTE: Temporary debug logging to file 'gradio_debug.log' is added to help
    diagnose hangs observed when this is invoked from the Gradio UI. This file
    logging is intentionally lightweight and will be removed once the root cause
    is identified.
    """
    import json

    # Lightweight file-based tracing to avoid stdout capture issues within Gradio workers
    try:
        with open("gradio_debug.log", "a", encoding="utf-8") as fh:
            fh.write(f"[parse_plot_specs] raw_repr={repr(raw)[:1000]}\n")
    except Exception:
        # Best-effort logging only; never fail parsing because logging couldn't be written
        pass

    if raw is None or str(raw).strip() == "":
        try:
            with open("gradio_debug.log", "a", encoding="utf-8") as fh:
                fh.write("[parse_plot_specs] raw blank -> returning []\n")
        except Exception:
            pass
        return []
    lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    parsed: List[PlotParams] = []
    for ln in lines:
        try:
            try:
                with open("gradio_debug.log", "a", encoding="utf-8") as fh:
                    fh.write(f"[parse_plot_specs] parsing line: {ln!r}\n")
            except Exception:
                pass

            # JSON-style spec
            if ln and ln[0] in ("{", "["):
                try:
                    spec_dict = json.loads(ln)
                except Exception as e:
                    raise ValueError(f"Invalid JSON in plot spec: {e}") from e
                params = PlotParams(
                    plot_layers=default_plot.plot_layers,
                    x_min=default_plot.x_min,
                    x_max=default_plot.x_max,
                    y_min=default_plot.y_min,
                    y_max=default_plot.y_max,
                )
                if "layers" in spec_dict:
                    params.plot_layers = _parse_plot_layers(spec_dict["layers"])
                # x_min
                if "x_min" in spec_dict:
                    if spec_dict["x_min"] is None:
                        params.x_min = None
                    else:
                        params.x_min = float(spec_dict["x_min"])
                # x_max supports the OMIT_FORT sentinel string
                if "x_max" in spec_dict:
                    v = spec_dict["x_max"]
                    if v is None:
                        params.x_max = None
                    elif isinstance(v, str) and str(v).strip().upper() == OMIT_FORT:
                        params.x_max = OMIT_FORT
                    else:
                        params.x_max = float(v)
                # y_min / y_max
                if "y_min" in spec_dict:
                    if spec_dict["y_min"] is None:
                        params.y_min = None
                    else:
                        params.y_min = float(spec_dict["y_min"])
                if "y_max" in spec_dict:
                    if spec_dict["y_max"] is None:
                        params.y_max = None
                    else:
                        params.y_max = float(spec_dict["y_max"])
                parsed.append(params)
            else:
                # key=value[,key=value...] style
                params = PlotParams(
                    plot_layers=default_plot.plot_layers,
                    x_min=default_plot.x_min,
                    x_max=default_plot.x_max,
                    y_min=default_plot.y_min,
                    y_max=default_plot.y_max,
                )
                # Split on commas but allow values containing '=' after the first
                for kv in [k.strip() for k in ln.split(",") if k.strip()]:
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
                        params.plot_layers = _parse_plot_layers(val_unq)
                    elif key == "x_min":
                        params.x_min = float(val_unq)
                    elif key == "x_max":
                        # accept sentinel text (case-insensitive)
                        if val_unq.upper() == OMIT_FORT:
                            params.x_max = OMIT_FORT
                        else:
                            params.x_max = float(val_unq)
                    elif key == "y_min":
                        params.y_min = float(val_unq)
                    elif key == "y_max":
                        params.y_max = float(val_unq)
                    else:
                        raise ValueError(f"Unknown key in plot spec: {key}")
                parsed.append(params)

            try:
                with open("gradio_debug.log", "a", encoding="utf-8") as fh:
                    fh.write(f"[parse_plot_specs] parsed line OK -> {parsed[-1]!r}\n")
            except Exception:
                pass
        except Exception as e:
            # Wrap and provide helpful message mentioning offending line
            try:
                with open("gradio_debug.log", "a", encoding="utf-8") as fh:
                    fh.write(f"[parse_plot_specs] ERROR parsing line {ln!r}: {e}\n")
            except Exception:
                pass
            raise ValueError(f"Failed to parse plot spec line: {ln!r} -> {e}") from e
    try:
        with open("gradio_debug.log", "a", encoding="utf-8") as fh:
            fh.write(f"[parse_plot_specs] completed, total_parsed={len(parsed)}\n")
    except Exception:
        pass
    return parsed


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
    plot_specs_raw: Optional[str] = None,
):
    """
    Execute pipeline and return (html_embed, zip_path, report_text) or (error_html, None, error_html) on failure.
    Added debug prints to trace progress when running via the Gradio UI.
    """
    t0 = time.time()
    print(
        f"[DEBUG {_short_ts(t0)}] _run_pipeline START - uploaded_file_path={uploaded_file_path!r}"
    )

    if not uploaded_file_path:
        msg = "<div style='color:red'>Error: No CSV file uploaded. Please upload a CSV file.</div>"
        print(f"[DEBUG {_short_ts(time.time())}] _run_pipeline EARLY_EXIT no file")
        return msg, None, msg

    # Load defaults and overlay UI params
    d_load, d_trans, d_plot = get_default_params()
    print(f"[DEBUG {_short_ts(time.time())}] Loaded default params")

    # Build LoadSliceParams
    lp = LoadSliceParams(
        log_path=Path(uploaded_file_path).resolve(),
        start_line=_parse_optional_int(start_line),
        end_line=_parse_optional_int(end_line),
        include_header=True,
    )
    print(f"[DEBUG {_short_ts(time.time())}] Built LoadSliceParams -> {lp}")

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
    )
    print(f"[DEBUG {_short_ts(time.time())}] Built TransformParams -> {tp}")

    try:
        print(f"[DEBUG {_short_ts(time.time())}] Calling build_run_identity...")
        abs_input_posix, short_hash, full_hash, effective_params = build_run_identity(
            lp, tp
        )
        print(
            f"[DEBUG {_short_ts(time.time())}] build_run_identity returned short_hash={short_hash}"
        )

        # Ensure run directory
        run_dir = Path("gradio_runs") / short_hash
        os.makedirs(run_dir, exist_ok=True)
        print(f"[DEBUG {_short_ts(time.time())}] Ensured run_dir={run_dir}")

        # Execute pipeline
        print(f"[DEBUG {_short_ts(time.time())}] Calling load_and_slice_csv...")
        df = load_and_slice_csv(lp)
        print(
            f"[DEBUG {_short_ts(time.time())}] load_and_slice_csv returned {len(df)} rows"
        )

        print(f"[DEBUG {_short_ts(time.time())}] Calling transform_pipeline...")
        transformed = transform_pipeline(df, tp)
        print(
            f"[DEBUG {_short_ts(time.time())}] transform_pipeline complete; df_range len={len(transformed.df_range)}"
        )

        print(f"[DEBUG {_short_ts(time.time())}] Calling summarize_and_model...")
        summary = summarize_and_model(transformed.df_range, tp)
        print(f"[DEBUG {_short_ts(time.time())}] summarize_and_model complete")

        # Build textual report from pipeline outputs
        try:
            best_label, table_text = build_model_comparison(
                summary.regression_diagnostics
            )
            report_text = assemble_text_report(
                df, transformed, summary, table_text, best_label
            )
            print(f"[DEBUG {_short_ts(time.time())}] Assembled textual report")
        except Exception:
            rpt_tb = traceback.format_exc()
            report_text = f"Failed to assemble report: {rpt_tb}"
            print(f"[DEBUG {_short_ts(time.time())}] Report assembly FAILED: {rpt_tb}")

        # Decide plot parameter list: parse UI-provided multiline specs or fallback to defaults
        try:
            print(
                f"[DEBUG {_short_ts(time.time())}] Parsing plot specs (raw={plot_specs_raw!r})"
            )
            parsed_list = parse_plot_specs(plot_specs_raw, d_plot)
            print(
                f"[DEBUG {_short_ts(time.time())}] parse_plot_specs returned {len(parsed_list)} entries"
            )
        except ValueError as ve:
            tb = traceback.format_exc()
            msg = f"<div style='color:red'><h3>Plot specs parse error</h3><pre>{str(ve)}</pre><pre>{tb}</pre></div>"
            print(f"[DEBUG {_short_ts(time.time())}] parse_plot_specs ERROR: {ve}")
            return msg, None, msg

        if parsed_list:
            list_plot_params = parsed_list
        else:
            # Use default single plot param (preserve defaults)
            list_plot_params = [d_plot]
        print(
            f"[DEBUG {_short_ts(time.time())}] Will render {len(list_plot_params)} plot(s)"
        )

        # Render plots (will create SVG files in current working directory)
        print(f"[DEBUG {_short_ts(time.time())}] Calling render_plots...")
        artifact_paths = render_plots(
            list_plot_params,
            transformed.df_range,
            summary,
            short_hash,
            transformed.df_excluded,
        )
        print(
            f"[DEBUG {_short_ts(time.time())}] render_plots returned {artifact_paths}"
        )

        # Move generated svgs into run_dir
        saved_svgs: List[Path] = []
        for p in artifact_paths:
            src = Path(p)
            if src.exists():
                dst = run_dir / src.name
                shutil.move(str(src), str(dst))
                saved_svgs.append(dst)
        print(f"[DEBUG {_short_ts(time.time())}] Saved svgs: {saved_svgs}")

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
                print(
                    f"[DEBUG {_short_ts(time.time())}] Async zip created at {zip_path_local}"
                )
            except Exception as e_zip:
                print(f"[DEBUG {_short_ts(time.time())}] Async zip failed: {e_zip}")

        try:
            thread = threading.Thread(
                target=_create_zip_async, args=(zip_path, list(saved_svgs)), daemon=True
            )
            thread.start()
            print(
                f"[DEBUG {_short_ts(time.time())}] Zip creation started in background thread"
            )
        except Exception as _e_thread:
            print(
                f"[DEBUG {_short_ts(time.time())}] Failed to start async zip thread: {_e_thread}"
            )
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

        print(
            f"[DEBUG {_short_ts(time.time())}] _run_pipeline COMPLETE (duration_ms={(time.time() - t0) * 1000:.1f})"
        )
        return html, str(zip_path), report_text

    except Exception as e:
        tb = traceback.format_exc()
        msg = f"<div style='color:red'><h3>Error running pipeline</h3><pre>{str(e)}</pre><pre>{tb}</pre></div>"
        print(f"[DEBUG {_short_ts(time.time())}] _run_pipeline EXCEPTION: {e}\n{tb}")
        return msg, None, msg


def _build_ui():
    with gr.Blocks() as demo:
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
                label="start_line (optional)", value=None, precision=0
            )
            end_line = gr.Number(label="end_line (optional)", value=None, precision=0)
        with gr.Row():
            zmin = gr.Number(
                label="zscore_min", value=get_default_params()[1].zscore_min
            )
            zmax = gr.Number(
                label="zscore_max", value=get_default_params()[1].zscore_max
            )
        with gr.Row():
            fort = gr.Number(
                label="input_data_fort",
                value=get_default_params()[1].input_data_fort,
                precision=0,
            )
            ignore = gr.Checkbox(
                label="ignore_resetticks",
                value=get_default_params()[1].ignore_resetticks,
            )
            verbose = gr.Checkbox(label="verbose_filtering", value=False)
        with gr.Row():
            delta = gr.Radio(
                label="delta_mode",
                choices=["PREVIOUS_CHUNK", "FIRST_CHUNK"],
                value=get_default_params()[1].delta_mode.name,
            )
        plot_specs = gr.Textbox(
            label="Plot specs (optional) - one per line (key=value,... or JSON)",
            placeholder='layers=DEFAULT\nlayers=DATA_SCATTER+OLS_PRED_LINEAR,x_max=200\n{"layers":"ALL_COST","x_max":null}',
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
            zmin_v,
            zmax_v,
            fort_v,
            ignore_v,
            verbose_v,
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
                zmin,
                zmax,
                fort,
                ignore,
                verbose,
                delta,
                plot_specs,
            ],
            outputs=[output_html, output_zip, report_code],
        )

    return demo


if __name__ == "__main__":
    demo = _build_ui()
    demo.launch()
