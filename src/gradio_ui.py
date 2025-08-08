"""Gradio UI wrapper for fort_calc pipeline.

Provides a simple interface to upload CSV and run pipeline producing SVGs and ZIP.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

try:
    from .main import (
        LoadSliceParams,
        TransformParams,
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
        LoadSliceParams,
        TransformParams,
        assemble_text_report,
        build_model_comparison,
        build_run_identity,
        get_default_params,
        load_and_slice_csv,
        render_plots,
        summarize_and_model,
        transform_pipeline,
    )

import gradio as gr


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


def _run_pipeline(
    uploaded_file_path: Optional[str],
    start_line: Optional[float],
    end_line: Optional[float],
    zscore_min: float,
    zscore_max: float,
    input_data_fort: int,
    ignore_resetticks: bool,
    delta_mode: str,
):
    """
    Execute pipeline and return (html_embed, zip_path, report_text) or (error_html, None, error_html) on failure.
    """
    if not uploaded_file_path:
        msg = "<div style='color:red'>Error: No CSV file uploaded. Please upload a CSV file.</div>"
        return msg, None, msg

    # Load defaults and overlay UI params
    d_load, d_trans, d_plot = get_default_params()

    # Build LoadSliceParams
    lp = LoadSliceParams(
        log_path=Path(uploaded_file_path).resolve(),
        start_line=_parse_optional_int(start_line),
        end_line=_parse_optional_int(end_line),
        include_header=True,
    )

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
        verbose_filtering=False,
        fail_on_any_invalid_timestamps=d_trans.fail_on_any_invalid_timestamps,
    )

    try:
        abs_input_posix, short_hash, full_hash, effective_params = build_run_identity(
            lp, tp
        )

        # Ensure run directory
        run_dir = Path("gradio_runs") / short_hash
        os.makedirs(run_dir, exist_ok=True)

        # Execute pipeline
        df = load_and_slice_csv(lp)
        transformed = transform_pipeline(df, tp)
        summary = summarize_and_model(transformed.df_range, tp)

        # Build textual report from pipeline outputs
        try:
            best_label, table_text = build_model_comparison(
                summary.regression_diagnostics
            )
            report_text = assemble_text_report(
                df, transformed, summary, table_text, best_label
            )
        except Exception:
            # If report assembly fails, still proceed but capture a short message
            import traceback

            rpt_tb = traceback.format_exc()
            report_text = f"Failed to assemble report: {rpt_tb}"

        # Use default single plot param (preserve defaults)
        list_plot_params = [d_plot]

        # Render plots (will create SVG files in current working directory)
        artifact_paths = render_plots(
            list_plot_params,
            transformed.df_range,
            summary,
            short_hash,
            transformed.df_excluded,
        )

        # Move generated svgs into run_dir
        saved_svgs: List[Path] = []
        for p in artifact_paths:
            src = Path(p)
            if src.exists():
                dst = run_dir / src.name
                shutil.move(str(src), str(dst))
                saved_svgs.append(dst)

        # Create zip archive inside run_dir: plots-{short_hash}.zip
        zip_base = run_dir / f"plots-{short_hash}"
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(run_dir))

        # Read SVGs and embed raw content
        parts = []
        for svg_path in sorted(saved_svgs):
            try:
                txt = svg_path.read_text(encoding="utf-8")
            except Exception:
                txt = f"<!-- Failed to read {svg_path} -->"
            parts.append(f"<div>{txt}</div>")
        html = "\n".join(parts)

        return html, str(zip_path), report_text

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        msg = f"<div style='color:red'><h3>Error running pipeline</h3><pre>{str(e)}</pre><pre>{tb}</pre></div>"
        return msg, None, msg


def _build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("### FORT Calculator GUI")
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
        with gr.Row():
            delta = gr.Radio(
                label="delta_mode",
                choices=["PREVIOUS_CHUNK", "FIRST_CHUNK"],
                value=get_default_params()[1].delta_mode.name,
            )
        run_button = gr.Button("Run")
        report_code = gr.Textbox(
            value="", lines=20, interactive=False, elem_id="report_box", label="Report"
        )
        output_html = gr.HTML(label="Plots")
        output_zip = gr.File(label="Download ZIP")

        def _click(file_obj, s_line, e_line, zmin_v, zmax_v, fort_v, ignore_v, delta_v):
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
                delta_v,
            )
            # gr.File accepts None to indicate no file available
            file_out = zip_p if zip_p is not None else None
            # report_p may be HTML or plain text; present as text block
            return html, file_out, report_p

        run_button.click(
            _click,
            inputs=[file_input, start_line, end_line, zmin, zmax, fort, ignore, delta],
            outputs=[output_html, output_zip, report_code],
        )

    return demo


if __name__ == "__main__":
    demo = _build_ui()
    demo.launch()
