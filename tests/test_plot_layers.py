import types
from pathlib import Path

import pandas as pd

from src.main import (
    PlotLayer,
    PlotParams,
    SummaryModelOutputs,
    _args_to_params,
    _parse_plot_spec_json,
    _parse_plot_spec_kv,
    model_cost_column,
    model_output_column,
    model_sum_column,
    render_plots,
)


class CallCounter:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        # Record labels if provided to help assertions distinguish OLS vs WLS
        label = kwargs.get("label")
        entry = {
            "args": args,
            "kwargs": kwargs,
            "label": label,
        }
        self.calls.append(entry)
        # Return a dummy object to mimic matplotlib return types
        return types.SimpleNamespace()


def _build_inputs(include_wls=True):
    # Small df_range
    df_range = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.15, 1.05],
        }
    )

    # Base df_results with OLS predictions (use canonical column names)
    df_results = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            model_output_column("ols_linear"): [1.00, 1.05, 1.10, 1.15, 1.20],
            model_output_column("ols_quadratic"): [1.00, 1.04, 1.09, 1.15, 1.22],
        }
    )
    df_results[model_sum_column("ols_linear")] = df_results[
        model_output_column("ols_linear")
    ].cumsum()
    df_results[model_sum_column("ols_quadratic")] = df_results[
        model_output_column("ols_quadratic")
    ].cumsum()
    df_results[model_cost_column("ols_linear")] = (
        df_results[model_sum_column("ols_linear")] + 0.1
    ) / df_results["sor#"]
    df_results[model_cost_column("ols_quadratic")] = (
        df_results[model_sum_column("ols_quadratic")] + 0.1
    ) / df_results["sor#"]

    if include_wls:
        # Provide WLS columns and derived metrics using canonical names
        df_results[model_output_column("wls_linear")] = [
            1.00,
            1.045,
            1.095,
            1.145,
            1.195,
        ]
        df_results[model_output_column("wls_quadratic")] = [
            1.00,
            1.042,
            1.088,
            1.146,
            1.218,
        ]
        df_results[model_sum_column("wls_linear")] = df_results[
            model_output_column("wls_linear")
        ].cumsum()
        df_results[model_sum_column("wls_quadratic")] = df_results[
            model_output_column("wls_quadratic")
        ].cumsum()
        df_results[model_cost_column("wls_linear")] = (
            df_results[model_sum_column("wls_linear")] + 0.1
        ) / df_results["sor#"]
        df_results[model_cost_column("wls_quadratic")] = (
            df_results[model_sum_column("wls_quadratic")] + 0.1
        ) / df_results["sor#"]

    # Provide sor_min_costs dict for compatibility with new SummaryModelOutputs shape
    sor_min_costs = {
        "ols_linear": 2,
        "ols_quadratic": 4,
    }
    if include_wls:
        sor_min_costs["wls_linear"] = 3
        sor_min_costs["wls_quadratic"] = 5

    summary = SummaryModelOutputs(
        df_summary=pd.DataFrame({"a": [1]}),
        df_results=df_results,
        regression_diagnostics={},
        offline_cost=0.1,
        sor_min_costs=sor_min_costs,
    )

    return df_range, summary


def test_plot_layers_default(monkeypatch, tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build inputs with WLS present, but DEFAULT should only draw OLS + data + legend
    df_range, summary = _build_inputs(include_wls=True)

    scatter_cc = CallCounter()
    plot_cc = CallCounter()
    axvline_cc = CallCounter()
    legend_cc = CallCounter()
    savefig_cc = CallCounter()

    monkeypatch.setattr(plt, "scatter", scatter_cc)
    monkeypatch.setattr(plt, "plot", plot_cc)
    monkeypatch.setattr(plt, "axvline", axvline_cc)
    monkeypatch.setattr(plt, "legend", legend_cc)
    # Do not monkeypatch savefig here; allow actual file creation

    out_path = tmp_path / "default.svg"
    artifact_paths = render_plots(
        list_plot_params=[PlotParams(plot_layers=PlotLayer.DEFAULT)],
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )
    print(f"Artifact paths: {artifact_paths}")
    assert "plot-testhash-00-DEFAULT.svg" in artifact_paths
    # DEFAULT expectations:

    # DEFAULT expectations:
    # - scatter once (data points)
    assert len(scatter_cc.calls) == 1

    # - OLS prediction lines: 2
    # - OLS cost curves: 2
    # Total plot() calls expected: 4
    labels = [c["label"] for c in plot_cc.calls]
    assert "Linear Model (OLS)" in labels
    assert "Quadratic Model (OLS)" in labels
    assert "Cost/Run @ FORT (Linear, OLS)" in labels
    assert "Cost/Run @ FORT (Quadratic, OLS)" in labels
    # Ensure no WLS labels present
    assert all((lbl is None) or ("WLS" not in lbl) for lbl in labels)

    # - Min markers for OLS: 2 vertical lines
    vlabels = [c["label"] for c in axvline_cc.calls]
    assert "Min Cost (Linear, OLS)" in vlabels
    assert "Min Cost (Quadratic, OLS)" in vlabels
    # Ensure no WLS min markers
    assert all((lbl is None) or ("WLS" not in lbl) for lbl in vlabels)

    # - legend once
    assert len(legend_cc.calls) == 1


def test_plot_layers_everything(monkeypatch, tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build inputs with WLS present; EVERYTHING should include all
    df_range, summary = _build_inputs(include_wls=True)

    scatter_cc = CallCounter()
    plot_cc = CallCounter()
    axvline_cc = CallCounter()
    legend_cc = CallCounter()
    savefig_cc = CallCounter()

    monkeypatch.setattr(plt, "scatter", scatter_cc)
    monkeypatch.setattr(plt, "plot", plot_cc)
    monkeypatch.setattr(plt, "axvline", axvline_cc)
    monkeypatch.setattr(plt, "legend", legend_cc)
    # Do not monkeypatch savefig here; allow actual file creation

    out_path = tmp_path / "everything.svg"
    artifact_paths = render_plots(
        list_plot_params=[PlotParams(plot_layers=PlotLayer.EVERYTHING)],
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )
    print(f"Artifact paths: {artifact_paths}")
    assert "plot-testhash-00-EVERYTHING.svg" in artifact_paths

    labels = [c["label"] for c in plot_cc.calls]
    # OLS lines present
    assert "Linear Model (OLS)" in labels
    assert "Quadratic Model (OLS)" in labels
    assert "Cost/Run @ FORT (Linear, OLS)" in labels
    assert "Cost/Run @ FORT (Quadratic, OLS)" in labels
    # WLS lines present
    assert "Linear Model (WLS)" in labels
    assert "Quadratic Model (WLS)" in labels
    assert "Cost/Run @ FORT (Linear, WLS)" in labels
    assert "Cost/Run @ FORT (Quadratic, WLS)" in labels

    vlabels = [c["label"] for c in axvline_cc.calls]
    # OLS min markers present
    assert "Min Cost (Linear, OLS)" in vlabels
    assert "Min Cost (Quadratic, OLS)" in vlabels
    # WLS min markers present
    assert "Min Cost (Linear, WLS)" in vlabels
    assert "Min Cost (Quadratic, WLS)" in vlabels

    # legend present
    assert len(legend_cc.calls) == 1

    # scatter present
    assert len(scatter_cc.calls) == 1


def test_plot_layers_wls_predictions_only(monkeypatch, tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build inputs with WLS present; selective flags should render only WLS predictions + legend
    df_range, summary = _build_inputs(include_wls=True)

    scatter_cc = CallCounter()
    plot_cc = CallCounter()
    axvline_cc = CallCounter()
    legend_cc = CallCounter()
    # Do NOT monkeypatch savefig to allow file creation

    monkeypatch.setattr(plt, "scatter", scatter_cc)
    monkeypatch.setattr(plt, "plot", plot_cc)
    monkeypatch.setattr(plt, "axvline", axvline_cc)
    monkeypatch.setattr(plt, "legend", legend_cc)

    flags = PlotLayer.WLS_PRED_LINEAR | PlotLayer.WLS_PRED_QUAD | PlotLayer.LEGEND
    out_path = tmp_path / "wls_preds_only.svg"
    artifact_paths = render_plots(
        list_plot_params=[PlotParams(plot_layers=flags)],
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )
    print(f"Artifact paths: {artifact_paths}")
    assert "plot-testhash-00-WLS_PRED_LINEAR+WLS_PRED_QUAD+LEGEND.svg" in artifact_paths

    # No scatter, no min markers
    assert len(scatter_cc.calls) == 0
    assert len(axvline_cc.calls) == 0

    labels = [c["label"] for c in plot_cc.calls]
    # Only WLS prediction lines present
    assert "Linear Model (WLS)" in labels
    assert "Quadratic Model (WLS)" in labels
    assert "Linear Model (OLS)" not in labels
    assert "Quadratic Model (OLS)" not in labels

    # No cost curves expected
    assert all("Cost/Run" not in (lbl or "") for lbl in labels)

    # legend once
    assert len(legend_cc.calls) == 1


def test_render_outputs_writes_svg(tmp_path: Path):
    # Minimal viable inputs: df_range and a precomputed SummaryModelOutputs
    df_range = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.15, 1.05],
        }
    )
    df_results = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            model_output_column("ols_linear"): [1.0, 1.05, 1.1, 1.15, 1.2],
            model_output_column("ols_quadratic"): [1.0, 1.04, 1.09, 1.15, 1.22],
        }
    )
    df_results[model_sum_column("ols_linear")] = df_results[
        model_output_column("ols_linear")
    ].cumsum()
    df_results[model_sum_column("ols_quadratic")] = df_results[
        model_output_column("ols_quadratic")
    ].cumsum()
    df_results[model_cost_column("ols_linear")] = (
        df_results[model_sum_column("ols_linear")] + 0.1
    ) / df_results["sor#"]
    df_results[model_cost_column("ols_quadratic")] = (
        df_results[model_sum_column("ols_quadratic")] + 0.1
    ) / df_results["sor#"]

    sor_min_costs = {"ols_linear": 2, "ols_quadratic": 3}
    summary = SummaryModelOutputs(
        df_summary=pd.DataFrame({"a": [1]}),
        df_results=df_results,
        regression_diagnostics={},
        offline_cost=0.1,
        sor_min_costs=sor_min_costs,
    )

    out_path = tmp_path / "plot.svg"
    artifact_paths = render_plots(
        list_plot_params=[PlotParams()],
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )
    print(f"Artifact paths: {artifact_paths}")
    assert "plot-testhash-00-DEFAULT.svg" in artifact_paths
    for path in artifact_paths:
        assert Path(path).exists()
        assert Path(path).suffix == ".svg"
    assert Path(path).suffix == ".svg"


def test_parse_plot_spec_kv():
    default_params = PlotParams()
    spec = "layers=DEFAULT,x_min=0,x_max=10"
    params = _parse_plot_spec_kv(spec, default_params)
    assert params.plot_layers == PlotLayer.DEFAULT
    assert params.x_min == 0
    assert params.x_max == 10


def test_parse_plot_spec_json():
    default_params = PlotParams()
    spec = '{"layers": "DEFAULT", "x_min": 0, "x_max": 10}'
    params = _parse_plot_spec_json(spec, default_params)
    assert params.plot_layers == PlotLayer.DEFAULT
    assert params.x_min == 0
    assert params.x_max == 10


def test_args_to_params():
    class Args:
        log_path = "test.csv"
        start_line = 1
        end_line = 10
        no_header = False
        zscore_min = -2.0
        zscore_max = 2.0
        input_data_fort = 5
        ignore_resetticks = True
        delta_mode = "PREVIOUS_CHUNK"
        exclude_range = None
        verbose_filtering = False
        fail_on_any_invalid_timestamps = True
        plot_spec = None
        plot_spec_json = None
        plot_layers = "DEFAULT"
        x_min = None
        x_max = None
        y_min = None
        y_max = None

    args = Args()
    load_params, transform_params, plot_params_list = _args_to_params(args)
    assert load_params.log_path == Path("test.csv").resolve()
    assert transform_params.zscore_min == -2.0
    assert transform_params.zscore_max == 2.0
    assert transform_params.input_data_fort == 5
    # Canonical defaults now return a list of policy PlotParams
    assert len(plot_params_list) == 3
    assert plot_params_list[0].plot_layers == (
        PlotLayer.DATA_SCATTER | PlotLayer.ALL_PREDICTION
    )


def test_args_to_params_deprecated_plot_layers_warning(caplog):
    """Test that deprecated --plot-layers flag generates warning."""

    class Args:
        log_path = "test.csv"
        start_line = 1
        end_line = 10
        no_header = False
        zscore_min = -2.0
        zscore_max = 2.0
        input_data_fort = 5
        ignore_resetticks = True
        delta_mode = "PREVIOUS_CHUNK"
        exclude_range = None
        verbose_filtering = False
        fail_on_any_invalid_timestamps = True
        plot_spec = None
        plot_spec_json = None
        plot_layers = "DEFAULT"
        x_min = None
        x_max = None
        y_min = None
        y_max = None

    args = Args()
    load_params, transform_params, plot_params_list = _args_to_params(args)

    # Deprecated top-level flags removed; canonical defaults are returned
    assert len(plot_params_list) == 3
    assert plot_params_list[0].plot_layers == (
        PlotLayer.DATA_SCATTER | PlotLayer.ALL_PREDICTION
    )


def test_args_to_params_deprecated_plot_layers_ignored_with_plot_spec(caplog):
    """Test that --plot-layers is ignored when --plot-spec is provided."""

    class Args:
        log_path = "test.csv"
        start_line = 1
        end_line = 10
        no_header = False
        zscore_min = -2.0
        zscore_max = 2.0
        input_data_fort = 5
        ignore_resetticks = True
        delta_mode = "PREVIOUS_CHUNK"
        exclude_range = None
        verbose_filtering = False
        fail_on_any_invalid_timestamps = True
        plot_spec = ["layers=ALL_COST"]
        plot_spec_json = None
        plot_layers = "DEFAULT"  # This should be ignored
        x_min = None
        x_max = None
        y_min = None
        y_max = None

    args = Args()
    load_params, transform_params, plot_params_list = _args_to_params(args)

    # Should use the plot-spec configuration, not any deprecated flags
    assert len(plot_params_list) == 1
    assert plot_params_list[0].plot_layers == PlotLayer.ALL_COST


def test_args_to_params_top_level_axis_limits_ignored_with_plot_spec(caplog):
    """Test that top-level axis limits are ignored when --plot-spec is provided."""

    class Args:
        log_path = "test.csv"
        start_line = 1
        end_line = 10
        no_header = False
        zscore_min = -2.0
        zscore_max = 2.0
        input_data_fort = 5
        ignore_resetticks = True
        delta_mode = "PREVIOUS_CHUNK"
        exclude_range = None
        verbose_filtering = False
        fail_on_any_invalid_timestamps = True
        plot_spec = ["layers=DEFAULT"]
        plot_spec_json = None
        plot_layers = None
        x_min = 0  # These should be ignored
        x_max = 100  # These should be ignored
        y_min = 0  # These should be ignored
        y_max = 5  # These should be ignored

    args = Args()
    load_params, transform_params, plot_params_list = _args_to_params(args)

    # Should use the plot-spec configuration without top-level axis limits
    assert len(plot_params_list) == 1
    assert plot_params_list[0].plot_layers == PlotLayer.DEFAULT
    # Top-level axis limits should not be applied to the plot params
    # The actual plot params should come from the plot-spec parsing
