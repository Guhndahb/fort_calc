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

    # Base df_results with OLS predictions
    df_results = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5],
            "linear_model_output": [1.00, 1.05, 1.10, 1.15, 1.20],
            "quadratic_model_output": [1.00, 1.04, 1.09, 1.15, 1.22],
        }
    )
    df_results["sum_lin"] = df_results["linear_model_output"].cumsum()
    df_results["sum_quad"] = df_results["quadratic_model_output"].cumsum()
    df_results["cost_per_run_at_fort_lin"] = (df_results["sum_lin"] + 0.1) / df_results[
        "sor#"
    ]
    df_results["cost_per_run_at_fort_quad"] = (
        df_results["sum_quad"] + 0.1
    ) / df_results["sor#"]

    if include_wls:
        # Provide WLS columns and derived metrics
        df_results["linear_model_output_wls"] = [1.00, 1.045, 1.095, 1.145, 1.195]
        df_results["quadratic_model_output_wls"] = [1.00, 1.042, 1.088, 1.146, 1.218]
        df_results["sum_lin_wls"] = df_results["linear_model_output_wls"].cumsum()
        df_results["sum_quad_wls"] = df_results["quadratic_model_output_wls"].cumsum()
        df_results["cost_per_run_at_fort_lin_wls"] = (
            df_results["sum_lin_wls"] + 0.1
        ) / df_results["sor#"]
        df_results["cost_per_run_at_fort_quad_wls"] = (
            df_results["sum_quad_wls"] + 0.1
        ) / df_results["sor#"]

    summary = SummaryModelOutputs(
        df_summary=pd.DataFrame({"a": [1]}),
        df_results=df_results,
        regression_diagnostics={},
        offline_cost=0.1,
        sor_min_cost_lin=2,
        sor_min_cost_quad=4,
        sor_min_cost_lin_wls=3 if include_wls else None,
        sor_min_cost_quad_wls=5 if include_wls else None,
    )

    return df_range, summary


def test_multi_plot_basic(monkeypatch, tmp_path: Path):
    """Test basic multi-plot functionality with two different plot configurations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build inputs
    df_range, summary = _build_inputs(include_wls=True)

    scatter_cc = CallCounter()
    plot_cc = CallCounter()
    axvline_cc = CallCounter()
    legend_cc = CallCounter()

    monkeypatch.setattr(plt, "scatter", scatter_cc)
    monkeypatch.setattr(plt, "plot", plot_cc)
    monkeypatch.setattr(plt, "axvline", axvline_cc)
    monkeypatch.setattr(plt, "legend", legend_cc)

    # Create two different plot configurations
    plot_params_list = [
        PlotParams(plot_layers=PlotLayer.DEFAULT),  # Default plot
        PlotParams(plot_layers=PlotLayer.ALL_COST),  # Cost curves only
    ]

    artifact_paths = render_plots(
        list_plot_params=plot_params_list,
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )

    # Should generate two plot files
    assert len(artifact_paths) == 2

    # Check file names follow the multi-plot pattern
    assert "plot-testhash-00-DEFAULT.svg" in artifact_paths
    assert "plot-testhash-01-ALL_COST.svg" in artifact_paths

    # Check that both plots were rendered
    for path in artifact_paths:
        assert Path(path).suffix == ".svg"


def test_multi_plot_with_axis_limits(monkeypatch, tmp_path: Path):
    """Test multi-plot with custom axis limits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build inputs
    df_range, summary = _build_inputs(include_wls=True)

    scatter_cc = CallCounter()
    plot_cc = CallCounter()
    axvline_cc = CallCounter()
    legend_cc = CallCounter()

    monkeypatch.setattr(plt, "scatter", scatter_cc)
    monkeypatch.setattr(plt, "plot", plot_cc)
    monkeypatch.setattr(plt, "axvline", axvline_cc)
    monkeypatch.setattr(plt, "legend", legend_cc)

    # Create plot configurations with axis limits
    plot_params_list = [
        PlotParams(plot_layers=PlotLayer.DEFAULT, x_min=0, x_max=100),
        PlotParams(plot_layers=PlotLayer.ALL_COST, y_min=0, y_max=5),
    ]

    artifact_paths = render_plots(
        list_plot_params=plot_params_list,
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )

    # Should generate two plot files
    assert len(artifact_paths) == 2

    # Check file names follow the multi-plot pattern
    assert "plot-testhash-00-DEFAULT.svg" in artifact_paths
    assert "plot-testhash-01-ALL_COST.svg" in artifact_paths


def test_parse_plot_spec_kv_multi_plot():
    """Test parsing multiple plot specifications from key=value format."""
    default_params = PlotParams()

    # Test first plot spec
    spec1 = "layers=DEFAULT,x_min=0,x_max=100"
    params1 = _parse_plot_spec_kv(spec1, default_params)
    assert params1.plot_layers == PlotLayer.DEFAULT
    assert params1.x_min == 0
    assert params1.x_max == 100

    # Test second plot spec
    spec2 = "layers=ALL_COST,y_min=0,y_max=5"
    params2 = _parse_plot_spec_kv(spec2, default_params)
    assert params2.plot_layers == PlotLayer.ALL_COST
    assert params2.y_min == 0
    assert params2.y_max == 5


def test_parse_plot_spec_json_multi_plot():
    """Test parsing multiple plot specifications from JSON format."""
    default_params = PlotParams()

    # Test first JSON spec
    spec1 = '{"layers": "DEFAULT", "x_min": 0, "x_max": 100}'
    params1 = _parse_plot_spec_json(spec1, default_params)
    assert params1.plot_layers == PlotLayer.DEFAULT
    assert params1.x_min == 0
    assert params1.x_max == 100

    # Test second JSON spec
    spec2 = '{"layers": "ALL_COST", "y_min": 0, "y_max": 5}'
    params2 = _parse_plot_spec_json(spec2, default_params)
    assert params2.plot_layers == PlotLayer.ALL_COST
    assert params2.y_min == 0
    assert params2.y_max == 5


def test_args_to_params_with_plot_spec():
    """Test CLI argument parsing with --plot-spec flags."""

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
        plot_spec = [
            "layers=DEFAULT,x_min=0,x_max=100",
            "layers=ALL_COST,y_min=0,y_max=5",
        ]
        plot_spec_json = None
        plot_layers = None
        x_min = None
        x_max = None
        y_min = None
        y_max = None

    args = Args()
    load_params, transform_params, plot_params_list = _args_to_params(args)

    # Should have two plot configurations
    assert len(plot_params_list) == 2

    # First plot should have DEFAULT layers and x limits
    assert plot_params_list[0].plot_layers == PlotLayer.DEFAULT
    assert plot_params_list[0].x_min == 0
    assert plot_params_list[0].x_max == 100

    # Second plot should have ALL_COST layers and y limits
    assert plot_params_list[1].plot_layers == PlotLayer.ALL_COST
    assert plot_params_list[1].y_min == 0
    assert plot_params_list[1].y_max == 5


def test_args_to_params_with_plot_spec_json():
    """Test CLI argument parsing with --plot-spec-json flags."""

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
        plot_spec_json = [
            '{"layers": "DEFAULT", "x_min": 0, "x_max": 100}',
            '{"layers": "ALL_COST", "y_min": 0, "y_max": 5}',
        ]
        plot_layers = None
        x_min = None
        x_max = None
        y_min = None
        y_max = None

    args = Args()
    load_params, transform_params, plot_params_list = _args_to_params(args)

    # Should have two plot configurations
    assert len(plot_params_list) == 2

    # First plot should have DEFAULT layers and x limits
    assert plot_params_list[0].plot_layers == PlotLayer.DEFAULT
    assert plot_params_list[0].x_min == 0
    assert plot_params_list[0].x_max == 100

    # Second plot should have ALL_COST layers and y limits
    assert plot_params_list[1].plot_layers == PlotLayer.ALL_COST
    assert plot_params_list[1].y_min == 0
    assert plot_params_list[1].y_max == 5


def test_args_to_params_with_mixed_plot_specs():
    """Test CLI argument parsing with mixed --plot-spec and --plot-spec-json flags."""

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
        plot_spec = ["layers=DEFAULT,x_min=0,x_max=100"]
        plot_spec_json = ['{"layers": "ALL_COST", "y_min": 0, "y_max": 5}']
        plot_layers = None
        x_min = None
        x_max = None
        y_min = None
        y_max = None

    args = Args()
    load_params, transform_params, plot_params_list = _args_to_params(args)

    # Should have two plot configurations
    assert len(plot_params_list) == 2

    # First plot should have DEFAULT layers and x limits
    assert plot_params_list[0].plot_layers == PlotLayer.DEFAULT
    assert plot_params_list[0].x_min == 0
    assert plot_params_list[0].x_max == 100

    # Second plot should have ALL_COST layers and y limits
    assert plot_params_list[1].plot_layers == PlotLayer.ALL_COST
    assert plot_params_list[1].y_min == 0
    assert plot_params_list[1].y_max == 5


def test_deprecated_plot_layers_warning(caplog):
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

    # Should generate a warning about deprecated --plot-layers
    assert "deprecated" in caplog.text
    assert "--plot-layers is deprecated" in caplog.text


def test_deprecated_plot_layers_ignored_with_plot_spec(caplog):
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

    # Should generate a warning about --plot-layers being ignored
    assert "ignored" in caplog.text
    assert "--plot-layers is ignored" in caplog.text

    # Should use the plot-spec configuration, not the deprecated plot-layers
    assert len(plot_params_list) == 1
    assert plot_params_list[0].plot_layers == PlotLayer.ALL_COST


def test_top_level_axis_limits_ignored_with_plot_spec(caplog):
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

    # Should generate a warning about top-level axis limits being ignored
    assert "ignored" in caplog.text
    assert "Top-level x/y min/max are ignored" in caplog.text

    # Should use the plot-spec configuration without top-level axis limits
    assert len(plot_params_list) == 1
    assert plot_params_list[0].plot_layers == PlotLayer.DEFAULT
    # Top-level axis limits should not be applied to the plot params
    # The actual plot params should come from the plot-spec parsing


def test_single_plot_filename_format(monkeypatch, tmp_path: Path):
    """Test that single plot uses the correct filename format."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build inputs
    df_range, summary = _build_inputs(include_wls=True)

    scatter_cc = CallCounter()
    plot_cc = CallCounter()
    axvline_cc = CallCounter()
    legend_cc = CallCounter()

    monkeypatch.setattr(plt, "scatter", scatter_cc)
    monkeypatch.setattr(plt, "plot", plot_cc)
    monkeypatch.setattr(plt, "axvline", axvline_cc)
    monkeypatch.setattr(plt, "legend", legend_cc)

    # Create single plot configuration
    plot_params_list = [
        PlotParams(plot_layers=PlotLayer.DEFAULT),
    ]

    artifact_paths = render_plots(
        list_plot_params=plot_params_list,
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )

    # Should generate one plot file with single-plot filename format
    assert len(artifact_paths) == 1
    assert "plot-testhash-00-DEFAULT.svg" in artifact_paths


def test_multi_plot_filename_format(monkeypatch, tmp_path: Path):
    """Test that multiple plots use the correct filename format."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build inputs
    df_range, summary = _build_inputs(include_wls=True)

    scatter_cc = CallCounter()
    plot_cc = CallCounter()
    axvline_cc = CallCounter()
    legend_cc = CallCounter()

    monkeypatch.setattr(plt, "scatter", scatter_cc)
    monkeypatch.setattr(plt, "plot", plot_cc)
    monkeypatch.setattr(plt, "axvline", axvline_cc)
    monkeypatch.setattr(plt, "legend", legend_cc)

    # Create multiple plot configurations
    plot_params_list = [
        PlotParams(plot_layers=PlotLayer.DEFAULT),
        PlotParams(plot_layers=PlotLayer.ALL_COST),
        PlotParams(plot_layers=PlotLayer.ALL_WLS),
    ]

    artifact_paths = render_plots(
        list_plot_params=plot_params_list,
        df_included=df_range,
        summary=summary,
        short_hash="testhash",
    )

    # Should generate three plot files with multi-plot filename format
    assert len(artifact_paths) == 3
    assert "plot-testhash-00-DEFAULT.svg" in artifact_paths
    assert "plot-testhash-01-ALL_COST.svg" in artifact_paths
    assert "plot-testhash-02-ALL_WLS.svg" in artifact_paths
