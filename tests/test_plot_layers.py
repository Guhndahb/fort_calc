import types
from pathlib import Path

import pandas as pd

from src.main import PlotLayer, SummaryModelOutputs, render_outputs


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
    render_outputs(
        df_range, summary, output_svg=str(out_path), plot_layers=PlotLayer.DEFAULT
    )

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

    # - file saved
    assert out_path.exists()


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
    render_outputs(
        df_range, summary, output_svg=str(out_path), plot_layers=PlotLayer.EVERYTHING
    )

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

    # file saved
    assert out_path.exists()


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
    render_outputs(df_range, summary, output_svg=str(out_path), plot_layers=flags)

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

    # file saved
    assert out_path.exists()
