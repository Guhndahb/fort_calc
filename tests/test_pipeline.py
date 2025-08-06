from pathlib import Path

import pandas as pd

from src.csv_processor import CSVRangeProcessor

# Import from the refactored module
from src.main import (
    DeltaMode,
    LoadSliceParams,
    SummaryModelOutputs,
    TransformOutputs,
    TransformParams,
    load_and_slice_csv,
    summarize_and_model,
    transform_pipeline,
)


def make_temp_csv(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


class DummyCSVRangeProcessor(CSVRangeProcessor):
    """Subclass to expose read_range behavior with a provided file; used implicitly."""


def test_load_and_slice_csv_reads_given_range(tmp_path: Path, monkeypatch):
    # Prepare CSV with 10 rows
    rows = [
        {
            "timestamp": 20250101000000 + i,
            "ignore": "FALSE",
            "sor#": i + 1,
            "runticks": 1000 + i * 10,
            "resetticks": 0,
            "notes": "" if i != 0 else "",
        }
        for i in range(10)
    ]
    csv_path = make_temp_csv(tmp_path, rows)

    params = LoadSliceParams(
        log_path=csv_path, start_line=1, end_line=5, include_header=True
    )
    df = load_and_slice_csv(params)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_transform_pipeline_filters_and_returns_contract():
    # Construct minimal DataFrame pre-CSV to avoid IO in unit test
    df = pd.DataFrame(
        {
            "timestamp": [
                20250101000001,
                20250101000002,
                20250101000003,
                20250101000004,
                20250101000005,
                20250101000006,
            ],
            "ignore": ["FALSE", "FALSE", "FALSE", "FALSE", "FALSE", "FALSE"],
            "sor#": [1, 2, 3, 4, 5, 6],
            "runticks": [1000, 1010, 1020, 1030, 1040, 1050],
            "resetticks": [0, 0, 0, 0, 0, 0],
            "notes": ["", "", "", "", "", ""],
        }
    )

    params = TransformParams(
        zscore_min=-2.0,
        zscore_max=2.0,
        input_data_fort=6,
        ignore_resetticks=True,
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
        exclude_timestamp_ranges=None,
        verbose_filtering=False,
    )

    out: TransformOutputs = transform_pipeline(df, params)
    assert isinstance(out, TransformOutputs)
    assert isinstance(out.df_range, pd.DataFrame)
    assert isinstance(out.df_excluded, pd.DataFrame)
    # After compute, adjusted_run_time should exist
    assert "adjusted_run_time" in out.df_range.columns
    # At least 5 rows should remain per guardrail in implementation
    assert len(out.df_range) >= 5


def test_summarize_and_model_contract():
    # Build a DataFrame with numeric adjusted_run_time and sor#
    # Contract change: require at least one row where sor# == input_data_fort
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5, 6, 7, 8],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.05, 1.0, 0.95, 0.9, 0.88],
        }
    )

    params = TransformParams(
        zscore_min=-2.0,
        zscore_max=2.0,
        input_data_fort=8,  # include offline cost row computed from summary
        ignore_resetticks=True,
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
        exclude_timestamp_ranges=None,
        verbose_filtering=False,
    )

    # Suppress statsmodels/scipy normality warnings due to small n in unit tests
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="`kurtosistest` p-value may be inaccurate",
        category=UserWarning,
    )
    summary: SummaryModelOutputs = summarize_and_model(df, params)
    assert isinstance(summary, SummaryModelOutputs)
    assert {
        "df_summary",
        "df_results",
        "regression_diagnostics",
        "offline_cost",
        "sor_min_cost_lin",
        "sor_min_cost_quad",
    }.issubset(set(summary.__dict__.keys()))

    # df_results expected columns
    for col in [
        "sor#",
        "linear_model_output",
        "quadratic_model_output",
        "sum_lin",
        "sum_quad",
        "cost_per_run_at_fort_lin",
        "cost_per_run_at_fort_quad",
    ]:
        assert col in summary.df_results.columns

    # offline_cost should be a float
    assert isinstance(summary.offline_cost, float)


# moved to tests/test_plot_layers.py: test_render_outputs_writes_svg


def test_regression_analysis_does_not_mutate_df_range_columns():
    # Prepare input DataFrame
    # Contract change: include a row with sor# == input_data_fort
    df = pd.DataFrame(
        {
            "sor#": [1, 2, 3, 4, 5, 6, 7, 8],
            "adjusted_run_time": [1.0, 1.1, 1.2, 1.05, 1.0, 0.95, 0.9, 0.88],
        }
    )
    # Capture original columns (order and set)
    original_cols = df.columns.tolist()

    # Call summarize_and_model which internally calls regression_analysis
    params = TransformParams(
        zscore_min=-2.0,
        zscore_max=2.0,
        input_data_fort=8,
        ignore_resetticks=True,
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
        exclude_timestamp_ranges=None,
        verbose_filtering=False,
    )
    # Suppress statsmodels/scipy normality warnings due to small n in unit tests
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="`kurtosistest` p-value may be inaccurate",
        category=UserWarning,
    )
    _ = summarize_and_model(df, params)

    # Assert columns unchanged in both content and order
    assert df.columns.tolist() == original_cols
