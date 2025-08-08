from pathlib import Path

import pandas as pd
import pytest

from src.main import (
    DeltaMode,
    LoadSliceParams,
    TransformParams,
    _args_to_params,
    _build_cli_parser,
    load_and_slice_csv,
    transform_pipeline,
)


def make_temp_csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_case_insensitive_mapping_and_pipeline(tmp_path: Path):
    df = pd.DataFrame(
        {
            "timestamp": [
                20250101000001,
                20250101000002,
                20250101000003,
                20250101000004,
                20250101000005,
            ],
            "ignore": ["FALSE"] * 5,
            "SorNum": [1, 2, 3, 4, 5],
            "RunTicks": [1000, 1010, 1020, 1030, 1040],
            "resetticks": [0] * 5,
            "notes": [""] * 5,
        }
    )
    csv_path = make_temp_csv(tmp_path, df)

    load = LoadSliceParams(
        log_path=csv_path,
        start_line=1,
        end_line=None,
        include_header=True,
        header_map={"sornum": "sor#", "runticks": "runticks"},
    )

    df_loaded = load_and_slice_csv(load)
    assert "sor#" in df_loaded.columns
    assert "runticks" in df_loaded.columns

    params = TransformParams(
        zscore_min=-2.0,
        zscore_max=2.0,
        input_data_fort=5,
        ignore_resetticks=True,
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
        exclude_timestamp_ranges=None,
        verbose_filtering=False,
    )

    out = transform_pipeline(df_loaded, params)
    assert "adjusted_run_time" in out.df_range.columns


def test_conflicting_provided_targets_raises(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = make_temp_csv(tmp_path, df)

    load = LoadSliceParams(
        log_path=csv_path,
        start_line=1,
        end_line=None,
        include_header=True,
        header_map={"a": "x", "b": "x"},
    )
    with pytest.raises(ValueError):
        load_and_slice_csv(load)


def test_rename_causes_duplicate_with_existing_column_raises(tmp_path: Path):
    df = pd.DataFrame({"existing": [1, 2, 3], "old1": [4, 5, 6]})
    csv_path = make_temp_csv(tmp_path, df)

    load = LoadSliceParams(
        log_path=csv_path,
        start_line=1,
        end_line=None,
        include_header=True,
        header_map={"old1": "existing"},
    )
    with pytest.raises(ValueError):
        load_and_slice_csv(load)


def test_cli_parsing_builds_header_map(tmp_path: Path):
    df = pd.DataFrame(
        {
            "timestamp": [1],
            "ignore": ["FALSE"],
            "sor#": [1],
            "runticks": [1000],
            "resetticks": [0],
            "notes": [""],
        }
    )
    csv_path = make_temp_csv(tmp_path, df)

    parser = _build_cli_parser()
    args = parser.parse_args(
        [
            "--log-path",
            str(csv_path),
            "--header-map",
            "Foo:bar",
            "--header-map",
            "Baz:qux",
        ]
    )
    load, _, _ = _args_to_params(args)
    assert load.header_map == {"Foo": "bar", "Baz": "qux"}
