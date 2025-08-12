from pathlib import Path

import pandas as pd
import pytest

from src.main import LoadSliceParams, _args_to_params, load_and_slice_csv


def test_col_index_mapping_by_index(tmp_path: Path):
    # Build DataFrame with specified columns
    df = pd.DataFrame(
        {
            "ts": [
                20250101000001,
                20250101000002,
                20250101000003,
                20250101000004,
                20250101000005,
            ],
            "ignore": ["FALSE"] * 5,
            "actual_sor": [1, 2, 3, 4, 5],
            "actual_ticks": [1000, 1010, 1020, 1030, 1040],
            "notes": ["" for _ in range(5)],
        }
    )

    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    params = LoadSliceParams(
        log_path=csv_path,
        start_line=1,
        end_line=None,
        include_header=True,
        header_map={},
        col_sor=2,
        col_ticks=3,
    )

    df_loaded = load_and_slice_csv(params)

    # Assert canonical names present
    assert "sor#" in df_loaded.columns
    assert "runticks" in df_loaded.columns

    # Values should match the original actual_sor and actual_ticks
    assert list(df_loaded["sor#"].astype(int)) == [1, 2, 3, 4, 5]
    assert list(df_loaded["runticks"].astype(int)) == [1000, 1010, 1020, 1030, 1040]


def test_col_index_conflict_existing_name(tmp_path: Path):
    # CSV where canonical name 'sor#' already exists at index 0
    df = pd.DataFrame({"sor#": [1, 2], "a": [10, 20], "b": [100, 200]})
    csv_path = tmp_path / "conflict.csv"
    df.to_csv(csv_path, index=False)

    params = LoadSliceParams(
        log_path=csv_path,
        start_line=1,
        end_line=None,
        include_header=True,
        header_map={},
        col_sor=1,  # request index 1 while 'sor#' is at index 0 -> conflict
        col_ticks=None,
    )

    with pytest.raises(ValueError) as excinfo:
        load_and_slice_csv(params)

    msg = str(excinfo.value)
    assert "sor#" in msg
    assert "does not match requested" in msg


def test_col_index_out_of_range(tmp_path: Path):
    # Small CSV with 3 columns
    df = pd.DataFrame({"c0": [1], "c1": [2], "c2": [3]})
    csv_path = tmp_path / "small.csv"
    df.to_csv(csv_path, index=False)

    params = LoadSliceParams(
        log_path=csv_path,
        start_line=1,
        end_line=None,
        include_header=True,
        header_map={},
        col_sor=None,
        col_ticks=10,  # out-of-range
    )

    with pytest.raises(ValueError) as excinfo:
        load_and_slice_csv(params)

    assert "out of range" in str(excinfo.value)


def test_args_to_params_rejects_negative_and_equal_indices():
    class Args:
        pass

    # Base minimal attributes required by _args_to_params
    base_attrs = {
        "log_path": "test.csv",
        "start_line": None,
        "end_line": None,
        "header_map": None,
        "delta_mode": None,
        "exclude_range": None,
        "ignore_resetticks": None,
        "fail_on_any_invalid_timestamps": None,
        "zscore_min": None,
        "zscore_max": None,
        "input_data_fort": None,
        "verbose_filtering": None,
        "iqr_k_low": None,
        "iqr_k_high": None,
        "use_iqr_filtering": None,
        "plot_spec": None,
        "plot_spec_json": None,
        "plot_layers": None,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "no_header": None,
    }

    # Test negative index for col_sor
    args1 = Args()
    for k, v in base_attrs.items():
        setattr(args1, k, v)
    args1.col_sor = -1
    args1.col_ticks = None

    with pytest.raises(ValueError) as excinfo_neg:
        _args_to_params(args1)
    assert "non-negative" in str(excinfo_neg.value) or "must be a non-negative" in str(
        excinfo_neg.value
    )

    # Test identical indices for col_sor and col_ticks
    args2 = Args()
    for k, v in base_attrs.items():
        setattr(args2, k, v)
    args2.col_sor = 1
    args2.col_ticks = 1

    with pytest.raises(ValueError) as excinfo_eq:
        _args_to_params(args2)
    assert "cannot be the same" in str(
        excinfo_eq.value
    ) or "cannot be the same index" in str(excinfo_eq.value)
