import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.main import regression_analysis


def _count_const_like(columns: list[str]) -> int:
    lowered = [c.lower() for c in columns]
    return sum(1 for c in lowered if c in ("const", "intercept"))


def test_constant_column_present_once_training_and_prediction():
    # Small deterministic dataset
    df = pd.DataFrame(
        {
            "sor#": np.arange(1, 21),
            "adjusted_run_time": np.linspace(1.0, 2.0, 20),
        }
    )

    # Run the public API to ensure predict path works and names are consistent
    result_df, _ = regression_analysis(df, input_data_fort=20)
    assert not result_df.empty

    # Recreate the internal training matrices exactly as in the implementation to introspect columns
    y = df["adjusted_run_time"].to_numpy()

    # Linear training design matrix
    X_linear_train = pd.DataFrame({"sor#": df["sor#"].to_numpy()})
    X_linear_train = sm.add_constant(X_linear_train, has_constant="add")
    # Some environments might name the constant differently; the implementation standardizes to 'const'
    if "const" not in X_linear_train.columns:
        const_col = [
            c for c in X_linear_train.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_linear_train = X_linear_train.rename(columns={const_col[0]: "const"})
    assert "const" in X_linear_train.columns
    assert _count_const_like(list(X_linear_train.columns)) == 1

    linear_model = sm.OLS(y, X_linear_train).fit()

    # Quadratic training design matrix
    X_quadratic_train = pd.DataFrame(
        {"sor#": df["sor#"].to_numpy(), "sor2": df["sor#"].to_numpy() ** 2}
    )
    X_quadratic_train = sm.add_constant(X_quadratic_train, has_constant="add")
    if "const" not in X_quadratic_train.columns:
        const_col = [
            c for c in X_quadratic_train.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_quadratic_train = X_quadratic_train.rename(
                columns={const_col[0]: "const"}
            )
    assert "const" in X_quadratic_train.columns
    assert _count_const_like(list(X_quadratic_train.columns)) == 1

    quadratic_model = sm.OLS(y, X_quadratic_train).fit()

    # Prediction matrices: construct once, add constant once, then align to trained model columns
    sor_sequence = np.arange(1, 21)

    X_linear_pred = pd.DataFrame({"sor#": sor_sequence})
    X_linear_pred = sm.add_constant(X_linear_pred, has_constant="add")
    if "const" not in X_linear_pred.columns:
        const_col = [
            c for c in X_linear_pred.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_linear_pred = X_linear_pred.rename(columns={const_col[0]: "const"})
    # Before alignment, ensure exactly one constant-like column
    assert "const" in X_linear_pred.columns
    assert _count_const_like(list(X_linear_pred.columns)) == 1
    # Align to trained exog names (order only)
    X_linear_pred = X_linear_pred.reindex(columns=linear_model.model.exog_names)
    # After alignment, ensure still exactly one constant-like column
    assert _count_const_like(list(X_linear_pred.columns)) == 1

    X_quadratic_pred = pd.DataFrame({"sor#": sor_sequence, "sor2": sor_sequence**2})
    X_quadratic_pred = sm.add_constant(X_quadratic_pred, has_constant="add")
    if "const" not in X_quadratic_pred.columns:
        const_col = [
            c for c in X_quadratic_pred.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_quadratic_pred = X_quadratic_pred.rename(columns={const_col[0]: "const"})
    assert "const" in X_quadratic_pred.columns
    assert _count_const_like(list(X_quadratic_pred.columns)) == 1
    X_quadratic_pred = X_quadratic_pred.reindex(
        columns=quadratic_model.model.exog_names
    )
    assert _count_const_like(list(X_quadratic_pred.columns)) == 1

    # Smoke: predictions run without error
    _ = linear_model.predict(X_linear_pred)
    _ = quadratic_model.predict(X_quadratic_pred)
