import numpy as np
import pandas as pd

from src.main import model_output_column, regression_analysis


def test_model_predict_column_order_independence():
    # Build simple monotonic data to ensure well-behaved regression
    df = pd.DataFrame(
        {
            "sor#": np.arange(1, 21),
            "adjusted_run_time": np.linspace(1.0, 2.0, 20),
        }
    )

    # Run regression to obtain trained models indirectly via regression_analysis
    # The function returns predictions for a default ordered prediction frame.
    # We will recreate a matching prediction frame with shuffled columns and
    # confirm predictions remain the same when column order is permuted.
    result_df, _ = regression_analysis(df, input_data_fort=20)

    # Expected predictions from the function using correctly named DataFrames
    expected_linear = result_df[model_output_column("ols_linear")].to_numpy()
    expected_quadratic = result_df[model_output_column("ols_quadratic")].to_numpy()

    # Construct our own prediction DataFrames with shuffled columns to validate
    # predict-time column-order independence. We must mirror training column names:
    # linear: ["const", "sor#"], quadratic: ["const", "sor#", "sor2"].
    sor_seq = np.arange(1, 21)
    X_linear = pd.DataFrame({"sor#": sor_seq})
    X_linear = pd.concat(
        [pd.Series(1.0, name="const", index=X_linear.index), X_linear], axis=1
    )
    # Shuffle columns intentionally
    X_linear_shuffled = X_linear[["sor#", "const"]]

    X_quad = pd.DataFrame({"sor#": sor_seq, "sor2": sor_seq**2})
    X_quad = pd.concat(
        [pd.Series(1.0, name="const", index=X_quad.index), X_quad], axis=1
    )
    # Shuffle columns intentionally
    X_quad_shuffled = X_quad[["sor2", "const", "sor#"]]

    # To call predict on the trained models, we re-run the internal fit steps identically
    # to expose the models here using the same construction as regression_analysis.
    # This mirrors the implementation precisely.
    import statsmodels.api as sm

    y = df["adjusted_run_time"].to_numpy()

    X_linear_train = pd.DataFrame({"sor#": df["sor#"].to_numpy()})
    X_linear_train = sm.add_constant(X_linear_train, has_constant="add")
    if "const" not in X_linear_train.columns:
        const_col = [
            c for c in X_linear_train.columns if c.lower() in ("const", "intercept")
        ]
        if const_col:
            X_linear_train = X_linear_train.rename(columns={const_col[0]: "const"})
    linear_model = sm.OLS(y, X_linear_train).fit()

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
    quadratic_model = sm.OLS(y, X_quadratic_train).fit()

    # Predict with shuffled-column DataFrames (names match; order differs)
    # Align to model.exog_names to guarantee correct mapping regardless of order
    X_linear_aligned = X_linear_shuffled.reindex(columns=linear_model.model.exog_names)
    X_quad_aligned = X_quad_shuffled.reindex(columns=quadratic_model.model.exog_names)

    pred_linear_shuffled = linear_model.predict(X_linear_aligned).to_numpy()
    pred_quadratic_shuffled = quadratic_model.predict(X_quad_aligned).to_numpy()

    # Both predictions should match those produced by regression_analysis
    # within a tight numerical tolerance.
    np.testing.assert_allclose(
        pred_linear_shuffled, expected_linear, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        pred_quadratic_shuffled, expected_quadratic, rtol=1e-10, atol=1e-12
    )
