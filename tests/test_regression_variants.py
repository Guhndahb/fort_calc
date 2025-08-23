import numpy as np
import pandas as pd

from src.main import model_output_column, regression_analysis


def _make_monotone_df(n=30, noise=0.0, hetero=False, seed=123):
    rng = np.random.default_rng(seed)
    sor = np.arange(1, n + 1, dtype=float)
    base = 1.0 + 0.01 * sor  # mild upward trend to keep things well-conditioned
    if hetero:
        eps = rng.normal(loc=0.0, scale=0.01 * sor, size=n)  # variance grows with sor#
    else:
        eps = rng.normal(loc=0.0, scale=noise, size=n)
    y = base + eps
    return pd.DataFrame({"sor#": sor.astype(int), "adjusted_run_time": y})


def test_diagnostics_include_all_variants_linear_and_quadratic():
    # Smooth deterministic data with small noise ensures stable fitting
    df = _make_monotone_df(n=40, noise=0.02, hetero=False, seed=1)

    # input_data_fort is used to build prediction sequence 1..fort
    # and is independent of the training subset length here (we train on entire df)
    result_df, diagnostics = regression_analysis(df, input_data_fort=40)

    # Sanity on result_df schema invariants (unchanged pipeline contract)
    expected_cols = {
        "sor#",
        model_output_column("ols_linear"),
        model_output_column("ols_quadratic"),
    }
    assert expected_cols.issubset(set(result_df.columns))
    assert len(result_df) == 40

    # Diagnostics must include the new variants for both linear and quadratic
    for form in ("linear", "quadratic"):
        assert form in diagnostics, f"Missing top-level diagnostics key: {form}"
        form_diag = diagnostics[form]
        for variant in ("ols", "ols_hc1", "wls", "wls_hc1"):
            assert variant in form_diag or f"{variant}_error" in form_diag, (
                f"Expected diagnostics to include '{variant}' (or '{variant}_error') "
                f"for model form '{form}'"
            )
        # If WLS present, expect weights_spec annotation
        if "wls" in form_diag:
            assert "weights_spec" in form_diag["wls"]
        if "wls_hc1" in form_diag:
            assert "weights_spec" in form_diag["wls_hc1"]

    # New model diagnostics (isotonic, pchip, robust_linear) should be present at top-level
    for m in ("isotonic", "pchip", "robust_linear"):
        assert m in diagnostics, f"Missing diagnostics for new model: {m}"
        # Each model diagnostics must include a fit_message token per contract
        assert "fit_message" in diagnostics[m], f"No fit_message for {m}"


def test_heteroskedastic_behavior_robust_se_and_wls_smoke():
    # Construct heteroskedastic data where variance increases with sor#
    df = _make_monotone_df(n=60, hetero=True, seed=42)

    _, diagnostics = regression_analysis(df, input_data_fort=60)

    # For at least one model form, robust SEs should typically differ from classical SEs
    # under heteroskedasticity. We use a lenient check: any coefficient SE differs.
    def any_se_diff(form_diag: dict) -> bool:
        if "ols" not in form_diag or "ols_hc1" not in form_diag:
            return False
        se_ols = form_diag["ols"].get("Standard Errors")
        se_hc1 = form_diag["ols_hc1"].get("Standard Errors")
        if se_ols is None or se_hc1 is None:
            return False
        try:
            # Align indices robustly
            idx = se_ols.index.intersection(se_hc1.index)
            if len(idx) == 0:
                return False
            diffs = (se_ols.loc[idx] - se_hc1.loc[idx]).abs()
            return bool((diffs > 1e-12).any())
        except Exception:
            return False

    # At least one of linear/quadratic should show robust-vs-classical SE differences
    linear_diff = any_se_diff(diagnostics.get("linear", {}))
    quadratic_diff = any_se_diff(diagnostics.get("quadratic", {}))
    assert linear_diff or quadratic_diff, (
        "Expected robust SEs to differ from classical SEs on heteroskedastic data for at least one model form"
    )

    # WLS and WLS+HC1 should have been fit successfully (or provide an error entry).
    for form in ("linear", "quadratic"):
        form_diag = diagnostics.get(form, {})
        # Accept either fitted diagnostics or explicit error capture
        assert "wls" in form_diag or "wls_error" in form_diag, f"WLS missing for {form}"
        assert "wls_hc1" in form_diag or "wls_hc1_error" in form_diag, (
            f"WLS+HC1 missing for {form}"
        )
