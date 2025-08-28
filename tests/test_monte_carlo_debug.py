#!/usr/bin/env python3
"""Deterministic runner to debug monte carlo epsilon logic.

This loader avoids importing the package `src` using standard package semantics by
loading `src/monte_carlo.py` directly after injecting a fake `src.main` module into
sys.modules so the monte_carlo lazy imports succeed.
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

# --- Build a fake `src.main` module so run_monte_carlo lazily imports our deterministic stubs.
fm = types.ModuleType("src.main")
fm.MODEL_PRIORITY = ["robust_linear", "linear"]


class DeltaMode:
    MODEL_BASED = "MODEL_BASED"
    PREVIOUS_CHUNK = "PREVIOUS_CHUNK"


fm.DeltaMode = DeltaMode


def model_cost_column(token):
    return f"{token}_cost"


def model_output_column(token):
    return f"{token}_pred"


def model_sum_column(token):
    return f"{token}_sum"


fm.model_cost_column = model_cost_column
fm.model_output_column = model_output_column
fm.model_sum_column = model_sum_column


def synthesize_data(summary, transform_params, original_df, random_state=None):
    # deterministic per-sim seed derived from provided RNG
    sim_seed = (
        int(random_state.integers(0, 1_000_000)) if random_state is not None else 0
    )
    # minimal df_range carries the sim_seed so summarize_and_model can reconstruct deterministic costs
    df = pd.DataFrame({"sim_seed": [sim_seed]})
    synth_out = SimpleNamespace(df_range=df)
    synth_diag = {"sim_seed": sim_seed}
    return synth_out, synth_diag


fm.synthesize_data = synthesize_data


def summarize_and_model(df, params):
    # deterministic cost curve based on sim_seed
    sim_seed = int(df["sim_seed"].iloc[0]) if "sim_seed" in df.columns else 0
    rng = np.random.default_rng(sim_seed)
    sor = np.arange(1, 11)
    # convex-ish base with minimum near sor=7, plus tiny noise
    base = 50.0 - (sor - 7) ** 2 * 0.5
    noise = rng.normal(scale=0.1, size=sor.shape)
    costs = base + noise
    df_results = pd.DataFrame({"sor#": sor, "robust_linear_cost": costs})
    df_summary = pd.DataFrame([{"run_time_mean": float(np.mean(costs))}])
    sim_summary = SimpleNamespace(
        df_results=df_results, df_summary=df_summary, offline_cost=0.0
    )
    return sim_summary


fm.summarize_and_model = summarize_and_model


def regression_analysis(df, synth_fort):
    # not used for this test; return minimal expected structure
    df_results = pd.DataFrame({"sor#": np.arange(1, synth_fort + 1)})
    return df_results, {}


fm.regression_analysis = regression_analysis

# Inject fake module so monte_carlo lazy import in src/monte_carlo.py finds src.main
sys.modules["src.main"] = fm

# Load src/monte_carlo.py as module name "src.monte_carlo"
here = os.getcwd()
mc_path = os.path.join(here, "src", "monte_carlo.py")
spec = importlib.util.spec_from_file_location("src.monte_carlo", mc_path)
mc_mod = importlib.util.module_from_spec(spec)
sys.modules["src.monte_carlo"] = mc_mod
spec.loader.exec_module(mc_mod)

# Import the function under test from loaded module
run_monte_carlo = mc_mod.run_monte_carlo
MonteCarloParams = mc_mod.MonteCarloParams


def main():
    original_df = pd.DataFrame({"x": [1, 2, 3]})
    transform_params = SimpleNamespace(
        synthesize_fort=10,
        input_data_fort=10,
        synthesize_model="robust_linear",
        delta_mode=DeltaMode.PREVIOUS_CHUNK,
    )
    mc_params = MonteCarloParams(
        n_simulations=3, epsilon_fraction=0.01, random_seed=1234
    )
    rng = np.random.default_rng(42)
    mc_summary = run_monte_carlo(
        original_df,
        summary=None,
        transform_params=transform_params,
        mc_params=mc_params,
        random_state=rng,
    )

    print("Per-simulation summary (rows):")
    print(mc_summary.per_simulation_df)

    print("\nDebug per-sim cost arrays and epsilon masks:")
    per_sim = mc_summary.per_simulation_df
    for i, row in per_sim.iterrows():
        synth_diag = row["synth_diag"]
        sim_seed = (
            synth_diag.get("sim_seed", None) if isinstance(synth_diag, dict) else None
        )
        if sim_seed is None:
            print(f"sim {i}: no sim_seed in synth_diag")
            continue
        # regenerate costs via our deterministic summarize_and_model stub
        sim_summary = summarize_and_model(
            pd.DataFrame({"sim_seed": [sim_seed]}), transform_params
        )
        cost_col = model_cost_column("robust_linear")
        costs = pd.to_numeric(
            sim_summary.df_results[cost_col], errors="coerce"
        ).to_numpy()
        min_cost_auth = float(np.nanmin(costs[np.isfinite(costs)]))
        eps = float(mc_params.epsilon_fraction)
        threshold = min_cost_auth * (1.0 + eps)
        eps_mask = (costs <= threshold) & np.isfinite(costs)
        print(f"sim {i} seed={sim_seed}")
        print("sor#", sim_summary.df_results["sor#"].to_list())
        print("costs", np.round(costs, 4).tolist())
        print("min_cost_auth", min_cost_auth, "threshold", round(threshold, 4))
        print("eps_mask", eps_mask.tolist())
        print(
            "epsilon sors",
            sim_summary.df_results.loc[eps_mask, "sor#"].astype(int).tolist(),
        )
        print("---")


if __name__ == "__main__":
    main()
