#!/usr/bin/env python3
"""
Generate diagnostic plots for representative Monte Carlo simulations.

Usage:
  python tools/plot_mc_representative.py --run-dir output/20250827T221231

The script:
 - Loads MC artifacts (mc-per-sim CSV, debug CSV, mc-summary JSON, range CSV)
 - Reconstructs the per-simulation RNG stream (using seed from mc-summary)
 - Re-synthesizes each selected simulation deterministically via synthesize_data(...)
 - Re-runs summarize_and_model(...) on the synthesized frame to obtain full per-SOR cost curves
 - Plots cost/run vs SOR for each representative sim with:
     * vertical line at recommended_fort
     * horizontal threshold (final epsilon_used at min cost)
     * marker if epsilon_fallback_to_argmin was applied
 - Writes SVG and PNG outputs into the same run folder:
     plot-sim-{sim_id}-{short}.svg/png
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg", force=True)
# Ensure repository root is on sys.path so package-style imports work when script is run from /tools
import ast
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Attempt imports similar to runtime (package-first, then fallback to script-style)
try:
    from src.main import (  # type: ignore
        model_cost_column,
        summarize_and_model,
        synthesize_data,
    )
except Exception:
    try:
        from main import (  # type: ignore
            model_cost_column,
            summarize_and_model,
            synthesize_data,
        )
    except Exception:
        # Re-raise original error for visibility
        raise


def choose_representative_sims(
    per_sim_df: pd.DataFrame, debug_df: pd.DataFrame, n: int = 6
):
    # Prefer mixing fallback True and False sims
    try:
        with_fallback = per_sim_df.loc[per_sim_df["epsilon_fallback_to_argmin"] == True]
        without = per_sim_df.loc[per_sim_df["epsilon_fallback_to_argmin"] == False]
    except Exception:
        # Fallback to simple selection
        with_fallback = pd.DataFrame(columns=per_sim_df.columns)
        without = per_sim_df
    reps = []
    k1 = min(len(with_fallback), n // 2)
    k2 = n - k1
    reps.extend(list(with_fallback["sim_id"].astype(int).tolist()[:k1]))
    remaining = [
        int(x) for x in without["sim_id"].astype(int).tolist() if int(x) not in reps
    ]
    reps.extend(remaining[:k2])
    # If still short, fill from available sims
    if len(reps) < n:
        all_ids = [int(x) for x in per_sim_df["sim_id"].astype(int).tolist()]
        for sid in all_ids:
            if sid not in reps:
                reps.append(sid)
            if len(reps) >= n:
                break
    return list(reps[:n])


def plot_sim(sim_id: int, sim_summary, cost_col: str, debug_row: dict, out_dir: Path):
    df = pd.DataFrame(sim_summary.df_results)
    if cost_col not in df.columns:
        print(f"cost column {cost_col} not found in sim {sim_id} results; skipping")
        return
    sor = pd.to_numeric(df["sor#"], errors="coerce").to_numpy(dtype=float)
    costs = pd.to_numeric(df[cost_col], errors="coerce").to_numpy(dtype=float)

    recommended = int(debug_row.get("recommended_fort", int(sor[np.nanargmin(costs)])))
    epsilon_used = float(debug_row.get("epsilon_used", np.nan))
    min_cost_auth = float(debug_row.get("min_cost_auth", np.nan))
    threshold = float(debug_row.get("threshold", np.nan))
    fallback = bool(debug_row.get("epsilon_fallback_to_argmin", False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sor, costs, marker="o", linestyle="-", label="cost/run", color="#1f77b4")
    ax.axvline(
        recommended,
        color="green",
        linestyle="--",
        linewidth=1.8,
        label=f"recommended SOR {recommended}",
    )
    # threshold line
    if np.isfinite(threshold):
        ax.axhline(
            threshold,
            color="red",
            linestyle=":",
            linewidth=1.6,
            label=f"threshold ({epsilon_used:.2e})",
        )
    # mark argmin point
    try:
        idx = int(
            pd.to_numeric(df["sor#"], errors="coerce")
            .to_numpy()
            .tolist()
            .index(recommended)
        )
        ax.scatter(
            [recommended],
            [costs[idx]],
            color="orange",
            s=80,
            zorder=5,
            label="argmin point",
        )
    except Exception:
        pass

    title = f"Sim {sim_id} — recommended={recommended}  epsilon_used={epsilon_used:.2e}  fallback={fallback}"
    ax.set_title(title)
    ax.set_xlabel("SOR #")
    ax.set_ylabel("Cost / run (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")

    svg_path = out_dir / f"plot-sim-{sim_id}.svg"
    png_path = out_dir / f"plot-sim-{sim_id}.png"
    fig.savefig(str(svg_path), format="svg")
    fig.savefig(str(png_path), format="png", dpi=150)
    plt.close(fig)
    print(f"Wrote {svg_path} and {png_path}")


def main():
    p = argparse.ArgumentParser(description="Plot MC representative sims")
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run output directory (e.g., output/20250827T221231)",
    )
    p.add_argument(
        "--n", type=int, default=6, help="Number of representative sims to plot"
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    # Read artifacts (robust globbing to tolerate optional prefixes)
    per_sim_candidates = list(run_dir.glob("mc-per-sim-*.csv"))
    per_sim_path = per_sim_candidates[0] if per_sim_candidates else None
    debug_candidates = list(run_dir.glob("debug-mc-*.csv"))
    debug_path = debug_candidates[0] if debug_candidates else None
    summary_candidates = list(run_dir.glob("mc-summary-*.json"))
    summary_path = summary_candidates[0] if summary_candidates else None
    range_candidates = list(run_dir.glob("range-*.csv"))
    range_path = range_candidates[0] if range_candidates else None

    if not (per_sim_path and per_sim_path.exists()):
        raise FileNotFoundError("mc-per-sim CSV not found in run dir")
    if not (debug_path and debug_path.exists()):
        raise FileNotFoundError("debug-mc CSV not found in run dir")
    if not (summary_path and summary_path.exists()):
        raise FileNotFoundError("mc-summary JSON not found in run dir")
    if not (range_path and range_path.exists()):
        raise FileNotFoundError("range CSV not found in run dir")

    per_sim_df = pd.read_csv(per_sim_path)
    debug_df = pd.read_csv(debug_path)
    summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
    seed_used = summary_json.get("diagnostics", {}).get("seed_used", None)
    if seed_used is None:
        raise ValueError("seed_used not found in mc-summary diagnostics")

    # Load original range (used as the 'original_df' when synth) — contains sor# and adjusted_run_time at least
    original_df = pd.read_csv(range_path)

    # Build a baseline summary and params to pass into synthesize_data
    # Use get_default_monte_carlo_params for defaults; synthesize_data only needs summary.df_results presence
    # We'll call summarize_and_model(original_df, trans_params) to obtain summary used for synthesis
    try:
        # derive a basic TransformParams approximation by calling main.get_default_params if available
        from src.main import (  # type: ignore
            DeltaMode,
            get_default_params,
        )

        _, d_trans, _ = get_default_params()
        trans_params = d_trans
    except Exception:
        # fallback: construct a minimal SimpleNamespace with expected attributes
        trans_params = SimpleNamespace(
            zscore_min=-1.75,
            zscore_max=2.5,
            input_data_fort=int(original_df["sor#"].max())
            if "sor#" in original_df.columns
            else 150,
            ignore_resetticks=True,
            delta_mode=DeltaMode.MODEL_BASED
            if hasattr(DeltaMode, "MODEL_BASED")
            else None,
            exclude_timestamp_ranges=None,
            verbose_filtering=False,
            fail_on_any_invalid_timestamps=True,
            iqr_k_low=1.0,
            iqr_k_high=2.0,
            use_iqr_filtering=True,
            offline_cost_override=None,
            simulated_fort=None,
            synthesize_model=None,
            synthesize_fort=None,
        )

    # Try to infer a synthesize_model token from per-sim diagnostics or mc-summary.
    inferred_synth_model = None
    try:
        # Prefer extracting from per_sim_df.synth_diag if available
        if (
            "synth_diag" in per_sim_df.columns
            and not per_sim_df["synth_diag"].isna().all()
        ):
            # pick first non-null synth_diag value
            for raw in per_sim_df["synth_diag"].tolist():
                if pd.isna(raw):
                    continue
                diag_obj = {}
                if isinstance(raw, str):
                    # try JSON, then ast.literal_eval
                    try:
                        diag_obj = json.loads(raw)
                    except Exception:
                        try:
                            diag_obj = ast.literal_eval(raw)
                        except Exception:
                            diag_obj = {}
                elif isinstance(raw, dict):
                    diag_obj = raw
                if isinstance(diag_obj, dict):
                    inferred = diag_obj.get("synthesized", {}).get("model_used")
                    if inferred:
                        inferred_synth_model = inferred
                        break
    except Exception:
        inferred_synth_model = None

    # Fallback: check mc-summary diagnostics for a recorded synth token
    if not inferred_synth_model:
        try:
            inferred_synth_model = summary_json.get("diagnostics", {}).get(
                "synthesize_model"
            )
        except Exception:
            inferred_synth_model = None

    # Final fallback to a sensible default
    if not inferred_synth_model:
        inferred_synth_model = "robust_linear"

    # Ensure trans_params.synthesize_model is set for synthesize_data calls
    try:
        trans_params.synthesize_model = inferred_synth_model
    except Exception:
        setattr(trans_params, "synthesize_model", inferred_synth_model)

    # Build the canonical summary used as the "summary" argument to synthesize_data
    base_summary = summarize_and_model(original_df, trans_params)

    # Decide representative sims
    rep_ids = choose_representative_sims(per_sim_df, debug_df, n=args.n)
    print("Selected representative sim ids:", rep_ids)

    # Recreate the rng_master sequence up to max selected sim id
    max_id = int(max(rep_ids))
    rng_master = np.random.default_rng(int(seed_used))
    sim_seeds = [int(rng_master.integers(0, 2**31 - 1)) for _ in range(max_id + 1)]

    # Build output directory
    out_dir = run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in rep_ids:
        # find debug row for this sim
        dbg_row = debug_df.loc[debug_df["sim_id"] == int(sid)]
        if dbg_row.empty:
            print(f"No debug row for sim {sid}; skipping")
            continue
        dbg = dbg_row.iloc[0].to_dict()
        sim_seed = int(sim_seeds[int(sid)])
        sim_rng = np.random.default_rng(sim_seed)

        # Re-synthesize deterministically
        try:
            synth_out, synth_diag = synthesize_data(
                base_summary, trans_params, original_df, random_state=sim_rng
            )
        except Exception as e:
            print(f"Failed to synthesize sim {sid}: {e}")
            continue

        # Re-run summarize/model on synthesised frame (force PREVIOUS_CHUNK baseline as in run_monte_carlo)
        try:
            from dataclasses import replace as _replace

            params_for_synth = _replace(
                trans_params,
                input_data_fort=int(
                    synth_diag.get("synthesized", {}).get(
                        "fort", trans_params.input_data_fort
                    )
                ),
                delta_mode=DeltaMode.PREVIOUS_CHUNK,
            )
        except Exception:
            params_for_synth = trans_params

        try:
            sim_summary = summarize_and_model(synth_out.df_range, params_for_synth)
        except Exception as e:
            print(f"Failed to summarize sim {sid}: {e}")
            continue

        # Determine authoritative token from debug row (preferred) or fall back to synth_model
        auth_token = (
            dbg.get("authoritative_token")
            or getattr(trans_params, "synthesize_model", None)
            or "robust_linear"
        )
        cost_col = model_cost_column(auth_token)

        # Plot
        plot_sim(int(sid), sim_summary, cost_col, dbg, out_dir)

    print("Done plotting representative sims.")


if __name__ == "__main__":
    main()
