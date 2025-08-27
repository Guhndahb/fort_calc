from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MonteCarloParams:
    n_simulations: int = 1000
    epsilon_fraction: float = 0.005
    random_seed: Optional[int] = None
    max_attempts: Optional[int] = None
    heteroskedastic_resampling: bool = False
    selection_policy: str = "priority"  # "priority" or "synthesize"
    output_prefix: Optional[str] = None


@dataclass
class MonteCarloSummary:
    pmf: Dict[int, float] = field(default_factory=dict)
    mode: Optional[int] = None
    median: Optional[float] = None
    central_50_interval: Optional[Tuple[int, int]] = None
    entropy_nats: Optional[float] = None
    iqr: Optional[float] = None
    epsilon_freq: Dict[int, float] = field(default_factory=dict)
    robust_choice: Optional[int] = None
    expected_regret_mean: Optional[float] = None
    expected_regret_p95: Optional[float] = None
    per_simulation_df: Optional[pd.DataFrame] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def get_default_monte_carlo_params() -> MonteCarloParams:
    """
    Derive Monte Carlo defaults. Keep conservative defaults independent of pipeline
    defaults while allowing deterministic behavior via seed when needed.
    """
    # Keep lightweight here to avoid importing src.main at module import time.
    return MonteCarloParams(
        n_simulations=1000,
        epsilon_fraction=0.005,
        random_seed=None,
        max_attempts=None,
        heteroskedastic_resampling=False,
        selection_policy="priority",
        output_prefix=None,
    )


def _safe_idxmin(series: pd.Series) -> Optional[int]:
    try:
        if series.dropna().empty:
            return None
        return int(series.dropna().idxmin())
    except Exception:
        return None


def _entropy_nats(probs: np.ndarray) -> float:
    # natural log entropy
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _central_contiguous_interval_50(
    pmf_map: Dict[int, float],
) -> Optional[Tuple[int, int]]:
    """
    Return the smallest contiguous integer interval [a, b] (inclusive) covering >=50% mass.
    If PMF empty return None.
    """
    if not pmf_map:
        return None
    # Build full integer range between min and max keys
    keys = sorted(pmf_map.keys())
    min_k, max_k = keys[0], keys[-1]
    # Build cumulative over the full integer range (fill missing with 0)
    values = np.array(
        [pmf_map.get(k, 0.0) for k in range(min_k, max_k + 1)], dtype=float
    )
    n = len(values)
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    target = 0.5
    best = None
    best_width = None
    for i in range(n):
        # binary search for minimal j >= i so prefix[j+1]-prefix[i] >= target
        lo = i
        hi = n - 1
        found = None
        while lo <= hi:
            mid = (lo + hi) // 2
            mass = prefix[mid + 1] - prefix[i]
            if mass >= target:
                found = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if found is not None:
            width = found - i
            if (best is None) or (width < best_width):
                best = (i, found)
                best_width = width
    if best is None:
        # fallback: return full span
        return (min_k, max_k)
    start_idx, end_idx = best
    return (min_k + start_idx, min_k + end_idx)


def run_monte_carlo(
    original_df: pd.DataFrame,
    summary: Any,
    transform_params: Any,
    mc_params: MonteCarloParams,
    random_state: Optional[np.random.Generator] = None,
    run_output_dir: Optional[Path] = None,
    short_hash: Optional[str] = None,
) -> MonteCarloSummary:
    """
    Run Monte Carlo simulations by repeatedly calling synthesize_data(...) and summarize_and_model(...).

    Important: imports of functions from src.main are done lazily inside this function
    to avoid circular import issues.
    """
    # Lazy imports to avoid circular dependency at module import time
    try:
        from src.main import (
            MODEL_PRIORITY,
            DeltaMode,
            model_cost_column,
            model_output_column,
            model_sum_column,
            regression_analysis,
            summarize_and_model,
            synthesize_data,
        )
    except Exception:
        # Fallback import style if package import fails (script mode)
        from main import (
            MODEL_PRIORITY,
            DeltaMode,
            model_cost_column,
            model_output_column,
            model_sum_column,
            regression_analysis,
            summarize_and_model,
            synthesize_data,
        )

    rng_master = (
        random_state
        if random_state is not None
        else np.random.default_rng(mc_params.random_seed)
    )

    n_target = int(mc_params.n_simulations)
    max_attempts = (
        mc_params.max_attempts
        if mc_params.max_attempts is not None
        else max(10, n_target * 2)
    )

    collected = 0
    attempts = 0
    failures: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    eps_counts: Dict[int, int] = {}
    recommend_counts: Dict[int, int] = {}
    regrets: List[float] = []

    # Determine synth target fort used for parameter copying
    synth_fort = (
        transform_params.synthesize_fort
        if getattr(transform_params, "synthesize_fort", None) is not None
        else transform_params.input_data_fort
    )

    while collected < n_target and attempts < max_attempts:
        attempts += 1
        try:
            # For each simulation, create an independent RNG stream for reproducibility
            sim_seed = rng_master.integers(0, 2**31 - 1)
            sim_rng = np.random.default_rng(int(sim_seed))

            # Synthesize using injected RNG
            synth_out, synth_diag = synthesize_data(
                summary, transform_params, original_df, random_state=sim_rng
            )

            # Build params for summary/modeling of synthesized frame
            try:
                from dataclasses import replace

                # Force PREVIOUS_CHUNK baseline to avoid MODEL_BASED semantics inside per-simulation summaries.
                params_for_synth = replace(
                    transform_params,
                    input_data_fort=int(synth_fort),
                    delta_mode=DeltaMode.PREVIOUS_CHUNK,
                )
            except Exception:
                # Best-effort fallback: attempt to set attribute directly, else use original params.
                try:
                    transform_params.delta_mode = DeltaMode.PREVIOUS_CHUNK  # type: ignore[attr-defined]
                except Exception:
                    pass
                params_for_synth = transform_params

            sim_summary = summarize_and_model(synth_out.df_range, params_for_synth)

            # Emulate MODEL_BASED offline-cost computation for this simulation when the
            # top-level transform_params requested MODEL_BASED. We avoid changing the
            # per-simulation delta_mode used by summarize_and_model (kept PREVIOUS_CHUNK
            # for stability), and instead derive per-model offline costs by running a
            # dedicated regression_analysis on the synthesized frame and computing:
            #   per_model_offline_cost = mean_fort - prediction(prev_k)
            # This reproduces MODEL_BASED semantics without requiring the summarize_and_model
            # function to be invoked with MODEL_BASED (which caused circular/enum issues).
            try:
                if (
                    getattr(transform_params, "delta_mode", None)
                    == DeltaMode.MODEL_BASED
                ):
                    try:
                        # Run regression to obtain model predictions on 1..synth_fort
                        df_results_reg, _ = regression_analysis(
                            synth_out.df_range, int(synth_fort)
                        )
                        # Extract mean_fort from sim_summary's df_summary final row if present
                        mean_fort = None
                        if (
                            hasattr(sim_summary, "df_summary")
                            and not sim_summary.df_summary.empty
                        ):
                            mean_fort = float(
                                sim_summary.df_summary.iloc[-1]["run_time_mean"]
                            )
                        prev_k = int(synth_fort) - 1
                        per_model_offline_costs: dict = {}
                        if mean_fort is not None and prev_k >= 1:
                            for token in MODEL_PRIORITY:
                                pred_col = model_output_column(token)
                                if pred_col in df_results_reg.columns:
                                    sel = df_results_reg.loc[
                                        df_results_reg["sor#"] == prev_k, pred_col
                                    ]
                                    if not sel.empty:
                                        try:
                                            val = float(
                                                pd.to_numeric(
                                                    sel.squeeze(), errors="coerce"
                                                )
                                            )
                                            if np.isfinite(val):
                                                per_model_offline_costs[token] = float(
                                                    mean_fort - val
                                                )
                                        except Exception:
                                            continue
                        # Recompute cost-per-run columns on df_results_reg using derived per-model offline costs
                        for token in MODEL_PRIORITY:
                            sum_col = model_sum_column(token)
                            if sum_col in df_results_reg.columns:
                                cost_col = model_cost_column(token)
                                offline = per_model_offline_costs.get(token, None)
                                offline_val = (
                                    float(offline)
                                    if (offline is not None and np.isfinite(offline))
                                    else float(
                                        getattr(sim_summary, "offline_cost", 0.0)
                                    )
                                )
                                # Avoid division by zero and preserve numeric conversion
                                try:
                                    df_results_reg[cost_col] = (
                                        pd.to_numeric(
                                            df_results_reg[sum_col], errors="coerce"
                                        )
                                        + offline_val
                                    ) / pd.to_numeric(
                                        df_results_reg["sor#"], errors="coerce"
                                    )
                                except Exception:
                                    # best-effort: skip updating this cost column on error
                                    pass
                        # Replace sim_summary.df_results with the updated regression results and attach per_model_offline_costs
                        try:
                            sim_summary.df_results = df_results_reg
                            sim_summary.per_model_offline_costs = (
                                per_model_offline_costs
                            )
                            # Recompute sor_min_costs defensively
                            sor_min = {}
                            for token in MODEL_PRIORITY:
                                cc = model_cost_column(token)
                                if cc in df_results_reg.columns:
                                    try:
                                        idx = _safe_idxmin(
                                            pd.to_numeric(
                                                df_results_reg[cc], errors="coerce"
                                            )
                                        )
                                        if idx is not None:
                                            sor_min[token] = int(
                                                df_results_reg.loc[idx, "sor#"]
                                            )
                                    except Exception:
                                        pass
                            sim_summary.sor_min_costs = sor_min
                        except Exception:
                            pass
                    except Exception:
                        # Emulation failed; fall back to sim_summary as-is
                        pass
            except Exception:
                # Top-level guard: do not let emulation errors fail the simulation loop
                pass

            # Determine authoritative model token based on selection_policy
            sel_policy = mc_params.selection_policy or "priority"
            authoritative_token = None
            if sel_policy == "synthesize":
                # Prefer the model used for synthesis when available
                auth_candidate = getattr(transform_params, "synthesize_model", None)
                if auth_candidate in MODEL_PRIORITY:
                    authoritative_token = auth_candidate
            if authoritative_token is None:
                # Fallback to priority ordering: first token present in df_results
                for tok in MODEL_PRIORITY:
                    if (
                        model_cost_column(tok)
                        in getattr(sim_summary, "df_results", pd.DataFrame()).columns
                    ):
                        authoritative_token = tok
                        break

            if authoritative_token is None:
                # Cannot determine authoritative model -> record failure and continue
                failures.append(
                    {"attempt": attempts, "reason": "no_authoritative_model"}
                )
                continue

            cost_col = model_cost_column(authoritative_token)
            df_results = sim_summary.df_results.copy()
            if cost_col not in df_results.columns or "sor#" not in df_results.columns:
                failures.append(
                    {"attempt": attempts, "reason": f"missing_cost_col_{cost_col}"}
                )
                continue

            # Ensure numeric
            df_results["sor#"] = pd.to_numeric(df_results["sor#"], errors="coerce")
            cost_series = pd.to_numeric(df_results[cost_col], errors="coerce")

            if not np.isfinite(cost_series.dropna().to_numpy()).any():
                failures.append({"attempt": attempts, "reason": "no_finite_costs"})
                continue

            # Compute recommended sor (argmin)
            idx_min = _safe_idxmin(cost_series)
            if idx_min is None:
                failures.append({"attempt": attempts, "reason": "idxmin_failed"})
                continue
            try:
                recommended_sor = int(df_results.loc[idx_min, "sor#"])
                candidate_cost = float(cost_series.loc[idx_min])
            except Exception:
                failures.append(
                    {"attempt": attempts, "reason": "extract_recommend_failed"}
                )
                continue

            # Compute min cost overall across all models (to compute regret)
            # Gather all cost columns present
            all_cost_cols = [
                model_cost_column(t)
                for t in MODEL_PRIORITY
                if model_cost_column(t) in df_results.columns
            ]
            min_cost_overall = float("inf")
            for cc in all_cost_cols:
                arr = pd.to_numeric(df_results[cc], errors="coerce").to_numpy(
                    dtype=float
                )
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    min_cost_overall = min(min_cost_overall, float(np.nanmin(finite)))
            if not np.isfinite(min_cost_overall):
                failures.append({"attempt": attempts, "reason": "no_global_min_cost"})
                continue

            regret = float(candidate_cost - min_cost_overall)
            regrets.append(regret)

            # Epsilon-optimal set for this simulation (within epsilon fraction of the authoritative model's min)
            eps = float(mc_params.epsilon_fraction)
            min_cost_auth = float(np.nanmin(cost_series.dropna().to_numpy()))
            threshold = min_cost_auth * (1.0 + eps)
            eps_mask = (cost_series <= threshold) & np.isfinite(cost_series)
            eps_sors = (
                df_results.loc[eps_mask, "sor#"]
                .dropna()
                .astype(int)
                .to_numpy(dtype=int)
                .tolist()
            )

            # Record counts
            recommend_counts[recommended_sor] = (
                recommend_counts.get(recommended_sor, 0) + 1
            )
            for s in eps_sors:
                eps_counts[s] = eps_counts.get(s, 0) + 1

            rows.append(
                {
                    "sim_id": collected,
                    "attempt": attempts,
                    "recommended_sor": int(recommended_sor),
                    "candidate_cost": float(candidate_cost),
                    "min_cost_auth": float(min_cost_auth),
                    "min_cost_overall": float(min_cost_overall),
                    "regret": float(regret),
                    "epsilon_optimal_sors": eps_sors,
                    "synth_diag": synth_diag,
                }
            )

            collected += 1

        except Exception as e:
            failures.append(
                {"attempt": attempts, "reason": "exception", "exception": str(e)}
            )
            # continue attempts until we hit max_attempts
            continue

    # Build per-simulation DataFrame
    per_sim_df = pd.DataFrame(rows)

    # Compute PMF over recommended choices
    total = float(max(collected, 1))
    pmf_map: Dict[int, float] = {k: v / total for k, v in recommend_counts.items()}

    # Mode (most frequent)
    mode_val = None
    if recommend_counts:
        mode_val = int(max(recommend_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])

    # Median & IQR of recommended integers
    recommendations = (
        np.array(sorted([int(r) for r in per_sim_df["recommended_sor"].to_numpy()]))
        if not per_sim_df.empty
        else np.array([], dtype=int)
    )
    median_val = float(np.median(recommendations)) if recommendations.size > 0 else None
    if recommendations.size > 0:
        q75 = float(np.percentile(recommendations, 75))
        q25 = float(np.percentile(recommendations, 25))
        iqr_val = float(q75 - q25)
    else:
        iqr_val = None

    # Central contiguous 50% interval
    central50 = _central_contiguous_interval_50(pmf_map)

    # Entropy in nats
    probs = np.array(list(pmf_map.values()), dtype=float)
    entropy_val = float(_entropy_nats(probs)) if probs.size > 0 else 0.0

    # Epsilon frequencies (fraction across collected sims)
    eps_freq_map: Dict[int, float] = {
        k: v / max(collected, 1) for k, v in eps_counts.items()
    }

    # Robust choice: smallest sor# with epsilon frequency >= 0.70
    robust_choice = None
    threshold_freq = 0.70
    if eps_freq_map:
        candidates_sorted = sorted(eps_freq_map.items(), key=lambda kv: kv[0])
        for sor, freq in candidates_sorted:
            if freq >= threshold_freq:
                robust_choice = int(sor)
                break

    # Expected regret stats
    regrets_arr = (
        np.array(regrets, dtype=float) if regrets else np.array([], dtype=float)
    )
    expected_regret_mean = float(np.mean(regrets_arr)) if regrets_arr.size > 0 else 0.0
    expected_regret_p95 = (
        float(np.percentile(regrets_arr, 95)) if regrets_arr.size > 0 else 0.0
    )

    # Diagnostics
    diagnostics = {
        "requested": int(mc_params.n_simulations),
        "collected": int(collected),
        "attempts": int(attempts),
        "failed_attempts": len(failures),
        "failures_sample": failures[:10],
        "seed_used": int(mc_params.random_seed)
        if mc_params.random_seed is not None
        else None,
        "selection_policy": mc_params.selection_policy,
        "hetero_resampling": bool(mc_params.heteroskedastic_resampling),
    }

    # Persist artifacts if run_output_dir provided
    try:
        if run_output_dir is not None:
            run_output_dir.mkdir(parents=True, exist_ok=True)
            # per-sim CSV
            if per_sim_df is not None and not per_sim_df.empty:
                csv_path = run_output_dir / (
                    f"mc-per-sim-{short_hash}.csv" if short_hash else "mc-per-sim.csv"
                )
                per_sim_df.to_csv(csv_path, index=False)
            # JSON summary
            import json

            summary_obj = {
                "pmf": pmf_map,
                "mode": mode_val,
                "median": median_val,
                "central_50_interval": central50,
                "entropy_nats": entropy_val,
                "iqr": iqr_val,
                "epsilon_freq": eps_freq_map,
                "robust_choice": robust_choice,
                "expected_regret_mean": expected_regret_mean,
                "expected_regret_p95": expected_regret_p95,
                "diagnostics": diagnostics,
            }
            json_path = run_output_dir / (
                f"mc-summary-{short_hash}.json" if short_hash else "mc-summary.json"
            )
            json_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort: do not fail the returned summary if persistence fails
        diagnostics["persistence_error"] = "failed_to_write_artifacts"

    mc_summary = MonteCarloSummary(
        pmf=pmf_map,
        mode=mode_val,
        median=median_val,
        central_50_interval=central50,
        entropy_nats=entropy_val,
        iqr=iqr_val,
        epsilon_freq=eps_freq_map,
        robust_choice=robust_choice,
        expected_regret_mean=expected_regret_mean,
        expected_regret_p95=expected_regret_p95,
        per_simulation_df=per_sim_df,
        diagnostics=diagnostics,
    )

    return mc_summary
