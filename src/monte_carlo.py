from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MonteCarloParams:
    n_simulations: int = 1000
    epsilon_start: float = 0.005
    random_seed: Optional[int] = None
    max_attempts: Optional[int] = None
    heteroskedastic_resampling: bool = False
    selection_policy: str = "priority"  # "priority" or "synthesize"
    output_prefix: Optional[str] = None
    # Adaptive epsilon tuning parameters (per-simulation)
    # Note: defaults are chosen to balance reducing large ε-optimal sets while avoiding
    # excessive fallbacks to argmin. Keep these synchronized with get_default_monte_carlo_params().
    epsilon_min: float = 1e-6
    epsilon_shrink_factor: float = 0.8
    epsilon_target_size: int = 3
    epsilon_max_iterations: int = 10
    # Automated epsilon selection mode: when True, ignore epsilon_min / epsilon_shrink_factor /
    # epsilon_max_iterations and deterministically select at most epsilon_target_size items
    # based only on the relative-cost ranking. Default True for automated behavior.
    epsilon_auto: bool = True


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

    NOTE: This function is the single point of authority for CLI-visible Monte Carlo
    defaults (so CLI help / _args_to_params can read them and keep behavior consistent).
    """
    # Keep lightweight here to avoid importing src.main at module import time.
    return MonteCarloParams(
        n_simulations=1000,
        epsilon_start=0.005,
        random_seed=None,
        max_attempts=None,
        heteroskedastic_resampling=False,
        selection_policy="priority",
        output_prefix=None,
        # Adaptive epsilon defaults (centralized here)
        epsilon_min=1e-6,
        epsilon_shrink_factor=0.8,
        epsilon_target_size=3,
        epsilon_max_iterations=10,
        # Automated epsilon selection enabled by default (see MonteCarloParams.epsilon_auto)
        epsilon_auto=True,
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
    debug_rows: List[Dict[str, Any]] = []

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
                recommended_fort = int(df_results.loc[idx_min, "sor#"])
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

            # Adaptive / automatic epsilon-optimal set selection per simulation.
            # Two modes:
            #  - epsilon_auto == False: preserve existing adaptive multiplicative shrink loop
            #    exactly as before (uses epsilon_start, epsilon_min, epsilon_shrink_factor, epsilon_max_iterations).
            #  - epsilon_auto == True: deterministic selection based solely on ranking of costs.
            #    Ignore epsilon_min / epsilon_shrink_factor / epsilon_max_iterations in this mode.
            #
            # Deterministic algorithm (auto mode):
            #  1) Sort SORs by their authoritative cost ascending.
            #  2) Let best_val = cost of the best SOR.
            #  3) Compute relative gap delta = (cost - best_val) / best_val for each SOR.
            #  4) Select the smallest prefix (best-to-worst) of sorted SORs whose size is <= epsilon_target_size.
            #     This guarantees the selected set size is never larger than epsilon_target_size and is deterministic.
            #  5) epsilon_used is the delta of the worst element included (0.0 when only argmin is included).
            #
            # Note: epsilon_min / epsilon_shrink_factor / epsilon_max_iterations are intentionally ignored in auto mode.
            target_size = int(getattr(mc_params, "epsilon_target_size", 5))
            min_cost_auth = float(np.nanmin(cost_series.dropna().to_numpy()))
            # Ensure max_iters is defined for downstream fallback checks and diagnostic logging.
            max_iters = int(getattr(mc_params, "epsilon_max_iterations", 10))

            epsilon_used = float(getattr(mc_params, "epsilon_start", 0.0))
            eps_sors = []
            epsilon_iterations = 0
            # Fallback trackers for diagnostic persistence
            epsilon_fallback_to_argmin = False
            epsilon_fallback_reason = None
            last_non_empty_eps_sors: Optional[List[int]] = None
            last_non_empty_epsilon: Optional[float] = None

            if bool(getattr(mc_params, "epsilon_auto", True)):
                # AUTO MODE: deterministic selection based on cost ranking.
                # Build (sor, cost) pairs for finite costs only.
                try:
                    cost_df = pd.DataFrame(
                        {
                            "sor#": df_results["sor#"],
                            "__cost": pd.to_numeric(cost_series, errors="coerce"),
                        }
                    )
                    # Keep only finite cost rows
                    cost_df = cost_df.loc[np.isfinite(cost_df["__cost"])].copy()
                    # Sort ascending by cost
                    cost_df = cost_df.sort_values(
                        by="__cost", ascending=True
                    ).reset_index(drop=True)
                except Exception:
                    cost_df = pd.DataFrame(columns=["sor#", "__cost"])

                # If no finite costs, deterministic fallback to argmin
                if cost_df.empty:
                    eps_sors = [int(recommended_fort)]
                    epsilon_used = 0.0
                    epsilon_iterations = len(eps_sors)
                    epsilon_fallback_reason = "auto_mode"
                    epsilon_fallback_to_argmin = True
                else:
                    # If target size == 1, return just the argmin deterministically
                    if int(target_size) <= 1:
                        # argmin is the first row in sorted cost_df
                        try:
                            argmin_sor = int(cost_df.iloc[0]["sor#"])
                        except Exception:
                            argmin_sor = int(recommended_fort)
                        eps_sors = [argmin_sor]
                        epsilon_used = 0.0
                        epsilon_iterations = 1
                        epsilon_fallback_reason = "auto_mode"
                        epsilon_fallback_to_argmin = False
                    else:
                        # Select up to target_size best items (prefix)
                        take_n = min(int(target_size), int(len(cost_df)))
                        selected = cost_df.iloc[0:take_n]
                        try:
                            eps_sors = (
                                selected["sor#"]
                                .dropna()
                                .astype(int)
                                .to_numpy()
                                .tolist()
                            )
                        except Exception:
                            eps_sors = [int(recommended_fort)]
                        # Compute epsilon_used as relative gap of worst included element
                        try:
                            best_val = float(selected["__cost"].iloc[0])
                            worst_val = float(selected["__cost"].iloc[-1])
                            if best_val == 0.0:
                                # Avoid division by zero: if both zero then gap is 0.0, else use inf
                                epsilon_used = 0.0 if worst_val == 0.0 else float("inf")
                            else:
                                epsilon_used = float((worst_val - best_val) / best_val)
                        except Exception:
                            epsilon_used = float(
                                getattr(mc_params, "epsilon_start", 0.0)
                            )
                        epsilon_iterations = len(eps_sors)
                        epsilon_fallback_reason = "auto_mode"
                        epsilon_fallback_to_argmin = False
                # Record last_non_empty for parity with legacy diagnostics (best-effort)
                if isinstance(eps_sors, (list, tuple)) and len(eps_sors) > 0:
                    last_non_empty_eps_sors = list(eps_sors)
                    last_non_empty_epsilon = float(epsilon_used)
            else:
                # LEGACY ADAPTIVE MODE: preserve original multiplicative shrinking behavior exactly.
                eps = float(getattr(mc_params, "epsilon_start", 0.005))
                # Adaptive parameters (with safe fallbacks)
                eps_min = float(getattr(mc_params, "epsilon_min", 1e-6))
                shrink_factor = float(getattr(mc_params, "epsilon_shrink_factor", 0.5))

                epsilon_used = eps
                eps_sors = []
                epsilon_iterations = 0
                # Track the last non-empty epsilon-optimal set (and the epsilon that produced it)
                last_non_empty_eps_sors = None
                last_non_empty_epsilon = None
                # Fallback trackers for diagnostic persistence
                epsilon_fallback_to_argmin = False
                epsilon_fallback_reason = None
                for _ in range(max_iters):
                    epsilon_iterations += 1
                    threshold = min_cost_auth * (1.0 + eps)
                    eps_mask = (cost_series <= threshold) & np.isfinite(cost_series)
                    try:
                        eps_sors = (
                            df_results.loc[eps_mask, "sor#"]
                            .dropna()
                            .astype(int)
                            .to_numpy(dtype=int)
                            .tolist()
                        )
                    except Exception:
                        eps_sors = []

                    # Record last non-empty set for potential fallback
                    if isinstance(eps_sors, (list, tuple)) and len(eps_sors) > 0:
                        last_non_empty_eps_sors = list(eps_sors)
                        last_non_empty_epsilon = float(eps)

                    # Always accept if only the argmin is selected
                    if len(eps_sors) == 1:
                        epsilon_used = eps
                        break
                    # Accept if within desired target size (and at least one)
                    if 1 <= len(eps_sors) <= target_size:
                        epsilon_used = eps
                        break
                    # Shrink epsilon and continue
                    eps = eps * shrink_factor
                    if eps < eps_min:
                        # Floor reached -> try to use last non-empty epsilon set if available,
                        # otherwise fall back to argmin.
                        eps = eps_min
                        if last_non_empty_eps_sors is not None:
                            # Use the most recent non-empty epsilon set we observed
                            eps_sors = list(last_non_empty_eps_sors)
                            epsilon_used = float(last_non_empty_epsilon)
                            # Do not mark this as fallback-to-argmin since we selected an epsilon-set
                            epsilon_fallback_to_argmin = False
                            epsilon_fallback_reason = (
                                "epsilon_floor_reached_used_last_nonempty"
                            )
                        else:
                            # No epsilon set ever found -> deterministic fallback to argmin
                            epsilon_used = float(eps)
                            eps_sors = [int(recommended_fort)]
                            epsilon_fallback_to_argmin = True
                            epsilon_fallback_reason = (
                                "epsilon_floor_reached_no_nonempty"
                            )
                        break
                    # update epsilon_used to current candidate (in case loop exits without break)
                    epsilon_used = eps

            # If loop exhausted without breaking, enforce deterministic fallback when appropriate
            if epsilon_iterations == 0:
                epsilon_iterations = 0
            # If we reached max iterations but still have too many epsilon members, prefer
            # the last non-empty epsilon set instead of immediately falling back to argmin.
            if (
                (epsilon_iterations >= max_iters)
                and isinstance(eps_sors, (list, tuple))
                and len(eps_sors) > target_size
            ):
                if last_non_empty_eps_sors is not None:
                    # Use the last non-empty epsilon set we observed
                    eps_sors = list(last_non_empty_eps_sors)
                    epsilon_used = float(last_non_empty_epsilon)
                    epsilon_fallback_to_argmin = False
                    epsilon_fallback_reason = (
                        "max_iterations_exceeded_used_last_nonempty"
                    )
                else:
                    # No epsilon set observed at all -> fallback to argmin
                    epsilon_fallback_to_argmin = True
                    epsilon_fallback_reason = "max_iterations_exceeded_no_nonempty"
                    eps_sors = [int(recommended_fort)]
                    epsilon_used = float(eps if eps >= eps_min else eps_min)
            # If nothing selected at all, try last_non_empty, else argmin as absolute fallback
            if not eps_sors:
                if last_non_empty_eps_sors is not None:
                    eps_sors = list(last_non_empty_eps_sors)
                    epsilon_used = float(last_non_empty_epsilon)
                else:
                    eps_sors = [int(recommended_fort)]
                    # ensure epsilon_used was set sensibly
                    epsilon_used = float(eps if eps >= eps_min else eps_min)

            # Expose epsilon_used for auditing in debug/per-sim outputs
            # Ensure threshold and max_iters exist so debug logging and later fallback checks
            # remain stable regardless of mode (auto vs legacy). For auto-mode threshold is
            # derived from the chosen epsilon_used for diagnostics (min_cost_auth*(1+epsilon_used)).
            # max_iters is defined above; no need to lazily rebind here.
            try:
                threshold  # type: ignore[name-defined]
            except Exception:
                try:
                    threshold = float(min_cost_auth * (1.0 + float(epsilon_used)))
                except Exception:
                    threshold = float("nan")

            # Collect debug row for this simulation (sample up to first 10 cost values)
            try:
                cost_sample = (
                    pd.to_numeric(cost_series, errors="coerce")
                    .dropna()
                    .to_numpy(dtype=float)[:10]
                    .tolist()
                )
            except Exception:
                cost_sample = []
            debug_rows.append(
                {
                    "sim_id": int(collected),
                    "attempt": int(attempts),
                    "authoritative_token": authoritative_token,
                    "cost_col": cost_col,
                    "min_cost_auth": float(min_cost_auth),
                    "threshold": float(threshold) if np.isfinite(threshold) else None,
                    "recommended_fort": int(recommended_fort),
                    "candidate_cost": float(candidate_cost),
                    "epsilon_optimal_sors": eps_sors,
                    "cost_sample": cost_sample,
                    "epsilon_used": float(epsilon_used),
                    "epsilon_iterations": int(epsilon_iterations),
                    "epsilon_fallback_to_argmin": bool(epsilon_fallback_to_argmin),
                    "epsilon_fallback_reason": (
                        str(epsilon_fallback_reason)
                        if epsilon_fallback_reason is not None
                        else None
                    ),
                }
            )

            # Diagnostic persistence: if the epsilon-optimal set is large, persist the full
            # df_results for offline inspection so we can validate sor#/cost alignment,
            # duplicated sor entries, and exact per-row cost values.
            try:
                if (
                    run_output_dir is not None
                    and isinstance(eps_sors, (list, tuple))
                    and len(eps_sors) > 20
                ):
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                    sim_id_for_dump = int(collected)
                    # Honor optional output_prefix when persisting large debug df-results dumps
                    prefix_str = (
                        f"{mc_params.output_prefix}-"
                        if getattr(mc_params, "output_prefix", None)
                        else ""
                    )
                    dump_name = (
                        f"{prefix_str}df-results-sim-{sim_id_for_dump}-{short_hash}.csv"
                        if short_hash
                        else f"{prefix_str}df-results-sim-{sim_id_for_dump}.csv"
                    )
                    dump_path = run_output_dir / dump_name
                    try:
                        # sim_summary.df_results is expected to be a DataFrame-like object
                        pd.DataFrame(sim_summary.df_results).to_csv(
                            dump_path, index=False
                        )
                    except Exception:
                        # best-effort fallback: coerce to strings and write
                        try:
                            pd.DataFrame(sim_summary.df_results).astype(str).to_csv(
                                dump_path, index=False
                            )
                        except Exception:
                            # swallow errors — diagnostics must not break MC loop
                            pass
            except Exception:
                # Never allow diagnostic persistence to break the main simulation loop
                pass

            # Record counts
            recommend_counts[recommended_fort] = (
                recommend_counts.get(recommended_fort, 0) + 1
            )
            for s in eps_sors:
                eps_counts[s] = eps_counts.get(s, 0) + 1

            rows.append(
                {
                    "sim_id": collected,
                    "attempt": attempts,
                    "recommended_fort": int(recommended_fort),
                    "candidate_cost": float(candidate_cost),
                    "min_cost_auth": float(min_cost_auth),
                    "min_cost_overall": float(min_cost_overall),
                    "regret": float(regret),
                    "epsilon_optimal_sors": eps_sors,
                    "synth_diag": synth_diag,
                    "epsilon_used": float(epsilon_used),
                    "epsilon_iterations": int(epsilon_iterations),
                    "epsilon_fallback_to_argmin": bool(epsilon_fallback_to_argmin),
                    "epsilon_fallback_reason": (
                        str(epsilon_fallback_reason)
                        if epsilon_fallback_reason is not None
                        else None
                    ),
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
        np.array(sorted([int(r) for r in per_sim_df["recommended_fort"].to_numpy()]))
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

    # Robust choice: pick a single robust SOR based on epsilon-frequency statistics.
    # Previous behavior selected the smallest SOR with freq >= threshold (which often
    # returned a low boundary when epsilon-optimal sets were contiguous upward).
    # New heuristic:
    #  1. Prefer SORs with epsilon frequency >= threshold_freq.
    #  2. If none meet the threshold, pick the SOR(s) with the maximum epsilon frequency.
    #  3. If multiple candidates remain, choose the one closest to the PMF mode (if known),
    #     otherwise choose the smallest (deterministic) SOR among candidates.
    robust_choice = None
    threshold_freq = 0.70
    if eps_freq_map:
        # Normalize items to (int_sor, freq)
        candidates = [(int(k), float(v)) for k, v in eps_freq_map.items()]
        # Prefer those meeting the frequency threshold
        above_thresh = [s for s, f in candidates if f >= threshold_freq]
        if len(above_thresh) > 0:
            filtered = above_thresh
        else:
            # Fall back to SOR(s) with maximum frequency
            max_freq = max(f for _, f in candidates)
            filtered = [s for s, f in candidates if f == max_freq]
        # Tie-breaker: choose candidate closest to the PMF mode if available
        if mode_val is not None:
            robust_choice = int(min(filtered, key=lambda s: abs(s - mode_val)))
        else:
            # deterministic fallback
            robust_choice = int(min(filtered))

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
            # honor optional output_prefix for all Monte Carlo artifacts
            prefix_str = (
                f"{mc_params.output_prefix}-"
                if getattr(mc_params, "output_prefix", None)
                else ""
            )
            # per-sim CSV
            if per_sim_df is not None and not per_sim_df.empty:
                csv_path = run_output_dir / (
                    f"{prefix_str}mc-per-sim-{short_hash}.csv"
                    if short_hash
                    else f"{prefix_str}mc-per-sim.csv"
                )
                per_sim_df.to_csv(csv_path, index=False)
            # debug per-sim sampled numeric diagnostics
            try:
                if debug_rows:
                    dbg_path = run_output_dir / (
                        f"{prefix_str}debug-mc-{short_hash}.csv"
                        if short_hash
                        else f"{prefix_str}debug-mc.csv"
                    )
                    pd.DataFrame(debug_rows).to_csv(dbg_path, index=False)
            except Exception:
                # do not fail persistence for debug CSV
                pass
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
                f"{prefix_str}mc-summary-{short_hash}.json"
                if short_hash
                else f"{prefix_str}mc-summary.json"
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
