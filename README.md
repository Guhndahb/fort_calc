---
title: Fort Calc
emoji: 🐨
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: cc0-1.0
short_description: Idle Champions Optimal FORT Calculator
---

# Idle Champions Optimal FORT Calculator

Estimate the optimal number of online runs before performing a restart (FORT) to minimize average time per run in a repeated process whose run-time tends to grow with continued online runs. The system uses regression to smooth noisy run durations, estimates the offline restart cost from data, and selects the FORT that minimizes:

average_cost(k) = (sum of predicted online run times for runs 1..k + offline_cost) / k

The project is implemented as a reasonably pure, functional pipeline in Python with clear, testable stages:

- Loading CSV slices
- Data cleaning and transformations
- Summarization and offline cost estimation
- Regression modeling (OLS and empirical WLS)
- Cost-per-run curve construction and FORT selection
- Plot rendering and reporting

## Why FORT matters

When gem farming in Idle Champions of the Forgotten Realms, repeated online runs gradually slow down due to reasons. A restart (usually an offline Briv stack) resets those degradations but costs a fixed amount of time. You want a cadence that amortizes the restart cost against the increasing online run durations. Because measurements are noisy, we model the online trend with regression, estimate offline restart cost from observed data at the fort boundary, then compute the average cost curve to pick the best FORT.

## Installation

Python 3.10+ recommended.

- Clone repo
- Change directory into local repo
- Recommended: use a virtual environment:
  - `python -m venv .venv`
  - `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix) or `.\.venv\Scripts\Activate.ps1` (Powershell - requires execution policy)
  - `pip install -e .`
- Required dependencies (already completed if `pip install -e .` was performed):
  - pandas, numpy, matplotlib, statsmodels

Run tests:

- pytest

## CLI Usage

Run as a module:

- python -m src.main --log-path PATH/TO/log.csv [options]

Required:

- --log-path: CSV file

Slice options:

- --start-line N, --end-line M
- --header-map OLD:NEW (repeatable)

Transform options:

- --zscore-min FLOAT, --zscore-max FLOAT
- --fort INT (input_data_fort)
- --ignore-resetticks / --use-resetticks
- --delta-mode {PREVIOUS_CHUNK, FIRST_CHUNK}
- --exclude-range START,END (repeatable; format YYYYMMDDHHMMSS). Requires a timestamp column parseable to datetime.
- --verbose-filtering
- --no-fail-on-invalid-ts (when a timestamp column exists and --exclude-range is not used, you can relax invalid timestamp failures)

Plot options:

- --plot-spec key=value[,key=value...] (repeatable)
  - layers=PRESET_OR_FLAGS
  - x_min=FLOAT, x_max=FLOAT, y_min=FLOAT, y_max=FLOAT
  - Example: --plot-spec layers=DEFAULT,x_min=0,x_max=100
- --plot-spec-json JSON_OBJECT (repeatable)
  - JSON object with keys: layers, x_min, x_max, y_min, y_max
  - Example: --plot-spec-json '{"layers":"ALL_COST","y_min":0}'
- --plot-layers PRESET_OR_FLAGS (deprecated)
  - Presets: DEFAULT, ALL_OLS, ALL_WLS, ALL_PREDICTION, ALL_COST, MIN_MARKERS_ONLY, EVERYTHING
  - Or combine atomic flags: DATA_SCATTER+OLS_PRED_LINEAR+LEGEND
- --x-min, --x-max, --y-min, --y-max (deprecated when using --plot-spec)

Defaults preview:

- python -m src.main --print-defaults
- Help with policy-defaults injected:
  - python -m src.main --help

Example:

- python -m src.main --log-path ./ICScriptHub/log-reset.csv --fort 100 --ignore-resetticks --zscore-min -1.5 --zscore-max 3
- python -m src.main --log-path ./ICScriptHub/log-reset.csv --fort 50 --ignore-resetticks --header-map run:sor# --header-map "duration ms:runticks"

Multi-plot examples:

- python -m src.main --log-path ./ICScriptHub/log-reset.csv --fort 100 --ignore-resetticks --zscore-min -1.5 --zscore-max 3 --plot-spec layers=DEFAULT --plot-spec layers=ALL_COST
- python -m src.main --log-path ./ICScriptHub/log-reset.csv --fort 100 --ignore-resetticks --zscore-min -1.5 --zscore-max 3 --plot-spec layers=DEFAULT,x_min=0,x_max=100 --plot-spec-json '{"layers":"ALL_COST","y_min":0}'

## Plot layer flags reference

Use --plot-spec with layers=PRESET_OR_FLAGS or --plot-spec-json with "layers":"PRESET_OR_FLAGS" or use the deprecated --plot-layers with either:

- A preset name; or
- A +-joined list of flags. Matching is case-insensitive.

Presets:

- DEFAULT: data points, OLS predictions (linear and quadratic), OLS cost-per-run curves, OLS min-cost markers, legend
- EVERYTHING: DEFAULT plus all WLS overlays
- NONE: no layers (useful when you only want axis limits or a blank canvas)
- ALL_DATA: only the data points
- ALL_OLS: all OLS items — predictions, cost curves, min markers, legend
- ALL_WLS: all WLS items — predictions, cost curves, min markers, legend
- ALL_PREDICTION: only prediction lines (both OLS and WLS) with legend
- ALL_COST: only cost-per-run curves (both OLS and WLS) with legend
- MIN_MARKERS_ONLY: only min-cost vertical markers (both OLS and WLS) with legend

Atomic flags:

- DATA_SCATTER: scatter of included data points
- DATA_SCATTER_EXCLUDED: scatter of points removed by z-score filtering (red x), shown only if you include this flag and excluded data exists
- OLS_PRED_LINEAR: OLS linear prediction line
- OLS_PRED_QUAD: OLS quadratic prediction line
- OLS_COST_LINEAR: cost-per-run curve from the OLS linear model
- OLS_COST_QUAD: cost-per-run curve from the OLS quadratic model
- OLS_MIN_LINEAR: vertical marker at the lowest cost/run for the OLS linear model
- OLS_MIN_QUAD: vertical marker at the lowest cost/run for the OLS quadratic model
- WLS_PRED_LINEAR: WLS linear prediction line
- WLS_PRED_QUAD: WLS quadratic prediction line
- WLS_COST_LINEAR: cost-per-run curve from the WLS linear model
- WLS_COST_QUAD: cost-per-run curve from the WLS quadratic model
- WLS_MIN_LINEAR: vertical marker at the lowest cost/run for the WLS linear model
- WLS_MIN_QUAD: vertical marker at the lowest cost/run for the WLS quadratic model
- LEGEND: show the legend

--plot-spec Examples:

- Using new --plot-spec with key=value:
  - python -m src.main --log-path ./ICScriptHub/log-reset.csv --plot-spec layers=DEFAULT
- Using multiple --plot-spec flags:
  - python -m src.main --log-path ./ICScriptHub/log-reset.csv --plot-spec layers=DEFAULT --plot-spec layers=ALL_COST,x_min=0
- Using --plot-spec-json:
  - python -m src.main --log-path ./ICScriptHub/log-reset.csv --plot-spec-json '{"layers":"ALL_COST","y_min":0}'

## Input data Schema

Required for core modeling:

- sor#: positive integer in [1..FORT]
- runticks: per-run time in ticks (milliseconds)

Optional:

- resetticks: time in ticks spent on the Modron reset; if not ignored, it is subtracted from runticks
- timestamp: optional unless you use --exclude-range; parsing format %Y%m%d%H%M%S (as string or numeric convertible to that string form)
- ignore: "TRUE"/"FALSE" (case-insensitive); TRUE rows are dropped

Note: Column headers can be remapped via CLI

## How it works

1. Load and slice CSV

- Use a dedicated range reader to ingest a chosen portion of a CSV log.
- The CSV must include columns: sor#, runticks (or have equivalent columns that can be remapped via CLI command)
- The CSV may include columns: ignore, resetticks, notes
- The CSV may include timestamp; it is optional unless you use --exclude-range, in which case a parseable timestamp column is required
- Additional columns will be ignored

2. Transform pipeline

- Clean rows marked ignore and enforce sor# bounds in [1..FORT].
- Compute adjusted_run_time (seconds):
  - If ignore_resetticks=True: adjusted = runticks / 1000
  - Else: adjusted = (runticks − resetticks) / 1000
    (This calculation is part of the transformation pipeline.)
- If a timestamp column is present, parse to datetime with metrics for invalid timestamps.
- Optionally remove timestamp ranges when --exclude-range is provided; valid parsed timestamps are required for this step.
- Ensure the first notes value is present.
- Remove outliers by z-score on adjusted_run_time while always keeping the fort row. Degenerate variance is auto-handled.
- Enforce that at least 5 rows remain after filtering; otherwise fail early.

3. Summarize and estimate offline_cost

- Partition sor ∈ [1..FORT−1] into 4 bins (balanced), compute mean adjusted_run_time per bin, and create a degenerate last row for fort (sor==FORT).
- Compute offline_cost as the last row’s run_time_delta under a selected policy; default and recommended policy is PREVIOUS_CHUNK:
  offline_cost = mean(fort) − mean(last bin) when delta_mode==PREVIOUS_CHUNK
- Validate the summary (no NaNs in run_time_mean, fort row present).

4. Regression modeling

- Fit OLS linear and quadratic models of adjusted_run_time ~ sor and adjusted_run_time ~ sor + sor^2
- Compute diagnostics (R², Adj R², AIC, BIC, RMSE), and robust HC1 variants for reference
- Empirical WLS: Estimate heteroskedastic variance power p̂ by regressing log(resid^2) on log(sor); then fit WLS with weights 1/(sor^p̂)
- Predictions cover sor ∈ [1..FORT] for cost computation, while training uses the filtered data you pass in (which excludes the fort effect from the fit)
  (Performed during the regression analysis phase.)

5. Cost-per-run curve and FORT selection

- Construct cumulative sums of predictions Σ(k) for k ∈ [1..FORT], add offline_cost, and divide by k
- Compute curves for OLS linear/quadratic, and WLS linear/quadratic (if WLS converged)
- Select the k that minimizes average cost for each model family:
  sor_min_cost_lin, sor_min_cost_quad, sor_min_cost_lin_wls, sor_min_cost_quad_wls
  (Computed during model summarization.)

6. Plot and reporting

- Flexible layer flags let you render data, predictions, cost curves, and min markers, in OLS and WLS variants.
- One or more plots can be generated with customizable configurations.
- Plot filenames are generated with stable, descriptive names based on their configurations.
- A manifest JSON is written with parameters, counts, and artifacts.
- A human-readable text report is printed with data heads/tails, results, summary, and a compact model-comparison table with ranked FORTs.

## Modeling details

- OLS linear: adjusted_run_time = const + β1\*sor
- OLS quadratic: adjusted_run_time = const + β1*sor + β2*sor^2
- Empirical WLS:
  - Estimate p̂ from OLS residuals via log(resid^2 + eps) ~ log(sor)
  - Weights w = 1 / (sor^p̂), median-imputed for non-finite values
- Robust HC1 covariance fits reported in diagnostics for OLS and WLS variants
- Predictions span sor ∈ [1..FORT] for cost evaluation
- Training is performed on the filtered rows you supply (which, by design, exclude the fort effect from the modeling inputs)

## Offline cost estimation

- Summary bins for sor ∈ [1..FORT−1] + degenerate fort row
- Delta policy (recommended): PREVIOUS_CHUNK
  offline_cost = mean(fort) − mean(last bin)
- Alternative: FIRST_CHUNK
  offline_cost = mean(fort) − mean(first bin)
- Final offline_cost must be finite; otherwise the pipeline fails with a clear message

## Decision policy and interpretations

- The cost curve is often very flat near the minimum: multiple k around the min can have negligible cost differences
- Consider using a tolerance band in practice (policy-level; not enforced by code):

  - Pick the smallest k with cost ≤ 1% above the minimum to reduce risk of long online runs
  - Add hysteresis: only change FORT when the recommended k shifts by ≥ Δk_min (e.g., 5) or when the min cost improves by ≥ 0.5–1.0%

- Model selection guidance:

  - Prefer the simpler model when ΔBIC ≤ 2
  - Use WLS when p̂ is stable across runs; if unstable, OLS may be more robust
  - If quadratic curvature implies extreme changes across 1..FORT, fall back to linear

- Offline_cost robustness:
  - Ensure adequate samples near fort and in the last bin
  - If bins are sparse, consider adaptive bin merging to hit minimum support thresholds

## Artifacts

- SVG plots using the selected plot layers (one or more plots can be generated).
- Manifest JSON with parameters, counts, and artifact names.
- Text report to stdout summarizing data, results, summary, and model/FORT comparisons.

## Testing

The tests cover:

- Cleaning and timestamp-range filtering behavior
- Timestamp parsing diagnostics
- Regression prediction column order
- Plot layer parsing and suffix logic
- Z-score edge cases
- Regression variants and diagnostics

See tests/ for details.

## FAQ

- Why is almost everything in main.py?
  Because this was a tiny project that suffered massive feature creep and I haven't gotten around to refactoring. That's my excuse, anyways, and I'm sticking to it.

- How useful are the results?
  I strongly believe, but have no way to prove, that they are going to be more accurate than general BPH testing which is going to be subject to all the same poor data problems. This program attempts to mitigate these problems with maffs. How it does so is explained in detail in this README.

- Then why is it giving me ideal FORT values that aren't very close?
  The cost curve is VERY flat where the "dip" that represents the ideal FORT resides. This means tiny changes (namely switching between linear and quadratic curves, and insufficient clean input - particularly at the testing FORT value) can cause significant shifts. The good news is that because it's so flat, choosing one over another is unlikely to impact your run speed appreciably.

- Why exclude the fort run from training?
  Because the fort run includes the offline restart cost; we want to model only the online trending behavior.

- What if z-score standard deviation is zero?
  The pipeline treats all z-scores as 0, retains only the fort row, and will fail with a “too few rows” error if insufficient data remains.

- Can I switch offline delta policy?
  Yes; set --delta-mode FIRST_CHUNK to compute offline_cost relative to the first bin mean. Neither is ideal and I'm open to suggested improvements since this is very important.

- How do I generate multiple plots with different configurations?
  Use the new --plot-spec or --plot-spec-json flags to specify multiple plot configurations. Each can have its own layers and axis limits.

## Roadmap ideas

- Add CLI option for output directory for artifacts (plots, manifest)
- IQR-based outlier filtering as an alternative to z-score: add an option to filter points outside a configurable multiple of the interquartile range (IQR). This is more robust to skewed distributions than z-score filtering, should always preserve the fort row, and include flags/doc examples for choosing between IQR and z-score methods.
- Optional adaptive binning for offline_cost stability
- Spline/GAM model options for flexible yet smooth trends
- Bootstrap CI for FORT: add a bootstrap resampling workflow to produce confidence intervals for recommended FORT and plot uncertainty bands around cost curves / Monte Carlo simulations
- Policy helpers: tolerance band selection and hysteresis built into the report

## Multi-plot functionality

The multi-plot functionality allows you to generate multiple plots with different configurations in a single run:

- Use --plot-spec key=value[,key=value...] to specify plot parameters in key=value format (repeatable)
- Use --plot-spec-json JSON_OBJECT to specify plot parameters as a JSON object (repeatable)
- Each plot specification can have its own layers, axis limits, and other parameters
- Multiple plots will be generated with descriptive filenames based on their configurations
- The deprecated --plot-layers flag still works for single plot generation but is no longer recommended

## License

This project is released under the CC0 1.0 Universal Public Domain Dedication.

- See the local ["LICENSE"](LICENSE) file and the Creative Commons summary: ["CC0 1.0 Universal (summary)"](https://creativecommons.org/publicdomain/zero/1.0/).
