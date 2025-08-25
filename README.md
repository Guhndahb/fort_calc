---
title: fort_calc
emoji: 📈
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: cc0-1.0
short_description: Idle Champions Optimal FORT Calculator
---

## Why FORT matters

When gem farming in Idle Champions of the Forgotten Realms, repeated online runs gradually slow down due to _reasons_. A restart (usually an offline Briv stack) resets those degradations but costs a fixed amount of time. You want a cadence that amortizes the restart cost against the increasing online run durations. Because measurements are noisy, we model the online trend with regression, estimate offline restart cost from observed data at the fort boundary, then compute the average cost curve to pick the best FORT.

# Idle Champions Optimal FORT Calculator

Estimate the optimal number of online runs before performing a restart (FORT) to minimize average time per run. The tool models run-time drift, estimates restart (offline) cost, and finds the run count k that minimizes:
average_cost(k) = (sum of predicted run times for runs 1..k + offline_cost) / k

## Installation

Python 3.10+ recommended.

- Clone the repository and change into it.
- Recommended: create and use a virtual environment
  - python -m venv .venv
  - Windows (cmd): .venv\Scripts\activate
  - Windows (PowerShell): .\.venv\Scripts\Activate.ps1
  - Unix/macOS: source .venv/bin/activate
- Install dependencies:
  - pip install -e .
- The project requires typical data/science packages (see requirements.txt). If you do not use a virtual environment, install into your system Python using pip.

Run tests:

- pytest

## Quick CLI usage

Run as a module:

- python -m src.main --log-path PATH/TO/log.csv [options]

Show canonical defaults (machine-readable):

- python -m src.main --print-defaults

Show help (with default values displayed):

- python -m src.main --help

Minimum required:

- --log-path PATH (CSV file)
- --fort INT — target SOR value to analyze

Examples:

- python -m src.main --log-path ./log.csv --fort 100 --ignore-resetticks --zscore-min -1.5 --zscore-max 3
- python -m src.main --log-path ./log.csv --fort 50 --ignore-resetticks --header-map run:sor# --header-map "duration ms:runticks"
- Multiple plots in one run:
  - python -m src.main --log-path ./log.csv --plot-spec layers=DEFAULT --plot-spec layers=ALL_FORT_COST,x_min=0
  - python -m src.main --log-path ./log.csv --plot-spec-json '{"layers":"ALL_FORT_COST","y_min":0}'

## Important CLI flags (names and behavior)

Slicing and column selection

- --log-path PATH (CSV file)
- --start-line N / --end-line M — 1-based inclusive data-row slice (header excluded)
- --header-map OLD:NEW — repeatable; remap input header OLD → NEW (case-insensitive)
- --col-sor INT — zero-based column index to use for the SOR/run index (overrides header mapping)
- --col-ticks INT — zero-based column index to use for runticks/duration (overrides header mapping)

Transform & filtering

- --fort INT — target SOR value to analyze
- --ignore-resetticks / --use-resetticks — boolean pair: ignore or subtract reset ticks from runticks
- --delta-mode {PREVIOUS_CHUNK, FIRST_CHUNK, MODEL_BASED} — policy for computing summary deltas and offline cost
- --exclude-range START,END — repeatable; exclude timestamp ranges (format YYYYMMDDHHMMSS)
- --iqr-k-low FLOAT, --iqr-k-high FLOAT — multipliers for IQR-based outlier filtering
- --use-iqr-filtering / --use-zscore-filtering — choose IQR or z-score outlier filtering
- --zscore-min FLOAT, --zscore-max FLOAT — bounds for z-score filtering
- --verbose-filtering — show extra diagnostics during filtering
- --no-fail-on-invalid-ts — relax strict timestamp parsing failures
- --offline-cost-override FLOAT — force a scalar offline cost
- --simulated-fort INT — simulate a smaller fort (requires --offline-cost-override)

Plotting

- --plot-spec key=value[,key=value...] — repeatable. Supported keys: layers, x_min, x_max, y_min, y_max. Example: layers=DEFAULT,x_min=0,x_max=100
- --plot-spec-json JSON_OBJECT — repeatable JSON object with the same keys. Example: '{"layers":"ALL_FORT_COST","y_min":0}'
- When no plot-spec flags are supplied, the program emits a canonical list of default plot configurations.

Special plotting sentinel

- The literal string OMIT_FORT may be passed as an x_max value in a plot specification to request omission of the final SOR/FORT point from the rendered plot.

## Canonical defaults (high level)

The canonical transform defaults used when no overrides are provided include (summary):

- z-score bounds: -1.75 (min), 2.5 (max)
- default nominal FORT: 100
- default behavior: ignore reset ticks
- default delta mode: MODEL_BASED
- default outlier method: IQR filtering enabled
- default IQR multipliers: low = 1.0, high = 2.0

Use --print-defaults to get the exact defaults in JSON.

## Input data schema

Required columns:

- sor# — integer run index, 1..FORT
- runticks — run duration in ticks (milliseconds)

Optional columns:

- resetticks — reset duration in ticks (used only if reset ticks are subtracted)
- timestamp — required only if using exclude-range; must parse as YYYYMMDDHHMMSS
- ignore — "TRUE" rows are dropped

Header mapping & precedence:

- Header remapping is case-insensitive.
- Numeric column index overrides (--col-sor, --col-ticks) take precedence over header remapping and are validated for collisions and range.

## Pipeline overview

High-level stages:

1. Load a CSV slice.
2. Clean rows flagged as ignored and enforce valid SOR bounds.
3. Compute adjusted run time in seconds (either runticks/1000 or (runticks − resetticks)/1000).
4. Parse timestamps if present and optionally exclude ranges.
5. Ensure first note is present.
6. Remove outliers using IQR or z-score filtering while always preserving the final fort row(s).
7. Require a minimum number of rows after filtering (fails early if too few remain).

Outlier behavior:

- IQR filtering (default) uses lower/upper bounds derived from non-fort rows; degenerate IQR → conservative fallback.
- Z-score filtering computes z-scores from non-fort mean/std; degenerate variance → conservative fallback.
- The fort row(s) are always preserved by filtering.

## Summarization and offline-cost estimation

Summarization:

- The range 1..(FORT−1) is partitioned into four contiguous bins as evenly as possible and the final degenerate fort row is appended. Each bin’s mean run time is computed and deltas are derived per selected policy.

Offline-cost policies:

- PREVIOUS_CHUNK: offline estimate = mean(fort) − mean(last bin)
- FIRST_CHUNK: offline estimate = mean(fort) − mean(first bin)
- MODEL_BASED: derives per-model offline estimates from model predictions at the penultimate run and falls back to a summary delta when model-based estimates are unavailable
- A CLI override can force the offline cost to a provided scalar

## Modeling summary

Modeling approaches attempted (in canonical order):

- Robust linear (Theil–Sen)
- Isotonic regression
- Piecewise cubic interpolation (PCHIP)
- Ordinary least squares: linear and quadratic
- Empirical weighted least squares variants using estimated variance behavior: linear and quadratic

Behavior:

- Predictions are produced for the full run grid 1..FORT.
- For each model that produced predictions, cumulative sums and per-run cost curves are computed.
- The program identifies the run index that minimizes the average cost for each model and reports per-model recommended FORTs.
- Convexity of per-run cost curves is checked; non-convex or omitted models are noted in the output diagnostics.

## Plotting layers and presets (user tokens)

Layer tokens accepted (examples):

- DATA_SCATTER, DATA_SCATTER_EXCLUDED, PREDICTION, FORT_COST, MIN_COST, LEGEND, LEGEND_FILTERING

Common presets:

- DEFAULT, EVERYTHING, NONE, ALL_DATA, ALL_SCATTER, ALL_PREDICTION, ALL_FORT_COST, ALL_MIN_COST, SCATTER_PREDICTION

Layer strings are case-insensitive and accept either a preset name or a '+'-joined list of atomic tokens.

## Artifacts and naming

Per-run directory:

- output/{timestamp}, where timestamp is formatted YYYYMMDDTHHMMSS

Files written:

- manifest-{short_hash}.json — run manifest with counts, effective params, and artifact list
- plot-{short_hash}-{ii}-{suffix}.svg — plot files (ii is zero-based index, suffix reflects chosen layers)
- report-{short_hash}.txt — human-readable report printed to stdout and saved in the run directory

## Multi-plot usage

- Multiple --plot-spec or --plot-spec-json flags create multiple distinct plots in a single run.
- If none provided, the program uses a canonical set of default plot configurations.

## Roadmap ideas

- Add an option to choose an output directory
- Bootstrap CI for FORT: add a bootstrap resampling workflow to produce confidence intervals for recommended FORT and plot uncertainty bands around cost curves / Monte Carlo simulations
- Policy helpers: tolerance band selection and hysteresis built into the report

## License

This project is released under the CC0 1.0 Universal Public Domain Dedication.

- See the local ["LICENSE"](LICENSE) file and the Creative Commons summary: ["CC0 1.0 Universal (summary)"](https://creativecommons.org/publicdomain/zero/1.0/).
