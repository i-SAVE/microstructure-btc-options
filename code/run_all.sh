#!/usr/bin/env bash
# Reproduce every number, table and figure of the article from the public Binance archives.
# Usage: bash code/run_all.sh [ROOT]   (ROOT = repository folder, default: current folder)
set -euo pipefail
ROOT="${1:-.}"
PY="${PYTHON:-python3}"
$PY "$ROOT/code/download_data.py" "$ROOT"          # inputs + SHA-256 check (data/input_manifest.csv)
$PY "$ROOT/code/data_prep.py" "$ROOT"              # panel, spot, counts.json
if [ ! -f "$ROOT/data/spot_spread.parquet" ]; then # spot spread proxy; downloads daily aggTrades (several GB)
  $PY "$ROOT/code/spot_spread.py" "$ROOT" 2023-09-01 2023-10-24
fi
$PY "$ROOT/code/estimation.py" "$ROOT"             # Table 1 (mark IV), regimes
$PY "$ROOT/code/estimation_mid.py" "$ROOT"         # Table 1 (quote IV)
$PY "$ROOT/code/backtest.py" "$ROOT"               # Tables 2-3, Pareto fronts, bootstrap
$PY "$ROOT/code/extra.py" "$ROOT"                  # ablations, hourly hedging by model, post-hoc band
$PY "$ROOT/code/hedge_quote_iv.py" "$ROOT"         # hedging with quote-IV deltas
$PY "$ROOT/code/figures.py" "$ROOT"                # figs/Fig1-4 (.png 600 dpi, .pdf; Fig1 also .svg)
$PY "$ROOT/code/table_values.py" "$ROOT"           # prints all table cells for cross-checking
