# Two-level value of market microstructure information: IV estimation and delta hedging of Bitcoin options

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, derived results and figures for the article

> Kistin I.A., Isaev Sh.M. **Двухуровневая оценка ценности микроструктурной информации при оценивании
> подразумеваемой волатильности и дельта-хеджировании опционов на биткоин** (Two-Level Assessment of the Value of
> Market Microstructure Information for Implied Volatility Estimation and Delta-Hedging of Bitcoin Options). Manuscript, 2026.

Authors: Ilya A. Kistin (Moscow Institute of Physics and Technology; Tsentralny University), ORCID
[0009-0001-8300-6461](https://orcid.org/0009-0001-8300-6461); Sharip M. Isaev (ITMO University), ORCID
[0009-0006-8141-5110](https://orcid.org/0009-0006-8141-5110).

Release **v2.0.0** corresponds to the article. Every number, table and figure of the article is produced by the
scripts in `code/`; the committed outputs in `data/` and `figs/` are the ones reported in the article.

## What the study does

The same market-state information is added (i) to the implied-volatility (IV) estimator and (ii) to the hedge
rebalancing rule, and its value is measured with common risk and cost metrics:

* online loop at hour *t*: input data → estimation (σ̂ₜ, Δₜ, Γₜ) → decision (φₜ, uₜ) → execution in the spot market
  (Fig. 1, eqs. 1–3 of the article);
* offline evaluation on the test sample: IV estimation error, step result ℓₜ₊₁, metrics *J* = (RMSE, VaR95, ES95,
  costs) and the value of information ΔJᴱ, ΔJᴰ (eqs. 4–6).

Estimation models: constant volatility, parametric surface, gradient boosting and random forest on contract geometry,
random forest on the full state (geometry + quote state + realized volatility); a forest with same-quote fields is
reported only as a leakage diagnostic. Rebalancing rules: hourly, Leland, fixed threshold, fixed band,
Whalley–Wilmott band, adaptive threshold (plus a post-hoc adaptive band).

## Data

Public Binance archives (no credentials), not redistributed here:

* Binance Options EOHSummary, BTCUSDT, daily files 2023-09-01 … 2023-10-23
  (`https://data.binance.vision/data/option/daily/EOHSummary/BTCUSDT/`); 41 files exist in this window
  (8–18 and 25 September are missing from the archive);
* Binance Spot BTCUSDT 1h klines, monthly files 2023-08 … 2023-10
  (`https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/`);
* Binance Spot BTCUSDT daily aggTrades (only for the spot-spread proxy, see below).

`code/download_data.py` downloads the files and checks them against `data/input_manifest.csv` (SHA-256).
After filtering: 91,314 observations of 308 contracts; chronological split into training (51,084, up to
2023-10-04 16:00 UTC), validation (12,822, up to 2023-10-11 15:00 UTC) and test (27,408, up to 2023-10-24 00:00 UTC)
parts; the test hedging backtest has 217 trajectories and 27,191 steps.

Derived data committed to the repository:

| File | Content |
|---|---|
| `data/counts.json` | filtering counts and split boundaries |
| `data/spot_spread.parquet` | hourly spot spread proxy (median of opposite-side aggTrades pairs, last 5 minutes of each hour) |
| `data/estimation_results.json` | Table 1 (mark IV), regimes, quartile edges |
| `data/estimation_mid_results.json` | Table 1 (IV inverted from the mid quote) |
| `data/backtest_results.json` | Tables 2–3, Pareto fronts, bootstrap, regimes |
| `data/extra_results.json` | feature-group ablation, hourly hedging by model, post-hoc adaptive band |
| `data/hedge_quote_iv.json` | hedging with quote-IV deltas (mid-quote option leg) |
| `data/input_manifest.csv` | SHA-256 of all raw input files |

## Reproduce

```bash
pip install -r requirements.txt          # Python 3.13; versions as used for the article
bash code/run_all.sh .                   # download + checksums, then every stage below
```

| Step | Script | Output |
|---|---|---|
| 1 | `download_data.py` | `binance/` (raw zips), checksum check |
| 2 | `data_prep.py` | `data/panel.parquet`, `data/spot.parquet`, `data/counts.json` |
| 3 | `spot_spread.py` | `data/spot_spread.parquet` (skipped if present; recomputing downloads several GB of aggTrades) |
| 4 | `estimation.py`, `estimation_mid.py` | Table 1, regimes |
| 5 | `backtest.py`, `extra.py`, `hedge_quote_iv.py` | Tables 2–3, Fig. 4 data, robustness |
| 6 | `figures.py` (+ `fig_scheme.py`) | `figs/Fig1`–`Fig4` (`.png` 600 dpi, `.pdf`; Fig. 1 also `.svg`) |
| 7 | `table_values.py` | prints every table cell for cross-checking |

A clean run of `run_all.sh` (download included, precomputed spot spread) took about 10 minutes on a 2-core machine
and reproduced all committed JSON results exactly. Fixed seeds: models 42; bootstrap 42 (estimation, hedging), 11
(quote IV), 7 (ablations), 5 (quote-IV hedging). `supplementary/table_values.txt` is the output of step 7.

## Design notes

* **Time alignment.** An EOHSummary row with `hour = h` is the end-of-hour snapshot at (h+1):00 UTC and is matched to
  the close of the spot candle opened at h; the underlying price is the spot close; r = 0 reproduces mark prices.
* **Leakage control.** Features exclude bid/ask/mark prices of the same contract, best bid/ask IV and exchange Greeks;
  lags and realized volatility use data dated t or earlier; normalisations, medians and quartile edges use only the
  data available for training (for the test: training + validation parts).
* **Costs.** ℓₜ₊₁ = (Cₜ₊₁ − Cₜ) − φₜ(Sₜ₊₁ − Sₜ) − cₜ|uₜ|Sₜ with cₜ = c_f + ½·(spot relative spread); the option leg is
  valued at the exchange mark price (mid-quote valuation is a robustness check); c_f = 2.5 bp in the main
  specification, 1 / 5 / 10 bp as robustness.
* **Tuning.** Rule parameters are chosen on the validation part by minimum RMSE subject to at most 10 % trading steps;
  test Pareto fronts are descriptive. Confidence intervals: cluster bootstrap over contracts, 1000 replications.

## Repository layout

```
code/            pipeline scripts (see the table above) and run_all.sh
data/            derived results (JSON), spot-spread proxy, input checksums
figs/            Fig1–Fig4 of the article (PNG 600 dpi, PDF; Fig1 also SVG)
supplementary/   table_values.txt (all table cells printed by code/table_values.py)
CITATION.cff     citation metadata
LICENSE          MIT
```

## Version history

* **v2.0.0** (2026-10-23) — matches the article: corrected time alignment and spot underlying, leakage-controlled
  features, spot-spread cost model, validation-based tuning, Whalley–Wilmott benchmark, bootstrap inference,
  two IV targets (mark IV and quote IV).
* **v1.0.0** (2026-07-25) — earlier pipeline, superseded. It used the option mid quote as the underlying price, a
  70/30 split and the option spread as a cost proxy for the spot hedge; its numbers do not correspond to the article.

## Citation and license

Please cite the article (see `CITATION.cff`). Code: MIT License. Market data © Binance, subject to Binance's terms of
use; no raw data are redistributed in this repository.
