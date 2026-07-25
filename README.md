# Microstructure-Aware Volatility Surfaces and Cost-Efficient Delta-Hedging in Bitcoin Options

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21570772.svg)](https://doi.org/10.5281/zenodo.21570772)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproducibility repository for the paper:

> **Pricing with the Order Book: Microstructure-Aware Volatility Surfaces and Cost-Efficient Delta-Hedging in Bitcoin Options.**
> Ilya A. Kistin (MIPT) and Sh. M. Isaev. Working paper, 2026.

The pipeline studies how top-of-book market microstructure information affects two separate layers of the option-trading problem — (i) estimating the implied-volatility (IV) surface and (ii) choosing a delta-hedging rebalancing policy under realistic transaction costs — on hourly Binance BTCUSDT option snapshots (September–October 2023). A single script downloads the public data, builds four pricing models and five rebalancing policies, and regenerates every table and figure in the paper.

## Repository layout

```
microstructure-btc-options/
├── src/
│   └── rebuild_thesis_artifacts.py     # end-to-end pipeline: data → models → policies → tables/figures
├── notebooks/
│   ├── Microstructure_option_pricing.ipynb   # exploratory / full pricing + hedging pipeline
│   └── thesis_hedging_experiments.ipynb      # assembles the tables and figures used in the paper
├── build/
│   └── results/                        # generated tables (CSV / LaTeX) land here
├── assets/
│   └── figures/                        # generated figures (PNG) land here
├── requirements.txt
├── CITATION.cff
├── LICENSE                             # MIT
└── README.md
```

## Data

All inputs are **publicly available** from the Binance historical-data archive — no credentials required:

- **Options:** Binance Options *End-of-Hour Summary* (EOHSummary), underlying `BTCUSDT`,
  `https://data.binance.vision/data/option/daily/EOHSummary/BTCUSDT/`
- **Spot:** Binance Spot `BTCUSDT` 1-hour klines from the same archive.
- **Window:** 1 September – 31 October 2023 (a closed, fully archived period chosen for reproducibility).

The script fetches the daily zip files directly; nothing is committed to the repo. Of 226,039 raw snapshot rows, 90,852 valid observations survive cleaning; the chronological 70/30 split gives 63,596 train and 27,256 test observations (27,120 one-step hedging transitions).

Documentation: Binance public data repository <https://github.com/binance/binance-public-data>.

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install dependencies (Python 3.11)
pip install -r requirements.txt

# 3. reproduce all tables and figures
python src/rebuild_thesis_artifacts.py
```

Outputs are written to `build/results/` (tables) and `assets/figures/` (figures). A network connection is required on first run to download the Binance archives.

## What the pipeline produces

- **Pricing models (Table 1–2, Figures 2–3):** constant-volatility Black–Scholes, a Heston-type parametric surface, a gradient-boosting local surface, and a microstructure random forest that additionally sees bid–ask spread, volume and open interest.
- **Hedging backtest (Table 3):** one-step P&L across the three model-driven hedgers under an explicit cost model (exchange fee + slippage + spread-crossing), at base costs of 0.5 / 1.0 / 2.0 bps.
- **Rebalancing policies (Table 4, Figures 4–6):** time-based, Leland, delta-threshold, no-trade band, and the proposed microstructure-adaptive threshold rule.

## Software

Python 3.11 with NumPy, pandas, SciPy, scikit-learn, Matplotlib, requests and tqdm (see `requirements.txt`). Random Forest and Gradient Boosting are from scikit-learn; Black–Scholes IV inversion uses SciPy's Brent solver.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code or the derived pipeline.

## License

Code is released under the MIT License (see `LICENSE`). The underlying market data are © Binance and are subject to Binance's terms of use; this repository redistributes none of it.
