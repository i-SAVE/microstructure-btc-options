"""Estimation layer: IV models with an explicit, leakage-controlled information set.

Models (all predict exchange mark IV; prices/Greeks via Black-Scholes on spot, r=0):
  BS      - one constant volatility, least-squares on training mid prices
  PARAM   - parametric surface: OLS of implied variance on (1, T, |k|, k^2, T k^2)
  GB_GEO  - gradient boosting on geometry (k, T, call)
  RF_GEO  - random forest on geometry (k, T, call)          [same learner as RF_MICRO]
  RF_MICRO- random forest on geometry + non-price state (spread, volume, OI, best bid/ask quantities, RV, lags of spread and volume)
Diagnostics (not candidate models):
  RF_LEAK - the 13 automatically selected columns of the earlier public script (contains same-contract
            bid/ask price levels and exchange Greeks computed from the target mark IV)
  ablations of RF_MICRO by feature group
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

ROOT = sys.argv[1]
D = os.path.join(ROOT, 'data')
panel = pd.read_parquet(os.path.join(D, 'panel.parquet'))
SEED = 42
B = 1000

GEO = ['k', 'T', 'call']
MICRO = GEO + ['s_rel', 's_rel_lag1', 'log_vol', 'log_vol_lag1', 'log_oi', 'log_bq', 'log_aq', 'RV6']
LEAK = GEO + ['best_ask_price', 'best_ask_qty', 'best_bid_qty', 'best_bid_price', 'delta', 'gamma',
              'openinterest_contracts_raw', 'vega', 'volume_contracts', 'volume_usdt']
panel['openinterest_contracts_raw'] = panel['oi']
ABL = {
    'RF_GEO_RV': GEO + ['RV6'],
    'RF_GEO_LIQ': GEO + ['log_vol', 'log_vol_lag1', 'log_oi', 'log_bq', 'log_aq'],
    'RF_GEO_SPREAD': GEO + ['s_rel', 's_rel_lag1'],
    'RF_MICRO_NOSPREAD': [c for c in MICRO if not c.startswith('s_rel')],
}


def bs_price(S, K, T, s, call):
    s = np.maximum(s, 1e-4)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * s * s * T) / (s * sq)
    d2 = d1 - s * sq
    return np.where(call == 1, S * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(S, K, T, s, call):
    s = np.maximum(s, 1e-4)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * s * s * T) / (s * sq)
    delta = np.where(call == 1, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = norm.pdf(d1) / (S * s * sq)
    vega = S * norm.pdf(d1) * sq
    return delta, gamma, vega


def fit_predict(train, test, seed=SEED):
    out = {}
    # BS constant
    def obj(s):
        return np.mean((bs_price(train.S.values, train.K.values, train['T'].values, np.full(len(train), s),
                                 train.call.values) - train.mid.values) ** 2)
    sig = minimize_scalar(obj, bounds=(0.05, 3.0), method='bounded').x
    out['BS'] = (np.full(len(train), sig), np.full(len(test), sig))
    # parametric surface
    def X(d):
        k = d.k.values
        T = d['T'].values
        return np.column_stack([np.ones_like(k), T, np.abs(k), k ** 2, T * k ** 2])
    beta, *_ = np.linalg.lstsq(X(train), train.iv.values ** 2, rcond=None)
    out['PARAM'] = (np.sqrt(np.maximum(X(train) @ beta, 1e-6)), np.sqrt(np.maximum(X(test) @ beta, 1e-6)))
    # GB geometry
    gb = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, min_samples_leaf=20,
                                   random_state=seed).fit(train[GEO], train.iv)
    out['GB_GEO'] = (gb.predict(train[GEO]), gb.predict(test[GEO]))
    # RFs
    def rf(cols):
        m = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=20, random_state=seed,
                                  n_jobs=-1).fit(train[cols], train.iv)
        return m.predict(train[cols]), m.predict(test[cols])
    out['RF_GEO'] = rf(GEO)
    out['RF_MICRO'] = rf(MICRO)
    out['RF_LEAK'] = rf(LEAK)
    for name, cols in ABL.items():
        out[name] = rf(cols)
    return out, sig, beta


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cluster_boot(err2_by_obs, groups, B=B, seed=SEED):
    """Bootstrap RMSE over clusters. err2_by_obs: dict name->squared errors."""
    g = pd.Series(groups).astype('category').cat.codes.values
    G = g.max() + 1
    cnt = np.bincount(g, minlength=G)
    sums = {n: np.bincount(g, weights=e, minlength=G) for n, e in err2_by_obs.items()}
    rng = np.random.default_rng(seed)
    res = {n: [] for n in sums}
    for _ in range(B):
        idx = rng.integers(0, G, G)
        c = cnt[idx].sum()
        for n in sums:
            res[n].append(np.sqrt(sums[n][idx].sum() / c))
    return {n: np.array(v) for n, v in res.items()}


results = {}
preds = {}
for stage, (tr_mask, te_mask) in {'val': (panel.split == 'fit', panel.split == 'val'),
                                   'test': (panel.split != 'test', panel.split == 'test')}.items():
    tr = panel[tr_mask].reset_index(drop=True)
    te = panel[te_mask].reset_index(drop=True)
    out, sig, beta = fit_predict(tr, te)
    P = te[['symbol', 'time']].copy()
    for n, (ptr, pte) in out.items():
        P['iv_' + n] = pte
    preds[stage] = P
    P.to_parquet(os.path.join(D, f'preds_{stage}.parquet'), index=False)
    if stage == 'test':
        results['bs_sigma'] = float(sig)
        results['param_beta'] = [float(b) for b in beta]
        tab = {}
        e2 = {}
        for n, (ptr, pte) in out.items():
            pr = bs_price(te.S.values, te.K.values, te['T'].values, pte, te.call.values)
            _, _, veg = bs_greeks(te.S.values, te.K.values, te['T'].values, te.iv.values, te.call.values)
            vw = float(np.sqrt(np.sum(veg * (pr - te.mid.values) ** 2) / np.sum(veg)))
            tab[n] = {'rmse_train': rmse(ptr, tr.iv.values), 'rmse_test': rmse(pte, te.iv.values),
                      'vega_w_price_rmse': vw}
            e2[n] = (pte - te.iv.values) ** 2
        bt = cluster_boot(e2, te.symbol.values)
        for n in tab:
            tab[n]['ci_test'] = [float(np.percentile(bt[n], 2.5)), float(np.percentile(bt[n], 97.5))]
        for a, b in [('RF_GEO', 'RF_MICRO'), ('GB_GEO', 'RF_MICRO'), ('PARAM', 'RF_MICRO'), ('BS', 'RF_MICRO'),
                     ('RF_MICRO', 'RF_LEAK')]:
            d = bt[a] - bt[b]
            tab[f'diff_{a}_minus_{b}'] = {'point': tab[a]['rmse_test'] - tab[b]['rmse_test'],
                                          'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        results['estimation'] = tab
        # regimes: quartile edges from the training window
        reg = {}
        for var in ['s_rel', 'oi']:
            edges = np.quantile(tr[var].values, [0.25, 0.5, 0.75])
            q = np.searchsorted(edges, te[var].values, side='right')
            reg[var] = {'edges': [float(x) for x in edges], 'n': [int((q == i).sum()) for i in range(4)]}
            for n in ['BS', 'PARAM', 'GB_GEO', 'RF_GEO', 'RF_MICRO']:
                reg[var][n] = [rmse(out[n][1][q == i], te.iv.values[q == i]) for i in range(4)]
        # 2-D grid spread x OI: RF_GEO - RF_MICRO
        e_s = np.quantile(tr['s_rel'].values, [0.25, 0.5, 0.75])
        e_o = np.quantile(tr['oi'].values, [0.25, 0.5, 0.75])
        qs = np.searchsorted(e_s, te['s_rel'].values, side='right')
        qo = np.searchsorted(e_o, te['oi'].values, side='right')
        grid = np.full((4, 4), np.nan)
        ncell = np.zeros((4, 4), int)
        for i in range(4):
            for j in range(4):
                m = (qs == i) & (qo == j)
                ncell[j, i] = m.sum()
                if m.sum() >= 30:
                    grid[j, i] = rmse(out['RF_GEO'][1][m], te.iv.values[m]) - rmse(out['RF_MICRO'][1][m], te.iv.values[m])
        reg['grid_gain_rfgeo_minus_rfmicro'] = grid.tolist()
        reg['grid_n'] = ncell.tolist()
        results['regimes'] = reg
        results['test_iv_summary'] = {'median_iv': float(te.iv.median()), 'iv_p05': float(te.iv.quantile(.05)),
                                      'iv_p95': float(te.iv.quantile(.95)),
                                      'median_s_rel': float(te.s_rel.median()),
                                      'train_median_s_rel': float(tr.s_rel.median()),
                                      'train_median_oi': float(tr.oi.median()),
                                      'train_median_rv6': float(tr.RV6.median())}
json.dump(results, open(os.path.join(D, 'estimation_results.json'), 'w'), indent=1)
print(json.dumps(results['estimation'], indent=1))
print(json.dumps({k: v for k, v in results['regimes'].items() if k in ('s_rel', 'oi')}, indent=1))
