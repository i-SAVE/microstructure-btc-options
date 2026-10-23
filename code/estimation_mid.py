"""Estimation layer, second target: quote-implied IV (Black-Scholes IV inverted from the mid quote).
Defined only where the mid lies inside the no-arbitrage bounds of the BS price (invertible rows).
Same models, same information sets, same chronological split; cluster bootstrap over contracts."""
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
p = pd.read_parquet(os.path.join(D, 'panel.parquet'))
GEO = ['k', 'T', 'call']
MICRO = GEO + ['s_rel', 's_rel_lag1', 'log_vol', 'log_vol_lag1', 'log_oi', 'log_bq', 'log_aq', 'RV6']
LEAK = GEO + ['best_ask_price', 'best_ask_qty', 'best_bid_qty', 'best_bid_price', 'delta', 'gamma', 'oi', 'vega',
              'volume_contracts', 'volume_usdt']
SPREAD = GEO + ['s_rel', 's_rel_lag1']


def bsp(S, K, T, s, call):
    s = np.maximum(s, 1e-6)
    d1 = (np.log(S / K) + 0.5 * s * s * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    return np.where(call == 1, S * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - S * norm.cdf(-d1))


S, K, T, c, m = p.S.values, p.K.values, p['T'].values, p.call.values, p.mid.values
lo = np.full(len(p), 1e-4)
hi = np.full(len(p), 3.0)
ok = ((bsp(S, K, T, lo, c) - m) < 0) & ((bsp(S, K, T, hi, c) - m) > 0)
for _ in range(60):
    md = 0.5 * (lo + hi)
    f = bsp(S, K, T, md, c) - m
    lo = np.where(f < 0, md, lo)
    hi = np.where(f >= 0, md, hi)
p['iv_mid'] = np.where(ok, 0.5 * (lo + hi), np.nan)
q = p[np.isfinite(p.iv_mid)].copy()
res = {'n_invertible': int(len(q)), 'n_total': int(len(p)),
       'n_train': int((q.split != 'test').sum()), 'n_test': int((q.split == 'test').sum())}
tr, te = q[q.split != 'test'].reset_index(drop=True), q[q.split == 'test'].reset_index(drop=True)
y, yt = tr.iv_mid.values, te.iv_mid.values
pred = {}
sig = minimize_scalar(lambda s: np.mean((bsp(tr.S.values, tr.K.values, tr['T'].values, s, tr.call.values) - tr.mid.values) ** 2),
                      bounds=(0.05, 3), method='bounded').x
pred['BS'] = (np.full(len(tr), sig), np.full(len(te), sig))
Xp = lambda d: np.column_stack([np.ones(len(d)), d['T'], np.abs(d.k), d.k ** 2, d['T'] * d.k ** 2])
b, *_ = np.linalg.lstsq(Xp(tr), y ** 2, rcond=None)
pred['PARAM'] = (np.sqrt(np.maximum(Xp(tr) @ b, 1e-6)), np.sqrt(np.maximum(Xp(te) @ b, 1e-6)))
gb = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, min_samples_leaf=20, random_state=42).fit(tr[GEO], y)
pred['GB_GEO'] = (gb.predict(tr[GEO]), gb.predict(te[GEO]))
for name, cols in [('RF_GEO', GEO), ('RF_MICRO', MICRO), ('RF_GEO_SPREAD', SPREAD), ('RF_LEAK', LEAK)]:
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=20, random_state=42, n_jobs=-1).fit(tr[cols], y)
    pred[name] = (rf.predict(tr[cols]), rf.predict(te[cols]))
tab = {}
for n, (a, bb) in pred.items():
    tab[n] = {'rmse_train': float(np.sqrt(np.mean((a - y) ** 2))), 'rmse_test': float(np.sqrt(np.mean((bb - yt) ** 2)))}
codes = te.symbol.astype('category').cat.codes.values
G = codes.max() + 1
cnt = np.bincount(codes, minlength=G)
ss = {n: np.bincount(codes, weights=(v[1] - yt) ** 2, minlength=G) for n, v in pred.items()}
rng = np.random.default_rng(11)
Rb = {n: [] for n in pred}
for _ in range(1000):
    idx = rng.integers(0, G, G)
    cc = cnt[idx].sum()
    for n in pred:
        Rb[n].append(np.sqrt(ss[n][idx].sum() / cc))
Rb = {n: np.array(v) for n, v in Rb.items()}
for n in pred:
    tab[n]['ci_test'] = [float(np.percentile(Rb[n], 2.5)), float(np.percentile(Rb[n], 97.5))]
for a_, b_ in [('RF_GEO', 'RF_MICRO'), ('GB_GEO', 'RF_MICRO'), ('RF_GEO', 'RF_GEO_SPREAD'), ('RF_MICRO', 'RF_LEAK')]:
    d = Rb[a_] - Rb[b_]
    tab[f'diff_{a_}_minus_{b_}'] = {'point': tab[a_]['rmse_test'] - tab[b_]['rmse_test'],
                                    'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
edges = np.quantile(p[p.split != 'test'].s_rel.values, [0.25, 0.5, 0.75])
qs = np.searchsorted(edges, te.s_rel.values, side='right')
tab['by_spread_q'] = {n: [float(np.sqrt(np.mean((pred[n][1][qs == i] - yt[qs == i]) ** 2))) for i in range(4)]
                      for n in ['PARAM', 'GB_GEO', 'RF_GEO', 'RF_MICRO']}
tab['by_spread_q_n'] = [int((qs == i).sum()) for i in range(4)]
res['estimation'] = tab
res['iv_mid_vs_mark'] = {'median_abs_diff': float(np.median(np.abs(q.iv_mid - q.iv))),
                         'p95_abs_diff': float(np.quantile(np.abs(q.iv_mid - q.iv), 0.95)),
                         'rmse_diff': float(np.sqrt(np.mean((q.iv_mid - q.iv) ** 2)))}
out = te[['symbol', 'time']].copy()
for n, (a, bb) in pred.items():
    out['ivq_' + n] = bb
out.to_parquet(os.path.join(D, 'preds_test_mid.parquet'), index=False)
json.dump(res, open(os.path.join(D, 'estimation_mid_results.json'), 'w'), indent=1)
for n, v in tab.items():
    print(n, v)
print(res['iv_mid_vs_mark'], res['n_invertible'])
