"""Additional analyses: (1) bootstrap CIs for feature-group ablations; (2) hourly-hedging differences across IV models;
(3) post-hoc adaptive no-trade band (labelled exploratory); (4) adaptive-threshold variant with spread term forced."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
sys.argv = [sys.argv[0], ROOT]
sys.path.insert(0, os.path.join(ROOT, 'code'))
import backtest as bt  # noqa: E402

D = os.path.join(ROOT, 'data')
panel = pd.read_parquet(os.path.join(D, 'panel.parquet'))
te = panel[panel.split == 'test'].reset_index(drop=True)
P = pd.read_parquet(os.path.join(D, 'preds_test.parquet'))
te = te.merge(P, on=['symbol', 'time'])
out = {}

# (1) estimation ablations: paired cluster bootstrap over contracts
names = ['RF_GEO', 'RF_GEO_RV', 'RF_GEO_LIQ', 'RF_GEO_SPREAD', 'RF_MICRO_NOSPREAD', 'RF_MICRO', 'GB_GEO', 'RF_LEAK']
codes = te.symbol.astype('category').cat.codes.values
G = codes.max() + 1
cnt = np.bincount(codes, minlength=G)
ss = {n: np.bincount(codes, weights=(te['iv_' + n] - te.iv) ** 2, minlength=G) for n in names}
rng = np.random.default_rng(7)
Rb = {n: [] for n in names}
for _ in range(1000):
    idx = rng.integers(0, G, G)
    c = cnt[idx].sum()
    for n in names:
        Rb[n].append(np.sqrt(ss[n][idx].sum() / c))
Rb = {n: np.array(v) for n, v in Rb.items()}
abl = {}
for n in names:
    if n == 'RF_GEO':
        continue
    d = Rb['RF_GEO'] - Rb[n]
    abl[n] = {'rmse': float(np.sqrt(ss[n].sum() / cnt.sum())), 'gain_vs_RF_GEO': float(np.sqrt(ss['RF_GEO'].sum() / cnt.sum()) - np.sqrt(ss[n].sum() / cnt.sum())),
              'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
out['ablation'] = abl

# (2) hourly hedging by IV model: paired bootstrap of RMSE and VaR proxy differences vs BS
Xt = bt.prepare('test')
Xv = bt.prepare('val')
outs = {m: bt.run_policy(Xt, 'hourly', {}, bt.MAIN_FEE, model=m) for m in bt.MODELS}
sym = outs['BS']['symbol']
cc = pd.Series(sym).astype('category').cat.codes.values
G2 = cc.max() + 1
cnt2 = np.bincount(cc, minlength=G2)
s2 = {m: np.bincount(cc, weights=outs[m]['pnl'] ** 2, minlength=G2) for m in bt.MODELS}
R2 = {m: [] for m in bt.MODELS}
for _ in range(1000):
    idx = rng.integers(0, G2, G2)
    c = cnt2[idx].sum()
    for m in bt.MODELS:
        R2[m].append(np.sqrt(s2[m][idx].sum() / c))
hed = {}
for m in bt.MODELS:
    d = np.array(R2['BS']) - np.array(R2[m])
    hed[m] = {'rmse': float(np.sqrt(np.mean(outs[m]['pnl'] ** 2))), 'gain_vs_BS': float(np.sqrt(np.mean(outs['BS']['pnl'] ** 2)) - np.sqrt(np.mean(outs[m]['pnl'] ** 2))),
              'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
              'mean_abs_delta_diff_vs_RF_MICRO': float(np.mean(np.abs(Xt['delta_' + m] - Xt['delta_RF_MICRO'])))}
out['hourly_models'] = hed


# (3) post-hoc adaptive band: band rule (trade to edge) with state-dependent half-width
def run_adaptive_band(X, prm, fee):
    med = X['med']
    m = ((X['s_rel'] / med['s_rel']) ** prm['a'] * (med['oi'] / np.maximum(X['oi'], 1.0)) ** prm['b']
         * (X['RV6'] / med['RV6']) ** prm['c'])
    width = np.clip(prm['h0'] * m, prm['h0'] / 4.0, 8.0 * prm['h0'])
    c = fee * 1e-4 + 0.5 * X['sp']
    delta = X['delta_' + bt.POLICY_MODEL]
    phi, prev, trade = bt.simulate(2, delta, width, X['newtraj'], X['hasnext'])
    hn = X['hasnext']
    S = X['S']
    dS = np.r_[S[1:] - S[:-1], 0.0]
    cost = c * np.abs(phi - prev) * S
    dCm = np.r_[X['Cmark'][1:] - X['Cmark'][:-1], 0.0]
    dCmid = np.r_[X['C'][1:] - X['C'][:-1], 0.0]
    gross = dCm - phi * dS
    srel_next = np.r_[X['s_rel'][1:], np.inf]
    midok = (X['s_rel'] <= 0.5) & (srel_next <= 0.5)
    return {'pnl': (gross - cost)[hn], 'gross': gross[hn], 'cost': cost[hn], 'trade': trade[hn],
            'turn': np.abs(phi - prev)[hn], 'pnl_mid': (dCmid - phi * dS - cost)[hn], 'midok': midok[hn],
            'symbol': X['symbol'][hn], 'width': width}


grid = [{'h0': h, 'a': a, 'b': b, 'c': c} for h in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
        for a in [0.0, 0.5, 1.0] for b in [0.0, 0.25, 0.5] for c in [0.0, 0.5, 1.0]]
best = None
for prm in grid:
    mv = bt.metrics(run_adaptive_band(Xv, prm, bt.MAIN_FEE))
    if mv['share'] <= bt.CAP and (best is None or mv['rmse'] < best[1]['rmse']):
        best = (prm, mv)
ab_test = run_adaptive_band(Xt, best[0], bt.MAIN_FEE)
out['adaptive_band_posthoc'] = {'params': best[0], 'val': best[1], 'test': bt.metrics(ab_test)}
# frontier of the adaptive band with the selected elasticities
out['adaptive_band_frontier'] = [{'h0': h, 'test': bt.metrics(run_adaptive_band(Xt, dict(best[0], h0=h), bt.MAIN_FEE))}
                                 for h in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2]]
# paired bootstrap: adaptive band vs fixed band and WW (selected configs)
sel = json.load(open(os.path.join(D, 'backtest_results.json')))['policies'][str(bt.MAIN_FEE)]
o_band = bt.run_policy(Xt, 'band', sel['band']['params'], bt.MAIN_FEE)
o_ww = bt.run_policy(Xt, 'ww', sel['ww']['params'], bt.MAIN_FEE)
cc = pd.Series(ab_test['symbol']).astype('category').cat.codes.values
G3 = cc.max() + 1
cnt3 = np.bincount(cc, minlength=G3)
ser = {'ab': ab_test, 'band': o_band, 'ww': o_ww}
s3 = {k: np.bincount(cc, weights=v['pnl'] ** 2, minlength=G3) for k, v in ser.items()}
c3 = {k: np.bincount(cc, weights=v['cost'], minlength=G3) for k, v in ser.items()}
D3 = {k: ([], []) for k in ['band', 'ww']}
for _ in range(1000):
    idx = rng.integers(0, G3, G3)
    c = cnt3[idx].sum()
    rab = np.sqrt(s3['ab'][idx].sum() / c)
    for k in ['band', 'ww']:
        D3[k][0].append(rab - np.sqrt(s3[k][idx].sum() / c))
        D3[k][1].append(c3['ab'][idx].sum() / c3[k][idx].sum())
out['adaptive_band_vs'] = {k: {'rmse_diff_ci': [float(np.percentile(v[0], 2.5)), float(np.percentile(v[0], 97.5))],
                               'cost_ratio_ci': [float(np.percentile(v[1], 2.5)), float(np.percentile(v[1], 97.5))]}
                           for k, v in D3.items()}
json.dump(out, open(os.path.join(D, 'extra_results.json'), 'w'), indent=1, default=float)
print(json.dumps(out['ablation'], indent=1))
print(json.dumps(out['hourly_models'], indent=1))
print(json.dumps({k: out[k] for k in ['adaptive_band_posthoc', 'adaptive_band_vs']}, indent=1, default=float)[:2500])
for r in out['adaptive_band_frontier']:
    t = r['test']
    print(r['h0'], round(t['share'], 4), round(t['rmse'], 3), round(t['cost']))
