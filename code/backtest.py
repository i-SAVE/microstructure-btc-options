"""Decision layer: unified hedging backtest (one P&L definition, one cost formula for every table).

System chain per contract trajectory (consecutive hourly snapshots of one option), eqs. (2)-(4) of the article:
  sigma_hat_t = f(x_t^E); Delta_t, Gamma_t from Black-Scholes on spot (r = 0)
  phi_t = pi(sigma_hat_t, Delta_t, Gamma_t, phi_{t-1}, x_t^D); u_t = phi_t - phi_{t-1}
  l_{t+1} = (C_{t+1}-C_t) - phi_t (S_{t+1}-S_t) - c_t |u_t| S_t
Option leg: exchange mark price (robustness: mid quote, relative spread <= 50 % at t and t+1).
Hedge instrument: BTCUSDT spot.
Cost: c_t = c_fee + 0.5 * s^spot_t  (s^spot from aggTrades; c_fee scenarios 1, 2.5, 5, 10 bps).
The initial hedge at the start of a trajectory is established without cost (common to all policies).
"""
import itertools
import json
import os
import sys

import numba
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = sys.argv[1]
D = os.path.join(ROOT, 'data')
panel = pd.read_parquet(os.path.join(D, 'panel.parquet'))
spread = pd.read_parquet(os.path.join(D, 'spot_spread.parquet'))
MODELS = ['BS', 'PARAM', 'GB_GEO', 'RF_GEO', 'RF_MICRO']
POLICY_MODEL = 'RF_MICRO'
FEES = [1.0, 2.5, 5.0, 10.0]
MAIN_FEE = 2.5
CAP = 0.10             # max share of trading steps for tuned policies (identical for all families)
DT = 1.0 / (24 * 365)
B = 1000
SEED = 42

GRID = {
    'threshold': [{'theta': t} for t in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4]],
    'band': [{'h': h} for h in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]],
    'ww': [{'gamma': g} for g in np.geomspace(1e-7, 1e-1, 25)],
    'adaptive': [{'theta0': t0, 'a': a, 'b': b, 'c': c}
                 for t0 in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2]
                 for a in [0.0, 0.5, 1.0] for b in [0.0, 0.25, 0.5] for c in [0.0, 0.5, 1.0]],
}


def bs_delta_gamma(S, K, T, s, call):
    s = np.maximum(s, 1e-4)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * s * s * T) / (s * sq)
    delta = np.where(call == 1, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = norm.pdf(d1) / (S * s * sq)
    return delta, gamma


@numba.njit(cache=True)
def simulate(mode, target, width, newtraj, hasnext):
    """mode 0: follow target; 1: threshold (full rebalance); 2: band (to nearest edge)."""
    n = target.shape[0]
    phi = np.empty(n)
    prev = np.empty(n)
    trade = np.zeros(n, dtype=np.bool_)
    p = 0.0
    for t in range(n):
        if newtraj[t]:
            p = target[t]
            phi[t] = p
            prev[t] = p
            continue
        prev[t] = p
        if not hasnext[t]:
            phi[t] = p
            continue
        g = target[t] - p
        if mode == 0:
            p_new = target[t]
        elif mode == 1:
            p_new = target[t] if abs(g) > width[t] else p
        else:
            if abs(g) > width[t]:
                p_new = target[t] - width[t] * np.sign(g)
            else:
                p_new = p
        if p_new != p:
            trade[t] = True
        p = p_new
        phi[t] = p
    return phi, prev, trade


def prepare(stage):
    df = panel[panel.split == stage].copy()
    preds = pd.read_parquet(os.path.join(D, f'preds_{stage}.parquet'))
    df = df.merge(preds, on=['symbol', 'time'], how='inner')
    df = df.sort_values(['symbol', 'time']).reset_index(drop=True)
    dt_prev = df.groupby('symbol')['time'].diff()
    newtraj = (df['symbol'] != df['symbol'].shift(1)) | (dt_prev != pd.Timedelta('1h'))
    traj = np.cumsum(newtraj.values)
    hasnext = np.r_[~newtraj.values[1:], False]
    sp = spread['s_spot_rel'].reindex(df['time']).values
    sp = np.where(np.isnan(sp), np.nanmedian(spread['s_spot_rel'].values), sp)
    ref = panel[panel.split == 'fit'] if stage == 'val' else panel[panel.split != 'test']
    med = {'s_rel': float(ref.s_rel.median()), 'oi': float(ref.oi.median()), 'RV6': float(ref.RV6.median())}
    X = {'S': df.S.values, 'C': df.mid.values, 'Cmark': df.mark_price.values, 'newtraj': newtraj.values,
         'hasnext': hasnext, 'traj': traj, 'symbol': df.symbol.values, 's_rel': df.s_rel.values,
         'oi': df.oi.values, 'RV6': df.RV6.values, 'sp': sp, 'time': df.time.values, 'med': med,
         'K': df.K.values, 'T': df['T'].values, 'call': df.call.values}
    for m in MODELS:
        d, g = bs_delta_gamma(df.S.values, df.K.values, df['T'].values, df['iv_' + m].values, df.call.values)
        X['delta_' + m] = d
        X['gamma_' + m] = g
        X['iv_' + m] = df['iv_' + m].values
    return X


def run_policy(X, family, prm, fee_bps, model=POLICY_MODEL):
    c = fee_bps * 1e-4 + 0.5 * X['sp']
    delta = X['delta_' + model]
    n = len(delta)
    if family == 'hourly':
        mode, target, width = 0, delta, np.zeros(n)
    elif family == 'leland':
        s = X['iv_' + model]
        sL = s * np.sqrt(1.0 + np.sqrt(2.0 / np.pi) * (2.0 * c) / (s * np.sqrt(DT)))
        target, _ = bs_delta_gamma(X['S'], X['K'], X['T'], sL, X['call'])
        mode, width = 0, np.zeros(n)
    elif family == 'threshold':
        mode, target, width = 1, delta, np.full(n, prm['theta'])
    elif family == 'band':
        mode, target, width = 2, delta, np.full(n, prm['h'])
    elif family == 'ww':
        G = X['gamma_' + model]
        width = (1.5 * c * X['S'] * G ** 2 / prm['gamma']) ** (1.0 / 3.0)
        mode, target = 2, delta
    elif family == 'adaptive':
        med = X['med']
        m = ((X['s_rel'] / med['s_rel']) ** prm['a'] * (med['oi'] / np.maximum(X['oi'], 1.0)) ** prm['b']
             * (X['RV6'] / med['RV6']) ** prm['c'])
        width = np.clip(prm['theta0'] * m, prm['theta0'] / 4.0, 8.0 * prm['theta0'])
        mode, target = 1, delta
    else:
        raise ValueError(family)
    phi, prev, trade = simulate(mode, target, width, X['newtraj'], X['hasnext'])
    hn = X['hasnext']
    S = X['S']
    dS = np.r_[S[1:] - S[:-1], 0.0]
    cost = c * np.abs(phi - prev) * S
    dCmark = np.r_[X['Cmark'][1:] - X['Cmark'][:-1], 0.0]      # primary: exchange mark price
    dCmid = np.r_[X['C'][1:] - X['C'][:-1], 0.0]               # robustness: mid quote
    gross = dCmark - phi * dS
    pnl = gross - cost
    srel_next = np.r_[X['s_rel'][1:], np.inf]
    midok = (X['s_rel'] <= 0.5) & (srel_next <= 0.5)
    out = {'pnl': pnl[hn], 'gross': gross[hn], 'cost': cost[hn], 'trade': trade[hn], 'turn': np.abs(phi - prev)[hn],
           'phi': phi, 'width': width, 'symbol': X['symbol'][hn], 'idx': np.where(hn)[0],
           'pnl_mid': (dCmid - phi * dS - cost)[hn], 'midok': midok[hn]}
    return out


def metrics(o):
    p = o['pnl']
    return {'rmse': float(np.sqrt(np.mean(p ** 2))), 'var95': float(-np.quantile(p, 0.05)),
            'es95': float(-p[p <= np.quantile(p, 0.05)].mean()), 'mean': float(p.mean()),
            'gross_rmse': float(np.sqrt(np.mean(o['gross'] ** 2))), 'trades': int(o['trade'].sum()),
            'share': float(o['trade'].mean()), 'turnover': float(o['turn'].sum()), 'cost': float(o['cost'].sum()),
            'steps': int(len(p)),
            'rmse_mid_filtered': float(np.sqrt(np.mean(o['pnl_mid'][o['midok']] ** 2))),
            'var95_mid_filtered': float(-np.quantile(o['pnl_mid'][o['midok']], 0.05)),
            'share_mid_ok': float(o['midok'].mean())}


def tune(Xv, family, fee, grid=None):
    grid = grid if grid is not None else GRID[family]
    best, rows = None, []
    for prm in grid:
        mt = metrics(run_policy(Xv, family, prm, fee))
        rows.append((prm, mt))
        if mt['share'] <= CAP and (best is None or mt['rmse'] < best[1]['rmse']):
            best = (prm, mt)
    return best, rows


def boot_policy(outs, names, base, B=B, seed=SEED):
    """Cluster bootstrap over option contracts: CI of RMSE and of differences vs `base`."""
    sym = outs[names[0]]['symbol']
    codes = pd.Series(sym).astype('category').cat.codes.values
    G = codes.max() + 1
    cnt = np.bincount(codes, minlength=G)
    ss = {n: np.bincount(codes, weights=outs[n]['pnl'] ** 2, minlength=G) for n in names}
    cs = {n: np.bincount(codes, weights=outs[n]['cost'], minlength=G) for n in names}
    rng = np.random.default_rng(seed)
    R = {n: [] for n in names}
    Cc = {n: [] for n in names}
    for _ in range(B):
        idx = rng.integers(0, G, G)
        c = cnt[idx].sum()
        for n in names:
            R[n].append(np.sqrt(ss[n][idx].sum() / c))
            Cc[n].append(cs[n][idx].sum())
    res = {}
    for n in names:
        r = np.array(R[n])
        res[n] = {'rmse_ci': [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
        if n != base:
            d = np.array(R[base]) - r
            dc = np.array(Cc[base]) / np.array(Cc[n])
            res[n]['diff_rmse_base_minus_this_ci'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            res[n]['cost_ratio_base_over_this_ci'] = [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))]
    return res


if __name__ == '__main__':
    Xv = prepare('val')
    Xt = prepare('test')
    R = {'n_steps_test': int(Xt['hasnext'].sum()), 'n_traj_test': int(Xt['traj'].max()),
         'n_steps_val': int(Xv['hasnext'].sum()), 'med_test': Xt['med'], 'med_val': Xv['med']}
    # --- estimation -> hourly hedging (attenuation), main fee
    R['hourly_by_model'] = {}
    for m in MODELS:
        R['hourly_by_model'][m] = metrics(run_policy(Xt, 'hourly', {}, MAIN_FEE, model=m))
    # --- policy comparison for every fee scenario
    R['policies'] = {}
    R['frontier'] = {}
    sel_outs_main = None
    for fee in FEES:
        res = {}
        outs = {}
        for fam in ['hourly', 'leland']:
            o = run_policy(Xt, fam, {}, fee)
            outs[fam] = o
            res[fam] = {'params': {}, 'test': metrics(o), 'val': metrics(run_policy(Xv, fam, {}, fee))}
        for fam in ['threshold', 'band', 'ww', 'adaptive']:
            (prm, mv), rows = tune(Xv, fam, fee)
            o = run_policy(Xt, fam, prm, fee)
            outs[fam] = o
            res[fam] = {'params': prm, 'val': mv, 'test': metrics(o)}
            if fee == MAIN_FEE:
                R['frontier'][fam] = [{'params': p, 'test': metrics(run_policy(Xt, fam, p, fee))} for p, _ in rows
                                      if fam != 'adaptive' or (p['a'], p['b'], p['c']) in [(1.0, 0.5, 0.0), (prm['a'], prm['b'], prm['c'])]]
        # ablations of the adaptive rule (decision-layer value of each state variable)
        if fee == MAIN_FEE:
            abl = {}
            for name, fix in {'spread_only': {'b': 0.0, 'c': 0.0}, 'oi_only': {'a': 0.0, 'c': 0.0},
                              'rv_only': {'a': 0.0, 'b': 0.0}}.items():
                g = [p for p in GRID['adaptive'] if all(p[k] == v for k, v in fix.items())]
                (prm, mv), _ = tune(Xv, 'adaptive', fee, grid=g)
                abl[name] = {'params': prm, 'val': mv, 'test': metrics(run_policy(Xt, 'adaptive', prm, fee))}
            R['adaptive_ablation'] = abl
            names = ['adaptive', 'hourly', 'leland', 'threshold', 'band', 'ww']
            R['bootstrap_main'] = boot_policy(outs, names, base='adaptive')
            sel_outs_main = outs
        R['policies'][str(fee)] = res
    # --- regimes on test (main fee): by spread quartile (train edges), RV tercile, test halves
    o = sel_outs_main
    idx = o['adaptive']['idx']
    med = Xt['med']
    ref = panel[panel.split != 'test']
    e_s = np.quantile(ref.s_rel.values, [0.25, 0.5, 0.75])
    e_r = np.quantile(ref.RV6.values, [1 / 3, 2 / 3])
    qs = np.searchsorted(e_s, Xt['s_rel'][idx], side='right')
    qr = np.searchsorted(e_r, Xt['RV6'][idx], side='right')
    tt = pd.to_datetime(Xt['time'][idx])
    half = (tt >= tt.min() + (tt.max() - tt.min()) / 2).astype(int)
    reg = {}
    for lab, q, K in [('spread_q', qs, 4), ('rv_t', qr, 3), ('half', np.asarray(half), 2)]:
        reg[lab] = {}
        for fam in ['hourly', 'threshold', 'band', 'ww', 'adaptive']:
            reg[lab][fam] = [{'rmse': float(np.sqrt(np.mean(o[fam]['pnl'][q == i] ** 2))),
                              'cost': float(o[fam]['cost'][q == i].sum()),
                              'share': float(o[fam]['trade'][q == i].mean()), 'n': int((q == i).sum())} for i in range(K)]
    R['regimes'] = reg
    # adaptive threshold mechanism: mean width by spread x OI quartile (train edges), test rows
    e_o = np.quantile(ref.oi.values, [0.25, 0.5, 0.75])
    w = o['adaptive']['width'] if len(o['adaptive']['width']) == len(Xt['s_rel']) else None
    qs_all = np.searchsorted(e_s, Xt['s_rel'], side='right')
    qo_all = np.searchsorted(e_o, Xt['oi'], side='right')
    grid = np.full((4, 4), np.nan)
    tr_share = np.full((4, 4), np.nan)
    hn = Xt['hasnext']
    for i in range(4):
        for j in range(4):
            m = (qs_all == i) & (qo_all == j)
            grid[j, i] = float(np.mean(w[m])) if m.sum() else np.nan
    for i in range(4):
        m = qs_all[hn] == i
        for fam in ['threshold', 'adaptive']:
            pass
    R['adaptive_width_grid'] = grid.tolist()
    R['adaptive_share_by_spread_q'] = {fam: [float(o[fam]['trade'][qs == i].mean()) for i in range(4)]
                                       for fam in ['hourly', 'threshold', 'band', 'ww', 'adaptive']}
    json.dump(R, open(os.path.join(D, 'backtest_results.json'), 'w'), indent=1, default=float)
    # compact print
    print('steps test', R['n_steps_test'], 'traj', R['n_traj_test'], 'steps val', R['n_steps_val'])
    for m, v in R['hourly_by_model'].items():
        print(f"hourly {m:9s} rmse={v['rmse']:.3f} var95={v['var95']:.3f} es95={v['es95']:.2f} mean={v['mean']:.3f} cost={v['cost']:.0f} rmse_mid={v['rmse_mid_filtered']:.3f}")
    for fee, res in R['policies'].items():
        print('--- fee', fee)
        for fam, v in res.items():
            t = v['test']
            print(f"{fam:10s} {str(v['params']):58s} sh={t['share']:.4f} rmse={t['rmse']:.3f} var={t['var95']:.2f} es={t['es95']:.1f} cost={t['cost']:.0f} turn={t['turnover']:.0f} mid={t['rmse_mid_filtered']:.2f} | val rmse={v['val']['rmse']:.3f} sh={v['val']['share']:.3f}")
    print(json.dumps(R['adaptive_ablation'], indent=0, default=float)[:3000])
