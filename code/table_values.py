"""Recompute the numbers printed in Tables 1-3 of the manuscript from the result files (consistency check)."""
import json, os, sys
D = os.path.join(sys.argv[1], 'data')
E = json.load(open(os.path.join(D, 'estimation_results.json')))['estimation']
EM = json.load(open(os.path.join(D, 'estimation_mid_results.json')))['estimation']
B = json.load(open(os.path.join(D, 'backtest_results.json')))
X = json.load(open(os.path.join(D, 'extra_results.json')))
f3 = lambda v: f'{v:.3f}'.replace('.', ',')
print('Table 1 (RMSE test [95% CI]): mark IV | quote IV')
for m in ['BS', 'PARAM', 'GB_GEO', 'RF_GEO', 'RF_MICRO', 'RF_LEAK']:
    a, b = E[m], EM[m]
    print(f"{m:9s} {f3(a['rmse_test'])} [{f3(a['ci_test'][0])}; {f3(a['ci_test'][1])}] | "
          f"{f3(b['rmse_test'])} [{f3(b['ci_test'][0])}; {f3(b['ci_test'][1])}]")
print('Table 2 (fee 2.5 bp): share %, RMSE, VaR95, cost')
P = B['policies']['2.5']
for n, v in P.items():
    t = v['test']
    print(f"{n:10s} {v['params']} {100*t['share']:.1f} {t['rmse']:.2f} {t['var95']:.2f} {t['cost']:.0f}")
ab = X.get('adaptive_band_posthoc')
if ab:
    t = ab['test'] if 'test' in ab else ab
    print('adaptive band (post hoc)', ab.get('params'), f"{100*t['share']:.1f} {t['rmse']:.2f} {t['var95']:.2f} {t['cost']:.0f}")
print('Table 3: RMSE increase % / cost reduction % vs hourly')
for fee in ['1.0', '2.5', '5.0', '10.0']:
    key = fee if fee in B['policies'] else fee.rstrip('0').rstrip('.')
    P = B['policies'][key]
    h = P['hourly']['test']
    cells = [f"{100*(P[n]['test']['rmse']/h['rmse']-1):+.1f} / {100*(P[n]['test']['cost']/h['cost']-1):.0f}".replace('.', ',')
             for n in ['threshold', 'band', 'ww', 'adaptive']]
    print(fee, f"{h['cost']:.0f}", cells)
