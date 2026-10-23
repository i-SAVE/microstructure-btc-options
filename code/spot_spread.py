"""Hourly quoted-spread proxy for BTCUSDT spot from Binance aggTrades.

For each end-of-hour timestamp t, take aggTrades in [t-300s, t). Pairs of consecutive trades with
opposite aggressor side within 50 ms approximate a simultaneous bid and ask; the spread proxy is the
median absolute price difference of such pairs, floored at the 0.01 USDT tick. Relative spread = proxy / price.
"""
import datetime as dt
import io
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
d0 = dt.date.fromisoformat(sys.argv[2])
d1 = dt.date.fromisoformat(sys.argv[3])
out_rows = []
d = d0
while d <= d1:
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{d}.zip"
    tmp = os.path.join(ROOT, 'binance', f'agg_{d}.zip')
    r = subprocess.run(['curl', '-sS', '-m', '300', '-o', tmp, '-w', '%{http_code}', url], capture_output=True, text=True)
    if r.stdout != '200':
        print('missing', d, r.stdout)
        d += dt.timedelta(1)
        continue
    with zipfile.ZipFile(tmp) as z:
        a = pd.read_csv(z.open(z.namelist()[0]), header=None,
                        names=['id', 'p', 'q', 'f', 'l', 'ts', 'buyer_maker', 'best'])
    os.remove(tmp)
    a['ts'] = pd.to_datetime(a['ts'], unit='ms')
    a = a.sort_values('ts').reset_index(drop=True)
    side = a['buyer_maker'].astype(bool).values           # True: seller-initiated (trade at bid)
    p = a['p'].values
    tsv = a['ts'].values.astype('datetime64[ms]').astype(np.int64)
    opp = np.r_[False, side[1:] != side[:-1]]
    close = np.r_[False, (tsv[1:] - tsv[:-1]) <= 50]
    pair = opp & close
    diff = np.r_[np.nan, np.abs(p[1:] - p[:-1])]
    a['pair_spread'] = np.where(pair, np.maximum(diff, 0.01), np.nan)
    a['t_end'] = a['ts'].dt.floor('h') + pd.Timedelta('1h')
    last5 = a[a['ts'] >= a['t_end'] - pd.Timedelta('300s')]
    g = last5.groupby('t_end').agg(spread=('pair_spread', 'median'), price=('p', 'last'), npairs=('pair_spread', 'count'))
    g['s_spot_rel'] = g['spread'] / g['price']
    out_rows.append(g)
    print(d, 'hours', len(g), 'median rel spread (bps)', round(float(g['s_spot_rel'].median() * 1e4), 5), flush=True)
    d += dt.timedelta(1)
res = pd.concat(out_rows)
res.index.name = 'time'
res.to_parquet(os.path.join(ROOT, 'data', 'spot_spread.parquet'))
print('overall median rel spread (bps):', res['s_spot_rel'].median() * 1e4, ' p95:', res['s_spot_rel'].quantile(0.95) * 1e4)
