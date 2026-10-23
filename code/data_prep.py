"""Corrected data stage: Binance BTCUSDT option EOHSummary + spot klines (Sep-Oct 2023).

Key corrections relative to the earlier pipeline:
  * timestamps: an EOHSummary row with field `hour`=h is the END-of-hour snapshot (h+1):00 UTC;
    it is aligned with the close of the spot candle opened at h (verified: corr of hourly changes 0.99);
  * the underlying S is the BTCUSDT spot close (EOHSummary contains no underlying price);
  * expiry at 08:00 UTC; maturities 5-60 days;
  * only two-sided, uncrossed quotes are kept (relative spread must be defined);
  * all rolling features use information dated t or earlier.
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
OPT_DIR = os.path.join(ROOT, 'binance', 'sepoct')
SPOT_DIR = os.path.join(ROOT, 'binance', 'spot')
OUT = os.path.join(ROOT, 'data')
os.makedirs(OUT, exist_ok=True)

counts = {}

# ---------------------------------------------------------------- options
frames = []
for f in sorted(glob.glob(os.path.join(OPT_DIR, 'BTCUSDT-EOHSummary-*.zip'))):
    with zipfile.ZipFile(f) as z:
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        frames.append(pd.read_csv(z.open(name)))
raw = pd.concat(frames, ignore_index=True)
counts['files'] = len(frames)
counts['raw_rows'] = len(raw)

df = raw.copy()
df['time'] = pd.to_datetime(df['date'].astype(str), errors='coerce') + pd.to_timedelta(df['hour'] + 1, unit='h')
df = df.dropna(subset=['time'])
df = df.drop_duplicates(subset=['symbol', 'time'])
counts['valid_time_dedup'] = len(df)

parts = df['symbol'].str.split('-', expand=True)
df['K'] = parts[2].astype(float)
df['call'] = (parts[3] == 'C').astype(int)
df['expiry'] = pd.to_datetime('20' + parts[1], format='%Y%m%d') + pd.Timedelta('8h')
df['T'] = (df['expiry'] - df['time']).dt.total_seconds() / (365.0 * 86400.0)

num = ['best_bid_price', 'best_ask_price', 'best_bid_qty', 'best_ask_qty', 'mark_price', 'mark_iv',
       'volume_contracts', 'openinterest_contracts']
for c in num:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df[(df.best_bid_price > 0) & (df.best_ask_price > df.best_bid_price)]
counts['two_sided_uncrossed'] = len(df)
df = df[(df.mark_price > 0) & (df.mark_iv > 0)]
counts['positive_mark'] = len(df)
df = df[(df['T'] >= 5 / 365.0) & (df['T'] <= 60 / 365.0)]
counts['dte_5_60'] = len(df)

# ---------------------------------------------------------------- spot
k = pd.concat([pd.read_csv(f, header=None) for f in sorted(glob.glob(os.path.join(SPOT_DIR, 'BTCUSDT-1h-*.csv')))])
k.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qvol', 'ntrades', 'tb', 'tq', 'ign']
k['t_end'] = pd.to_datetime(k['open_time'], unit='ms') + pd.Timedelta('1h')
spot = k.set_index('t_end')['close'].astype(float).sort_index()
spot = spot[~spot.index.duplicated()]
r = np.log(spot).diff()
rv6 = np.sqrt((r ** 2).rolling(6, min_periods=6).mean() * 24 * 365)    # six hourly returns ending at t
spot_df = pd.DataFrame({'S': spot, 'RV6': rv6})
spot_df.index.name = 'time'

df = df.merge(spot_df, left_on='time', right_index=True, how='inner')
df = df.dropna(subset=['S', 'RV6'])
counts['with_spot'] = len(df)

# ---------------------------------------------------------------- features
df['mid'] = 0.5 * (df.best_bid_price + df.best_ask_price)
df['s_rel'] = (df.best_ask_price - df.best_bid_price) / df.mid
df['s_abs'] = df.best_ask_price - df.best_bid_price
df['k'] = np.log(df.S / df.K)
df['log_vol'] = np.log1p(df.volume_contracts.fillna(0))
df['log_oi'] = np.log1p(df.openinterest_contracts.fillna(0))
df['oi'] = df.openinterest_contracts.fillna(0)
df['log_bq'] = np.log1p(df.best_bid_qty.fillna(0))
df['log_aq'] = np.log1p(df.best_ask_qty.fillna(0))
df = df.sort_values(['symbol', 'time']).reset_index(drop=True)
prev_t = df.groupby('symbol')['time'].shift(1)
consec = (df['time'] - prev_t) == pd.Timedelta('1h')
for c in ['s_rel', 'log_vol']:
    lag = df.groupby('symbol')[c].shift(1)
    df[c + '_lag1'] = np.where(consec, lag, df[c])          # first observation: current value (no look-ahead)
df['iv'] = df['mark_iv']

keep = ['symbol', 'time', 'expiry', 'K', 'call', 'T', 'S', 'RV6', 'k', 'mid', 'mark_price', 'iv',
        'best_bid_price', 'best_ask_price', 'best_bid_qty', 'best_ask_qty', 'delta', 'gamma', 'vega', 'theta',
        'best_buy_iv', 'best_sell_iv', 'close', 'volume_contracts', 'volume_usdt', 'openinterest_usdt',
        's_rel', 's_abs', 'log_vol', 'log_oi', 'oi', 'log_bq', 'log_aq', 's_rel_lag1', 'log_vol_lag1']
df = df[keep].sort_values(['time', 'symbol']).reset_index(drop=True)
counts['final'] = len(df)
counts['n_symbols'] = int(df.symbol.nunique())
counts['first_time'] = str(df.time.min())
counts['last_time'] = str(df.time.max())

# ---------------------------------------------------------------- chronological split at hour boundaries
times = df['time']
t_split = times.iloc[int(0.7 * len(df))]
train = df[times < t_split]
t_val = train['time'].iloc[int(0.8 * len(train))]
df['split'] = np.where(times >= t_split, 'test', np.where(times >= t_val, 'val', 'fit'))
counts['t_val'] = str(t_val)
counts['t_split'] = str(t_split)
for s in ['fit', 'val', 'test']:
    counts['n_' + s] = int((df.split == s).sum())
counts['n_train'] = counts['n_fit'] + counts['n_val']

df.to_parquet(os.path.join(OUT, 'panel.parquet'), index=False)
spot_df.to_parquet(os.path.join(OUT, 'spot.parquet'))
json.dump(counts, open(os.path.join(OUT, 'counts.json'), 'w'), indent=1)
print(json.dumps(counts, indent=1))
