"""Download the public Binance inputs used in the article and verify them against data/input_manifest.csv.

Options: Binance Options EOHSummary, BTCUSDT, daily files 2023-09-01 ... 2023-10-23 (days missing from the archive are
skipped; in this window 41 files exist). Spot: Binance Spot BTCUSDT 1h klines, monthly files 2023-08 ... 2023-10.
Usage: python code/download_data.py ROOT
"""
import csv
import datetime as dt
import hashlib
import os
import sys
import urllib.error
import urllib.request
import zipfile

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
BASE = 'https://data.binance.vision/data'
OPT = os.path.join(ROOT, 'binance', 'sepoct')
SPOT = os.path.join(ROOT, 'binance', 'spot')
os.makedirs(OPT, exist_ok=True)
os.makedirs(SPOT, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dst):
    if os.path.exists(dst):
        return True
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dst + '.part', 'wb') as f:
            f.write(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    os.replace(dst + '.part', dst)
    return True


d = dt.date(2023, 9, 1)
n_opt = 0
while d <= dt.date(2023, 10, 23):
    name = f'BTCUSDT-EOHSummary-{d}.zip'
    n_opt += fetch(f'{BASE}/option/daily/EOHSummary/BTCUSDT/{name}', os.path.join(OPT, name))
    d += dt.timedelta(days=1)
for m in ('2023-08', '2023-09', '2023-10'):
    name = f'BTCUSDT-1h-{m}.zip'
    dst = os.path.join(SPOT, name)
    fetch(f'{BASE}/spot/monthly/klines/BTCUSDT/1h/{name}', dst)
    with zipfile.ZipFile(dst) as z:
        z.extractall(SPOT)
print('option files:', n_opt)

man = os.path.join(ROOT, 'data', 'input_manifest.csv')
bad = []
with open(man) as f:
    for row in csv.DictReader(f):
        p = os.path.join(ROOT, row['path'])
        if not os.path.exists(p) or sha256(p) != row['sha256']:
            bad.append(row['path'])
print('checksum mismatches:', bad if bad else 'none')
sys.exit(1 if bad else 0)
