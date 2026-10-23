"""Hourly hedging with deltas from quote-IV models; option leg valued at mid (spread <= 50% at t and t+1)."""
import json, os, sys
import numpy as np, pandas as pd
ROOT=sys.argv[1]; sys.argv=[sys.argv[0],ROOT]; sys.path.insert(0,os.path.join(ROOT,'code'))
import backtest as bt
D=os.path.join(ROOT,'data')
panel=pd.read_parquet(os.path.join(D,'panel.parquet')); Pm=pd.read_parquet(os.path.join(D,'preds_test_mid.parquet'))
df=panel[panel.split=='test'].merge(Pm,on=['symbol','time']).sort_values(['symbol','time']).reset_index(drop=True)
newtraj=((df.symbol!=df.symbol.shift(1))|(df.groupby('symbol').time.diff()!=pd.Timedelta('1h'))).values
hasnext=np.r_[~newtraj[1:],False]
sp=pd.read_parquet(os.path.join(D,'spot_spread.parquet'))['s_spot_rel'].reindex(df.time).fillna(3.6e-7).values
c=2.5e-4+0.5*sp
S=df.S.values; C=df.mid.values; srel=df.s_rel.values
dS=np.r_[S[1:]-S[:-1],0]; dC=np.r_[C[1:]-C[:-1],0]
ok=hasnext&(srel<=0.5)&(np.r_[srel[1:],np.inf]<=0.5)
res={}; pn={}
for m in ['BS','PARAM','GB_GEO','RF_GEO','RF_MICRO']:
    d,g=bt.bs_delta_gamma(S,df.K.values,df['T'].values,df['ivq_'+m].values,df.call.values)
    phi,prev,tr=bt.simulate(0,d,np.zeros(len(d)),newtraj,hasnext)
    pnl=dC-phi*dS-c*np.abs(phi-prev)*S
    pn[m]=pnl[ok]; res[m]={'rmse':float(np.sqrt(np.mean(pnl[ok]**2))),'steps':int(ok.sum())}
sym=df.symbol.values[ok]; cc=pd.Series(sym).astype('category').cat.codes.values; G=cc.max()+1; cnt=np.bincount(cc,minlength=G)
ss={m:np.bincount(cc,weights=pn[m]**2,minlength=G) for m in pn}; rng=np.random.default_rng(5); Rb={m:[] for m in pn}
for _ in range(1000):
    idx=rng.integers(0,G,G); n=cnt[idx].sum()
    for m in pn: Rb[m].append(np.sqrt(ss[m][idx].sum()/n))
for m in pn:
    dd=np.array(Rb['BS'])-np.array(Rb[m]); res[m]['gain_vs_BS_ci']=[float(np.percentile(dd,2.5)),float(np.percentile(dd,97.5))]
    dd2=np.array(Rb['GB_GEO'])-np.array(Rb[m]); res[m]['gain_vs_GB_ci']=[float(np.percentile(dd2,2.5)),float(np.percentile(dd2,97.5))]
json.dump(res,open(os.path.join(D,'hedge_quote_iv.json'),'w'),indent=1); print(json.dumps(res,indent=1))
