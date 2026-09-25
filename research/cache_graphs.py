"""Cache historical graph inputs by year; restart skips completed files."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
import sys,json,time,warnings
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps')); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from features import reliable_graph
threadpool_limits(1)
OUT=ROOT/'data'/'graphs_L8'; OUT.mkdir(exist_ok=True)
with np.load(ROOT/'data/trade_arrays.npz') as archive:
    # NPZ is lazy: materialize once, not decompress full arrays for every sample.
    data={key:archive[key] for key in ['exporters','products','destinations','imports','flow','total']}
table=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
table=table[table.t.between(2010,2018)]
macro=pd.read_csv(ROOT/'data/macro_baci_mapping.csv'); macro=macro[macro.year.notna()].set_index(['country_code','year'])
ei={int(c):n for n,c in enumerate(data['exporters'])}; pi={str(c):n for n,c in enumerate(data['products'])}
dest=data['destinations']; n_nodes=len(dest); L=8
for t, samples in table.groupby('t',sort=True):
    target=OUT/f'{t}.npz'
    if target.exists():
        cached=np.load(target)
        if np.array_equal(cached['row_ids'],samples.index.to_numpy()):
            print('GRAPH_REUSE',t,flush=True); continue
        raise ValueError('Graph cache sample mismatch')
    start=time.perf_counter(); ti=int(t)-1995; nr=len(samples)
    rho=np.zeros((nr,n_nodes,n_nodes),np.float32); rel=np.zeros_like(rho)
    z=np.full((nr,n_nodes,9),np.nan,np.float32); shares=np.zeros((nr,n_nodes),np.float32)
    zm=np.full((n_nodes,3),np.nan)
    for j,code in enumerate(dest):
        if (code,t) in macro.index:
            r=macro.loc[(code,t)]
            zm[j]=[r.gdp_growth,np.log(r.gdp_usd) if r.gdp_usd>0 else np.nan,r.inflation]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for n,row in enumerate(samples.itertuples()):
            e=ei[row.i]; p=pi[row.k4]
            mh=data['imports'][ti-L:ti+1,p]-data['flow'][ti-L:ti+1,e,p]
            mh=np.where(mh>=1000,mh,np.nan)
            u=np.diff(np.log1p(mh),axis=0)
            corr,reliability,_=reliable_graph(u)
            rho[n]=corr; rel[n]=reliability
            valid=np.isfinite(u); last=np.where(valid,np.arange(L)[:,None],-1).max(axis=0)
            last_value=np.where(last>=0,u[np.maximum(last,0),np.arange(n_nodes)],np.nan)
            z[n]=np.column_stack([u[-1],np.nanmean(u,axis=0),np.nanstd(u,axis=0,ddof=1),
                valid.sum(0)/L,last_value,np.where(last>=0,L-1-last,L),zm])
            shares[n]=data['flow'][ti,e,p]/data['total'][ti,e,p]
    np.savez_compressed(target,row_ids=samples.index.to_numpy(),rho=rho,reliability=rel,z=z,shares=shares)
    print('GRAPH_CACHED',int(t),nr,'seconds',round(time.perf_counter()-start,1),flush=True)
(OUT/'schema.json').write_text(json.dumps({'window':8,'minimum_common_years':6,'topk_per_sign':5,
 'node_features':['last_change','mean_change','sd_change','observed_fraction','last_valid_change','last_valid_age','gdp_growth','log_gdp','inflation'],
 'relationship':'target exporter removed; unavailable or <1000 kusd markets unknown',
 'lambda':'removed','schema_version':1},indent=2),encoding='utf-8')
