"""Convert audited annual HS4 caches to arrays without inventing unobserved exits."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
screen=json.loads((ROOT/'audit/early_sample/sample_screen.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'audit/baci/manifest.json').read_text(encoding='utf-8'))
assert manifest['years_completed']==30
exporters=screen['exporters']; destinations=screen['destinations']; years=list(range(1995,2025))
products=pd.read_csv(DATA/'products_hs92.csv',dtype={'code':str})
headings=sorted(set(products.loc[products.code.str.startswith(('84','85')),'code'].str[:4]))
ei={v:i for i,v in enumerate(exporters)}; ji={v:i for i,v in enumerate(destinations)}; ki={v:i for i,v in enumerate(headings)}
shape=(len(years),len(exporters),len(headings))
totals=np.zeros(shape,dtype='float64'); flows=np.zeros(shape+(len(destinations),),dtype='float64')
imports=np.zeros((len(years),len(headings),len(destinations)),dtype='float64')
hhi=np.zeros(shape); n_dest=np.zeros(shape,dtype='int16'); top_share=np.zeros(shape)
for ti,t in enumerate(years):
    d=pd.read_csv(DATA/f'hs84_85_hs4_{t}.csv.gz',dtype={'k4':str},usecols=['i','j','k4','v'])
    nodes=d[d.j.isin(destinations)].groupby(['k4','j']).v.sum().reset_index()
    imports[ti,nodes.k4.map(ki).to_numpy(),nodes.j.map(ji).to_numpy()]=nodes.v
    own=d[d.i.isin(exporters)].copy()
    own['e']=own.i.map(ei); own['p']=own.k4.map(ki)
    group=own.groupby(['e','p']).v.agg(['sum','max','count'])
    e=group.index.get_level_values(0).to_numpy(); p=group.index.get_level_values(1).to_numpy()
    totals[ti,e,p]=group['sum']; top_share[ti,e,p]=group['max']/group['sum']; n_dest[ti,e,p]=group['count']
    own['share']=own.v/totals[ti,own.e,own.p]
    own['share_sq']=own.share**2
    hh=own.groupby(['e','p']).share_sq.sum()
    hhi[ti,hh.index.get_level_values(0),hh.index.get_level_values(1)]=hh
    keep=own[own.j.isin(destinations)]
    flows[ti,keep.e,keep.p,keep.j.map(ji)]=keep.v
    assert np.all(flows[ti].sum(-1)<=totals[ti]*(1+1e-10)+1e-6)
    print('ARRAY_YEAR',t,flush=True)
np.savez_compressed(DATA/'trade_arrays.npz',years=years,exporters=exporters,destinations=destinations,
    products=headings,total=totals,flow=flows,imports=imports,hhi=hhi,n_dest=n_dest,top_share=top_share)
(DATA/'trade_arrays_semantics.json').write_text(json.dumps({
 'zero':'zero contribution to observed BACI sum, not proof of true zero trade',
 'imports':'sum across all exporters in observed BACI; subtract target flow before constructing market changes',
 'totals':'all destinations including those outside graph',
 'source_sha256':manifest['sha256'],'sample_screen':screen},ensure_ascii=False,indent=2),encoding='utf-8')
print('SHAPE',shape,'NODES',len(destinations),'PRODUCTS',len(headings))
