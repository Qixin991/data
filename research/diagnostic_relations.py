"""Development-only grouped ablation and historical relation-window diagnosis."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import sys,json,pickle,warnings,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
from features import pair_corr
from baselines import RELATIONS,feature_names,preprocessor,tree,weights,yearly,GRID
threadpool_limits(1)
OUT=ROOT/'results/diagnosis/trees_threads4'; OUT.mkdir(parents=True,exist_ok=True)

def residualize(u):
    valid=np.isfinite(u); counts=valid.sum(1)[:,None]-valid
    sums=np.nansum(u,axis=1)[:,None]-np.nan_to_num(u)
    f=np.divide(sums,counts,out=np.full_like(u,np.nan),where=counts>0)
    residual=np.full_like(u,np.nan)
    for j in range(u.shape[1]):
        ok=np.isfinite(u[:,j])&np.isfinite(f[:,j])
        if ok.sum()<6: continue
        x=np.column_stack([np.ones(ok.sum()),f[ok,j]])
        if np.linalg.matrix_rank(x)<2: continue
        residual[ok,j]=u[ok,j]-x@np.linalg.lstsq(x,u[ok,j],rcond=None)[0]
    return residual

def summarize(u,s):
    rho,n,valid=pair_corr(u)
    pair=s[:,None]*s[None,:]; np.fill_diagonal(pair,0)
    coverage=float(pair[valid].sum())
    neg=np.where(np.isfinite(u),u<0,0).astype(float)
    joint=np.divide(neg.T@neg,np.maximum(n,1))*valid
    good=np.isfinite(u).sum(0)>=6
    port=np.nan
    if good.sum()>1:
        a=u[:,good]; a=np.where(np.isfinite(a),a,np.nanmean(a,axis=0))
        port=float(s[good]@LedoitWolf().fit(a).covariance_@s[good])
    return [port,float(s[good].sum()),float((pair*rho).sum()/coverage) if coverage else np.nan,
        float((pair*joint).sum()/coverage) if coverage else np.nan,coverage,
        float((pair*np.maximum(rho,0)).sum()/2),float((pair*np.maximum(-rho,0)).sum()/2)]

def build(data):
    target=ROOT/'data/diagnostic_relations_development.csv.gz'
    if target.exists():
        cached=pd.read_csv(target,dtype={'k4':str})
        assert cached[['t','i','k4']].equals(data[['t','i','k4']].reset_index(drop=True))
        return cached
    with np.load(ROOT/'data/trade_arrays.npz') as archive:
        a={k:archive[k] for k in ['exporters','products','flow','imports','total']}
    ei={int(v):n for n,v in enumerate(a['exporters'])}; pi={str(v):n for n,v in enumerate(a['products'])}
    rows=[]; start=time.perf_counter()
    for t,samples in data.groupby('t'):
        ti=int(t)-1995
        for row in samples.itertuples():
            e=ei[row.i]; p=pi[row.k4]; s=a['flow'][ti,e,p]/a['total'][ti,e,p]
            r={'t':row.t,'i':row.i,'k4':row.k4}
            for L in [8,12]:
                assert ti-L>=0
                mh=a['imports'][ti-L:ti+1,p]-a['flow'][ti-L:ti+1,e,p]
                u=np.diff(np.log1p(np.where(mh>=1000,mh,np.nan)),axis=0)
                for residual in [False,True]:
                    prefix=f'{"res" if residual else "raw"}{L}_'
                    values=summarize(residualize(u) if residual else u,s)
                    r.update(dict(zip([prefix+c for c in RELATIONS],values)))
            rows.append(r)
        print('RELATION_FEATURES',t,'elapsed',round(time.perf_counter()-start,1),flush=True)
    frame=pd.DataFrame(rows)
    for c in RELATIONS:
        np.testing.assert_allclose(frame['raw8_'+c],data[c].to_numpy(),rtol=1e-7,atol=1e-10,equal_nan=True)
    assert not np.isinf(frame.select_dtypes('number')).any().any()
    frame.to_csv(target,index=False,compression='gzip')
    return frame

def fit(data,cols,name):
    checkpoint=OUT/f'{name}_predictions.csv'
    val=data[data.t.between(2016,2018)]
    if checkpoint.exists():
        saved=pd.read_csv(checkpoint,dtype={'k4':str})
        assert saved[['t','i','k4']].equals(val[['t','i','k4']].reset_index(drop=True))
        print('REUSE',name,flush=True)
        return [{'model':name,**r} for r in yearly(val,saved.p.to_numpy())]
    train=data[data.t<=2015]; tr=preprocessor(cols)
    x=tr.fit_transform(train); v=tr.transform(val); trials=[]; best=None
    for config in GRID:
        with threadpool_limits(4):
            m=tree(config); m.fit(x,train.label_observed_continuing,sample_weight=weights(train))
            p=m.predict_proba(v)[:,1]
        score=np.mean([r['ap'] for r in yearly(val,p)])
        trials.append({**config,'mean_validation_ap':score})
        if best is None or score>best[0]: best=(score,m,p,config)
    pd.DataFrame(trials).to_csv(OUT/f'{name}_search.csv',index=False)
    pred=val[['t','i','k4','label_observed_continuing']].copy(); pred['p']=best[2]; pred.to_csv(checkpoint,index=False)
    with (OUT/f'{name}.pkl').open('wb') as f: pickle.dump((tr,best[1]),f)
    (OUT/f'{name}_features.json').write_text(json.dumps({'numeric_features':cols,'identity_features':['i','k4'],'parameters':best[3]},indent=2))
    print('DIAGNOSIS',name,'AP',best[0],best[3],flush=True)
    return [{'model':name,**r} for r in yearly(val,best[2])]

def main(groups_only=False):
    data=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
    data=data[data.t<=2018].reset_index(drop=True)
    concentration=['n_dest','hhi','top_share']
    markets=['market_change','market_history_mean','market_feature_coverage']
    volatility=['market_volatility']
    common=['global_factor_last','global_factor_sd']
    full=feature_names(data,False)
    base=[c for c in full if c not in concentration+markets+volatility+common]
    metrics=[]
    for name,cols in [('D0_history_macro',base),('D1_concentration',base+concentration),
                      ('D2_market_changes',base+concentration+markets),
                      ('D3_market_volatility',base+concentration+markets+volatility)]:
        metrics.extend(fit(data,cols,name))
    previous=pd.read_csv(ROOT/'results/development/validation_metrics.csv')
    metrics.extend(previous[previous.model.isin(['B2_XGBoost','B3_XGBoost_relations'])].to_dict('records'))
    if groups_only:
        pd.DataFrame(metrics).to_csv(OUT/'group_metrics.csv',index=False)
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        extra=build(data)
    for variant in ['raw12','res8','res12']:
        trial=data.copy()
        for c in RELATIONS: trial[c]=extra[variant+'_'+c].to_numpy()
        metrics.extend(fit(trial,feature_names(trial,True),'D4_'+variant))
    result=pd.DataFrame(metrics); result.to_csv(OUT/'validation_metrics.csv',index=False)
    print(result.groupby('model')[['ap','auc','brier']].mean().to_string(),flush=True)
    print('FINAL_HOLDOUT_NOT_EVALUATED',flush=True)

if __name__=='__main__': main('--groups-only' in sys.argv)
