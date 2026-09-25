"""Result-motivated development check of common-factor and relation contributions."""
from diagnostic_relations import *
data=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
data=data[data.t<=2018].reset_index(drop=True)
common=['global_factor_last','global_factor_sd']
strong=feature_names(data,True); d3=[c for c in feature_names(data,False) if c not in common]
fit(data,[c for c in strong if c not in common],'D5_relations_without_common')
train=data[data.t<=2015]; val=data[data.t.between(2016,2018)]; result=[]
for name,cols in [('D3',d3),('B2',feature_names(data,False)),('B3',strong),('D5',[c for c in strong if c not in common])]:
    # Preserve the original table feature order for this seed check.
    tr=preprocessor(cols); x=tr.fit_transform(train); v=tr.transform(val)
    for seed in [42,2024,2026]:
        target=OUT/f'seedcheck_{name}_{seed}.csv'
        if target.exists(): p=pd.read_csv(target).p.to_numpy()
        else:
            with threadpool_limits(4):
                m=tree({'max_depth':4,'n_estimators':150},seed)
                m.fit(x,train.label_observed_continuing,sample_weight=weights(train)); p=m.predict_proba(v)[:,1]
            pred=val[['t','i','k4','label_observed_continuing']].copy(); pred['p']=p; pred.to_csv(target,index=False)
        result.extend([{'model':name,'seed':seed,**r} for r in yearly(val,p)])
        print('SEEDCHECK',name,seed,np.mean([r['ap'] for r in yearly(val,p)]),flush=True)
pd.DataFrame(result).to_csv(OUT/'seedcheck_metrics.csv',index=False)
