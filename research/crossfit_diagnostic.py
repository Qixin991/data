"""Forward out-of-fold base predictions; fixed development-selected B3 settings."""
from baselines import *
OUT=ROOT/'results/diagnosis'
data=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
data=data[data.t<=2018]
params={'max_depth':3,'n_estimators':150}
rows=[]
for t in range(2010,2016):
    target=OUT/f'crossfit_{t}.csv'
    if target.exists():
        rows.append(pd.read_csv(target,dtype={'k4':str})); continue
    train=data[data.t<t]; val=data[data.t==t]
    assert train.t.max()+1<=t
    tr=preprocessor(feature_names(train,True)); x=tr.fit_transform(train)
    m=tree(params); m.fit(x,train.label_observed_continuing,sample_weight=weights(train))
    p=m.predict_proba(tr.transform(val))[:,1]
    r=val[['t','i','k4','label_observed_continuing']].copy(); r['p_base']=p
    r.to_csv(target,index=False); rows.append(r)
    print('FORWARD_BASE',t,'training_max',int(train.t.max()),flush=True)
old=pd.read_csv(ROOT/'results/development/validation_predictions.csv',dtype={'k4':str})
val=old[['t','i','k4','label_observed_continuing']].copy(); val['p_base']=old.B3_XGBoost_relations
rows.append(val)
pd.concat(rows).to_csv(OUT/'crossfit_base.csv',index=False)
(OUT/'crossfit_protocol.json').write_text(json.dumps({'parameters':params,'training_origins':list(range(2010,2016)),
 'rule':'origin t uses samples through t-1 (outcomes through t); preprocessing also fit strictly earlier',
 'caveat':'fixed parameters selected in prior development validation; this is OOF residual training, not a claim that these parameters were chosen in real time',
 'validation_base':'existing B3 trained 2007-2015; no validation labels in model fit'},indent=2))
