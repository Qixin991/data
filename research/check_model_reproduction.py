from baselines import *
from threadpoolctl import threadpool_limits
data=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
train=data[data.t<=2015]; val=data[data.t.between(2016,2018)]
old=pd.read_csv(ROOT/'results/development/validation_predictions.csv')
report={}
for name,strong in [('B2_XGBoost',False),('B3_XGBoost_relations',True)]:
    with (ROOT/f'results/development/{name}.pkl').open('rb') as f: tr,model=pickle.load(f)
    fresh=preprocessor(feature_names(train,strong)); x=fresh.fit_transform(train); v=fresh.transform(val)
    xx=tr.transform(train); vv=tr.transform(val)
    p=model.predict_proba(vv)[:,1]
    info={'feature_names_equal':list(tr.get_feature_names_out())==list(fresh.get_feature_names_out()),
          'train_transform_max_diff':float(np.max(np.abs(x-xx))),
          'val_transform_max_diff':float(np.max(np.abs(v-vv))),
          'stored_prediction_max_diff':float(np.max(np.abs(p-old[name]))),'params':model.get_params()}
    for threads in [1,4]:
        with threadpool_limits(threads):
            new=tree({'max_depth':model.max_depth,'n_estimators':model.n_estimators})
            new.fit(xx,train.label_observed_continuing,sample_weight=weights(train)); q=new.predict_proba(vv)[:,1]
        info[f'refit_threads{threads}']={'max_prediction_diff':float(np.max(np.abs(p-q))),
            'ap':float(np.mean([r['ap'] for r in yearly(val,q)]))}
    report[name]=info
(ROOT/'audit/model_reproduction.json').write_text(json.dumps(report,indent=2,default=str))
print(json.dumps(report,indent=2,default=str))
