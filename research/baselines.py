"""Development-only baseline selection; final holdout metrics remain unopened."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,json,pickle
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from xgboost import XGBClassifier

RELATIONS=['portfolio_variance','covariance_share_coverage','average_correlation','joint_down_frequency',
 'estimable_pair_exposure','positive_pair_exposure','negative_pair_exposure']
EXCLUDED=['t','i','k4','input_year','label_observed_continuing','outcome_growth']
GRID=[{'max_depth':d,'n_estimators':n} for d in [2,3,4] for n in [150,400]]

def feature_names(data,strong=True):
    return [c for c in data.columns if c not in EXCLUDED and (strong or c not in RELATIONS)]

def preprocessor(numeric):
    return ColumnTransformer([
        ('numeric',Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                             ('scale',StandardScaler())]),numeric),
        ('identity',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['i','k4'])],sparse_threshold=0)

def weights(data):
    w=1/data.t.map(data.t.value_counts()).to_numpy(dtype=float)
    return w/w.mean()

def tree(params,seed=42):
    return XGBClassifier(**params,learning_rate=.05,subsample=.9,colsample_bytree=.9,
          min_child_weight=5,reg_lambda=5,objective='binary:logistic',eval_metric='logloss',
          n_jobs=4,tree_method='hist',random_state=seed)

def metrics(y,p):
    # Fractional tie allocation gives the expected result of random tie-breaking.
    # This avoids arbitrary row-order precision for a constant risk-rate baseline.
    k=max(1,int(np.ceil(.1*len(y))))
    cutoff=np.sort(p)[-k]
    selected=(p>cutoff).astype(float)
    tied=(p==cutoff)
    selected[tied]=(k-selected.sum())/max(1,tied.sum())
    selected_positive=float(selected@y)
    return {'ap':float(average_precision_score(y,p)),
        'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        'brier':float(brier_score_loss(y,p)),'logloss':float(log_loss(y,p,labels=[0,1])),
        'precision_at_10':selected_positive/k,'recall_at_10':selected_positive/max(1,y.sum()),
        'risk_rate':float(y.mean()),'n':len(y)}

def yearly(data,p):
    return [{'year':int(t),**metrics(data.loc[data.t==t,'label_observed_continuing'].to_numpy(),p[data.t.to_numpy()==t])}
            for t in sorted(data.t.unique())]

def select_xgb(train,validation,strong):
    transform=preprocessor(feature_names(train,strong))
    x=transform.fit_transform(train); v=transform.transform(validation)
    y=train.label_observed_continuing.to_numpy()
    trials=[]; best=None
    for config in GRID:
        model=tree(config); model.fit(x,y,sample_weight=weights(train))
        p=model.predict_proba(v)[:,1]
        score=np.mean([r['ap'] for r in yearly(validation,p)])
        trials.append({**config,'mean_validation_ap':float(score)})
        if best is None or score>best[0]: best=(score,model,config,p)
    return transform,best[1],best[2],best[3],trials

def main():
    out=ROOT/'results'/'development'; out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
    train=data[data.t<=2015]; val=data[data.t.between(2016,2018)]
    rows=[]; preds=val[['t','i','k4','label_observed_continuing']].copy(); selected={}
    for strong,name in [(False,'B2_XGBoost'),(True,'B3_XGBoost_relations')]:
        transform,model,params,p,trials=select_xgb(train,val,strong)
        with (out/f'{name}.pkl').open('wb') as stream: pickle.dump((transform,model),stream)
        pd.DataFrame(trials).to_csv(out/f'{name}_search.csv',index=False)
        rows.extend([{'model':name,**r} for r in yearly(val,p)])
        preds[name]=p; selected[name]=params
        print('VALIDATION',name,params,yearly(val,p),flush=True)
    transform=preprocessor(feature_names(train,True)); x=transform.fit_transform(train); v=transform.transform(val)
    for name,model in [('B1_logistic',LogisticRegression(C=1,max_iter=3000)),
                       ('B1_random_forest',RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=10,n_jobs=4,random_state=42))]:
        model.fit(x,train.label_observed_continuing,sample_weight=weights(train))
        p=model.predict_proba(v)[:,1]
        with (out/f'{name}.pkl').open('wb') as stream: pickle.dump((transform,model),stream)
        rows.extend([{'model':name,**r} for r in yearly(val,p)]); preds[name]=p
    p=np.full(len(val),np.average(train.label_observed_continuing,weights=weights(train)))
    rows.extend([{'model':'B0_training_risk_rate',**r} for r in yearly(val,p)])
    preds['B0_training_risk_rate']=p
    pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
    preds.to_csv(out/'validation_predictions.csv',index=False)
    (out/'selected_parameters.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('model')[['ap','auc','brier']].mean().to_string(),flush=True)
    print('FINAL_HOLDOUT_NOT_EVALUATED',flush=True)

if __name__=='__main__': main()
