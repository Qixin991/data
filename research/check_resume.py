"""Cheap integrity checks of completed artifacts; never rebuild raw data."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from features import pair_corr
f=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
assert not f.duplicated(['t','i','k4']).any()
assert (f.input_year==f.t).all()
assert (f.label_observed_continuing==(f.outcome_growth<-.2).astype(int)).all()
assert f.graph_coverage.between(.9,1+1e-10).all()
assert f.hhi.between(0,1+1e-10).all()
assert (f.portfolio_variance.dropna()>=-1e-12).all()
assert f.average_correlation.dropna().between(-1-1e-10,1+1e-10).all()
assert f.joint_down_frequency.dropna().between(0,1+1e-10).all()
assert not np.isinf(f.select_dtypes('number')).any().any()
flow=pd.read_csv(ROOT/'audit/sample_flow_lag0_L8.csv')
assert (flow.potential==flow[['below_scale','low_graph_coverage','unknown_label','retained']].sum(axis=1)).all()
assert len(f)==flow.retained.sum()
u=np.random.default_rng(18).normal(size=(8,10)); u[0,0]=np.nan; u[2,1]=np.nan
rho,n,valid=pair_corr(u)
for j in range(10):
    for k in range(j+1,10):
        ok=np.isfinite(u[:,j])&np.isfinite(u[:,k])
        if ok.sum()>=6: assert np.isclose(rho[j,k],np.corrcoef(u[ok,j],u[ok,k])[0,1],atol=1e-10)
meta=[]
for year in range(1995,2025):
    d=json.loads((ROOT/f'sources/comtrade_availability_{year}.json').read_text(encoding='utf-8'))
    assert not d.get('error') and d['count']==len(d['data'])
    assert all(int(r['period'])==year for r in d['data'])
    meta.extend(d['data'])
pd.DataFrame(meta).to_csv(ROOT/'data/comtrade_availability_current.csv',index=False)
names=['gdp_growth','log_gdp','inflation']
train=f[f.t<=2015]
report={'rows':len(f),'training_rows':len(train),'years':sorted(f.t.unique().tolist()),
 'train_risk_rate':float(train.label_observed_continuing.mean()),'final_test_performance_opened':False,
 'feature_sha256':hashlib.sha256((ROOT/'data/features_lag0_L8.csv.gz').read_bytes()).hexdigest(),
 'checks':['unique keys','input-year cutoff','label identity','coverage bounds','PSD variance nonnegative',
           'correlation bounds','sample flow reconciliation','pairwise correlation against numpy','30 availability response counts'],
 'macro_training_median_coverage':{n:float(train[f'destination_{n}_coverage'].median()) for n in names}}
(ROOT/'audit/resume_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
