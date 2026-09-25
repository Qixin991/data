"""Audit cached development graph properties and base prediction distributions."""
from baselines import *
OUT=ROOT/'audit'
base=pd.read_csv(ROOT/'results/diagnosis/crossfit_base.csv',dtype={'k4':str})
rows=[]
for t,part in base.groupby('t'):
    p=part.p_base.to_numpy(); y=part.label_observed_continuing.to_numpy()
    rows.append({'year':int(t),'phase':'OOF_training' if t<=2015 else 'development_validation',
                 'n':len(p),'risk_rate':float(y.mean()),'base_mean_probability':float(p.mean()),**metrics(y,p)})
pd.DataFrame(rows).to_csv(OUT/'crossfit_prediction_distribution.csv',index=False)
with np.load(ROOT/'data/graphs_L8/2015.npz') as f:
    rho=f['rho']; rel=f['reliability']; z=f['z']; s=f['shares']; ids=f['row_ids']
assert np.isfinite(rho).all() and np.isfinite(rel).all()
assert (np.abs(rho)<=1).all() and ((rel>=0)&(rel<=1)).all()
np.testing.assert_allclose(rho,rho.transpose(0,2,1),atol=1e-7)
np.testing.assert_allclose(rel,rel.transpose(0,2,1),atol=1e-7)
assert (rho[:,np.arange(100),np.arange(100)]==0).all()
assert (rel[:,np.arange(100),np.arange(100)]==0).all()
assert (s>=0).all() and (s.sum(1)>=.9-1e-6).all() and (s.sum(1)<=1+1e-6).all()
table=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str}).loc[ids]
valid=np.isfinite(z[:,:,0]); coverage=(s*valid).sum(1)
last=np.divide((np.nan_to_num(z[:,:,0])*s).sum(1),coverage,out=np.full(len(s),np.nan),where=coverage>0)
np.testing.assert_allclose(last,table.market_change,rtol=1e-4,atol=1e-6,equal_nan=True)
pair=s[:,:,None]*s[:,None,:]; sparse=np.zeros_like(rho)
for sign in [1,-1]:
    a=np.maximum(sign*rho,0)*rel; keep=np.zeros_like(a,dtype=bool)
    np.put_along_axis(keep,np.argsort(a,axis=-1)[...,-5:],True,axis=-1)
    sparse+=a*(keep|keep.transpose(0,2,1))
den=(pair*np.abs(rho)).sum((1,2)); num=(pair*sparse).sum((1,2))
gate=np.divide(num,den,out=np.zeros_like(num),where=den>1e-12)
report={'audit_year':2015,'samples':len(s),'symmetry_bounds_diagonal_and_feature_reconciliation':'passed',
        'gate_quantiles':np.quantile(gate,[0,.1,.5,.9,1]).tolist(),
        'zero_gate_fraction':float((gate==0).mean()),
        'retained_undirected_edges_quantiles':np.quantile((sparse>0).sum((1,2))/2,[0,.1,.5,.9,1]).tolist(),
        'positive_edge_reliability_quantiles':np.quantile(rel[rel>0],[.1,.5,.9]).tolist(),
        'final_holdout_evaluated':False}
(OUT/'graph_diagnostic_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
