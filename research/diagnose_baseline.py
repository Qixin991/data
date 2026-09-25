"""Paired validation uncertainty and matched-budget diagnostic; no holdout access."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1'
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'));sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
threadpool_limits(1)
OUT=ROOT/'results/development'
p=pd.read_csv(OUT/'validation_predictions.csv',dtype={'k4':str})
groups=pd.MultiIndex.from_frame(p[['i','k4']]); codes,levels=pd.factorize(groups)
rng=np.random.default_rng(20260925); differences=[]
y=p.label_observed_continuing.to_numpy(); b=p.B2_XGBoost.to_numpy(); r=p.B3_XGBoost_relations.to_numpy()
years=[(p.t.to_numpy()==t) for t in sorted(p.t.unique())]
for _ in range(1000):
    frequencies=np.bincount(rng.integers(0,len(levels),len(levels)),minlength=len(levels))
    w=frequencies[codes]
    delta=[]
    for mask in years:
        delta.append(average_precision_score(y[mask],r[mask],sample_weight=w[mask])-
                     average_precision_score(y[mask],b[mask],sample_weight=w[mask]))
    differences.append(np.mean(delta))
base=pd.read_csv(OUT/'B2_XGBoost_search.csv')
rel=pd.read_csv(OUT/'B3_XGBoost_relations_search.csv')
paired=base.merge(rel,on=['max_depth','n_estimators'],suffixes=('_B2','_B3'))
paired['delta_ap']=paired.mean_validation_ap_B3-paired.mean_validation_ap_B2
paired.to_csv(OUT/'matched_parameter_differences.csv',index=False)
report={'comparison':'selected B3 minus selected B2; equal-weight mean yearly validation AP',
 'point_difference':float(rel.mean_validation_ap.max()-base.mean_validation_ap.max()),
 'conditional_cluster_bootstrap_95_percent_interval':np.quantile(differences,[.025,.975]).tolist(),
 'bootstrap_replicates':1000,'clusters':len(levels),'resampling_unit':'exporter-HS4 with all its validation years kept together',
 'limitation':'conditional on these 3 validation years and already-selected hyperparameters; not selection-adjusted, not a final-test interval; global/product cross-cluster dependence remains',
 'matched_parameter_positive_comparisons':int((paired.delta_ap>0).sum()),'matched_parameter_total':len(paired),
 'final_test_evaluated':False}
(OUT/'signal_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
print(paired.to_string(index=False))
