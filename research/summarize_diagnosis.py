"""Summarize completed development diagnostics without accessing held-out labels."""
from baselines import *
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
threadpool_limits(1)
OUT=ROOT/'results/diagnosis'
trees=OUT/'trees_threads4'; graphs=OUT/'graphs'
models=['no_graph','ordinary_graph','reliable_graph','reliable_pair']
seeds=[42,2024,2026]
rows=[]; predictions={}; selections={}
for mode in models:
    ps=[]
    for seed in seeds:
        file=graphs/f'{mode}_{seed}.csv'
        if not file.exists(): raise FileNotFoundError(file)
        p=pd.read_csv(file,dtype={'k4':str}); ps.append(p.p.to_numpy())
        sel=json.loads((graphs/f'{mode}_{seed}_selection.json').read_text())
        selections[f'{mode}_{seed}']=sel
        rows.extend([{'model':mode,'seed':seed,'selected_epoch':sel['epoch'],**r} for r in yearly(p,p.p.to_numpy())])
    predictions[mode]=np.stack(ps)
result=pd.DataFrame(rows); result.to_csv(OUT/'graph_validation_metrics.csv',index=False)
seedmetrics=result.groupby(['model','seed'],as_index=False)[['ap','auc','brier','logloss']].mean()
seedmetrics.to_csv(OUT/'graph_seed_metrics.csv',index=False)
summary=seedmetrics.groupby('model').agg(ap_mean=('ap','mean'),ap_sd=('ap','std'),ap_min=('ap','min'),ap_max=('ap','max'),auc=('auc','mean'),brier=('brier','mean'))
summary.to_csv(OUT/'graph_summary.csv')

# Conditional uncertainty for the difference in mean seed performance, not a newly selected ensemble.
codes,levels=pd.factorize(pd.MultiIndex.from_frame(p[['i','k4']]))
y=p.label_observed_continuing.to_numpy(); masks=[(p.t.to_numpy()==t) for t in sorted(p.t.unique())]
comparisons=[('ordinary_graph','no_graph'),('reliable_graph','ordinary_graph'),('reliable_pair','reliable_graph'),('reliable_pair','no_graph')]
bootstrap={}; rng=np.random.default_rng(20260925)
sorted_predictions={}
for mode,pred in predictions.items():
    prepared=[]
    for r in pred:
        for mask in masks:
            ids=np.flatnonzero(mask); order=np.argsort(-r[ids],kind='stable'); ids=ids[order]
            ends=np.r_[np.flatnonzero(np.diff(r[ids])!=0),len(ids)-1]
            prepared.append((ids,ends))
    sorted_predictions[mode]=prepared
def score(pred,w):
    values=[]
    for ids,ends in sorted_predictions[pred]:
        tp=np.cumsum(y[ids]*w[ids])[ends]; total=np.cumsum(w[ids])[ends]
        precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
        values.append(float(np.sum(np.diff(np.r_[0,tp])*precision)/tp[-1]) if tp[-1]>0 else 0.)
    return np.mean(values)
for mode,pred in predictions.items():
    checkweights=rng.integers(0,5,len(p)).astype(float)
    reference=np.mean([average_precision_score(y[mask],r[mask],sample_weight=checkweights[mask]) for r in pred for mask in masks])
    np.testing.assert_allclose(score(mode,checkweights),reference,rtol=1e-12,atol=1e-12)
for left,right in comparisons:
    draws=[]
    for _ in range(1000):
        freq=np.bincount(rng.integers(0,len(levels),len(levels)),minlength=len(levels)); w=freq[codes]
        draws.append(score(left,w)-score(right,w))
    bootstrap[left+' minus '+right]={'point':float(summary.loc[left,'ap_mean']-summary.loc[right,'ap_mean']),
        'conditional_95_percent_interval':np.quantile(draws,[.025,.975]).tolist()}
for mode in ['D3','D5']:
    prepared=[]
    for seed in seeds:
        q=pd.read_csv(trees/f'seedcheck_{mode}_{seed}.csv',dtype={'k4':str})
        assert q[['t','i','k4']].equals(p[['t','i','k4']])
        r=q.p.to_numpy()
        for mask in masks:
            ids=np.flatnonzero(mask); ids=ids[np.argsort(-r[ids],kind='stable')]
            prepared.append((ids,np.r_[np.flatnonzero(np.diff(r[ids])!=0),len(ids)-1]))
    sorted_predictions[mode]=prepared
draws=[]
for _ in range(1000):
    freq=np.bincount(rng.integers(0,len(levels),len(levels)),minlength=len(levels)); w=freq[codes]
    draws.append(score('D5',w)-score('D3',w))
bootstrap['trees D5 minus D3 (no common-factor features)']={'point':float(score('D5',np.ones(len(p)))-score('D3',np.ones(len(p)))),
    'conditional_95_percent_interval':np.quantile(draws,[.025,.975]).tolist()}
for left,right in [('no_graph','D5'),('reliable_pair','D5')]:
    draws=[]
    for _ in range(1000):
        freq=np.bincount(rng.integers(0,len(levels),len(levels)),minlength=len(levels)); w=freq[codes]
        draws.append(score(left,w)-score(right,w))
    bootstrap[left+' minus trees '+right]={'point':float(score(left,np.ones(len(p)))-score(right,np.ones(len(p)))),
        'conditional_95_percent_interval':np.quantile(draws,[.025,.975]).tolist()}
(OUT/'graph_comparison_intervals.json').write_text(json.dumps({'comparisons':bootstrap,'replicates':1000,
 'limitation':'Conditional on 2016-2018 and selected checkpoints; no selection correction; shared year/product dependence not removed. Not final-test significance.'},indent=2))

tree_metrics=pd.read_csv(trees/'validation_metrics.csv')
other=pd.read_csv(trees/'D5_relations_without_common_predictions.csv',dtype={'k4':str})
tree_metrics=pd.concat([tree_metrics,pd.DataFrame([{'model':'D5_relations_without_common',**r} for r in yearly(other,other.p.to_numpy())])],ignore_index=True)
tree_metrics.to_csv(OUT/'tree_validation_metrics_verified.csv',index=False)
tree_summary=tree_metrics.groupby('model')[['ap','auc','brier']].mean()
tree_summary.to_csv(OUT/'tree_summary_verified.csv')
print('TREES\n'+tree_summary.to_string())
print('GRAPHS\n'+summary.to_string())
print(json.dumps(bootstrap,indent=2))
print('ANNUAL_GRAPH_AP\n'+result.groupby(['model','year']).ap.mean().unstack().to_string())
print('FINAL_HOLDOUT_NOT_EVALUATED')
