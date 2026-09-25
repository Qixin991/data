"""Approved six-cell comparison: four existing cells reused, two new x three seeds."""
from baselines import *
from common_factor_candidates import build,NEW,sha
from threadpoolctl import threadpool_limits
from datetime import datetime,timezone
OUT=ROOT/'results/common_factor_aggregate_v1'
OLD=ROOT/'results/diagnosis/trees_threads4'
SEEDS=[42,2024,2026]
PARAMS={'max_depth':4,'n_estimators':150}
COMMON=['global_factor_last','global_factor_sd']

def main():
    OUT.mkdir(exist_ok=True)
    table=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
    table=table[table.t.between(2007,2018)].reset_index(drop=True)
    candidate=build()
    assert table[['t','i','k4']].equals(candidate[['t','i','k4']])
    inputs=[ROOT/'data/features_lag0_L8.csv.gz',ROOT/'data/trade_arrays.npz']
    inputs += [OLD/f'seedcheck_{name}_{seed}.csv' for name in ['D3','D5','B2','B3'] for seed in SEEDS]
    before={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
    val=table[table.t.between(2016,2018)].reset_index(drop=True)
    records=[]; predictions={}; source_records=[]
    for name,label in [('D3','none_no_rel'),('D5','none_rel'),('B2','equal_no_rel'),('B3','equal_rel')]:
        ps=[]
        for seed in SEEDS:
            file=OLD/f'seedcheck_{name}_{seed}.csv'; saved=pd.read_csv(file,dtype={'k4':str})
            assert saved[['t','i','k4','label_observed_continuing']].equals(val[['t','i','k4','label_observed_continuing']])
            p=saved.p.to_numpy(); ps.append(p)
            records.extend([{'model':label,'seed':seed,**r} for r in yearly(val,p)])
            source_records.append({'model':label,'seed':seed,'source':str(file.relative_to(ROOT)),'action':'reused, not retrained'})
        predictions[label]=np.stack(ps)
    # Replace in-place column positions; all other features keep original ordering.
    modified=table.copy()
    for old,new in zip(COMMON,NEW): modified[old]=candidate[new].to_numpy()
    modified=modified.rename(columns=dict(zip(COMMON,NEW)))
    for strong,label in [(False,'aggregate_no_rel'),(True,'aggregate_rel')]:
        cols=[c for c in modified.columns if c not in EXCLUDED and (strong or c not in RELATIONS)]
        train=modified[modified.t<=2015]; validation=modified[modified.t.between(2016,2018)]
        tr=preprocessor(cols); x=tr.fit_transform(train); v=tr.transform(validation); ps=[]
        for seed in SEEDS:
            target=OUT/f'{label}_{seed}_predictions.csv'; metadata=OUT/f'{label}_{seed}.json'
            spec={'features':cols,'parameters':PARAMS,'seed':seed,'threads':4,
                  'source_feature_sha256':before['data/features_lag0_L8.csv.gz'],
                  'candidate_sha256':sha(ROOT/'data/common_factor_aggregate_development.csv.gz')}
            if target.exists():
                assert json.loads(metadata.read_text())==spec
                saved=pd.read_csv(target,dtype={'k4':str})
                assert saved[['t','i','k4']].equals(val[['t','i','k4']])
                p=saved.p.to_numpy()
            else:
                with threadpool_limits(4):
                    m=tree(PARAMS,seed); m.fit(x,train.label_observed_continuing,sample_weight=weights(train))
                    p=m.predict_proba(v)[:,1]
                saved=val[['t','i','k4','label_observed_continuing']].copy(); saved['p']=p
                saved.to_csv(target,index=False)
                with (OUT/f'{label}_{seed}.pkl').open('wb') as f: pickle.dump((tr,m),f)
                metadata.write_text(json.dumps(spec,indent=2),encoding='utf-8')
            ps.append(p); records.extend([{'model':label,'seed':seed,**r} for r in yearly(val,p)])
            print('AGGREGATE_EXPERIMENT',label,seed,'AP',np.mean([r['ap'] for r in yearly(val,p)]),flush=True)
        predictions[label]=np.stack(ps)
    metrics_frame=pd.DataFrame(records); metrics_frame.to_csv(OUT/'annual_metrics.csv',index=False)
    seedmetrics=metrics_frame.groupby(['model','seed'],as_index=False)[['ap','auc','brier','logloss']].mean()
    seedmetrics.to_csv(OUT/'seed_metrics.csv',index=False)
    summary=seedmetrics.groupby('model').agg(ap_mean=('ap','mean'),ap_sd=('ap','std'),brier=('brier','mean'),auc=('auc','mean'))
    summary.to_csv(OUT/'summary.csv')
    # Exact weighted AP, sorting once to keep paired cluster bootstrap inexpensive.
    y=val.label_observed_continuing.to_numpy(); prepared={}
    for label,preds in predictions.items():
        prepared[label]=[]
        for pred in preds:
            for year in [2016,2017,2018]:
                ids=np.flatnonzero(val.t.to_numpy()==year); ids=ids[np.argsort(-pred[ids],kind='stable')]
                ends=np.r_[np.flatnonzero(np.diff(pred[ids])!=0),len(ids)-1]
                prepared[label].append((ids,ends))
    def score(label,w):
        values=[]
        for ids,ends in prepared[label]:
            tp=np.cumsum(w[ids]*y[ids])[ends]; n=np.cumsum(w[ids])[ends]
            precision=np.divide(tp,n,out=np.zeros_like(tp,dtype=float),where=n>0)
            values.append(float(np.diff(np.r_[0,tp])@precision/tp[-1]) if tp[-1]>0 else 0.)
        return np.mean(values)
    for label in prepared: np.testing.assert_allclose(score(label,np.ones(len(val))),summary.loc[label,'ap_mean'],atol=1e-12)
    checkw=np.random.default_rng(123).integers(0,5,len(val))
    for label,preds in predictions.items():
        reference=np.mean([average_precision_score(y[val.t.to_numpy()==t],p[val.t.to_numpy()==t],sample_weight=checkw[val.t.to_numpy()==t]) for p in preds for t in [2016,2017,2018]])
        np.testing.assert_allclose(score(label,checkw),reference,atol=1e-12)
    codes,levels=pd.factorize(pd.MultiIndex.from_frame(val[['i','k4']]))
    comparisons=[('aggregate_no_rel','equal_no_rel'),('aggregate_rel','equal_rel'),
                 ('aggregate_rel','aggregate_no_rel'),('aggregate_no_rel','none_no_rel'),('aggregate_rel','none_rel'),('none_rel','none_no_rel')]
    rng=np.random.default_rng(20260925); draws={f'{a} minus {b}':[] for a,b in comparisons}
    for _ in range(1000):
        counts=np.bincount(rng.integers(0,len(levels),len(levels)),minlength=len(levels)); w=counts[codes]
        scores={label:score(label,w) for label in prepared}
        for a,b in comparisons: draws[f'{a} minus {b}'].append(scores[a]-scores[b])
    intervals={}
    for a,b in comparisons:
        key=f'{a} minus {b}'
        perseed=seedmetrics[seedmetrics.model==a].set_index('seed').ap-seedmetrics[seedmetrics.model==b].set_index('seed').ap
        annual=metrics_frame.groupby(['model','year']).ap.mean()
        intervals[key]={'delta_ap':float(summary.loc[a,'ap_mean']-summary.loc[b,'ap_mean']),
            'conditional_95_interval':np.quantile(draws[key],[.025,.975]).tolist(),
            'positive_seeds':int((perseed>0).sum()),
            'year_differences':{str(t):float(annual.loc[a,t]-annual.loc[b,t]) for t in [2016,2017,2018]}}
    (OUT/'comparisons.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
    after={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}; assert before==after
    manifest={'completed_utc':datetime.now(timezone.utc).isoformat(),'new_model_fits':6,'reused_predictions':12,
              'preserved_inputs_sha256':before,'sources':source_records,'final_holdout_evaluated':False,
              'interval_limitation':'Conditional on reused validation years and fixed previously selected settings; not selection adjusted; shared product/global dependence remains',
              'code_sha256':{p.name:sha(p) for p in [ROOT/'common_factor_candidates.py',Path(__file__)]}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# 市场总额共同因素：获批修改与前后结果','',
       '用户确认后实施。仅新增候选特征及两个实验单元，未修改原始缓存、基准预测、图模型、标签或样本筛选。原始特征、数组和12份复用预测的SHA256前后完全一致。',
       '', '## 定义与研究假设','',
       '固定早期选定的100个目的地，先将剔除目标出口国后的可观测进口额求和，再计算log(1+金额)的一阶差分；金额单位千美元。第二项为截至样本年最近8次变化的样本标准差。保留各小市场的可观测金额贡献，不使用逐节点100万美元门槛截断总额；总额仍要求至少100万美元。它不是全球进口额，也不构成完整申报证明。',
       '假设：共同因素的度量方式可能影响联动指标的条件预测增量；不预设总额口径必然更优。',
       '', '## 实验','',
       '训练样本年2007—2015，验证2016—2018（结果年2017—2019），深度4、150棵树、种子42/2024/2026、四线程。无共同因素和简单平均口径直接复用同参数三种子预测；替换新因子时保留原有列位置。未打开2019—2020校准或末端测试性能。',
       '', '| 口径 | 关系指标 | 平均AP | 种子AP标准差 | Brier |','|---|---|---:|---:|---:|']
    for label in ['none_no_rel','none_rel','equal_no_rel','equal_rel','aggregate_no_rel','aggregate_rel']:
        r=summary.loc[label]; lines.append(f'| {label} | {"有" if label.endswith("_rel") and not label.endswith("no_rel") else "无"} | {r.ap_mean:.6f} | {r.ap_sd:.6f} | {r.brier:.6f} |')
    lines += ['', 'none=无共同因素；equal=原简单平均；aggregate=新市场总额。AP越高越好，Brier越低越好。',
        '', '| 对比 | AP差异 | 条件95%区间 | 正向种子数 | 2016 / 2017 / 2018差异 |','|---|---:|---|---:|---|']
    for key,r in intervals.items():
        lo,hi=r['conditional_95_interval']; yy=' / '.join(f'{v:+.6f}' for v in r['year_differences'].values())
        lines.append(f'| {key} | {r["delta_ap"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r["positive_seeds"]}/3 | {yy} |')
    lines += ['', '按出口国—HS4聚类重采样1000次，三个验证年一同保留。区间条件于既有开发验证年份和参数，未校正此前研究选择，也未消除共同产品/全球冲击依赖。不以最高分单独认定成功，不将负结果删除。',
       '', '## 文件与Git建议','',
       '新增代码：`common_factor_candidates.py`、`experiment_common_factor.py`。新增特征：`data/common_factor_aggregate_development.csv.gz`及`audit/common_factor_aggregate.json`。本目录保留全部六个新模型、逐样本预测、逐年和逐种子指标、配对区间及哈希清单。',
       '建议提交信息：`feat: add aggregate-market factor diagnostic with preserved baselines`。仅建议提交本次明确路径，不自动提交；现有Git暂存区还有此前工作，不应使用全量暂存/提交混入本次。']
    (OUT/'修改与实验记录.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(json.dumps(intervals,indent=2),flush=True)
    print('FINAL_HOLDOUT_NOT_EVALUATED; ORIGINAL_INPUTS_UNCHANGED',flush=True)

if __name__=='__main__': main()
