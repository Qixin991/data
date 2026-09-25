"""Approved A/B/C attribution, two factor policies, fixed parameters and three seeds."""
from baselines import *
from common_factor_candidates import NEW,sha
from decompose_relation_risk import build,B_EXTRA,C_EXTRA
from threadpoolctl import threadpool_limits
from datetime import datetime,timezone
OUT=ROOT/'results/relation_attribution_v1'
SEEDS=[42,2024,2026]
PARAMS={'max_depth':4,'n_estimators':150}
OLD=ROOT/'results/diagnosis/trees_threads4'
AGG=ROOT/'results/common_factor_aggregate_v1'
COMMON=['global_factor_last','global_factor_sd']

def main():
    OUT.mkdir(exist_ok=True)
    table=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
    table=table[table.t.between(2007,2018)].reset_index(drop=True)
    factors=pd.read_csv(ROOT/'data/common_factor_aggregate_development.csv.gz',dtype={'k4':str})
    assert factors[['t','i','k4']].equals(table[['t','i','k4']])
    paths=[ROOT/'data/features_lag0_L8.csv.gz',ROOT/'data/trade_arrays.npz',ROOT/'data/common_factor_aggregate_development.csv.gz']
    sources={}
    for seed in SEEDS:
        for label,path in [('none_A',OLD/f'seedcheck_D3_{seed}.csv'),('none_reference',OLD/f'seedcheck_D5_{seed}.csv'),
                           ('aggregate_A',AGG/f'aggregate_no_rel_{seed}_predictions.csv'),('aggregate_reference',AGG/f'aggregate_rel_{seed}_predictions.csv')]:
            sources[label,seed]=path; paths.append(path)
    before={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    gitindex=ROOT.parent/'.git/index'; index_before=sha(gitindex) if gitindex.exists() else None
    extra=build(); assert extra[['t','i','k4']].equals(table[['t','i','k4']])
    val=table[table.t.between(2016,2018)].reset_index(drop=True)
    records=[]; predictions={}; fit_count=0
    for label in ['none_A','aggregate_A','none_reference','aggregate_reference']:
        ps=[]
        for seed in SEEDS:
            saved=pd.read_csv(sources[label,seed],dtype={'k4':str})
            assert saved[['t','i','k4','label_observed_continuing']].equals(val[['t','i','k4','label_observed_continuing']])
            p=saved.p.to_numpy(); ps.append(p)
            records.extend([{'model':label,'seed':seed,**r} for r in yearly(val,p)])
        predictions[label]=np.stack(ps)
    for policy in ['none','aggregate']:
        data=table.copy()
        if policy=='aggregate':
            for old,new in zip(COMMON,NEW): data[old]=factors[new].to_numpy()
            data=data.rename(columns=dict(zip(COMMON,NEW)))
        basecols=[c for c in data.columns if c not in EXCLUDED+RELATIONS+COMMON]
        for name in ['diag_portfolio_variance','offdiag_portfolio_variance','independent_joint_down','excess_joint_down']:
            data[name]=extra[name].to_numpy()
        for group in ['B','C']:
            label=policy+'_'+group; cols=basecols+B_EXTRA+(C_EXTRA if group=='C' else [])
            assert len(cols)==len(set(cols)) and not any(c.startswith('check_') for c in cols)
            train=data[data.t<=2015]; validation=data[data.t.between(2016,2018)]
            tr=preprocessor(cols); x=tr.fit_transform(train); v=tr.transform(validation); ps=[]
            for seed in SEEDS:
                target=OUT/f'{label}_{seed}_predictions.csv'; meta=OUT/f'{label}_{seed}.json'
                spec={'features':cols,'parameters':PARAMS,'seed':seed,'threads':4,
                      'input_hashes':{k:v for k,v in before.items() if k.startswith('data/')},
                      'decomposition_sha256':sha(ROOT/'data/relation_attribution_development.csv.gz')}
                if target.exists():
                    assert json.loads(meta.read_text())==spec
                    saved=pd.read_csv(target,dtype={'k4':str}); assert saved[['t','i','k4']].equals(val[['t','i','k4']])
                    p=saved.p.to_numpy()
                else:
                    with threadpool_limits(4):
                        m=tree(PARAMS,seed); m.fit(x,train.label_observed_continuing,sample_weight=weights(train))
                        p=m.predict_proba(v)[:,1]
                    saved=val[['t','i','k4','label_observed_continuing']].copy(); saved['p']=p; saved.to_csv(target,index=False)
                    with (OUT/f'{label}_{seed}.pkl').open('wb') as f: pickle.dump((tr,m),f)
                    meta.write_text(json.dumps(spec,indent=2),encoding='utf-8'); fit_count+=1
                ps.append(p); records.extend([{'model':label,'seed':seed,**r} for r in yearly(val,p)])
                print('ATTRIBUTION',label,seed,'AP',np.mean([r['ap'] for r in yearly(val,p)]),flush=True)
            predictions[label]=np.stack(ps)
    frame=pd.DataFrame(records); frame.to_csv(OUT/'annual_metrics.csv',index=False)
    sm=frame.groupby(['model','seed'],as_index=False)[['ap','auc','brier','logloss']].mean(); sm.to_csv(OUT/'seed_metrics.csv',index=False)
    summary=sm.groupby('model').agg(ap_mean=('ap','mean'),ap_sd=('ap','std'),brier=('brier','mean'),auc=('auc','mean'))
    summary.to_csv(OUT/'summary.csv')
    y=val.label_observed_continuing.to_numpy(); prepared={}
    for label,ps in predictions.items():
        prepared[label]=[]
        for p in ps:
            for t in [2016,2017,2018]:
                ids=np.flatnonzero(val.t.to_numpy()==t); ids=ids[np.argsort(-p[ids],kind='stable')]
                prepared[label].append((ids,np.r_[np.flatnonzero(np.diff(p[ids])!=0),len(ids)-1]))
    def score(label,w):
        scores=[]
        for ids,ends in prepared[label]:
            tp=np.cumsum(w[ids]*y[ids])[ends]; n=np.cumsum(w[ids])[ends]
            precision=np.divide(tp,n,out=np.zeros_like(tp,dtype=float),where=n>0)
            scores.append(float(np.diff(np.r_[0,tp])@precision/tp[-1]) if tp[-1]>0 else 0.)
        return float(np.mean(scores))
    checkw=np.random.default_rng(123).integers(0,5,len(val))
    for label,ps in predictions.items():
        np.testing.assert_allclose(score(label,np.ones(len(val))),summary.loc[label,'ap_mean'],atol=1e-12)
        reference=np.mean([average_precision_score(y[val.t.to_numpy()==t],p[val.t.to_numpy()==t],sample_weight=checkw[val.t.to_numpy()==t]) for p in ps for t in [2016,2017,2018]])
        np.testing.assert_allclose(score(label,checkw),reference,atol=1e-12)
    comparisons=[(f'{p}_{a}',f'{p}_{b}') for p in ['none','aggregate'] for a,b in [('C','B'),('B','A'),('C','A')]]
    codes,levels=pd.factorize(pd.MultiIndex.from_frame(val[['i','k4']]))
    rng=np.random.default_rng(20260926); draws={f'{a} minus {b}':[] for a,b in comparisons}
    for _ in range(1000):
        counts=np.bincount(rng.integers(0,len(levels),len(levels)),minlength=len(levels)); w=counts[codes]
        scores={label:score(label,w) for label in prepared if not label.endswith('reference')}
        for a,b in comparisons: draws[f'{a} minus {b}'].append(scores[a]-scores[b])
    intervals={}; annual=frame.groupby(['model','year']).ap.mean()
    for a,b in comparisons:
        diff=sm[sm.model==a].set_index('seed').ap-sm[sm.model==b].set_index('seed').ap
        intervals[f'{a} minus {b}']={'delta_ap':float(summary.loc[a,'ap_mean']-summary.loc[b,'ap_mean']),
             'conditional_95_interval':np.quantile(draws[f'{a} minus {b}'],[.025,.975]).tolist(),
             'delta_brier':float(summary.loc[a,'brier']-summary.loc[b,'brier']),
             'seed_differences':{str(s):float(diff.loc[s]) for s in SEEDS},
             'year_differences':{str(t):float(annual.loc[a,t]-annual.loc[b,t]) for t in [2016,2017,2018]}}
    (OUT/'comparisons.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
    after={p.relative_to(ROOT).as_posix():sha(p) for p in paths}; assert before==after
    assert (sha(gitindex) if gitindex.exists() else None)==index_before
    manifest={'completed_utc':datetime.now(timezone.utc).isoformat(),'model_checkpoints':12,'new_fits_this_run':fit_count,
       'reused_A_predictions':6,'reused_reference_predictions':6,'preserved_inputs_sha256':before,
       'git_index_unchanged':True,'git_commit_executed':False,'final_holdout_evaluated':False,'hyperparameter_search':False,
       'code_sha256':{p.name:sha(p) for p in [ROOT/'decompose_relation_risk.py',Path(__file__)]}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# 联动信息增量来源分解：实验与修改记录','',
        '按用户批准的A/B/C×两种共同因素口径执行。新增12个模型，无参数搜索；A组6份预测和原完整关系模型6份预测直接复用。所有正负结果保留。',
        '', '## 研究问题及实现','',
        '检验控制单市场风险与数据覆盖后，联动信息是否仍有预测增量。主要对比C−B，B−A用于检查原先增益是否可由风险/覆盖补充解释；不作因果归因。',
        'A：原基础模型。B：A＋对角方差贡献、独立下行基准、协方差可用份额覆盖、可估计成对暴露。C：B＋非对角协方差贡献、平均相关性、超额共同下行、正/负相关暴露。reference：原完整关系模型，仅作已存在结果的参照。',
        '同一Ledoit-Wolf协方差矩阵拆分对角/非对角项，二者之和须等于原组合方差。其收缩估计使用联合样本，因此B不是完全不含跨市场估计信息的纯边际模型，不能将结果解释为彻底隔离所有依赖。',
        '每对市场仅用相同共同有效年份计算各自下行频率p_j、p_l；独立基准为p_j*p_l，超额共同下行为实际同时下行频率减该基准。使用原相关指标相同的有效对集合、份额乘积及归一化分母，保持缺失口径。该基准是假想独立参照，不是金融风险概率模型。',
        '复用原8年历史窗口、至少6个有效观测、既有市场门槛、样本与标签。训练2007—2015，开发验证2016—2018；深度4、150棵树、种子42/2024/2026、四线程。A列序保持不变，B和C按事先列出的顺序追加；填补/标准化仅拟合训练期。',
        '', '| 口径/组别 | 平均AP | 种子AP标准差 | 平均Brier | 平均AUC |','|---|---:|---:|---:|---:|']
    for label in ['none_A','none_B','none_C','none_reference','aggregate_A','aggregate_B','aggregate_C','aggregate_reference']:
        r=summary.loc[label]; lines.append(f'| {label} | {r.ap_mean:.6f} | {r.ap_sd:.6f} | {r.brier:.6f} | {r.auc:.6f} |')
    lines+=['','none=不加入共同因素；aggregate=已批准的市场总额口径。AP/AUC越高越好，Brier越低越好。',
       '', '| 配对对比 | AP差异 | 条件95%区间 | Brier差异 | 三个年度AP差异 |','|---|---:|---|---:|---|']
    for key,r in intervals.items():
        lo,hi=r['conditional_95_interval']; yy=' / '.join(f'{v:+.6f}' for v in r['year_differences'].values())
        lines.append(f'| {key} | {r["delta_ap"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r["delta_brier"]:+.6f} | {yy} |')
    lines+=['','年度顺序2016/2017/2018，对应结果年2017/2018/2019；种子明细见seed_metrics.csv及comparisons.json。',
        '区间按出口国—HS4聚类重采样1000次，同组跨年一起抽样，比较三种子性能平均。条件于已多次使用的验证年份、此前确定的参数和特征设计，未校正选择，亦未消除共同产品/全球冲击依赖。特征数量改变也会影响树拟合及随机列抽样；这是受控预测对照，不是因果或唯一贡献分配。',
        '', '## 检查与修改范围','',
        '检查通过：共同有效年份的手工成对频率对照、分解之和与原指标逐样本勾稽、数值边界/缺失审计、加权AP对sklearn校验，以及原始特征/数组/总额候选/12份复用预测修改前后SHA256一致。Git index前后哈希一致。',
        '新增decompose_relation_risk.py、experiment_relation_attribution.py、独立特征缓存及按年检查点、特征审计与本目录模型/预测/指标/清单。研究协议和恢复记录只追加本次工作。不修改旧模型、旧结果、标签或样本；未查看最终测试性能。',
        '', '## Git建议','',
        '`feat: decompose relation risk and record fixed-budget attribution experiments`',
        '仅建议纳入本次新代码、缓存/审计、结果目录及协议/恢复记录的对应增量；未运行git add或git commit，不将既有暂存改动混入本次。实验结论另存实验结论.md，重新汇总不覆盖人工研究判断。']
    (OUT/'修改与实验记录.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary.to_string(),flush=True); print(json.dumps(intervals,indent=2),flush=True)
    print('SCOPE_COMPLETE; HOLDOUT_UNOPENED; INPUTS_AND_GIT_INDEX_UNCHANGED',flush=True)

if __name__=='__main__': main()
