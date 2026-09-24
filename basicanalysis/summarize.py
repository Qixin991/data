"""汇总已运行的结果并校验公共数据及分类输出。"""
from pathlib import Path
import json,html,platform,importlib.metadata
import numpy as np
import pandas as pd
from common import ROOT,DATA,panel,config,save,note

def mdtable(frame):
    cols=list(frame.columns)
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)

d=panel();audit=pd.read_csv(DATA/'逐年数据审计.csv')
selected=pd.read_csv(DATA/'早期出口国选择.csv');selected=selected[selected.selected]
model_results=[];predictions=[]
for folder in ['05_决策树分类','06_逻辑回归分类','07_随机森林分类']:
    a=pd.read_csv(ROOT/folder/'年份等权测试指标.csv')
    model_results.append(a[a.model.ne('训练期风险率基线')])
    predictions.append(pd.read_csv(ROOT/folder/'测试集逐样本预测.csv',dtype={'hs4':str}))
model_results.append(a[a.model.eq('训练期风险率基线')])
comparison=pd.concat(model_results,ignore_index=True)
save(comparison,ROOT/'分类模型对比.csv')
cluster=pd.read_csv(ROOT/'02_标准化_PCA_KMeans'/'聚类数比较.csv')
rules=pd.read_csv(ROOT/'04_离散化_Apriori关联规则'/'下行风险关联规则.csv')
test=d[d.eligible & d.split.eq('test')]
checks=[]
def check(name,condition):
    checks.append({'check':name,'passed':bool(condition)})
    assert condition,name
check('完整30个年度文件',len(audit)==30 and set(audit.year)==set(range(1995,2025)))
check('HS84原始键无重复',audit.hs84_duplicate_keys.sum()==0)
check('面板键唯一',not d.duplicated(['t','i','hs4']).any())
check('2024标签均未知',d.loc[d.t.eq(2024),'risk'].isna().all())
check('已知标签符合下降20%定义',(d.loc[d.risk.notna(),'risk']==(d.loc[d.risk.notna(),'next_growth']<=-.2).astype(int)).all())
check('HHI在合法区间',d.hhi.dropna().between(0,1+1e-10).all())
check('宏观覆盖份额在合法区间',all(d[c].dropna().between(0,1+1e-10).all() for c in config()['features'] if c.endswith('_coverage')))
check('仅以早期完整12年选20出口国',len(selected)==20 and selected.observed_years.eq(12).all())
check('特征无未来标签列',not set(config()['features'])&{'risk','next_growth','next_export'})
next_actual=d[['t','i','hs4','export_value']].copy()
next_actual['t']=next_actual.t-1
next_actual=next_actual.rename(columns={'export_value':'expected_next'})
joined=d.merge(next_actual,on=['t','i','hs4'],how='left',validate='one_to_one')
check('下一年标签值与相邻日历年原始面板匹配',np.allclose(joined.next_export,joined.expected_next,equal_nan=True))
raw=pd.read_csv(DATA/'年度缓存'/'2024.csv.gz',dtype={'hs4':str})
for row in d[d.t.eq(2024)&d.export_value.notna()].sample(5,random_state=42).itertuples():
    flows=raw[(raw.i==row.i)&(raw.hs4==row.hs4)].v
    check(f'2024金额与HHI回查_{row.i}_{row.hs4}',np.isclose(flows.sum(),row.export_value) and np.isclose(((flows/flows.sum())**2).sum(),row.hhi))
key=['t','i','hs4']
for i,p in enumerate(predictions):
    check(f'模型{i+1}预测数量等于公共测试集',len(p)==len(test))
    check(f'模型{i+1}概率合法',p.probability.notna().all() and p.probability.between(0,1).all())
    check(f'模型{i+1}测试键一致',p[key].reset_index(drop=True).equals(predictions[0][key].reset_index(drop=True)))
save(pd.DataFrame(checks),ROOT/'运行校验.csv')
versions={pkg:importlib.metadata.version(pkg) for pkg in ['numpy','pandas','scikit-learn','scipy','matplotlib','xlrd']}
versions['python']=platform.python_version();note(ROOT/'运行环境.json',json.dumps(versions,ensure_ascii=False,indent=2))
note(ROOT/'requirements-lock.txt','\n'.join(f'{p}=={v}' for p,v in versions.items() if p!='python')+'\n')
best=cluster.loc[cluster.silhouette.idxmax()]
main=comparison[['model','AP','ROC_AUC','Brier','precision_at_10pct','recall_at_10pct']].rename(columns={'model':'模型','precision_at_10pct':'前10%精确率','recall_at_10pct':'前10%召回率'})
overallrisk=float(test.risk.mean())
winner=comparison[comparison.model.ne('训练期风险率基线')].sort_values('AP',ascending=False).iloc[0]
text=f'''# HS84贸易数据基础分析总览

已完成8组文件夹中的数据准备与课程基础方法分析。数据来自现有BACI HS92 V202601和三份WDI表；按研究方案从HS84开始，聚合到HS4。分类结果属于持续正出口样本的时间外基础预测，聚类属于2024截面描述。

## 数据规模与样本口径

- 扫描1995—2024共30个年度文件，原始记录{int(audit.all_rows.sum()):,}条；其中HS84记录{int(audit.hs84_raw_rows.sum()):,}条。
- 早期1995—2006平均规模与连续性选择20个出口经济体，HS4产品共{d.hs4.nunique()}类。
- 观测到正出口的出口国—产品—年份共{d.export_value.notna().sum():,}条；公共完整年度索引{len(d):,}条，缺失格保留为空。
- 分类训练{int((d.eligible & d.split.eq('train')).sum()):,}条、验证{int((d.eligible & d.split.eq('validation')).sum()):,}条、测试{len(test):,}条；测试风险率{overallrisk:.2%}。
- 出口经济体：{', '.join(selected.country_name)}。

## 分类方法实际结果

下表为2021、2022、2023三个测试年份的指标等权平均。AP越高越好，Brier越低越好。各年风险率不同，AP必须结合基线解读。

{mdtable(main)}

在本次固定协议下，三个基础分类器中AP最高的是{winner.model}（{winner.AP:.4f}）。这只是此数据与此划分下的比较，未经不确定性检验，不能据此声称显著领先或已验证研究方案中的创新。简单常数基线没有排序能力，其前10%指标只由并列处理决定。

## 聚类与关联规则

PCA后K-means在k=2至6中选出k={int(best.k)}，轮廓系数{best.silhouette:.4f}；详见第二组文件夹的簇画像和随机种子稳定性。DBSCAN另列参数敏感性和噪声候选；噪声不是已确认风险。

Apriori在训练期发现{len(rules)}条满足预设支持度、置信度、提升度条件的下行风险规则。规则CSV同时列出测试期支持前件样本数、置信度和提升度；不按测试效果删选规则。

## 文件组织与阅读顺序

1. `00_数据预处理与描述统计`：公共数据、原始来源、审计、缺失率、概览图和特征口径。
2. `01_OLAP多维聚合`：事实表、年度上卷、2024切片、双边下钻与透视。
3. `02_标准化_PCA_KMeans`：依次完成标准化、PCA、K-means，串行步骤放在一起。
4. `03_DBSCAN密度聚类`：独立密度聚类分析。
5. `04_离散化_Apriori关联规则`：离散化和Apriori归并。
6. `05_决策树分类`、`06_逻辑回归分类`、`07_随机森林分类`：独立算法分别保存，公用同一评价协议。

每组包含可重跑的Python脚本、结果表及分析说明；有适合的图形时提供PNG。运行方式见README。

## 解释边界和校验

缺失下一年贸易记录没有直接置零；目前分类仅针对持续正出口样本，无法反映全部退出风险。2024缺少2025标签，仅参与无监督描述。名义美元变化包含价格和数量因素，不能解释为实际销量或纯需求变化。宏观数据使用统一修订版本，时间划分不等于真实实时回测。

全部{len(checks)}项运行校验通过，包括年度完整性、HS84键唯一、标签边界、HHI/宏观覆盖区间以及三模型测试样本一致性。每年HS6到HS4聚合金额已核对守恒。没有进行统计显著性检验、完整退出识别、图模型训练或论文复现；课程最终项目中的最新论文实现和综述仍是后续工作。
'''
note(ROOT/'分析总览.md',text)
# Standalone local HTML overview, without network resources.
sections=''
for line in text.splitlines():
    if line.startswith('# '):sections+=f'<h1>{html.escape(line[2:])}</h1>'
    elif line.startswith('## '):sections+=f'<h2>{html.escape(line[3:])}</h2>'
    elif line.startswith('|'):continue
    elif line.strip():sections+=f'<p>{html.escape(line)}</p>'
    if line.startswith('下表为'):sections+=main.round(4).to_html(index=False,border=0)
images=[('00_数据预处理与描述统计/数据概览.png','数据概览'),('01_OLAP多维聚合/2024产品出口排名.png','产品结构'),('02_标准化_PCA_KMeans/聚类散点图.png','PCA与K-means'),('07_随机森林分类/测试集评价曲线.png','随机森林测试曲线')]
gallery=''.join(f'<h2>{title}</h2><img src="{path}" alt="{title}">' for path,title in images)
note(ROOT/'分析总览.html',f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HS84贸易数据基础分析</title><style>body{{max-width:1050px;margin:40px auto;padding:0 28px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#182738;background:#fafbfc}}h1{{font-size:30px}}h2{{margin-top:34px;font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{border:1px solid #dae1e8;padding:9px;text-align:right}}th{{background:#e8eff7}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;background:white;margin:15px 0}}p{{overflow-wrap:anywhere}}</style>{sections}{gallery}</html>''')
print('汇总与校验完成')
