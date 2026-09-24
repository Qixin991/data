from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
out=Path(__file__).resolve().parent
d=snapshot();cols=['log_export','destinations','hhi','top_share','growth','growth_mean3','growth_std3']
x=d[cols].clip(lower=d[cols].quantile(.01),upper=d[cols].quantile(.99),axis=1)
x=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(x))
rows=[];labels=[]
for eps in [.5,.75,1.,1.25,1.5,2.]:
 for n in [10,20]:
  lab=DBSCAN(eps=eps,min_samples=n).fit_predict(x);labels.append(lab);valid=lab>=0;k=len(set(lab[valid]))
  score=silhouette_score(x[valid],lab[valid]) if 1<k<valid.sum() else np.nan
  rows.append({'eps':eps,'min_samples':n,'clusters':k,'noise_rate':1-valid.mean(),'silhouette_excluding_noise':score})
scores=pd.DataFrame(rows);save(scores,out/'参数敏感性.csv')
eligible=scores[(scores.clusters>=2)&(scores.noise_rate<=.4)].dropna()
idx=int(eligible.silhouette_excluding_noise.idxmax()) if len(eligible) else 4
chosen=scores.loc[idx];d['cluster']=labels[idx];save(d,out/'2024密度聚类结果.csv');save(d[d.cluster.eq(-1)],out/'噪声候选样本.csv')
save(d.groupby('cluster')[cols].mean().reset_index(),out/'簇与噪声画像.csv')
plt.figure(figsize=(8,5))
for c,g in d.groupby('cluster'):
 plt.scatter(g.log_export,g.hhi,s=14,alpha=.6,label='噪声候选' if c==-1 else f'簇{c}',color='grey' if c==-1 else None)
plt.legend(fontsize=8);plt.xlabel('log(1+出口额 千美元)');plt.ylabel('HHI');plt.title('DBSCAN结果在出口规模与集中度上的投影');figure(out/'密度聚类投影.png')
note(out/'分析说明.md',f'''# DBSCAN密度聚类

对应第十、十一章。2024截面、七项特征、1%/99%截尾、中位数填补与标准化与K-means相同；DBSCAN直接使用七维标准化空间，不依赖PCA/K-means结果，因此单独放置。

展示12组参数敏感性。默认在至少2簇且噪声比例不超过40%的方案中选择排除噪声后的最高轮廓系数；如无合格方案回退eps=1、min_samples=10，并保留失败信息。实际eps={chosen.eps}，min_samples={int(chosen.min_samples)}，簇数{int(chosen.clusters)}，噪声比例{chosen.noise_rate:.2%}。

簇-1为低密度候选点，不等于错误数据、贸易违规或已确认风险。轮廓系数排除了噪声，不能直接与全部样本上的K-means分数比较。图是二维原始特征投影，不是DBSCAN实际距离空间。
''')
