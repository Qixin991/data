from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score,adjusted_rand_score
out=Path(__file__).resolve().parent
d=snapshot();cols=['log_export','destinations','hhi','top_share','growth','growth_mean3','growth_std3']
x=d[cols].clip(lower=d[cols].quantile(.01),upper=d[cols].quantile(.99),axis=1)
x=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(x))
pca=PCA(n_components=.90,svd_solver='full');z=pca.fit_transform(x)
rows=[];fits=[]
for k in range(2,7):
 m=KMeans(n_clusters=k,n_init=20,random_state=42).fit(z);fits.append(m)
 rows.append({'k':k,'silhouette':silhouette_score(z,m.labels_),'davies_bouldin':davies_bouldin_score(z,m.labels_),'inertia':m.inertia_})
scores=pd.DataFrame(rows);save(scores,out/'聚类数比较.csv');model=fits[int(scores.silhouette.argmax())]
d['cluster']=model.labels_;d['PC1']=z[:,0];d['PC2']=z[:,1];save(d,out/'2024聚类结果.csv')
save(d.groupby('cluster')[cols].mean().reset_index(),out/'各簇原始指标均值.csv')
save(d.groupby('cluster').size().rename('n').reset_index(),out/'各簇样本量.csv')
save(pd.DataFrame(pca.components_.T,index=cols,columns=[f'PC{i+1}' for i in range(z.shape[1])]).reset_index(names='feature'),out/'PCA载荷.csv')
save(pd.DataFrame({'component':range(1,z.shape[1]+1),'variance_ratio':pca.explained_variance_ratio_,'cumulative':pca.explained_variance_ratio_.cumsum()}),out/'PCA解释方差.csv')
stable=[]
for seed in [7,21,99]:
 other=KMeans(n_clusters=model.n_clusters,n_init=20,random_state=seed).fit_predict(z)
 stable.append({'seed':seed,'ARI_vs_seed42':adjusted_rand_score(model.labels_,other)})
save(pd.DataFrame(stable),out/'随机种子稳定性.csv')
plt.figure(figsize=(8,5));sc=plt.scatter(z[:,0],z[:,1],c=model.labels_,s=13,cmap='tab10',alpha=.6);plt.xlabel('PC1');plt.ylabel('PC2');plt.title('2024出口组合 PCA与K-means');plt.colorbar(sc,label='簇编号');figure(out/'聚类散点图.png')
note(out/'分析说明.md',f'''# 标准化 PCA K-means串行分析

对应第三章和第十、十一章。样本为2024年基期出口至少100万美元的{len(d)}个出口国—HS4产品组合。七项特征先在该截面进行1%/99%截尾、中位数填补、标准化，再PCA保留至少90%方差（实际{z.shape[1]}个主成分），最后K-means聚类。截尾仅影响聚类距离，结果均值表保留原始量纲和未截尾值。

在k=2至6中按轮廓系数选择k={model.n_clusters}，轮廓系数{scores.silhouette.max():.4f}；另报DB指标和随机种子ARI稳定性。这些指标是同一描述性截面内的内部评价，不代表未来预测或真实经济类别。簇编号无高低风险含义。二维图只展示前两个主成分，聚类使用全部保留成分。所有串行步骤集中在本文件夹。
''')
