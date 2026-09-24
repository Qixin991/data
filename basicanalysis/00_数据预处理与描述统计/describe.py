from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
out=Path(__file__).resolve().parent
d=panel();cols=config()['features'];observed=d[d.export_value.notna()]
save(observed[cols].describe(percentiles=[.01,.25,.5,.75,.99]).T.reset_index(names='feature'),out/'描述统计.csv')
save(pd.DataFrame({'feature':cols,'missing_rate':[observed[c].isna().mean() for c in cols]}),out/'特征缺失率.csv')
cor=observed[cols].corr(method='spearman');cor.to_csv(out/'Spearman相关矩阵.csv',encoding='utf-8-sig')
save(d[d.eligible].groupby('t').risk.agg(['size','mean']).reset_index().rename(columns={'size':'n','mean':'risk_rate'}),out/'逐年下行风险率.csv')
fig,axes=plt.subplots(2,2,figsize=(11,8))
axes[0,0].hist(observed.log_export,bins=50);axes[0,0].set(title='出口规模分布',xlabel='log(1+千美元)',ylabel='样本数')
axes[0,1].hist(observed.hhi,bins=40);axes[0,1].set(title='出口目的地集中度',xlabel='HHI',ylabel='样本数')
annual=observed.groupby('t').export_value.sum()/1e9
axes[1,0].plot(annual.index,annual.values);axes[1,0].set(title='选定经济体HS84出口总额',xlabel='年份',ylabel='万亿美元（名义）')
r=d[d.eligible].groupby('t').risk.mean();axes[1,1].plot(r.index,r.values);axes[1,1].set(title='次年下降至少20%的样本比例',xlabel='特征年份t',ylabel='风险率')
figure(out/'数据概览.png')
plt.figure(figsize=(10,8));plt.imshow(cor,vmin=-1,vmax=1,cmap='RdBu_r');plt.xticks(range(len(cols)),cols,rotation=90,fontsize=8);plt.yticks(range(len(cols)),cols,fontsize=8);plt.colorbar(label='Spearman相关系数');plt.title('合并面板的描述性相关性');figure(out/'相关性热力图.png')
aud=pd.read_csv(out/'逐年数据审计.csv')
note(out/'分析说明.md',f'''# 数据预处理与描述统计

对应第二、三章。读取BACI HS92 V202601的1995—2024年全部年度文件，共扫描{aud.all_rows.sum():,}条记录；保留HS84机械类{aud.hs84_raw_rows.sum():,}条，再按年份、出口国、进口国、HS4聚合。所有年度保留缓存，重新运行可跳过已完成年份。缓存随原始数据版本固定；更换输入版本时应使用新的输出目录。

出口经济体仅按1995—2006年HS84平均出口额排序，并要求12年均有观测，选前20。历史实体代码单独保留、不接续到今天国家。WDI按ISO3匹配，元数据Region为空的区域汇总项排除，历史实体不匹配宏观数据。宏观表年份仅取1995—2024；不使用2025数值。

金额单位为千现价美元；没有以CPI充当产品平减指数。宏观增速与通胀单位为百分点。目的地宏观变量按可观测目的地出口份额归一加权，同时保留覆盖份额；GDP先log(1+美元)再加权，不能解释成平均GDP的对数。HHI和最大份额使用全部目的地。选定20国并不代表全部世界贸易。

按完整年度索引计算增长和过去3年均值/标准差（至少2个有效增长），不跨缺失年份计算增长。次年未观测时标签未知，不补零。2024没有标签；基本分类仅使用基期至少1000千美元且次年仍有正记录的持续出口样本。这会漏掉无法确认的完全退出，必须保留该选择偏差。未实现研究方案的图覆盖90%门槛、目的地剔除本国面板和图学习，它们不属于本次课程基础方法。

描述统计、合并相关矩阵涵盖可观测1995—2024面板，只用于数据概览，不用于选模。缺失值原样保留于公共数据，分类模型内部只在训练期填补。原始文件不改写。HS84重复键检查与聚合金额守恒检查已经运行；未对非HS84进行完整重复键审计。
''')
