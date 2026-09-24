from pathlib import Path
from itertools import combinations
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
out=Path(__file__).resolve().parent
d=panel();d=d[d.eligible];train=d[d.split.eq('train')];test=d[d.split.eq('test')]
cols=['hhi','log_export','destinations','growth']
thresholds=train[cols].median()
note(out/'训练期离散化阈值.json',thresholds.to_json(indent=2))
def transactions(frame):
 items={}
 for c in cols:
  # Missing growth is a separate item, never silently assigned to low growth.
  items[c+'_high']=(frame[c]>thresholds[c]).to_numpy()
  items[c+'_low']=(frame[c]<=thresholds[c]).to_numpy()
  if frame[c].isna().any():items[c+'_missing']=frame[c].isna().to_numpy()
 items['risk_next_year']=(frame.risk==1).to_numpy()
 items['no_risk_next_year']=(frame.risk==0).to_numpy()
 return items
tx=transactions(train);te=transactions(test);supports={};previous=set();frequent=[]
min_support=.05
for size in range(1,4):
 current=set()
 for combo in combinations(sorted(tx),size):
  key=frozenset(combo)
  if size>1 and not all(frozenset(s) in previous for s in combinations(combo,size-1)):continue
  support=float(np.logical_and.reduce([tx[c] for c in combo]).mean())
  if support>=min_support:
   current.add(key);supports[key]=support;frequent.append({'items':' & '.join(combo),'size':size,'support':support,'count':int(round(support*len(train)))})
 previous=current
save(pd.DataFrame(frequent),out/'频繁项集.csv')
rules=[]
for key,support in supports.items():
 if 'risk_next_year' not in key or len(key)<2:continue
 antecedent=key-{'risk_next_year'};a=supports[antecedent];conf=support/a;lift=conf/supports[frozenset(['risk_next_year'])]
 if conf<.20 or lift<=1:continue
 trainmask=np.logical_and.reduce([tx[c] for c in antecedent]);testmask=np.logical_and.reduce([te.get(c,np.zeros(len(test),dtype=bool)) for c in antecedent])
 testconf=float(test.loc[testmask,'risk'].mean()) if testmask.sum() else np.nan
 rules.append({'antecedent':' & '.join(sorted(antecedent)),'consequent':'risk_next_year','train_support':support,'train_antecedent_n':int(trainmask.sum()),'train_confidence':conf,'train_lift':lift,'test_antecedent_n':int(testmask.sum()),'test_confidence':testconf,'test_lift':testconf/test.risk.mean()})
r=pd.DataFrame(rules,columns=['antecedent','consequent','train_support','train_antecedent_n','train_confidence','train_lift','test_antecedent_n','test_confidence','test_lift'])
if len(r):r=r.sort_values('train_lift',ascending=False)
save(r,out/'下行风险关联规则.csv')
note(out/'分析说明.md',f'''# 离散化 Apriori关联规则

对应第三章和第六章。每条交易是一个出口国—HS4产品—年份观测，不是购物订单。训练期2007—2015的HHI、规模、目的地数量和当期增长率按训练中位数转为高/低项，缺失单列；次年下降20%的标签作为后件。此处“高”只表示高于训练期中位数，并非经济政策门槛。

实现Apriori的逐层候选生成与子集剪枝，最多三项，最低支持度5%；只展示后件为下行风险、置信度至少20%、提升度大于1的规则。共{len(frequent)}个频繁项集、{len(r)}条满足展示条件的规则。阈值预先固定，仅训练集发现规则，测试2021—2023用于同规则描述性核验，没有按测试效果筛选。

支持度=P(前件且风险)，置信度=P(风险|前件)，提升度=置信度/总体风险率。重复经济体与产品使交易不独立；未经多重检验或聚类相关调整，规则不能作为显著性结论或因果机制。若CSV只有表头，表示设定下没有合格规则，不应降低阈值直至出现预期结果。
''')
