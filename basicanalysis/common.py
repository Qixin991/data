"""各方法共用的数据读取、作图与时间外分类评价。"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score,roc_auc_score,brier_score_loss,log_loss,
    precision_score,recall_score,f1_score,confusion_matrix,precision_recall_curve,roc_curve)

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'00_数据预处理与描述统计'
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def config(): return json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
def panel(): return pd.read_csv(DATA/'产品出口面板.csv',dtype={'hs4':str})
def save(df,path): df.to_csv(path,index=False,encoding='utf-8-sig')
def figure(path): plt.tight_layout();plt.savefig(path,dpi=160,bbox_inches='tight');plt.close()
def note(path,text): Path(path).write_text(text,encoding='utf-8')
def snapshot():
    d=panel();d=d[d.t.eq(2024)&d.export_value.ge(1000)].copy()
    return d

def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p); pred=p>=.5
    k=max(1,int(np.ceil(.1*len(y))));top=np.argsort(-p,kind='stable')[:k]
    return {'n':len(y),'risk_rate':y.mean(),'AP':average_precision_score(y,p) if y.sum() else np.nan,
      'ROC_AUC':roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
      'Brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),
      'precision_at_10pct':y[top].mean(),'recall_at_10pct':y[top].sum()/y.sum() if y.sum() else np.nan,
      'precision_0.5':precision_score(y,pred,zero_division=0),'recall_0.5':recall_score(y,pred,zero_division=0),
      'F1_0.5':f1_score(y,pred,zero_division=0)}

def classification(out, models, label, standardize=False):
    out=Path(out);d=panel();d=d[d.eligible].copy();cols=config()['features']
    train=d[d.split.eq('train')];val=d[d.split.eq('validation')];test=d[d.split.eq('test')]
    assert train.t.max()+1<=val.t.min() and val.t.max()+1<test.t.min()
    assert not set(cols)&{'risk','next_growth','next_export','split'}
    scores=[]; fitted=[]
    for name,model in models:
        steps=[('imputer',SimpleImputer(strategy='median',add_indicator=True))]
        if standardize:steps.append(('scale',StandardScaler()))
        steps.append(('model',model));pipe=Pipeline(steps)
        pipe.fit(train[cols],train.risk.astype(int));p=pipe.predict_proba(val[cols])[:,1]
        scores.append({'candidate':name,**metrics(val.risk,p)});fitted.append(pipe)
    selection=pd.DataFrame(scores);save(selection,out/'验证集选模.csv')
    best=int(selection.AP.to_numpy().argmax());pipe=fitted[best]
    # Freeze the train-fitted model. No refit or tuning on test/calibration observations.
    predictions=test[['t','i','country_name','hs4','risk','next_growth']].copy()
    predictions['probability']=pipe.predict_proba(test[cols])[:,1]
    predictions['prediction_0.5']=(predictions.probability>=.5).astype(int)
    save(predictions,out/'测试集逐样本预测.csv')
    rows=[]
    base=float(train.risk.mean())
    for year,g in predictions.groupby('t'):
        rows.append({'model':label,'year':year,**metrics(g.risk,g.probability)})
        rows.append({'model':'训练期风险率基线','year':year,**metrics(g.risk,np.full(len(g),base))})
    scores=pd.DataFrame(rows);save(scores,out/'逐年测试指标.csv')
    pooled=pd.DataFrame([{'model':label,**metrics(predictions.risk,predictions.probability)},
      {'model':'训练期风险率基线',**metrics(predictions.risk,np.full(len(predictions),base))}]);save(pooled,out/'合并测试指标.csv')
    avg=scores.groupby('model',as_index=False).mean(numeric_only=True).drop(columns=['year','n']);save(avg,out/'年份等权测试指标.csv')
    cm=confusion_matrix(predictions.risk,predictions['prediction_0.5'],labels=[0,1])
    save(pd.DataFrame(cm,columns=['预测非风险','预测风险']).assign(真实类别=['非风险','风险']),out/'混淆矩阵.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    precision,recall,_=precision_recall_curve(predictions.risk,predictions.probability)
    axes[0].plot(recall,precision);axes[0].axhline(predictions.risk.mean(),ls='--',color='grey');axes[0].set(xlabel='召回率',ylabel='精确率',title=f'{label} PR曲线')
    fpr,tpr,_=roc_curve(predictions.risk,predictions.probability);axes[1].plot(fpr,tpr);axes[1].plot([0,1],[0,1],'--',color='grey');axes[1].set(xlabel='假阳性率',ylabel='真阳性率',title='ROC曲线')
    figure(out/'测试集评价曲线.png')
    names=pipe.named_steps['imputer'].get_feature_names_out(cols)
    m=pipe.named_steps['model']
    values=m.feature_importances_ if hasattr(m,'feature_importances_') else m.coef_[0]
    importance=pd.DataFrame({'feature':names,'value':values});importance['absolute']=importance.value.abs();importance=importance.sort_values('absolute',ascending=False)
    save(importance,out/'特征重要性或系数.csv')
    if label=='决策树':
        from sklearn.tree import export_text,plot_tree
        note(out/'决策树规则.txt',export_text(m,feature_names=list(names)))
        plt.figure(figsize=(17,8));plot_tree(m,feature_names=list(names),class_names=['非风险','风险'],max_depth=2,filled=True,fontsize=8);figure(out/'决策树前两层.png')
    result=avg[avg.model.eq(label)].iloc[0]
    note(out/'分析说明.md',f'''# {label}基础分析

对应大纲第八章。目标为出口经济体—HS4产品在下一年名义美元出口额下降至少20%。仅纳入基期至少100万美元且次年仍有正出口记录的样本；未知退出不补零，因此存在持续出口样本选择偏差。

训练2007—2015（{len(train)}条），验证2016—2018（{len(val)}条），测试2021—2023（{len(test)}条）。2019—2020保留未用于本次基础模型校准。缺失中位数及标准化仅在训练集拟合，增加缺失指示。选模依据验证集AP；选中 {models[best][0]}，随后冻结模型。阈值固定0.5，未用测试集调参。所有模型使用相同13项输入及样本。

测试年份等权AP为 {result.AP:.4f}，ROC-AUC为 {result.ROC_AUC:.4f}，Brier为 {result.Brier:.4f}。AP采用Average Precision，不是梯形PR曲线面积。另附逐年、合并指标和训练期风险率基线。相同分数的前10%采用稳定行顺序打破并列，基线的该项不具排序意义。

特征重要性/系数反映模型关联，不是因果效应。未进行概率校准、国家外推、置信区间或图模型创新验证；本次是课程基础方法结果。统一修订版历史数据的时间外测试不等于真实发布时点的回测。

运行：`python "{out.relative_to(ROOT)}/run.py"`（工作目录为基础分析，或使用脚本绝对路径）。
''')
    return pipe
