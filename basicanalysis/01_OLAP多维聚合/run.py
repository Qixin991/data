from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
out=Path(__file__).resolve().parent
d=pd.read_csv(DATA/'选定出口国_HS4双边贸易.csv.gz',dtype={'hs4':str})
cube=d.groupby(['t','i','hs4'],as_index=False).v.sum()
save(cube,out/'年_出口国_产品事实表.csv')
save(cube.groupby('t',as_index=False).v.sum(),out/'上卷_年度总额.csv')
save(cube[cube.t.eq(2024)].sort_values('v',ascending=False),out/'切片_2024出口国产品.csv')
save(d[d.t.eq(2024)].groupby(['i','j'],as_index=False).v.sum().sort_values('v',ascending=False),out/'下钻_2024双边流量.csv')
pivot=cube.pivot_table(index='t',columns='i',values='v',aggfunc='sum');pivot.to_csv(out/'透视_年度出口国.csv',encoding='utf-8-sig')
assert np.isclose(cube.v.sum(),d.v.sum())
top=cube[cube.t.eq(2024)].groupby('hs4').v.sum().nlargest(15).sort_values()
plt.figure(figsize=(9,5));(top/1e6).plot.barh();plt.xlabel('十亿美元（名义）');plt.ylabel('HS4产品');plt.title('2024年选定20个出口经济体的HS84产品出口额');figure(out/'2024产品出口排名.png')
note(out/'分析说明.md','''# OLAP多维分析

对应第四、五章的事实表、上卷、切片、下钻和透视。原始HS6向HS4聚合在公共预处理完成；这里在“年份—出口国—进口国—产品”维度上生成可直接查看的聚合结果。全部金额CSV单位为千现价美元，图中转换为十亿美元。统计范围是早期窗口选出的20个经济体及HS84，不是全球全部贸易。事实表和双边明细总额已核对一致。

未将普通groupby包装为BUC或Multi-way算法实现。数量q没有加总，因为不同产品的重量并不具有相同经济含义。
''')
