"""Screen samples using 1995-2006 only; does not access later labels or outcomes."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT/'data', ROOT/'audit'/'early_sample'
OUT.mkdir(parents=True,exist_ok=True)
parts=[]
for t in range(1995,2007):
    frame=pd.read_csv(DATA/f'hs84_85_hs4_{t}.csv.gz',dtype={'k4':str},usecols=['t','i','j','k4','v'])
    parts.append(frame[frame.k4.str.startswith('84')])
d=pd.concat(parts,ignore_index=True)
countries=pd.read_csv(DATA/'countries.csv',keep_default_na=False)
rank=d.groupby(['i','t']).v.sum().reset_index().groupby('i').agg(
    early_value_kusd=('v','sum'), positive_years=('t','nunique')).reset_index()
rank['mean_annual_kusd']=rank.early_value_kusd/12
rank=rank.merge(countries,left_on='i',right_on='country_code',validate='one_to_one')
rank=rank.sort_values('mean_annual_kusd',ascending=False)
rank['eligible_continuity']=rank.positive_years.eq(12)
selected=rank[rank.eligible_continuity].head(20)
rank.to_csv(OUT/'exporter_ranking_1995_2006.csv',index=False,encoding='utf-8-sig')
dest=d.groupby('j').v.sum().sort_values(ascending=False)
dest.rename('early_imports_kusd').reset_index().merge(countries,left_on='j',right_on='country_code',validate='one_to_one').to_csv(OUT/'destination_ranking_1995_2006.csv',index=False,encoding='utf-8-sig')
x=d[d.i.isin(selected.i)].copy()
tot=x.groupby(['t','i','k4']).v.sum().rename('x')
tot=tot[tot>=1000]
coverage=[]
chosen=None
for n in [40,60,80,100,120]:
    nodes=dest.head(n).index.tolist()
    s=x[x.j.isin(nodes)].groupby(['t','i','k4']).v.sum().reindex(tot.index,fill_value=0)/tot
    stats={'n_destinations':n,'eligible_early_samples':len(s),
           'fraction_coverage_ge_90':float((s>=.90).mean()),
           'coverage_p05':float(s.quantile(.05)),'coverage_median':float(s.median()),
           'weighted_coverage':float((s*tot).sum()/tot.sum())}
    coverage.append(stats)
    if chosen is None and stats['fraction_coverage_ge_90']>=.95:
        chosen=n
pd.DataFrame(coverage).to_csv(OUT/'destination_coverage_options.csv',index=False)
decision={'early_years':[1995,2006],'chapter':'84','base_min_kusd':1000,
          'exporters':selected.i.astype(int).tolist(),
          'exporter_rule':'largest mean annual HS84 exports among economies present in all 12 early years; continuity does not prove reporting completeness',
          'destination_rule':'smallest of 40/60/80/100/120 covering >=90% exports in >=95% early eligible economy-product-year observations',
          'n_destinations':chosen,
          'destinations':dest.head(chosen).index.astype(int).tolist() if chosen else [],
          'status':'EARLY_COVERAGE_SCREEN_ONLY_NO_LABELS'}
(OUT/'sample_screen.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf-8')
print('EXPORTERS',selected[['i','country_name','positive_years']].to_string(index=False))
print('EXCLUDED_TOP',rank.head(30).loc[~rank.head(30).eligible_continuity,['country_name','positive_years']].to_string(index=False))
print('COVERAGE',json.dumps(coverage))
print('CHOSEN_N',chosen)
