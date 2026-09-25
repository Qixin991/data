"""Normalize supplied WDI sheets without imputation or merging historical entities."""
from pathlib import Path
import sys
import json
import hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
sys.stdout.reconfigure(encoding='utf-8')
import xlrd
import pandas as pd
OUT=ROOT/'data'
rows=[]; metadata=[]; manifests=[]
for file in (ROOT.parent/'paper').glob('*.xls'):
    book=xlrd.open_workbook(str(file))
    data=book.sheet_by_name('Data')
    metas=book.sheet_by_name('Metadata - Countries')
    meta={str(metas.cell_value(r,0)):metas.row_values(r) for r in range(1,metas.nrows)}
    for r in range(4,data.nrows):
        name,iso,ind_name,ind=data.row_values(r)[:4]
        if not iso: continue
        rowmeta=meta.get(iso,[iso,'','','',''])
        for c in range(4,data.ncols):
            year=int(data.cell_value(3,c))
            value=data.cell_value(r,c)
            rows.append({'iso3':iso,'country_name_wdi':name,'year':year,'indicator':ind,
                         'value':None if value=='' else float(value),
                         'region':rowmeta[1],'income_group':rowmeta[2]})
    raw_date=data.cell_value(1,1)
    date=xlrd.xldate_as_datetime(raw_date,book.datemode).date().isoformat() if isinstance(raw_date,(int,float)) else str(raw_date)
    manifests.append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'last_updated':date})
long=pd.DataFrame(rows)
if long.duplicated(['iso3','year','indicator']).any(): raise ValueError('Duplicate macro keys')
wide=long.pivot(index=['iso3','year'],columns='indicator',values='value').reset_index().rename(columns={
 'NY.GDP.MKTP.CD':'gdp_usd','NY.GDP.MKTP.KD.ZG':'gdp_growth','FP.CPI.TOTL.ZG':'inflation'})
wide.to_csv(OUT/'macro_wdi_all.csv',index=False)
countries=pd.read_csv(OUT/'countries.csv',keep_default_na=False)
# Current ISO codes on historical entities must not silently attach modern macro values.
countries['historical_entity']=countries.country_name.str.contains(r'\.\.\.',regex=True)
map_table=countries.merge(wide,left_on='country_iso3',right_on='iso3',how='left',validate='many_to_many')
for field in ['gdp_usd','gdp_growth','inflation']:
    map_table.loc[map_table.historical_entity,field]=float('nan')
map_table.to_csv(OUT/'macro_baci_mapping.csv',index=False,encoding='utf-8-sig')
mapped=set(wide.iso3)
unmapped=countries[~countries.country_iso3.isin(mapped) | countries.historical_entity]
unmapped.to_csv(ROOT/'audit'/'macro_unmapped_entities.csv',index=False,encoding='utf-8-sig')
(ROOT/'audit'/'macro_sources.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf-8')
print('WDI_KEYS',len(wide),'BACI_MAPPED_ROWS',len(map_table))
print('UNMAPPED_ENTITIES',unmapped[['country_code','country_name','country_iso3','historical_entity']].to_string(index=False))
print('VERSIONS',manifests)
