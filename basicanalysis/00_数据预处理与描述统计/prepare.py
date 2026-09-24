"""流式读取原始压缩包，保留HS84并聚合至HS4；不修改原始文件。"""
from pathlib import Path
import sys, json, zipfile, re, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / '新建文件夹'
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
CACHE = OUT / '年度缓存'
CACHE.mkdir(exist_ok=True)

def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def main():
    archive = SOURCE / 'BACI_HS92_V202601.zip'
    with zipfile.ZipFile(archive) as z:
        countries = pd.read_csv(z.open('country_codes_V202601.csv'))
        save(countries, '国家代码.csv')
        products = pd.read_csv(z.open('product_codes_HS92_V202601.csv'), dtype=str)
        save(products, '产品代码.csv')
        (OUT/'BACI原始说明.txt').write_bytes(z.read('Readme.txt'))
        audits=[]
        for year in range(1995, 2025):
            target=CACHE/f'{year}.csv.gz'
            audit_path=CACHE/f'{year}_audit.json'
            if target.exists() and audit_path.exists():
                audits.append(json.loads(audit_path.read_text())); continue
            member=f'BACI_HS92_Y{year}_V202601.csv'
            parts=[]; total=0; invalid_all=0
            with z.open(member) as f:
                for chunk in pd.read_csv(f, chunksize=500000, dtype={'k':'string'}):
                    total+=len(chunk)
                    invalid_all+=int((~np.isfinite(chunk.v) | (chunk.v<=0)).sum())
                    sub=chunk.loc[chunk.k.str.startswith('84')].copy()
                    parts.append(sub)
            sub=pd.concat(parts, ignore_index=True)
            duplicates=int(sub.duplicated(['t','i','j','k']).sum())
            if duplicates: raise ValueError(f'{year}: HS84 has duplicate keys; resolve before aggregation')
            if not sub.t.eq(year).all(): raise ValueError('year mismatch')
            bad=(~np.isfinite(sub.v)) | (sub.v<=0)
            audit={'year':year,'all_rows':total,'all_invalid_value':invalid_all,
                   'hs84_raw_rows':len(sub),'hs84_duplicate_keys':duplicates,
                   'hs84_invalid_value':int(bad.sum()),'hs84_quantity_missing':int(sub.q.isna().sum()),
                   'hs84_quantity_nonpositive':int((sub.q<=0).sum())}
            sub=sub.loc[~bad].copy(); sub['hs4']=sub.k.str[:4]
            agg=sub.groupby(['t','i','j','hs4'],as_index=False,observed=True).v.sum()
            audit.update(hs4_rows=len(agg),exporters=int(agg.i.nunique()),importers=int(agg.j.nunique()),
                         value_thousand_usd=float(agg.v.sum()),
                         exporter_mapping_missing=int((~agg.i.isin(countries.country_code)).sum()),
                         importer_mapping_missing=int((~agg.j.isin(countries.country_code)).sum()))
            assert np.isclose(sub.v.sum(),agg.v.sum(),rtol=1e-10)
            agg.to_csv(target,index=False,compression='gzip')
            audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
            audits.append(audit)
            print(f'完成 {year}: {total:,} 原始行; HS84 {len(sub):,} 行',flush=True)
    save(pd.DataFrame(audits),'逐年数据审计.csv')
    # Historical country entities are kept separate and excluded from macro mapping.
    historical=countries.country_name.str.contains(r'\.\.\.|\d{4}',regex=True,na=False)
    current=countries.loc[~historical].copy()
    save(countries.loc[historical], '历史实体不合并清单.csv')
    if current.country_code.duplicated().any(): raise ValueError('country code duplicate')
    macros=[]
    names={'NY.GDP.MKTP.CD':'gdp_usd','NY.GDP.MKTP.KD.ZG':'gdp_growth_pct','FP.CPI.TOTL.ZG':'inflation_pct'}
    for path in sorted(SOURCE.glob('*.xls')):
        data=pd.read_excel(path,sheet_name='Data',skiprows=3)
        metadata=pd.read_excel(path,sheet_name='Metadata - Countries')
        real=metadata.loc[metadata.Region.notna(),'Country Code']
        data=data.loc[data['Country Code'].isin(real)]
        indicator=data['Indicator Code'].iloc[0]; col=names[indicator]
        years=[str(y) for y in range(1995,2025)]
        d=data.melt(id_vars=['Country Code'],value_vars=years,var_name='t',value_name=col)
        d.t=d.t.astype(int); d=d.rename(columns={'Country Code':'country_iso3'})
        macros.append(d)
    macro=macros[0]
    for m in macros[1:]: macro=macro.merge(m,on=['country_iso3','t'],validate='one_to_one',how='outer')
    macro=current.merge(macro,on='country_iso3',how='left',validate='many_to_many')
    macro=macro.dropna(subset=['t']);macro.t=macro.t.astype(int)
    assert not macro.duplicated(['country_code','t']).any()
    save(macro,'国家宏观面板.csv')
    # Use only 1995-2006 to fix exporter scope, requiring 12 years of observed exports.
    early=[]
    for year in range(1995,2007):
        d=pd.read_csv(CACHE/f'{year}.csv.gz',dtype={'hs4':str})
        early.append(d.groupby('i',as_index=False).v.sum().assign(t=year))
    ranking=pd.concat(early).groupby('i').agg(early_mean=('v','mean'),observed_years=('t','nunique')).reset_index()
    ranking=ranking.merge(current,left_on='i',right_on='country_code',how='inner',validate='one_to_one')
    ranking['eligible']=ranking.observed_years.eq(12)
    ranking=ranking.sort_values('early_mean',ascending=False)
    ids=ranking.loc[ranking.eligible].head(20).i.tolist(); ranking['selected']=ranking.i.isin(ids)
    save(ranking,'早期出口国选择.csv')
    panel_parts=[]; bilateral=[]
    macro_cols=list(names.values())
    for year in range(1995,2025):
        d=pd.read_csv(CACHE/f'{year}.csv.gz',dtype={'hs4':str})
        d=d[d.i.isin(ids)].copy();bilateral.append(d)
        d['total']=d.groupby(['i','hs4']).v.transform('sum');d['share']=d.v/d.total
        d['share_sq']=d.share**2
        p=d.groupby(['t','i','hs4'],as_index=False).agg(export_value=('v','sum'),destinations=('j','nunique'),hhi=('share_sq','sum'),top_share=('share','max'))
        dm=d.merge(macro[['country_code','t']+macro_cols],left_on=['j','t'],right_on=['country_code','t'],how='left',validate='many_to_one')
        for c in macro_cols:
            val=np.log1p(dm[c].where(dm[c]>0)) if c=='gdp_usd' else dm[c]
            dm['weighted']=dm.share*val
            dm['coverage']=dm.share.where(val.notna(),0)
            a=dm.groupby(['t','i','hs4']).agg(weighted=('weighted',lambda s:s.sum(min_count=1)),coverage=('coverage','sum')).reset_index()
            name='dest_log_gdp' if c=='gdp_usd' else 'dest_'+c
            a[name]=a.weighted/a.coverage.where(a.coverage>0)
            a=a.rename(columns={'coverage':name+'_coverage'}).drop(columns='weighted')
            p=p.merge(a,on=['t','i','hs4'],validate='one_to_one')
        panel_parts.append(p)
    panel=pd.concat(panel_parts,ignore_index=True)
    panel=panel.merge(current[['country_code','country_name','country_iso3']],left_on='i',right_on='country_code',validate='many_to_one')
    panel=panel.sort_values(['i','hs4','t'])
    # A full year index stops rolling history and labels from jumping over missing years.
    filled=[]
    for (i,k),g in panel.groupby(['i','hs4']):
        g=g.set_index('t').reindex(range(1995,2025));g['i']=i;g['hs4']=k
        g['growth']=g.export_value/g.export_value.shift(1)-1
        g['growth_mean3']=g.growth.rolling(3,min_periods=2).mean()
        g['growth_std3']=g.growth.rolling(3,min_periods=2).std()
        g['next_export']=g.export_value.shift(-1)
        g['next_growth']=g.next_export/g.export_value-1
        g['risk']=np.where(g.next_growth.notna(),(g.next_growth<=-.20).astype(float),np.nan)
        filled.append(g.reset_index().rename(columns={'index':'t'}))
    panel=pd.concat(filled,ignore_index=True)
    panel['log_export']=np.log1p(panel.export_value)
    panel['split']=np.select([panel.t.between(2007,2015),panel.t.between(2016,2018),panel.t.between(2019,2020),panel.t.between(2021,2023)],['train','validation','calibration_unused','test'],default='history_or_unlabeled')
    panel['eligible']=panel.export_value.ge(1000)&panel.risk.notna()
    save(panel,'产品出口面板.csv')
    pd.concat(bilateral,ignore_index=True).to_csv(OUT/'选定出口国_HS4双边贸易.csv.gz',index=False,compression='gzip')
    counts=panel.groupby('t').agg(observed=('export_value','count'),size_eligible=('export_value',lambda s:int(s.ge(1000).sum())),known_label=('risk','count'),eligible=('eligible','sum')).reset_index()
    save(counts,'逐年样本流转.csv')
    features=['log_export','destinations','hhi','top_share','growth','growth_mean3','growth_std3','dest_log_gdp','dest_gdp_growth_pct','dest_inflation_pct','dest_log_gdp_coverage','dest_gdp_growth_pct_coverage','dest_inflation_pct_coverage']
    (ROOT/'config.json').write_text(json.dumps({'seed':42,'features':features,'exporters':ids,'hs2':'84','years':[1995,2024],'risk_threshold':-.2,'minimum_thousand_usd':1000},ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[{'file':p.name,'bytes':p.stat().st_size} for p in SOURCE.iterdir() if p.suffix in ['.xls','.zip']]
    (OUT/'来源清单.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'预处理完成：{len(panel):,} 面板行，{len(ids)} 个出口经济体',flush=True)

if __name__=='__main__': main()
