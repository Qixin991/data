"""Approved aggregate-market factor; historical development inputs only."""
from baselines import ROOT,np,pd,json
import hashlib

TARGET=ROOT/'data/common_factor_aggregate_development.csv.gz'
AUDIT=ROOT/'audit/common_factor_aggregate.json'
NEW=['selected_market_total_growth','selected_market_total_growth_sd8']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    source=ROOT/'data/features_lag0_L8.csv.gz'
    source_hash=sha(source)
    table=pd.read_csv(source,dtype={'k4':str})
    table=table[table.t.between(2007,2018)].reset_index(drop=True)
    if TARGET.exists():
        audit=json.loads(AUDIT.read_text(encoding='utf-8'))
        assert audit['source_feature_sha256']==source_hash and audit['candidate_sha256']==sha(TARGET)
        cached=pd.read_csv(TARGET,dtype={'k4':str})
        assert cached[['t','i','k4']].equals(table[['t','i','k4']])
        print('REUSE_CANDIDATE',len(cached),flush=True)
        return cached
    with np.load(ROOT/'data/trade_arrays.npz') as archive:
        years=archive['years']; end=int(np.flatnonzero(years==2018)[0])+1
        imports=archive['imports'][:end]; flow=archive['flow'][:end]
        exporters=archive['exporters']; products=archive['products']; destinations=archive['destinations']
    assert len(destinations)==100
    ei={int(v):j for j,v in enumerate(exporters)}; ki={str(v):j for j,v in enumerate(products)}
    # Keep all observed contributions to the total, including individually small markets.
    outside=imports[:,None,:,:]-flow
    assert np.isfinite(outside).all() and outside.min()>=-1e-7
    outside=np.maximum(outside,0)  # Only arithmetic round-off, tolerance checked above.
    total=outside.sum(-1)
    levels=np.where(total>=1000,total,np.nan)
    changes=np.diff(np.log1p(levels),axis=0)
    rows=[]
    for r in table.itertuples():
        ti=int(r.t)-int(years[0]); e=ei[r.i]; k=ki[r.k4]
        assert r.input_year==r.t and ti-8>=0
        history=changes[ti-8:ti,e,k]
        assert len(history)==8 and np.isfinite(history).all(), 'Incomplete aggregate window: pause and inspect'
        rows.append([r.t,r.i,r.k4,float(history[-1]),float(history.std(ddof=1))])
    result=pd.DataFrame(rows,columns=['t','i','k4',*NEW])
    assert not result.duplicated(['t','i','k4']).any()
    assert np.isfinite(result[NEW]).all().all()
    # Independent arithmetic identity: sum of leave-exporter-out contributions.
    np.testing.assert_allclose(total,imports.sum(-1)[:,None,:]-flow.sum(-1),rtol=1e-10,atol=1e-5)
    # Hand-check first and last retained observations against explicit historical slicing.
    for idx in [0,len(table)-1]:
        r=table.iloc[idx]; ti=int(r.t)-1995; e=ei[r.i]; k=ki[r.k4]
        explicit=np.array([sum(imports[j,k,d]-flow[j,e,k,d] for d in range(100)) for j in range(ti-8,ti+1)])
        h=np.diff(np.log1p(explicit))
        np.testing.assert_allclose(result.loc[idx,NEW].to_numpy(dtype=float),[h[-1],h.std(ddof=1)],rtol=1e-9,atol=1e-10)
    result.to_csv(TARGET,index=False,compression='gzip')
    audit={'source_feature_sha256':source_hash,'candidate_sha256':sha(TARGET),'rows':len(result),
       'sample_years':[2007,2018],'last_input_year':2018,'destinations':destinations.tolist(),
       'definition':'log1p(sum of observed imports into fixed 100 destinations, excluding target exporter)) first difference; sd over latest 8 changes, ddof=1',
       'units':'thousand USD','individual_small_market_filter':False,'minimum_aggregate_level':1000,
       'coverage_limitation':'Observed BACI totals; fixed destination set does not prove complete reporting; not global total imports',
       'checks':'finite complete historical windows, unique keys, total reconciliation, explicit endpoint checks passed',
       'final_test_evaluated':False}
    AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    assert sha(source)==source_hash
    print('CANDIDATE_COMPLETED',len(result),flush=True)
    return result

if __name__=='__main__': build()
