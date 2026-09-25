"""Approved diagonal/off-diagonal and pairwise downside decomposition."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
from baselines import ROOT,np,pd,json
from common_factor_candidates import sha
from features import pair_corr
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
import warnings
threadpool_limits(1)
TARGET=ROOT/'data/relation_attribution_development.csv.gz'
AUDIT=ROOT/'audit/relation_attribution_features.json'
B_EXTRA=['diag_portfolio_variance','independent_joint_down','covariance_share_coverage','estimable_pair_exposure']
C_EXTRA=['offdiag_portfolio_variance','average_correlation','excess_joint_down','positive_pair_exposure','negative_pair_exposure']

def decompose(u,s):
    _,n,valid=pair_corr(u)
    mask=np.isfinite(u).astype(float); neg=np.where(np.isfinite(u),u<0,0).astype(float)
    # Each endpoint's frequency is evaluated on exactly the pair's shared years.
    negative_counts=neg.T@mask
    marginal=np.divide(negative_counts,n,out=np.zeros_like(n),where=n>0)
    independent=marginal*marginal.T
    observed=np.divide(neg.T@neg,n,out=np.zeros_like(n),where=n>0)
    pair=s[:,None]*s[None,:]; np.fill_diagonal(pair,0)
    exposure=float(pair[valid].sum())
    expected=float((pair*independent*valid).sum()/exposure) if exposure else np.nan
    joint=float((pair*observed*valid).sum()/exposure) if exposure else np.nan
    good=np.isfinite(u).sum(0)>=6
    diagonal=offdiagonal=total=np.nan
    if good.sum()>1:
        x=u[:,good]; x=np.where(np.isfinite(x),x,np.nanmean(x,axis=0))
        cov=LedoitWolf().fit(x).covariance_; weights=s[good]
        diagonal=float((weights**2)@np.diag(cov))
        off=cov.copy(); np.fill_diagonal(off,0)
        offdiagonal=float(weights@off@weights)
        total=float(weights@cov@weights)
    return {'diag_portfolio_variance':diagonal,'offdiag_portfolio_variance':offdiagonal,
            'independent_joint_down':expected,'excess_joint_down':joint-expected,
            'check_portfolio_variance':total,'check_joint_down_frequency':joint}

def check_pair_arithmetic():
    u=np.array([[-1,2,1],[2,-1,2],[-2,-3,3],[1,4,-2],[-1,-2,-1],[3,1,1],[np.nan,-2,2],[1,np.nan,-3]],float)
    s=np.array([.2,.3,.4]); result=decompose(u,s)
    rho,n,valid=pair_corr(u); den=expect=joint=0.
    for j in range(3):
        for k in range(3):
            if not valid[j,k]: continue
            keep=np.isfinite(u[:,j])&np.isfinite(u[:,k]); w=s[j]*s[k]
            expect+=w*(u[keep,j]<0).mean()*(u[keep,k]<0).mean()
            joint+=w*((u[keep,j]<0)&(u[keep,k]<0)).mean(); den+=w
    np.testing.assert_allclose([result['independent_joint_down'],result['excess_joint_down']],
                              [expect/den,(joint-expect)/den],atol=1e-12)
    np.testing.assert_allclose(result['diag_portfolio_variance']+result['offdiag_portfolio_variance'],result['check_portfolio_variance'],atol=1e-12)

def build():
    check_pair_arithmetic()
    source=ROOT/'data/features_lag0_L8.csv.gz'; source_hash=sha(source)
    table=pd.read_csv(source,dtype={'k4':str}); table=table[table.t.between(2007,2018)].reset_index(drop=True)
    if TARGET.exists():
        info=json.loads(AUDIT.read_text()); cached=pd.read_csv(TARGET,dtype={'k4':str})
        assert info['source_feature_sha256']==source_hash and info['candidate_sha256']==sha(TARGET)
        assert cached[['t','i','k4']].equals(table[['t','i','k4']])
        print('REUSE_DECOMPOSITION',len(cached),flush=True); return cached
    with np.load(ROOT/'data/trade_arrays.npz') as f:
        a={k:f[k] for k in ['flow','imports','total','exporters','products']}
    ei={int(c):i for i,c in enumerate(a['exporters'])}; ki={str(c):i for i,c in enumerate(a['products'])}
    parts=[]; checkpoint=ROOT/'data/relation_attribution_years'; checkpoint.mkdir(exist_ok=True)
    for year,part in table.groupby('t'):
        path=checkpoint/f'{year}.csv.gz'
        if path.exists():
            got=pd.read_csv(path,dtype={'k4':str})
            assert got[['t','i','k4']].equals(part[['t','i','k4']].reset_index(drop=True))
            parts.append(got); print('REUSE_DECOMPOSITION_YEAR',year,flush=True); continue
        rows=[]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for r in part.itertuples():
                ti=int(r.t)-1995; e=ei[r.i]; k=ki[r.k4]
                assert r.input_year==r.t and ti-8>=0
                mh=a['imports'][ti-8:ti+1,k]-a['flow'][ti-8:ti+1,e,k]
                u=np.diff(np.log1p(np.where(mh>=1000,mh,np.nan)),axis=0)
                s=a['flow'][ti,e,k]/a['total'][ti,e,k]
                rows.append({'t':r.t,'i':r.i,'k4':r.k4,**decompose(u,s)})
        got=pd.DataFrame(rows); got.to_csv(path,index=False,compression='gzip'); parts.append(got)
        print('DECOMPOSITION_YEAR',year,len(got),flush=True)
    result=pd.concat(parts,ignore_index=True)
    for old,new in [('portfolio_variance','check_portfolio_variance'),('joint_down_frequency','check_joint_down_frequency')]:
        np.testing.assert_allclose(result[new],table[old],rtol=1e-7,atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(result.diag_portfolio_variance+result.offdiag_portfolio_variance,table.portfolio_variance,rtol=1e-7,atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(result.independent_joint_down+result.excess_joint_down,table.joint_down_frequency,rtol=1e-7,atol=1e-10,equal_nan=True)
    assert not np.isinf(result.select_dtypes('number')).any().any()
    assert result.diag_portfolio_variance.dropna().ge(-1e-12).all()
    assert result.independent_joint_down.dropna().between(0,1).all()
    result.to_csv(TARGET,index=False,compression='gzip')
    AUDIT.write_text(json.dumps({'source_feature_sha256':source_hash,'candidate_sha256':sha(TARGET),'rows':len(result),
       'sample_years':[2007,2018],'window':8,'pairwise_minimum':6,
       'checks':'explicit shared-year pair arithmetic, old portfolio/joint frequency reconciliation, both decompositions, bounds passed',
       'missing_counts':result.isna().sum().to_dict(),
       'limitation':'Ledoit-Wolf diagonal uses joint shrinkage estimation; B is not an estimator entirely independent of cross-market information. Pairwise frequencies also condition on common observed years.',
       'final_holdout_evaluated':False},indent=2),encoding='utf-8')
    assert sha(source)==source_hash
    return result

if __name__=='__main__': build()
