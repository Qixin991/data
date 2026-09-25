"""Historical feature construction; labels are not used to fit any transformation."""
from pathlib import Path
import argparse
import json
import sys
import warnings
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def pair_corr(u, minimum=6):
    m=np.isfinite(u).astype(float); a=np.nan_to_num(u,nan=0.)
    n=m.T@m; sums=a.T@m; sq=(a*a).T@m; prod=a.T@a
    den_n=np.maximum(n,1)
    numerator=prod-sums*sums.T/den_n
    variance=np.maximum(sq-sums*sums/den_n,0)
    denom=np.sqrt(variance*variance.T)
    valid=(n>=minimum)&(denom>1e-12)
    corr=np.divide(numerator,denom,out=np.zeros_like(denom),where=valid)
    corr=np.clip(corr,-1,1); np.fill_diagonal(corr,0); np.fill_diagonal(valid,False)
    return corr,n,valid

def reliable_graph(u, k=5, block=1):
    rho,n,valid=pair_corr(u)
    stability=np.zeros_like(rho); considered=np.zeros_like(rho)
    for a in range(len(u)-block+1):
        removed=np.zeros(len(u),bool); removed[a:a+block]=True
        ri,_,vi=pair_corr(u[~removed],minimum=max(3,6-block))
        observed=np.isfinite(u[a:a+block]).all(axis=0)
        at_risk=observed[:,None]&observed[None,:]
        stability += at_risk*vi*(np.sign(ri)==np.sign(rho))*np.maximum(0,1-np.abs(ri-rho)/2)
        considered += at_risk
    stability=np.divide(stability,considered,out=np.zeros_like(stability),where=considered>0)
    reliability=(n/len(u))*stability*valid
    weights=[]
    for sign in [1,-1]:
        a=reliability*np.maximum(sign*rho,0)
        keep=np.zeros_like(a,dtype=bool)
        idx=np.argsort(a,axis=1)[:,-k:]
        np.put_along_axis(keep,idx,True,axis=1)
        keep=keep|keep.T
        weights.append(a*keep)
    return rho,reliability,weights

def finite_mean(x):
    valid=np.isfinite(x)
    return float(x[valid].mean()) if valid.any() else np.nan

def finite_std(x):
    valid=np.isfinite(x)
    return float(x[valid].std(ddof=1)) if valid.sum()>1 else np.nan

def weighted(x,s):
    ok=np.isfinite(x); cov=float(s[ok].sum())
    return (float((x[ok]*s[ok]).sum()/cov) if cov>0 else np.nan),cov

def build(lag=0, window=8):
    data=np.load(ROOT/'data/trade_arrays.npz')
    years=data['years']; exporters=data['exporters']; dest=data['destinations']; products=data['products']
    totals=data['total']; flows=data['flow']; imports=data['imports']
    macro=pd.read_csv(ROOT/'data/macro_baci_mapping.csv')
    # Non-matched macro entities have no year; retain missing values instead of dropping trade.
    macro=macro[macro.year.notna()]
    if macro.duplicated(['country_code','year']).any(): raise ValueError('Macro merge is not many-to-one')
    mm=macro.set_index(['country_code','year'])
    out=[]; transitions=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for ti,t in enumerate(years):
            if t<2007 or t>2023: continue
            ai=ti-lag
            zmacro=[]
            for code in dest:
                if (code,years[ai]) in mm.index:
                    r=mm.loc[(code,years[ai])]
                    zmacro.append([r.gdp_growth,np.log(r.gdp_usd) if r.gdp_usd>0 else np.nan,r.inflation])
                else: zmacro.append([np.nan]*3)
            zmacro=np.asarray(zmacro)
            count={'year':int(t),'input_lag':lag,'potential':0,'below_scale':0,'low_graph_coverage':0,'unknown_label':0,'retained':0}
            for e,code in enumerate(exporters):
                for p,product in enumerate(products):
                    if not str(product).startswith('84'): continue
                    count['potential']+=1
                    xa=totals[ai,e,p]; xt=totals[ti,e,p]; xn=totals[ti+1,e,p]
                    # Lagged task selection is based on the available input year, not a future scale filter.
                    if xa<1000: count['below_scale']+=1; continue
                    share=flows[ai,e,p]/xa
                    coverage=float(share.sum())
                    if coverage<.9: count['low_graph_coverage']+=1; continue
                    if xt<=0 or xn<=0: count['unknown_label']+=1; continue
                    count['retained']+=1
                    hist=totals[max(0,ai-5):ai+1,e,p].copy()
                    hist[hist<=0]=np.nan
                    growth=np.diff(np.log(hist))
                    mh=imports[ai-window:ai+1,p]-flows[ai-window:ai+1,e,p]
                    # Tiny or absent observed markets are unknown graph inputs, never silently zero growth.
                    mh=np.where(mh>=1000,mh,np.nan)
                    u=np.diff(np.log1p(mh),axis=0)
                    mean=np.nanmean(u,axis=0); std=np.nanstd(u,axis=0,ddof=1)
                    rho,n,valid=pair_corr(u)
                    pair=share[:,None]*share[None,:]; np.fill_diagonal(pair,0)
                    rel_cov=float(pair[valid].sum())
                    avg_corr=float((pair*rho).sum()/rel_cov) if rel_cov>0 else np.nan
                    a=np.isfinite(u).astype(float); negative=np.where(np.isfinite(u),u<0,0).astype(float)
                    codown=np.divide(negative.T@negative,np.maximum(n,1))*valid
                    joint=float((pair*codown).sum()/rel_cov) if rel_cov>0 else np.nan
                    good=np.isfinite(u).sum(0)>=6
                    if good.sum()>1:
                        complete=np.where(np.isfinite(u[:,good]),u[:,good],mean[good])
                        cov=LedoitWolf().fit(complete).covariance_
                        port_var=float(share[good]@cov@share[good])
                    else: port_var=np.nan
                    last,market_coverage=weighted(u[-1],share)
                    hist_market,_=weighted(mean,share)
                    market_vol,_=weighted(std,share)
                    global_factor=np.nanmean(u,axis=1)
                    row={'t':int(t),'i':int(code),'k4':str(product),'input_year':int(years[ai]),
                         'label_observed_continuing':int(xn/xt-1<-.2),'outcome_growth':float(xn/xt-1),
                         'log_export':float(np.log(xa)), 'growth_1':float(growth[-1]),
                         'growth_mean3':finite_mean(growth[-3:]),'growth_mean5':finite_mean(growth),
                         'growth_sd3':finite_std(growth[-3:]),'growth_sd5':finite_std(growth),
                         'n_dest':int(data['n_dest'][ai,e,p]),'hhi':float(data['hhi'][ai,e,p]),
                         'top_share':float(data['top_share'][ai,e,p]),'graph_coverage':coverage,
                         'market_change':last,'market_history_mean':hist_market,'market_volatility':market_vol,
                         'market_feature_coverage':market_coverage,'global_factor_last':float(global_factor[-1]),
                         'global_factor_sd':finite_std(global_factor),'portfolio_variance':port_var,
                         'covariance_share_coverage':float(share[good].sum()),'average_correlation':avg_corr,
                         'joint_down_frequency':joint,'estimable_pair_exposure':rel_cov,
                         'positive_pair_exposure':float((pair*np.maximum(rho,0)).sum()/2),
                         'negative_pair_exposure':float((pair*np.maximum(-rho,0)).sum()/2)}
                    for c,name in enumerate(['gdp_growth','log_gdp','inflation']):
                        row['destination_'+name],row['destination_'+name+'_coverage']=weighted(zmacro[:,c],share)
                    out.append(row)
            transitions.append(count)
            print('FEATURES_YEAR',t,'retained',count['retained'],flush=True)
    table=pd.DataFrame(out)
    table.to_csv(ROOT/f'data/features_lag{lag}_L{window}.csv.gz',index=False,compression='gzip')
    pd.DataFrame(transitions).to_csv(ROOT/f'audit/sample_flow_lag{lag}_L{window}.csv',index=False)
    # Keep final-year target rates out of progress output until model selection is frozen.
    train=table[table.t<=2015]
    print('TRAINING_ONLY_RISK_RATE',train.label_observed_continuing.mean(),'TRAIN_ROWS',len(train))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--lag',type=int,default=0); parser.add_argument('--window',type=int,default=8)
    args=parser.parse_args(); build(args.lag,args.window)
