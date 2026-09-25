"""Bounded signed graph residual comparison, development data only; CPU reproducible."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import sys,json,time,copy,argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps')); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from baselines import yearly,weights
torch.set_num_threads(2)
OUT=ROOT/'results/diagnosis/graphs'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,2024,2026]
EPOCHS=15

def prune(a):
    keep=np.zeros_like(a,dtype=bool)
    idx=np.argsort(a,axis=-1)[...,-5:]
    np.put_along_axis(keep,idx,True,axis=-1)
    return a*(keep|keep.transpose(0,2,1))

def load(mode):
    base=pd.read_csv(ROOT/'results/diagnosis/crossfit_base.csv',dtype={'k4':str})
    table=pd.read_csv(ROOT/'data/features_lag0_L8.csv.gz',dtype={'k4':str})
    parts={k:[] for k in ['z','s','a','rel','gate','control']}
    for t,rows in base.groupby('t'):
        with np.load(ROOT/f'data/graphs_L8/{t}.npz') as f:
            r=f['rho']; reliability=f['reliability']; s=f['shares']; z=f['z']; ids=f['row_ids']
        selected=table.loc[ids]
        assert selected[['t','i','k4']].reset_index(drop=True).equals(rows[['t','i','k4']].reset_index(drop=True))
        pair=s[:,:,None]*s[:,None,:]
        reliable=prune(np.maximum(r,0)*reliability)+prune(np.maximum(-r,0)*reliability)
        den=(pair*np.abs(r)).sum((1,2)); num=(pair*reliable).sum((1,2))
        gate=np.divide(num,den,out=np.zeros_like(num),where=den>1e-12).clip(0,1)
        # Same reliability gate and metadata for all variants: isolate representation.
        raw=np.ones_like(reliability) if mode=='ordinary_graph' else reliability
        ap=prune(np.maximum(r,0)*raw); an=prune(np.maximum(-r,0)*raw)
        parts['z'].append(z); parts['s'].append(s)
        parts['a'].append(np.stack([ap,an],axis=1).astype(np.float16))
        parts['rel'].append(reliability.astype(np.float16))
        parts['gate'].append(gate)
        parts['control'].append(np.column_stack([selected.hhi,selected.graph_coverage,num/2,gate]))
    arrays={k:np.concatenate(v) for k,v in parts.items()}
    train=(base.t<=2015).to_numpy()
    z=arrays['z']; valid=np.isfinite(z)
    median=np.nanmedian(z[train],axis=(0,1)); median=np.nan_to_num(median)
    z=np.where(valid,z,median)
    mean=z[train].mean((0,1)); sd=z[train].std((0,1)); sd=np.maximum(sd,1e-6)
    arrays['z']=np.concatenate([(z-mean)/sd,(~valid).astype(np.float32)],axis=-1).astype(np.float32)
    p=base.p_base.to_numpy().clip(1e-6,1-1e-6)
    arrays['base']=np.log(p/(1-p)).astype(np.float32)
    arrays['y']=base.label_observed_continuing.to_numpy(dtype=np.float32)
    arrays['w']=weights(base).astype(np.float32)
    arrays['control']=arrays['control'].astype(np.float32)
    np.testing.assert_array_less(arrays['gate'],1.00001)
    assert np.isfinite(arrays['z']).all() and (arrays['gate']>=0).all()
    return base,arrays,{'median':median.tolist(),'mean':mean.tolist(),'sd':sd.tolist()}

class Correction(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.mode=mode
        self.node=nn.Sequential(nn.Linear(18,32),nn.ReLU())
        self.self_layer=nn.Linear(32,32)
        self.pos=nn.Linear(32,32,bias=False); self.neg=nn.Linear(32,32,bias=False)
        # Symmetric endpoint features: product, absolute difference, sign, reliability.
        self.pair=nn.Sequential(nn.Linear(66,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(32+16*2+5,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self,z,s,a,rel,gate,control,base):
        h=self.node(z)
        if self.mode!='no_graph':
            norm=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
            h=torch.relu(self.self_layer(h)+self.pos(torch.bmm(norm[:,0],h))+self.neg(torch.bmm(norm[:,1],h)))
        else: h=torch.relu(self.self_layer(h))
        pooled=(h*s.unsqueeze(-1)).sum(1)
        pairparts=[]
        for sign in range(2):
            out=torch.zeros((len(z),16),dtype=h.dtype)
            if self.mode=='reliable_pair':
                b,j,k=torch.where(torch.triu(a[:,sign],diagonal=1)>0)
                if len(b):
                    hj=h[b,j]; hk=h[b,k]
                    edge=torch.stack([torch.full_like(rel[b,j,k],1. if sign==0 else -1.),rel[b,j,k]],dim=1)
                    emb=self.pair(torch.cat([hj*hk,torch.abs(hj-hk),edge],dim=1))
                    exposure=s[b,j]*s[b,k]*a[b,sign,j,k]
                    out.index_add_(0,b,emb*exposure[:,None])
            pairparts.append(out)
        raw=self.head(torch.cat([pooled,*pairparts,control,base[:,None]],dim=1)).squeeze(-1)
        correction=2*gate*torch.tanh(raw)
        return base+correction,correction

def batch(a,idx):
    return [torch.from_numpy(a[k][idx].astype(np.float32)) for k in ['z','s','a','rel','gate','control','base']]

def predict(model,a,ids):
    model.eval(); values=[]; corrections=[]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            logits,delta=model(*batch(a,ids[start:start+64]))
            values.append(torch.sigmoid(logits).numpy()); corrections.append(delta.numpy())
    return np.concatenate(values),np.concatenate(corrections)

def properties(model,a):
    model.eval(); x=batch(a,np.arange(3)); x[4]=torch.zeros_like(x[4])
    with torch.no_grad():
        output,delta=model(*x)
        assert torch.equal(output,x[-1]) and torch.count_nonzero(delta)==0
    # Permute all destination axes together: invariant economic representation.
    x=batch(a,np.arange(3)); p=torch.randperm(100)
    xp=[x[0][:,p],x[1][:,p],x[2][:,:,p][:,:,:,p],x[3][:,p][:,:,p],*x[4:]]
    with torch.no_grad():
        y,d=model(*x); yp,dp=model(*xp)
        torch.testing.assert_close(y,yp,rtol=1e-5,atol=1e-6)
        assert torch.all(torch.abs(d)<=2*x[4]+1e-6)

def main(mode,epochs):
    base,a,scaling=load(mode); train=np.flatnonzero((base.t<=2015).to_numpy()); valid=np.flatnonzero((base.t>=2016).to_numpy())
    val=base.iloc[valid]; allmetrics=[]
    for seed in SEEDS:
        stem=OUT/f'{mode}_{seed}'
        if stem.with_suffix('.csv').exists():
            print('REUSE_GRAPH_MODEL',mode,seed,flush=True); continue
        torch.manual_seed(seed); rng=np.random.default_rng(seed)
        model=Correction(mode); properties(model,a)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        p,delta=predict(model,a,valid); bestscore=np.mean([r['ap'] for r in yearly(val,p)])
        best=copy.deepcopy(model.state_dict()); bestepoch=0; history=[]; start=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train(); order=rng.permutation(train); loss_sum=0; count=0
            for offset in range(0,len(order),64):
                ids=order[offset:offset+64]; y=torch.from_numpy(a['y'][ids]); w=torch.from_numpy(a['w'][ids])
                logits,delta=model(*batch(a,ids))
                loss=(nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).mean()+.01*(delta**2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                loss_sum+=loss.item()*len(ids); count+=len(ids)
            # Fixed four checkpoint choices including the exact base fallback.
            if epoch in [5,10,15] or epoch==epochs:
                p,delta=predict(model,a,valid); annual=yearly(val,p); score=np.mean([r['ap'] for r in annual])
                history.append({'epoch':epoch,'training_objective':loss_sum/count,'validation_ap':score,
                                'validation_brier':np.mean([r['brier'] for r in annual]),'mean_abs_delta':float(np.abs(delta).mean())})
                print('GRAPH_DIAG',mode,seed,epoch,'AP',round(score,6),'seconds',round(time.perf_counter()-start,1),flush=True)
                if score>bestscore: bestscore=score; bestepoch=epoch; best=copy.deepcopy(model.state_dict())
                pd.DataFrame(history).to_csv(OUT/f'{mode}_{seed}_history.csv',index=False)
        model.load_state_dict(best); properties(model,a); p,delta=predict(model,a,valid)
        pred=val[['t','i','k4','label_observed_continuing','p_base']].copy(); pred['p']=p; pred['logit_correction']=delta
        pred.to_csv(stem.with_suffix('.csv'),index=False)
        torch.save({'state_dict':best,'scaling':scaling,'mode':mode,'seed':seed,'epoch':bestepoch},stem.with_suffix('.pt'))
        inactive=[]
        if mode=='no_graph': inactive.extend([model.pos,model.neg])
        if mode!='reliable_pair': inactive.append(model.pair)
        total=sum(x.numel() for x in model.parameters())
        inactive_count=sum(x.numel() for module in inactive for x in module.parameters())
        (OUT/f'{mode}_{seed}_selection.json').write_text(json.dumps({'epoch':bestepoch,'mean_validation_ap':bestscore,'total_parameters':total,'parameters_in_active_modules':total-inactive_count,'property_checks_passed':True},indent=2))
        allmetrics.extend([{'model':mode,'seed':seed,'epoch':bestepoch,**r} for r in yearly(val,p)])
    print('COMPLETE',mode,'FINAL_HOLDOUT_NOT_EVALUATED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['no_graph','ordinary_graph','reliable_graph','reliable_pair'],required=True)
    parser.add_argument('--epochs',type=int,default=EPOCHS); args=parser.parse_args(); main(args.mode,args.epochs)
