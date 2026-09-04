"""Is "the geometry MLP takes 2.94 s" separable, or is it inside the evaluation noise?

Table 1's surveyed rows are a single deterministic sequence split and carry no spread, yet the
paper reads an ordering off them.  The one row where the gap is modest is h=2.94 s: the table
reads 0.563 and the MLP 0.358.  The held-out set is made of whole driving sequences, so the
sequence is the independent unit and the comparison can be bootstrapped over them without
retraining anything -- the predictors are fixed, only the evaluation sample is resampled.
"""
import numpy as np, json, torch, torch.nn as nn
torch.set_num_threads(16)
src=open('experiments/_surveyed_common.py').read()
exec(src[src.index('V2I=[31'):src.index('DATA={s:load(s)')])
K=32; B=4000
rng=np.random.default_rng(20260901)

def table_pf(D,tr,ei,k,KM,HIST):
    m=tr&D['good']; mx,my,mP=D['x'][m],D['y'][m],D['Pn'][m]
    sc=k/(HIST-1)
    qx=D['x'][ei]+(D['x'][ei]-D['x'][ei-HIST+1])*sc
    qy=D['y'][ei]+(D['y'][ei]-D['y'][ei-HIST+1])*sc
    KM=min(KM,len(mx)-1); Pf=D['Pn'][ei+k]; out=np.empty(len(ei))
    for a in range(0,len(ei),256):
        sl=slice(a,min(a+256,len(ei)))
        d2=(qx[sl,None]-mx[None])**2+(qy[sl,None]-my[None])**2
        nb=np.argpartition(d2,KM,axis=1)[:,:KM]
        w=1.0/(np.sqrt(np.take_along_axis(d2,nb,1))+0.5); w/=w.sum(1,keepdims=True)
        S=(mP[nb]*w[:,:,None]).sum(1)
        out[sl]=Pf[np.arange(sl.start,sl.stop),S.argmax(1)]
    return out

def mlp_pf(D,tr,ei,k,seed):
    def feat(i): return np.stack([D['x'][i],D['y'][i],D['x'][i]-D['x'][i-SEQ+1],
                                  D['y'][i]-D['y'][i-SEQ+1]],1).astype(np.float32)
    itr=eval_idx(D,tr,k,SEQ); mu,sd=feat(itr).mean(0),feat(itr).std(0)+1e-6
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(4,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,D['K']))
    opt=torch.optim.Adam(net.parameters(),3e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
    X=torch.tensor((feat(itr)-mu)/sd); y=torch.tensor(D['beam'][itr+k]).long()
    for _ in range(200):
        p=torch.randperm(len(X))
        for a in range(0,len(p),4096):
            b=p[a:a+4096]; opt.zero_grad()
            nn.functional.cross_entropy(net(X[b]),y[b]).backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad(): pr=net(torch.tensor((feat(ei)-mu)/sd)).argmax(1).numpy()
    return D['Pn'][ei+k][np.arange(len(ei)),pr]

pf_t, pf_m, sq_t, sq_m = [], [], [], []
for s in V2I:
    D=load(s)
    tr,va,te=seqsplit(D,np.random.default_rng(20260901+s))   # the Table-1 split
    best=None
    for KM in KMAP_G:
        for H in HIST_G:
            ev=eval_idx(D,va,K,H)
            if len(ev)<80: continue
            v=-10*np.log10(max(table_pf(D,tr,ev,K,KM,H).mean(),1e-12))
            if best is None or v<best[0]: best=(v,KM,H)
    _,KM,H=best
    eiT=eval_idx(D,te,K,H); eiM=eval_idx(D,te,K,SEQ)
    pf_t.append(table_pf(D,tr,eiT,K,KM,H)); sq_t.append(D['seq'][eiT]+100*s)
    pf_m.append(mlp_pf(D,tr,eiM,K,20260901+s)); sq_m.append(D['seq'][eiM]+100*s)
    print(f'  s{s}: table {-10*np.log10(pf_t[-1].mean()):.3f}  MLP {-10*np.log10(pf_m[-1].mean()):.3f}'
          f'  ({len(np.unique(sq_t[-1]))} held-out sequences)', flush=True)

pt=np.concatenate(pf_t); pm=np.concatenate(pf_m)
st=np.concatenate(sq_t); sm=np.concatenate(sq_m)
dB=lambda v: -10*np.log10(max(float(np.mean(v)),1e-12))
print(f'\npoint: table {dB(pt):.3f} dB   MLP {dB(pm):.3f} dB   gap {dB(pt)-dB(pm):.3f} dB')
u=np.unique(st); um=np.unique(sm)
diffs=[]
for _ in range(B):
    d1=rng.choice(u,len(u),True); d2=rng.choice(um,len(um),True)
    a=dB(np.concatenate([pt[st==x] for x in d1])); b=dB(np.concatenate([pm[sm==x] for x in d2]))
    diffs.append(a-b)
lo,hi=np.percentile(diffs,[2.5,97.5])
print(f'sequence bootstrap of the gap: {np.mean(diffs):.3f} dB, 95% CI [{lo:.3f}, {hi:.3f}]')
print(f'MLP better in {100*np.mean(np.array(diffs)>0):.1f}% of resamples')
json.dump(dict(table=dB(pt), mlp=dB(pm), gap=dB(pt)-dB(pm), ci=[float(lo),float(hi)],
               frac_mlp_better=float(np.mean(np.array(diffs)>0)),
               n_seq_table=int(len(u)), n_seq_mlp=int(len(um))),
          open('results/surveyed_sequence_bootstrap.json','w'), indent=1)
print('saved -> results/surveyed_sequence_bootstrap.json')
