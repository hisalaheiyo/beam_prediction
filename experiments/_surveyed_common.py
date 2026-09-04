"""Table 1's surveyed rows carry no uncertainty at all, yet the paper reads an ordering off them.

The unsurveyed rows are averaged over block permutations and quote a standard error; the
surveyed rows are one deterministic sequence split per deployment.  The claim that rests on the
smallest surveyed gap is "the geometry MLP takes 2.94 s" (0.358 against the table's 0.563).

The surveyed split holds out whole driving sequences, so the sequence is the independent unit
and the split itself can be resampled.  Re-run the two cheap predictors over many random
sequence splits and report the spread.  TransFuser cannot be included -- one training run is
~680 h on this machine -- so this bounds the ordering only between the table and the MLP.
"""
import numpy as np, json, torch, torch.nn as nn

V2I=[31,32,33,34]; HOR=[8,16,32]; SEQ=5; NSPLIT=24
KMAP_G, HIST_G = [1,5,15,25], [3,5,8]
torch.set_num_threads(8)

def load(s):
    d=np.load(f'data/cache/s{s}.npz',allow_pickle=True); P=d['pwr'].astype(np.float64)
    return dict(Pn=P/np.maximum(np.nanmax(P,1,keepdims=True),1e-30), beam=d['beam'].astype(int),
                seq=d['seq'].astype(int), x=d['x'].astype(float), y=d['y'].astype(float),
                n=len(d['beam']), K=P.shape[1],
                good=~np.isnan(P).any(1)&~np.isnan(d['x'])&~np.isnan(d['y']))

def seqsplit(D,rng):
    u=np.unique(D['seq'][D['good']]); p=rng.permutation(len(u))
    a,b=int(.5*len(u)),int(.7*len(u))
    return [np.isin(D['seq'],u[p[:a]]), np.isin(D['seq'],u[p[a:b]]), np.isin(D['seq'],u[p[b:]])]

def eval_idx(D,mask,k,hist):
    n,seq,good=D['n'],D['seq'],D['good']
    fut=np.minimum(np.arange(n)+k,n-1)
    v=(np.arange(n)+k<n)&(seq[fut]==seq)&good&good[fut]&mask&mask[fut]
    ei=np.where(v)[0]; ei=ei[ei>=hist]; ei=ei[seq[ei-hist+1]==seq[ei]]
    return ei[good[ei-hist+1]]

dbl=lambda v: float(-10*np.log10(max(float(np.mean(v)),1e-12)))

def table(D,tr,ei,k,KM,HIST):
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
    return dbl(out)

def mlp(D,tr,ei,k,rng):
    def feat(i): return np.stack([D['x'][i],D['y'][i],D['x'][i]-D['x'][i-SEQ+1],
                                  D['y'][i]-D['y'][i-SEQ+1]],1).astype(np.float32)
    itr=eval_idx(D,tr,k,SEQ)
    if len(itr)<50: return None
    mu,sd=feat(itr).mean(0),feat(itr).std(0)+1e-6
    torch.manual_seed(int(rng.integers(1<<30)))
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
    with torch.no_grad():
        pr=net(torch.tensor((feat(ei)-mu)/sd)).argmax(1).numpy()
    return dbl(D['Pn'][ei+k][np.arange(len(ei)),pr])

DATA={s:load(s) for s in V2I}
res={k:{'table':[], 'mlp':[]} for k in HOR}
for r in range(NSPLIT):
    rng=np.random.default_rng(1000+r)
    for k in HOR:
        tb, ml = [], []
        for s in V2I:
            D=DATA[s]; tr,va,te=seqsplit(D,rng)
            best=None
            for KM in KMAP_G:
                for H in HIST_G:                      # tuned on VALIDATION, as in Table 1
                    ev=eval_idx(D,va,k,H)
                    if len(ev)<80: continue
                    v=table(D,tr,ev,k,KM,H)
                    if best is None or v<best[0]: best=(v,KM,H)
            if best is None: continue
            _,KM,H = best
            eiT=eval_idx(D,te,k,H)
            if len(eiT)<80: continue
            tb.append(table(D,tr,eiT,k,KM,H))
            eiM=eval_idx(D,te,k,SEQ)
            if len(eiM)>=80:
                v=mlp(D,tr,eiM,k,rng)
                if v is not None: ml.append(v)
        if tb: res[k]['table'].append(float(np.mean(tb)))
        if ml: res[k]['mlp'].append(float(np.mean(ml)))
    print(f'  split {r+1}/{NSPLIT} done', flush=True)

out={}
print(f'\n{"h[s]":>5} | {"table mean(sd)":>18} {"MLP mean(sd)":>18} | {"MLP wins":>9} {"verdict":>22}')
for k in HOR:
    a=np.array(res[k]['table']); b=np.array(res[k]['mlp'])
    if not len(a) or not len(b): continue
    w=int((b<a).sum())
    v='MLP better' if w>0.9*len(b) else ('table better' if w<0.1*len(b) else 'not separated')
    out[str(k)]=dict(table_mean=float(a.mean()), table_sd=float(a.std(ddof=1)),
                     mlp_mean=float(b.mean()), mlp_sd=float(b.std(ddof=1)),
                     n=len(a), mlp_wins=w, verdict=v)
    print(f'{k*0.092:5.2f} | {a.mean():>10.3f}({a.std(ddof=1):.3f}) {b.mean():>10.3f}({b.std(ddof=1):.3f})'
          f' | {w:>4}/{len(b)} {v:>22}')
json.dump(out, open('results/surveyed_spread.json','w'), indent=1)
print('\nsaved -> results/surveyed_spread.json')
