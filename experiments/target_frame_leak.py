"""Does the blocked 'unsurveyed' split leak at long horizons?

The block mask is applied to the frame index t, but the map is queried at the position the
vehicle will occupy at t+h.  At 10 m/s and h = 2.94 s that is about 30 m ahead, and the blocks
are twelfths of the route.  If the target position falls back inside a training block, the
query is not off-support at all and the split does not mean what the paper says it means.
"""
import numpy as np

V2I=[31,32,33,34]; HOR=[8,16,32]; HIST=3

def load(s):
    d=np.load(f'data/cache/s{s}.npz',allow_pickle=True); P=d['pwr'].astype(np.float64)
    return dict(beam=d['beam'].astype(int),seq=d['seq'].astype(int),x=d['x'].astype(float),
                y=d['y'].astype(float),n=len(d['beam']),
                good=~np.isnan(P).any(1)&~np.isnan(d['x'])&~np.isnan(d['y']))

def blocks(D,rng,nb=12):
    g=D['good']; xy=np.stack([D['x'],D['y']],1)
    ax=np.linalg.svd(xy[g]-xy[g].mean(0),full_matrices=False)[2][0]
    t=(xy-xy[g].mean(0))@ax
    blk=np.clip(np.digitize(t,np.quantile(t[g],np.linspace(0,1,nb+1))[1:-1]),0,nb-1)
    p=rng.permutation(nb); a=int(.7*nb)
    return blk, np.isin(blk,p[:a]), np.isin(blk,p[a:])

def evalidx(D,m,k):
    n,seq,good=D['n'],D['seq'],D['good']
    fut=np.minimum(np.arange(n)+k,n-1)
    v=(np.arange(n)+k<n)&(seq[fut]==seq)&good&good[fut]&m
    ei=np.where(v)[0]; ei=ei[ei>=HIST]; ei=ei[seq[ei-HIST+1]==seq[ei]]
    return ei[good[ei-HIST+1]]

print('is the TARGET position (t+h) inside a training block?')
print(f'{"scen":>5} {"h[s]":>5} | {"target in train blk":>20} {"med dist target->train":>24} '
      f'{"med dist current->train":>25}')
agg={}
for s in V2I:
    D=load(s); blk,tr,te=blocks(D,np.random.default_rng(20260901+s))
    trpts=np.stack([D['x'][tr&D['good']],D['y'][tr&D['good']]],1)
    for k in HOR:
        ei=evalidx(D,te,k)
        if len(ei)<200: continue
        tgt=ei+k
        frac=float(np.mean(tr[tgt]))
        def med(idx):
            B=np.stack([D['x'][idx],D['y'][idx]],1); out=np.empty(len(B))
            for a in range(0,len(B),512):
                sl=slice(a,min(a+512,len(B)))
                out[sl]=np.sqrt(((B[sl,None]-trpts[None])**2).sum(2)).min(1)
            return float(np.median(out))
        d_t, d_c = med(tgt), med(ei)
        agg.setdefault(k,[]).append((frac,d_t,d_c))
        print(f'{s:>5} {k*0.092:5.2f} | {frac:19.1%} {d_t:23.2f}m {d_c:24.2f}m')
print()
print(f'{"h[s]":>5} | {"target in train blk":>20} {"med dist target":>17} {"med dist current":>18}')
for k in HOR:
    v=agg[k]
    print(f'{k*0.092:5.2f} | {np.mean([x[0] for x in v]):19.1%} '
          f'{np.mean([x[1] for x in v]):16.2f}m {np.mean([x[2] for x in v]):17.2f}m')
import json
out={str(k): dict(frac=float(np.mean([x[0] for x in agg[k]])),
                  med_target=float(np.mean([x[1] for x in agg[k]])),
                  med_current=float(np.mean([x[2] for x in agg[k]])),
                  per_cell_frac=[float(x[0]) for x in agg[k]],
                  zero_median_cells=int(sum(x[1] < 1e-9 for x in agg[k])))
     for k in HOR}
out['n_zero_median_cells_total']=int(sum(out[str(k)]['zero_median_cells'] for k in HOR))
out['n_cells_total']=int(sum(len(agg[k]) for k in HOR))
json.dump(out, open('results/target_frame_leak.json','w'), indent=1)
print('\nsaved -> results/target_frame_leak.json')
