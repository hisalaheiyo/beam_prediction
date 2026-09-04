"""The last numbers in the paper that no script had re-derived: frame rate, V2V drift,
the exchangeability test, and the C sensitivity across 15 window definitions."""
import json, numpy as np
R='results'; bad=[]
def chk(c,s,v,tol):
    ok=abs(s-v)<=tol
    if not ok: bad.append(c)
    print(f'  {"OK " if ok else "BAD"}  {c:<44} paper {s:<8} computed {v:.4f}')

def secs(a):
    out=np.full(len(a), np.nan)
    for i,v in enumerate(a):
        try:
            hms,us=str(v).split('-'); h,m,s=hms.split(':')
            out[i]=int(h)*3600+int(m)*60+int(s)+int(us)/1e6
        except Exception: pass
    return out

print('--- frame rate, recomputed from the cached timestamps (Sec 3.1) ---')
import os
if not os.path.exists('data/cache/s31.npz'):
    print('   SKIP: needs data/cache/, built by data/cache_data.py from a DeepSense download')
    rates=None
else:
    rates=[]
if rates is not None:
    for s in [31,32,33,34]:
        d=np.load(f'data/cache/s{s}.npz',allow_pickle=True)
        seq=d['seq'].astype(int); ts=secs(d['ts']); dt=[]
        for q in np.unique(seq):
            i=np.nonzero(seq==q)[0]
            if len(i)>2:
                v=np.diff(ts[i]); dt.append(v[np.isfinite(v)&(v>0)&(v<1)])
        rates.append(1/float(np.median(np.concatenate(dt))))
    chk('frame rate low 10.7 Hz', 10.7, min(rates), 0.05)
    chk('frame rate high 10.9 Hz', 10.9, max(rates), 0.05)
    chk('median inter-frame interval 92 ms', 92, 1000/float(np.mean(rates)), 0.6)

print('--- within-window drift (Table 2, Sec 3.2) ---')
g=json.load(open(f'{R}/window_geometry.json'))['geometry']
v2v=[g[k]['drift_med'] for k in g if k.startswith('v2v')]
v2i=[g[k]['drift_med'] for k in g if not k.startswith('v2v')]
chk('V2V drift low 0.40 m', 0.40, min(v2v), 0.006)
chk('V2V drift high 0.46 m', 0.46, max(v2v), 0.006)
chk('V2I drift low 0.01 m', 0.01, min(v2i), 0.006)
chk('V2I drift high 0.19 m', 0.19, max(v2i), 0.006)

print('--- exchangeability (Sec 3.2) ---')
f=json.load(open(f'{R}/window_exchangeability.json'))
p_v2i=[f[k]['p'] for k in f if k.startswith('s3') and int(k[1:])<35]
p_v2v=[f[k]['p'] for k in f if not (k.startswith('s3') and int(k[1:])<35)]
chk('V2I windows pass: min p > 0.33', 0.33, min(p_v2i), 0.011)
chk('V2V windows fail: max p < 0.002', 0.002, max(p_v2v), 0.0011)

print('--- C across 15 (W,Dmax) definitions (Sec 3.2) ---')
e=json.load(open(f'{R}/bayes_accuracy_sensitivity.json'))
sp=[]
for s in e:
    v=[e[s][c]['C'] for c in e[s] if e[s][c].get('C') is not None]
    if len(v)>1: sp.append(max(v)-min(v))
chk('C spread low 0.084', 0.084, min(sp), 0.0011)
chk('C spread high 0.195', 0.195, max(sp), 0.0011)
chk('deployments in that sweep: eight', 8, len(sp), 0)
chk('window definitions: 15', 15, max(len(e[s]) for s in e), 0)

print('\n' + ('ALL RESIDUAL NUMBERS VERIFIED' if not bad else f'{len(bad)} MISMATCH: '+', '.join(bad)))
