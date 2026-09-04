"""The floor, estimated out of sample.

fixed_beam_reference selects the power-optimal fixed beam by argmax of the mean normalised power
over the same W=11 frames it is then scored on. With K=64 and W=11 that selection is optimistic,
and the direction matters: an under-estimated floor makes the addressable remainder look larger
than it is, which is the direction the paper's own argument leans on.

Leave-one-out removes the selection bias: for each frame, the fixed beam is chosen from the
other W-1 frames of its window and scored on the held-out frame. Reported per deployment,
together with the sensitivity of the floor to (W, DMAX) that the earlier sweep measured for C
but not for the floor itself.
"""
import numpy as np, json

V2I=[31,32,33,34]
GRID=[(7,0.25),(9,0.25),(11,0.25),(7,0.5),(9,0.5),(11,0.5),(15,0.5),(21,0.5),(11,1.0),(15,1.0)]


def load(s):
    d=np.load(f'data/cache/s{s}.npz',allow_pickle=True); P=d['pwr'].astype(np.float64)
    return dict(Pn=P/np.maximum(np.nanmax(P,1,keepdims=True),1e-30), beam=d['beam'].astype(int),
                seq=d['seq'].astype(int), x=d['x'].astype(float), y=d['y'].astype(float),
                n=len(d['beam']), K=P.shape[1],
                good=~np.isnan(P).any(1)&~np.isnan(d['x'])&~np.isnan(d['y']))


def windows(D,W,DMAX):
    seq,good,x,y,n=D['seq'],D['good'],D['x'],D['y'],D['n']
    inw=np.zeros(n,bool)
    for st in range(n-W):
        i=np.arange(st,st+W)
        if good[i].all() and seq[i[0]]==seq[i[-1]] and np.hypot(x[i]-x[i[0]],y[i]-y[i[0]]).max()<=DMAX:
            inw[i]=True
    out,st=[],0
    while st<=n-W:
        i=np.arange(st,st+W)
        if inw[i].all() and seq[i[0]]==seq[i[-1]] and np.hypot(x[i]-x[i[0]],y[i]-y[i[0]]).max()<=DMAX:
            out.append(i); st+=W
        else: st+=1
    return out


def floor_est(D,W,DMAX):
    ws=windows(D,W,DMAX)
    if len(ws)<5: return None
    ins,loo,err_i,err_l=[],[],[],[]
    for i in ws:
        P=D['Pn'][i]; b=D['beam'][i]
        bi=int(np.nanmean(P,0).argmax())
        ins.append(P[:,bi]); err_i.append(b!=bi)
        for j in range(len(i)):
            m=np.ones(len(i),bool); m[j]=False
            bj=int(np.nanmean(P[m],0).argmax())
            loo.append(P[j,bj]); err_l.append(b[j]!=bj)
    f=lambda v:-10*np.log10(max(float(np.mean(np.concatenate(v) if isinstance(v[0],np.ndarray) else v)),1e-12))
    return dict(n_win=len(ws), ins=f(ins), loo=f(loo),
                err_ins=float(np.mean(np.concatenate(err_i))), err_loo=float(np.mean(err_l)))


out={}
print('the floor, in-sample vs leave-one-out, at the paper\'s window definition')
print(f'{"site":>6} {"windows":>8} {"in-sample":>10} {"LOO":>8} {"inflation":>10} | '
      f'{"err IS":>7} {"err LOO":>8}')
for s in V2I:
    D=load(s); r=floor_est(D,11,0.5); out[f's{s}']=r
    print(f'{s:>6} {r["n_win"]:8d} {r["ins"]:10.4f} {r["loo"]:8.4f} '
          f'{100*(r["loo"]/r["ins"]-1):9.0f}% | {100*r["err_ins"]:6.1f}% {100*r["err_loo"]:7.1f}%')
mi=float(np.mean([out[f's{s}']['ins'] for s in V2I])); ml=float(np.mean([out[f's{s}']['loo'] for s in V2I]))
print(f'{"mean":>6} {"":>8} {mi:10.4f} {ml:8.4f} {100*(ml/mi-1):9.0f}%')
print(f'  per-deployment LOO floor spans {min(out[f"s{s}"]["loo"] for s in V2I):.3f}-'
      f'{max(out[f"s{s}"]["loo"] for s in V2I):.3f} dB')
print(f'  LOO error rate spans {100*min(out[f"s{s}"]["err_loo"] for s in V2I):.0f}-'
      f'{100*max(out[f"s{s}"]["err_loo"] for s in V2I):.0f}%')

print('\nsensitivity of the LOO floor to the window definition (never measured before)')
print(f'{"W":>4} {"DMAX":>6} | ' + ' '.join(f'{"s"+str(s):>8}' for s in V2I) + f' {"mean":>8}')
sens={}
for W,DM in GRID:
    row=[]
    for s in V2I:
        r=floor_est(load(s),W,DM); row.append(r['loo'] if r else np.nan)
    sens[f'W{W}_d{DM}']=row
    print(f'{W:>4} {DM:6.2f} | ' + ' '.join(f'{x:8.4f}' if np.isfinite(x) else f'{"n/a":>8}' for x in row)
          + f' {np.nanmean(row):8.4f}')
mn=[np.nanmean(v) for v in sens.values()]
print(f'  floor over 10 window definitions: {min(mn):.3f}-{max(mn):.3f} dB (spread {max(mn)-min(mn):.3f})')
json.dump(dict(per_site=out, sensitivity=sens), open('results/floor_out_of_sample.json','w'),
          indent=1, default=float)
print('\nsaved -> results/floor_out_of_sample.json')
