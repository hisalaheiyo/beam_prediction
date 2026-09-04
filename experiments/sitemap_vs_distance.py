"""Replace the two split labels with one curve: loss against distance to the nearest survey point.

Two agents independently found that the blocked "unsurveyed" numbers come from a single block
permutation whose variance is 4-14x, because the permutation decides how far off-support the
queries land. If that is the mechanism, then distance-to-support -- not the split label -- is
the variable that actually orders the loss, and reporting against it should be stable where
the split label is not.

Two things measured:
  (1) permutation variance of the pooled unsurveyed budget, over many block permutations;
  (2) loss binned by the distance from the QUERIED position to the nearest surveyed frame,
      separately per horizon and per split, to see whether one curve covers both.
"""
import numpy as np, json

V2I = [31, 32, 33, 34]
HOR, HIST, KMAP, W, DMAX = [8, 16, 32], 3, 15, 11, 0.5
NPERM = 12
BINS = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 1e9]


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
                y=d['y'].astype(float), n=len(d['beam']), K=P.shape[1],
                good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))


def dbl(v):
    return float(-10 * np.log10(max(float(np.mean(v)), 1e-12)))


def split(D, rng, mode, nb=12):
    if mode == 'seq':
        u = np.unique(D['seq'][D['good']]); p = rng.permutation(len(u)); a = int(.7 * len(u))
        return np.isin(D['seq'], u[p[:a]]), np.isin(D['seq'], u[p[a:]])
    g = D['good']; xy = np.stack([D['x'], D['y']], 1)
    ax = np.linalg.svd(xy[g] - xy[g].mean(0), full_matrices=False)[2][0]
    t = (xy - xy[g].mean(0)) @ ax
    blk = np.clip(np.digitize(t, np.quantile(t[g], np.linspace(0, 1, nb + 1))[1:-1]), 0, nb - 1)
    p = rng.permutation(nb); a = int(.7 * nb)
    return np.isin(blk, p[:a]), np.isin(blk, p[a:])


def evalidx(D, m, k):
    n, seq, good = D['n'], D['seq'], D['good']
    fut = np.minimum(np.arange(n) + k, n - 1)
    v = (np.arange(n) + k < n) & (seq[fut] == seq) & good & good[fut] & m & m[fut]
    ei = np.where(v)[0]; ei = ei[ei >= HIST]
    ei = ei[seq[ei - HIST + 1] == seq[ei]]
    return ei[good[ei - HIST + 1]]


def lookup(D, tr, qx, qy):
    m = tr & D['good']
    mx, my, mP = D['x'][m], D['y'][m], D['Pn'][m]
    out = np.empty((len(qx), D['K'])); dm = np.empty(len(qx))
    for a in range(0, len(qx), 256):
        sl = slice(a, min(a + 256, len(qx)))
        d2 = (qx[sl, None] - mx[None]) ** 2 + (qy[sl, None] - my[None]) ** 2
        kk = min(KMAP, d2.shape[1] - 1)
        nb = np.argpartition(d2, kk, axis=1)[:, :kk]
        dd = np.take_along_axis(d2, nb, 1)
        dm[sl] = np.sqrt(dd.min(1))
        w = 1.0 / (np.sqrt(dd) + 0.5); w /= w.sum(1, keepdims=True)
        out[sl] = (mP[nb] * w[:, :, None]).sum(1)
    return out, dm


rows, pooled = [], {}
for perm in range(NPERM):
    for mode in ['seq', 'spatial']:
        for s in V2I:
            D = load(s)
            tr, te = split(D, np.random.default_rng(1000 * perm + s), mode)
            for k in HOR:
                ei = evalidx(D, te, k)
                if len(ei) < 150:
                    continue
                Pf = D['Pn'][ei + k]; ii = np.arange(len(ei))
                # loss at the TRUE future position: isolates the map term from tracking
                S, dm = lookup(D, tr, D['x'][ei + k], D['y'][ei + k])
                g = Pf[ii, S.argmax(1)]
                pooled.setdefault((mode, k), []).append(dbl(g))
                b = np.digitize(dm, BINS[1:-1])
                for bi in range(len(BINS) - 1):
                    sel = b == bi
                    if sel.sum() >= 60:
                        rows.append(dict(perm=perm, split=mode, scen=s, k=k, bin=bi,
                                         n=int(sel.sum()), dist=float(np.median(dm[sel])),
                                         loss=dbl(g[sel])))
    print(f'  permutation {perm+1}/{NPERM} done', flush=True)

print('\n(1) permutation variance of the pooled loss at the true future position')
print(f'{"split":>8} {"h[s]":>5} {"n perms":>8} {"min":>7} {"median":>8} {"max":>7} {"max/min":>8}')
for mode in ['seq', 'spatial']:
    for k in HOR:
        v = pooled.get((mode, k), [])
        if len(v) < 4: continue
        print(f'{mode:>8} {k*0.092:5.2f} {len(v):8d} {min(v):7.3f} {np.median(v):8.3f} '
              f'{max(v):7.3f} {max(v)/max(min(v),1e-9):8.1f}x')

print('\n(2) loss against distance to the nearest surveyed frame')
print(f'{"dist band [m]":>15} {"n cells":>8} | ' + ' '.join(f'{"h="+str(round(k*0.092,2)):>9}' for k in HOR)
      + ' | ' + ' '.join(f'{m:>9}' for m in ['seq','spatial']))
lab = ['<0.25','0.25-0.5','0.5-1','1-2','2-4','4-8','8-16','>16']
for bi in range(len(BINS) - 1):
    v = [r for r in rows if r['bin'] == bi]
    if len(v) < 6: continue
    byk = [np.mean([r['loss'] for r in v if r['k'] == k]) for k in HOR]
    bym = [np.mean([r['loss'] for r in v if r['split'] == m]) for m in ['seq','spatial']]
    print(f'{lab[bi]:>15} {len(v):8d} | ' + ' '.join(f'{x:9.3f}' if np.isfinite(x) else f'{"--":>9}' for x in byk)
          + ' | ' + ' '.join(f'{x:9.3f}' if np.isfinite(x) else f'{"--":>9}' for x in bym))
print('\nif the columns agree within a band, distance -- not the split label -- is the variable')
json.dump(rows, open('results/sitemap_vs_distance.json','w'), indent=1)
print('saved -> results/sitemap_vs_distance.json')
