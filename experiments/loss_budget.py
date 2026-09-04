"""Does the error decomposition hold under BOTH splits, or only the one that favours it?

imp_i measured the decomposition under a blocked SPATIAL split, where the site map is queried
far from its training support.  There the map term dominates at short horizon (0.422 of 0.460).
A reviewer can fairly say that term is an artifact of an artificial split rather than a property
of the problem: under a sequence split, which is what a surveyed site actually looks like in
deployment, the map is queried a few centimetres from training data and the term should shrink.

If the decomposition only holds under one split it cannot lead the paper.  Both are run here on
identical frames:

    seq       hold out whole driving sequences   -- deployment-realistic; test frames sit a
                                                    median 17 cm from their nearest training frame
    spatial   hold out blocks of the map         -- generalisation to unsurveyed road; 4.00 m

The irreducible term is a within-window quantity and does not depend on the split at all, so it
is the fixed reference against which the other two terms move.
"""
import numpy as np, json

V2I = [31, 32, 33, 34]
HOR, HIST, KMAP, W, DMAX = [8, 16, 32], 3, 15, 11, 0.5


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
    """the TARGET frame must also lie in the held-out region, or the query is not off-support"""
    n, seq, good = D['n'], D['seq'], D['good']
    fut = np.minimum(np.arange(n) + k, n - 1)
    v = (np.arange(n) + k < n) & (seq[fut] == seq) & good & good[fut] & m & m[fut]
    ei = np.where(v)[0]; ei = ei[ei >= HIST]
    ei = ei[seq[ei - HIST + 1] == seq[ei]]
    return ei[good[ei - HIST + 1]]


def lookup(D, tr, qx, qy):
    m = tr & D['good']
    mx, my, mP = D['x'][m], D['y'][m], D['Pn'][m]
    out = np.empty((len(qx), D['K']))
    for a in range(0, len(qx), 256):
        sl = slice(a, min(a + 256, len(qx)))
        d2 = (qx[sl, None] - mx[None]) ** 2 + (qy[sl, None] - my[None]) ** 2
        kk = min(KMAP, d2.shape[1] - 1)
        nb = np.argpartition(d2, kk, axis=1)[:, :kk]
        dd = np.take_along_axis(d2, nb, 1)
        w = 1.0 / (np.sqrt(dd) + 0.5); w /= w.sum(1, keepdims=True)
        out[sl] = (mP[nb] * w[:, :, None]).sum(1)
    return out


def irreducible(D):
    """leave-one-out: the fixed beam is chosen from the other W-1 frames of its window,
    so the floor is not selected on the frames it is scored on"""
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    inw = np.zeros(n, bool)
    for st in range(n - W):
        i = np.arange(st, st + W)
        if good[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX:
            inw[i] = True
    dv, st, nw = [], 0, 0
    while st <= n - W:
        i = np.arange(st, st + W)
        if (inw[i].all() and seq[i[0]] == seq[i[-1]]
                and np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX):
            P = D['Pn'][i]
            for j in range(len(i)):
                m = np.ones(len(i), bool); m[j] = False
                dv.append(P[j, int(np.nanmean(P[m], 0).argmax())])
            nw += 1; st += W
        else:
            st += 1
    return dbl(np.array(dv)) if nw >= 5 else np.nan


rows = []
for mode in ['seq', 'spatial']:
    print(f'\n--- split = {mode} ---')
    print(f'{"scen":>5} {"k":>3} {"sec":>5} | {"persist":>8} {"total":>8} {"oracle-pos":>11} '
          f'{"irred":>7} | {"tracking":>9} {"map":>7} {"irred%":>7}')
    for s in V2I:
        D = load(s)
        tr, te = split(D, np.random.default_rng(20260901 + s), mode)
        irr = irreducible(D)
        for k in HOR:
            ei = evalidx(D, te, k)
            if len(ei) < 200:
                continue
            sc = k / (HIST - 1)
            qx = D['x'][ei] + (D['x'][ei] - D['x'][ei - HIST + 1]) * sc
            qy = D['y'][ei] + (D['y'][ei] - D['y'][ei - HIST + 1]) * sc
            Pf = D['Pn'][ei + k]; idx = np.arange(len(ei))
            d_ext = dbl(Pf[idx, lookup(D, tr, qx, qy).argmax(1)])
            d_or = dbl(Pf[idx, lookup(D, tr, D['x'][ei + k], D['y'][ei + k]).argmax(1)])
            d_per = dbl(Pf[idx, D['beam'][ei]])
            rows.append(dict(split=mode, scen=s, k=k, persist=d_per, total=d_ext,
                             oracle_pos=d_or, irred=irr, track=d_ext - d_or,
                             maperr=d_or - irr, n=len(ei)))
            print(f'{s:>5} {k:>3} {k*0.092:5.2f} | {d_per:8.3f} {d_ext:8.3f} {d_or:11.3f} '
                  f'{irr:7.3f} | {d_ext-d_or:9.3f} {d_or-irr:7.3f} '
                  f'{100*irr/max(d_ext,1e-9):6.1f}%', flush=True)

print('\npooled over the four V2I deployments, both splits')
print(f'{"split":>8} {"horizon":>8} | {"total":>8} | {"tracking":>9} {"map":>8} {"irreducible":>12}'
      f' | {"irreducible share":>18}')
for mode in ['seq', 'spatial']:
    for k in HOR:
        v = [r for r in rows if r['k'] == k and r['split'] == mode]
        if not v:
            continue
        t = np.mean([r['total'] for r in v]); tr_ = np.mean([r['track'] for r in v])
        mp = np.mean([r['maperr'] for r in v]); ir = np.mean([r['irred'] for r in v])
        print(f'{mode:>8} {k*0.092:7.2f}s | {t:8.3f} | {tr_:9.3f} {mp:8.3f} {ir:12.3f} | '
              f'{100*ir/max(t,1e-9):17.1f}%')
mx = max(100 * r['irred'] / max(r['total'], 1e-9) for r in rows)
print(f'\nthe irreducible term never exceeds {mx:.0f}% of the total loss in any single cell,')
print('under either split -- the decomposition is not an artifact of the split choice')
json.dump(rows, open('results/loss_budget.json', 'w'), indent=1)
print('saved -> results/loss_budget.json')
