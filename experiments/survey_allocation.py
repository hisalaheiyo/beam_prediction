"""A method that is not a predictor: where should the survey go?

The measured curve says loss is a function of distance to the nearest surveyed point. If the
beam field were equally hard everywhere, minimising loss under a budget would be plain uniform
coverage and there would be nothing to propose. The question worth asking is whether the field
varies at different rates in different places, so that a NON-UNIFORM survey beats a uniform one
at the same cost.

The local difficulty of a place is estimable from the survey itself, without labels from
anywhere else: inside the already-surveyed part, measure how fast the optimal beam changes with
position. Places where it changes fast need denser sampling; places where it is flat do not.

Four allocation policies, all given the same budget and all buildable by an operator:
    uniform     farthest-point over the route                 (the coverage baseline)
    random      random subset                                 (the floor)
    gradient    density proportional to the local rate of beam change, estimated from a
                small pilot survey and then extrapolated                (the proposal)
    oracle-grad the same but using the rate measured on the full map     (upper bound)

PRE-REGISTERED, fixed before running: the gradient policy must beat uniform by >= 0.02 dB
pooled over the four deployments at two or more of the three budgets, without losing at any.
Otherwise it is reported as a further negative result.
"""
import numpy as np, json

V2I = [31, 32, 33, 34]
HOR, HIST, KMAP = [8, 16, 32], 3, 15
FRACS = [0.25, 0.12, 0.06]
NPERM = 6


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
                y=d['y'].astype(float), n=len(d['beam']), K=P.shape[1],
                good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))


def dbl(v):
    return float(-10 * np.log10(max(float(np.mean(v)), 1e-12)))


def split(D, rng, nb=12):
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


def beam_rate(D, idx, kk=12):
    """local rate of change of the optimal beam, per metre, from labels at idx only"""
    x, y, b = D['x'][idx], D['y'][idx], D['beam'][idx]
    r = np.zeros(len(idx))
    for a in range(0, len(idx), 256):
        sl = slice(a, min(a + 256, len(idx)))
        d2 = (x[sl, None] - x[None]) ** 2 + (y[sl, None] - y[None]) ** 2
        nb = np.argpartition(d2, kk, axis=1)[:, :kk]
        dd = np.sqrt(np.take_along_axis(d2, nb, 1)) + 0.25
        db = np.abs(b[nb] - b[sl, None]).astype(float)
        r[sl] = (db / dd).mean(1)
    return r


def farthest(pts, keep, rng):
    sel = [int(rng.integers(len(pts)))]
    d = np.hypot(pts[:, 0] - pts[sel[0], 0], pts[:, 1] - pts[sel[0], 1])
    while len(sel) < keep:
        j = int(d.argmax()); sel.append(j)
        d = np.minimum(d, np.hypot(pts[:, 0] - pts[j, 0], pts[:, 1] - pts[j, 1]))
    return np.array(sel)


def weighted_farthest(pts, w, keep, rng):
    """farthest-point, but distances are scaled by local difficulty: hard places get denser"""
    sc = np.clip(w / max(np.median(w), 1e-9), 0.3, 3.0)
    sel = [int(rng.integers(len(pts)))]
    d = np.hypot(pts[:, 0] - pts[sel[0], 0], pts[:, 1] - pts[sel[0], 1]) * sc
    while len(sel) < keep:
        j = int(d.argmax()); sel.append(j)
        d = np.minimum(d, np.hypot(pts[:, 0] - pts[j, 0], pts[:, 1] - pts[j, 1]) * sc)
    return np.array(sel)


def lookup(D, midx, qx, qy):
    mx, my, mP = D['x'][midx], D['y'][midx], D['Pn'][midx]
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


res = []
for perm in range(NPERM):
    for s in V2I:
        D = load(s)
        tr, te = split(D, np.random.default_rng(1000 * perm + s))
        idx = np.where(tr & D['good'])[0]
        pts = np.stack([D['x'][idx], D['y'][idx]], 1)
        rate_full = beam_rate(D, idx)
        rng = np.random.default_rng(77 + perm)
        pilot = rng.choice(len(idx), max(int(0.05 * len(idx)), 40), replace=False)
        # extrapolate the pilot's difficulty to the whole route by nearest pilot point
        d2p = ((pts[:, None] - pts[pilot][None]) ** 2).sum(2)
        rate_pilot = beam_rate(D, idx[pilot])[d2p.argmin(1)]
        for f in FRACS:
            keep = max(int(len(idx) * f), 40)
            pol = {'random': idx[rng.choice(len(idx), keep, replace=False)],
                   'uniform': idx[farthest(pts, keep, rng)],
                   'gradient': idx[weighted_farthest(pts, rate_pilot, keep, rng)],
                   'oracle-grad': idx[weighted_farthest(pts, rate_full, keep, rng)]}
            for nm, midx in pol.items():
                per = []
                for k in HOR:
                    ei = evalidx(D, te, k)
                    if len(ei) < 150: continue
                    Pf = D['Pn'][ei + k]; ii = np.arange(len(ei))
                    S = lookup(D, midx, D['x'][ei + k], D['y'][ei + k])
                    per.append(dbl(Pf[ii, S.argmax(1)]))
                if per:
                    res.append(dict(perm=perm, scen=s, frac=f, policy=nm,
                                    n_map=len(midx), loss=float(np.mean(per))))
    print(f'  permutation {perm+1}/{NPERM}', flush=True)

print('\nsurvey allocation at matched budget, median over permutations and deployments')
print(f'{"budget":>8} {"frames":>7} | ' + ' '.join(f'{p:>12}' for p in
      ['random','uniform','gradient','oracle-grad']))
POL = ['random','uniform','gradient','oracle-grad']
for f in FRACS:
    v = {p: [r['loss'] for r in res if r['frac'] == f and r['policy'] == p] for p in POL}
    n = np.mean([r['n_map'] for r in res if r['frac'] == f])
    print(f'{f:8.2f} {n:7.0f} | ' + ' '.join(f'{np.median(v[p]):12.3f}' for p in POL))
print()
ok = 0
for f in FRACS:
    u = np.median([r['loss'] for r in res if r['frac'] == f and r['policy'] == 'uniform'])
    g = np.median([r['loss'] for r in res if r['frac'] == f and r['policy'] == 'gradient'])
    o = np.median([r['loss'] for r in res if r['frac'] == f and r['policy'] == 'oracle-grad'])
    ok += (u - g) >= 0.02
    print(f'  budget {f:.2f}: gradient {u-g:+.3f} dB vs uniform   (oracle {u-o:+.3f} dB)')
print(f'\nPRE-REGISTERED: gradient beats uniform by >=0.02 dB at >=2 of 3 budgets, losing none')
print('RESULT: ' + ('PASS' if ok >= 2 else 'FAIL -- reported as a further negative result'))
json.dump(res, open('results/survey_allocation.json','w'), indent=1)
print('saved -> results/survey_allocation.json')
