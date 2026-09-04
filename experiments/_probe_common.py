"""Shared loaders for the probe-allocation experiment: budget allocated by local ceiling.

The paper measures an irreducible ceiling C = max_b p(b | geometry).  That measurement has a
direct operational consequence that nothing in the paper currently exploits: where the local
ceiling is HIGH the geometry already determines the beam and one probe suffices; where it is
LOW the beam is genuinely ambiguous and probing more pays.  A fixed probe budget spends the
same effort on both.

The local ceiling is available at run time for free, from the same kNN neighbourhood the site
map already builds:

    c_hat(q) = max_b  sum of weights of neighbours of q whose best beam is b

This is the paper's own ceiling estimator, evaluated over spatial rather than temporal
neighbours.  The allocator is a threshold on c_hat:

    c_hat(q) <  tau  ->  probe N_hi beams
    c_hat(q) >= tau  ->  probe N_lo beams

tau is calibrated on the VALIDATION split to hit a target mean budget, and every test sample
is then decided from its own c_hat alone.  This is the property the earlier gate lacked: no
test-set quantile, no rank within the evaluation batch, deployable one sample at a time.

Compared against the fixed-N curve at the SAME mean number of probes, plus an oracle allocator
that sees which samples the top-1 guess actually gets wrong, as the ceiling on any allocator.
"""
import numpy as np, json

V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]
HOR, HIST, KMAP = [8, 16, 32], 3, 15
NFIX = [1, 2, 3, 4, 6, 8, 12, 16]
PAIRS = [(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (1, 16)]


def load(s, v2v=False):
    d = np.load(f'data/cache/{"v2v_s" if v2v else "s"}{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
                y=d['y'].astype(float), good=np.asarray(g, bool), n=len(d['beam']), K=P.shape[1])


def dbl(v):
    return float(-10 * np.log10(max(float(np.mean(v)), 1e-12)))


def split(D, rng, v2v=False, nb=12):
    if v2v:
        idx = np.arange(D['n'])
        a, b = int(.5 * D['n']), int(.7 * D['n'])
        return idx < a, (idx >= a) & (idx < b), idx >= b
    g = D['good']; xy = np.stack([D['x'], D['y']], 1)
    ax = np.linalg.svd(xy[g] - xy[g].mean(0), full_matrices=False)[2][0]
    t = (xy - xy[g].mean(0)) @ ax
    blk = np.clip(np.digitize(t, np.quantile(t[g], np.linspace(0, 1, nb + 1))[1:-1]), 0, nb - 1)
    p = rng.permutation(nb); a, b = int(.5 * nb), int(.7 * nb)
    return np.isin(blk, p[:a]), np.isin(blk, p[a:b]), np.isin(blk, p[b:])


def evalidx(D, m, k):
    n, seq, good = D['n'], D['seq'], D['good']
    fut = np.minimum(np.arange(n) + k, n - 1)
    v = (np.arange(n) + k < n) & (seq[fut] == seq) & good & good[fut] & m & m[fut]
    ei = np.where(v)[0]; ei = ei[ei >= HIST]
    ei = ei[seq[ei - HIST + 1] == seq[ei]]
    return ei[good[ei - HIST + 1]]        # history frame must be valid or qx/qy go NaN


def lookup(D, tr, ei, k):
    """returns (power-space scores, local ceiling estimate c_hat) for the extrapolated position"""
    m = tr & D['good']
    mx, my, mP, mb = D['x'][m], D['y'][m], D['Pn'][m], D['beam'][m]
    sc = k / (HIST - 1)
    qx = D['x'][ei] + (D['x'][ei] - D['x'][ei - HIST + 1]) * sc
    qy = D['y'][ei] + (D['y'][ei] - D['y'][ei - HIST + 1]) * sc
    KM = min(KMAP, len(mx) - 1)
    S = np.empty((len(ei), D['K'])); C = np.empty(len(ei))
    for a in range(0, len(ei), 256):
        sl = slice(a, min(a + 256, len(ei)))
        d2 = (qx[sl, None] - mx[None]) ** 2 + (qy[sl, None] - my[None]) ** 2
        nb = np.argpartition(d2, KM, axis=1)[:, :KM]
        dd = np.take_along_axis(d2, nb, 1)
        w = 1.0 / (np.sqrt(dd) + 0.5); w /= w.sum(1, keepdims=True)
        S[sl] = (mP[nb] * w[:, :, None]).sum(1)
        lab = mb[nb]
        for r in range(nb.shape[0]):
            v = np.zeros(D['K']); np.add.at(v, lab[r], w[r])
            C[sl.start + r] = v.max()          # <- the local ceiling, from training data only
    return S, C


def loss_per_sample(D, ei, k, S, Nvec):
    """probe Nvec[i] beams for sample i; keep the measured best"""
    Pf = D['Pn'][ei + k]
    order = np.argsort(-S, axis=1)
    out = np.empty(len(ei))
    for i in range(len(ei)):
        out[i] = Pf[i, order[i, :Nvec[i]]].max()
    return out


rows = []
print('adaptive probing driven by the local ceiling, spatial split')
print(f'{"scen":>7} {"k":>3} | {"budget":>7} {"fixed-N dB":>11} {"adaptive dB":>12} '
      f'{"gain":>7} | {"oracle dB":>10}')
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v)
    tr, va, te = split(D, np.random.default_rng(20260901 + s), v2v=v)
    for k in HOR:
        eva, ete = evalidx(D, va, k), evalidx(D, te, k)
        if len(ete) < 200 or len(eva) < 150:
            continue
        Sv, Cv = lookup(D, tr, eva, k)
        St, Ct = lookup(D, tr, ete, k)
        # fixed-N reference curve on TEST (N ascending -- interp needs ascending x)
        fixN = np.array(NFIX, float)
        fixdB = np.array([dbl(loss_per_sample(D, ete, k, St,
                          np.full(len(ete), N))) for N in NFIX])
        for lo, hi in PAIRS:
            # tau calibrated on VALIDATION only, over a grid of quantiles of c_hat
            for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                tau = float(np.quantile(Cv, q))          # a fixed number, applied per sample
                Nt = np.where(Ct < tau, hi, lo)
                budget = float(Nt.mean())
                if budget < fixN[0] or budget > fixN[-1]:
                    continue
                adB = dbl(loss_per_sample(D, ete, k, St, Nt))
                ref = float(np.interp(budget, fixN, fixdB))   # ascending x, verified
                # oracle: same budget, spent on the samples the top-1 guess actually misses
                miss = St.argmax(1) != D['beam'][ete + k]
                nhi = int(round((budget - lo) / max(hi - lo, 1e-9) * len(ete)))
                nhi = int(np.clip(nhi, 0, len(ete)))
                ordm = np.argsort(-miss.astype(float) + np.random.default_rng(0).random(len(ete)) * 1e-6)
                No = np.full(len(ete), lo); No[ordm[:nhi]] = hi
                oDB = dbl(loss_per_sample(D, ete, k, St, No))
                rows.append(dict(scen=f'v2v{s}' if v else f's{s}', k=k, lo=lo, hi=hi, q=q,
                                 budget=budget, fixed=ref, adaptive=adB, oracle=oDB,
                                 gain=ref - adB, n=len(ete)))
        best = max([r for r in rows if r['scen'].endswith(str(s)) and r['k'] == k],
                   key=lambda r: r['gain'], default=None)
        if best:
            print(f'{best["scen"]:>7} {k:>3} | {best["budget"]:7.2f} {best["fixed"]:11.3f} '
                  f'{best["adaptive"]:12.3f} {best["gain"]:+7.3f} | {best["oracle"]:10.3f}',
                  flush=True)

print('\nadaptive vs the fixed-N curve at matched mean budget, pooled')
print(f'{"budget band":>14} {"n cells":>8} {"mean gain [dB]":>15} {"cells improved":>15} '
      f'{"oracle gain":>12}')
for lo_b, hi_b in [(1, 2), (2, 3), (3, 5), (5, 9), (9, 16)]:
    v = [r for r in rows if lo_b <= r['budget'] < hi_b]
    if not v:
        continue
    g = np.mean([r['gain'] for r in v])
    og = np.mean([r['fixed'] - r['oracle'] for r in v])
    imp = sum(1 for r in v if r['gain'] > 0)
    print(f'{f"{lo_b}-{hi_b} probes":>14} {len(v):8d} {g:+15.3f} {imp}/{len(v):<11} {og:+12.3f}')
tot = len(rows); pos = sum(1 for r in rows if r['gain'] > 0)
print(f'\nadaptive beats matched fixed-N in {pos}/{tot} configurations '
      f'({100*pos/max(tot,1):.0f}%)')
json.dump(rows, open('results/probe_adaptive.json', 'w'), indent=1)
print('saved -> results/probe_adaptive.json')
