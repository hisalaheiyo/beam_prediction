"""What is the addressable loss actually worth?

The paper partitions a budget of roughly 0.5 dB without ever saying whether 0.5 dB matters.
A reviewer is entitled to read that as partitioning a negligible quantity. Three conversions,
each standard and each computable from the measured power vectors:

  (1) spectral efficiency. A loss of L dB in delivered power moves post-beamforming SNR by L,
      so the rate cost depends on the operating SNR: large in the linear region, small when
      already high. Reported at 0, 10 and 20 dB.
  (2) outage. Fraction of frames whose delivered power falls more than 1 dB and 3 dB below the
      per-frame optimum -- the tail that an average in dB hides.
  (3) the beam-sweep it replaces. Predicting instead of sweeping saves K probe slots; the
      question is how many beams a predictor must probe to match a given loss.
"""
import numpy as np, json, glob

V2I = [31, 32, 33, 34]
HOR, HIST, KMAP = [8, 16, 32], 3, 15
SNRS = [0, 10, 20]


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
                y=d['y'].astype(float), n=len(d['beam']), K=P.shape[1],
                good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))


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


rows = []
for mode in ['seq', 'spatial']:
    for s in V2I:
        D = load(s)
        tr, te = split(D, np.random.default_rng(20260901 + s), mode)
        for k in HOR:
            ei = evalidx(D, te, k)
            if len(ei) < 200:
                continue
            sc = k / (HIST - 1)
            qx = D['x'][ei] + (D['x'][ei] - D['x'][ei - HIST + 1]) * sc
            qy = D['y'][ei] + (D['y'][ei] - D['y'][ei - HIST + 1]) * sc
            Pf = D['Pn'][ei + k]; ii = np.arange(len(ei))
            S = lookup(D, tr, qx, qy)
            g = Pf[ii, S.argmax(1)]                       # delivered / optimum, per frame
            row = dict(scen=s, split=mode, k=k, n=len(ei),
                       dB=float(-10*np.log10(max(g.mean(),1e-12))),
                       out1=float((g < 10**-0.1).mean()), out3=float((g < 10**-0.3).mean()))
            for snr in SNRS:
                lin = 10 ** (snr / 10.0)
                se_opt = float(np.mean(np.log2(1 + lin)))
                se_got = float(np.mean(np.log2(1 + lin * g)))
                row[f'se{snr}'] = se_opt - se_got
                row[f'serel{snr}'] = 100 * (se_opt - se_got) / se_opt
            # probes needed to reach the same delivered power by sweeping the top-N of the map
            ordr = np.argsort(-S, axis=1)
            for N in [1, 2, 3, 4, 8]:
                row[f'probe{N}'] = float(-10*np.log10(max(
                    np.take_along_axis(Pf, ordr[:, :N], 1).max(1).mean(), 1e-12)))
            rows.append(row)

print('what the loss is worth, pooled over the four V2I deployments')
print(f'{"split":>8} {"h[s]":>5} {"loss[dB]":>9} | ' +
      ' '.join(f'{"SE@"+str(s)+"dB":>10}' for s in SNRS) + f' | {"P(>1dB)":>8} {"P(>3dB)":>8}')
for mode in ['seq', 'spatial']:
    for k in HOR:
        v = [r for r in rows if r['split'] == mode and r['k'] == k]
        if not v: continue
        print(f'{mode:>8} {k*0.092:5.2f} {np.mean([r["dB"] for r in v]):9.3f} | ' +
              ' '.join(f'{np.mean([r[f"serel{s}"] for r in v]):9.2f}%' for s in SNRS) +
              f' | {np.mean([r["out1"] for r in v]):7.1%} {np.mean([r["out3"] for r in v]):7.1%}')
print()
print('absolute spectral-efficiency loss [bit/s/Hz]')
for mode in ['seq', 'spatial']:
    for k in HOR:
        v = [r for r in rows if r['split'] == mode and r['k'] == k]
        if not v: continue
        print(f'  {mode:>8} h={k*0.092:.2f}s  ' +
              '  '.join(f'{s:2d} dB SNR: {np.mean([r[f"se{s}"] for r in v]):.4f}' for s in SNRS))
print()
print('probing instead of predicting: loss after sweeping the map\'s top-N')
print(f'{"split":>8} {"h[s]":>5} | ' + ' '.join(f'{"N="+str(n):>8}' for n in [1,2,3,4,8]))
for mode in ['seq', 'spatial']:
    for k in HOR:
        v = [r for r in rows if r['split'] == mode and r['k'] == k]
        if not v: continue
        print(f'{mode:>8} {k*0.092:5.2f} | ' + ' '.join(
            f'{np.mean([r[f"probe{n}"] for r in v]):8.3f}' for n in [1,2,3,4,8]))
json.dump(rows, open('results/rate_cost_and_probing.json','w'), indent=1)
print('\nsaved -> results/rate_cost_and_probing.json')
