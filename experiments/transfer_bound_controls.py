"""Revisit floor, controlling for the confounds that inflate it.

Two frames at the same (x,y) on different passes are the same geometry only if the vehicle is
also pointing the same way -- opposite lanes share coordinates but not geometry.  Sweep the
heading tolerance and the cluster radius to see how much of the stop-to-revisit gap is a real
transfer gap and how much is heterogeneous geometry inside a cluster.
"""
import numpy as np, json, itertools

V2I = [31, 32, 33, 34]
MINPASS, VMIN = 6, 1.0


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True); P = d['pwr'].astype(np.float64)
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int),
                x=d['x'].astype(float), y=d['y'].astype(float), n=len(d['beam']),
                good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))


def kinematics(D, dt=0.092):
    v = np.full(D['n'], np.nan); th = np.full(D['n'], np.nan)
    for s in np.unique(D['seq']):
        i = np.nonzero(D['seq'] == s)[0]
        if len(i) < 3: continue
        dx = D['x'][i[2:]] - D['x'][i[:-2]]; dy = D['y'][i[2:]] - D['y'][i[:-2]]
        v[i[1:-1]] = np.hypot(dx, dy) / (2 * dt)
        th[i[1:-1]] = np.degrees(np.arctan2(dy, dx))
    return v, th


def loo(P):
    m = len(P); out = []
    for j in range(m):
        k = np.ones(m, bool); k[j] = False
        out.append(P[j, int(np.nanmean(P[k], 0).argmax())])
    return out


def run(D, v, th, R, HTOL):
    ok = D['good'] & (v > VMIN) & ~np.isnan(th)
    idx = np.nonzero(ok)[0]
    x, y, sq, h = D['x'][idx], D['y'][idx], D['seq'][idx], th[idx]
    used = np.zeros(len(idx), bool); g = []
    for a in range(len(idx)):
        if used[a]: continue
        dh = np.abs((h - h[a] + 180) % 360 - 180)
        d = np.hypot(x - x[a], y - y[a])
        near = np.nonzero((d <= R) & (dh <= HTOL) & ~used)[0]
        if len(near) < MINPASS: continue
        pick = [near[sq[near] == s][np.argmin(d[near[sq[near] == s]])]
                for s in np.unique(sq[near])]
        if len(pick) < MINPASS: continue
        used[pick] = True
        g.append(loo(D['Pn'][idx[np.array(pick)]]))
    if not g: return None, 0
    return -10 * np.log10(max(float(np.mean(np.concatenate(g))), 1e-12)), len(g)


stop = json.load(open('results/floor_out_of_sample.json'))['per_site']
sf = float(np.mean([stop[f's{s}']['loo'] for s in V2I]))
DATA = {s: (lambda D: (D,) + kinematics(D))(load(s)) for s in V2I}

print(f'stop-window floor (reference) = {sf:.4f} dB\n')
print(f'{"R[m]":>5} {"heading tol":>12} {"clusters":>9} {"revisit floor":>14} {"ratio":>7}')
out = {}
for R, HTOL in itertools.product([0.5, 0.35, 0.25], [180, 45, 20]):
    fs, ns = [], 0
    for s in V2I:
        D, v, th = DATA[s]
        f, n = run(D, v, th, R, HTOL)
        if f is not None: fs.append(f); ns += n
    if not fs: continue
    m = float(np.mean(fs))
    tag = 'any heading' if HTOL == 180 else f'+/-{HTOL} deg'
    out[f'R{R}_H{HTOL}'] = dict(floor=m, n_cluster=ns, ratio=m / sf)
    print(f'{R:>5} {tag:>12} {ns:>9} {m:>14.4f} {m/sf:>6.2f}x')

json.dump(dict(stop_floor=sf, sweep=out), open('results/transfer_bound_controls.json', 'w'), indent=1)
print('\nsaved -> results/transfer_bound_controls.json')
