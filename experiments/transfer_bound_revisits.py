"""Validate the stops-to-motion transfer that the whole floor argument rests on.

The floor is measured in windows where the vehicle is nearly stationary and then imported to
moving frames.  The paper states that as an assumption and never tests it.  It is testable:
each deployment drives the route many times, so one location is visited on several separate
passes.  Take at most ONE frame per pass, so the members of a cluster are independent traversals
of the same geometry AT SPEED, and run the same leave-one-out fixed-beam estimator on them.

If the revisit floor agrees with the stop floor, the transfer is supported.  If the revisit
floor is much larger, motion adds ambiguity that geometry does not resolve and the reported
floor is an under-estimate for moving frames.

Also answers two claims the paper asserts without evidence:
  (B) is the within-window beam variation predictable from the previous frame?  If it were,
      the "irreducible" term would just be untracked structure (e.g. a blocker moving through).
  (C) do scenarios 5-9 really carry no stopping windows, as the conclusion states?
"""
import numpy as np, json

V2I = [31, 32, 33, 34]
W, DMAX = 11, 0.5
R = 0.5          # cluster radius, matched to the stop-window drift cap
MINPASS = 6      # a cluster needs this many distinct passes
VMIN = 1.0       # m/s, members must actually be moving


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True); P = d['pwr'].astype(np.float64)
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int),
                x=d['x'].astype(float), y=d['y'].astype(float), n=len(d['beam']),
                good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))


def speed(D, dt=0.092):
    v = np.full(D['n'], np.nan)
    for s in np.unique(D['seq']):
        i = np.nonzero(D['seq'] == s)[0]
        if len(i) < 3: continue
        v[i[1:-1]] = np.hypot(D['x'][i[2:]] - D['x'][i[:-2]],
                              D['y'][i[2:]] - D['y'][i[:-2]]) / (2 * dt)
    return v


def loo_floor(Pn_rows):
    """Leave-one-out delivered power of the best fixed beam, same estimator as the stop windows."""
    m = len(Pn_rows); out = []
    for j in range(m):
        k = np.ones(m, bool); k[j] = False
        out.append(Pn_rows[j, int(np.nanmean(Pn_rows[k], 0).argmax())])
    return out


def revisit_clusters(D, v):
    ok = D['good'] & (v > VMIN)
    idx = np.nonzero(ok)[0]
    x, y, sq = D['x'][idx], D['y'][idx], D['seq'][idx]
    used = np.zeros(len(idx), bool)
    cl = []
    for a in range(len(idx)):
        if used[a]: continue
        d = np.hypot(x - x[a], y - y[a])
        near = np.nonzero((d <= R) & ~used)[0]
        if len(near) < MINPASS: continue
        pick = []                                   # one frame per pass: the closest
        for s in np.unique(sq[near]):
            c = near[sq[near] == s]
            pick.append(c[np.argmin(d[c])])
        if len(pick) < MINPASS: continue
        used[pick] = True
        cl.append(idx[np.array(pick)])
    return cl


print('(A) floor from spatial revisits AT SPEED vs floor from temporal stops')
print(f'{"site":>6} {"clusters":>9} {"passes/cl":>10} {"med speed":>10} {"revisit floor":>14} {"stop floor":>11}')
stop = json.load(open('results/floor_out_of_sample.json'))['per_site']
res = {}
for s in V2I:
    D = load(s); v = speed(D)
    cl = revisit_clusters(D, v)
    if not cl:
        print(f'{s:>6} {"none":>9}'); continue
    g = np.concatenate([loo_floor(D['Pn'][c]) for c in cl])
    f = -10 * np.log10(max(float(np.mean(g)), 1e-12))
    sp = float(np.median(np.concatenate([v[c] for c in cl])))
    res[f's{s}'] = dict(n_cluster=len(cl), n_frames=int(sum(len(c) for c in cl)),
                        med_speed=sp, revisit_floor=f, stop_floor=stop[f's{s}']['loo'])
    print(f'{s:>6} {len(cl):>9} {np.mean([len(c) for c in cl]):>10.1f} {sp:>9.1f}m/s '
          f'{f:>13.4f} {stop[f"s{s}"]["loo"]:>11.4f}')
if res:
    rv = float(np.mean([r['revisit_floor'] for r in res.values()]))
    sf = float(np.mean([r['stop_floor'] for r in res.values()]))
    print(f'\n  mean revisit floor {rv:.4f} dB   vs   mean stop floor {sf:.4f} dB   '
          f'ratio {rv/sf:.2f}x')

print('\n(B) is the within-window variation predictable from the previous frame?')
print(f'{"site":>6} {"fixed(LOO)":>11} {"hold-prev":>10} {"verdict":>28}')
for s in V2I:
    D = load(s)
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    inw = np.zeros(n, bool)
    for st in range(n - W):
        i = np.arange(st, st + W)
        if good[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX: inw[i] = True
    ws, st = [], 0
    while st <= n - W:
        i = np.arange(st, st + W)
        if inw[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX: ws.append(i); st += W
        else: st += 1
    fx = np.concatenate([loo_floor(D['Pn'][i]) for i in ws])
    hp = np.concatenate([D['Pn'][i][1:, :][np.arange(W - 1), D['beam'][i][:-1]] for i in ws])
    a = -10 * np.log10(max(float(np.mean(fx)), 1e-12))
    b = -10 * np.log10(max(float(np.mean(hp)), 1e-12))
    res.setdefault(f's{s}', {}).update(fixed_loo=a, hold_prev=b)
    print(f'{s:>6} {a:>11.4f} {b:>10.4f} {"history does not help" if b >= a else "HISTORY HELPS":>28}')

print('\n(C) do scenarios 5-9 carry any stopping windows?')
for s in [5, 6, 7, 8, 9]:
    D = load(s)
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    cnt = 0
    for st in range(n - W):
        i = np.arange(st, st + W)
        if good[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX: cnt += 1
    print(f'  s{s}: {cnt} qualifying windows out of {n} frames')

json.dump(res, open('results/transfer_bound_revisits.json', 'w'), indent=1)
print('\nsaved -> results/transfer_bound_revisits.json')
