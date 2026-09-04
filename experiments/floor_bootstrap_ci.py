"""Cluster bootstrap CI for the leave-one-out floor, resampling distinct stopping events.

The method section promises exactly this ("Confidence intervals are bootstrapped over distinct
stopping events, not over windows or frames") but no interval is reported anywhere.  0.039 dB
is the paper's headline number and currently carries no uncertainty beyond the (W,DMAX) sweep.

Windows inside one stopping event are repeated looks at one geometry, so the event is the
independent unit: resample events with replacement, keep every window of a drawn event.
"""
import numpy as np, json

V2I = [31, 32, 33, 34]
W, DMAX, B = 11, 0.5, 4000
rng = np.random.default_rng(0)


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True); P = d['pwr'].astype(np.float64)
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int),
                x=d['x'].astype(float), y=d['y'].astype(float), n=len(d['beam']),
                good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))


def collect(D):
    """Per non-overlapping frozen window: the W held-out delivered powers, and its event id."""
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    inw = np.zeros(n, bool)
    for st in range(n - W):
        i = np.arange(st, st + W)
        if good[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX:
            inw[i] = True
    ev = np.zeros(n, int); e = 0; prev = False
    for t in range(n):
        if inw[t] and not prev: e += 1
        ev[t] = e if inw[t] else 0; prev = inw[t]
    gains, eid, st = [], [], 0
    while st <= n - W:
        i = np.arange(st, st + W)
        if (inw[i].all() and ev[i[0]] == ev[i[-1]]
                and np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX):
            P = D['Pn'][i]; g = []
            for j in range(W):
                m = np.ones(W, bool); m[j] = False
                g.append(P[j, int(np.nanmean(P[m], 0).argmax())])
            gains.append(np.array(g)); eid.append(ev[i[0]]); st += W
        else:
            st += 1
    return gains, np.array(eid)


dB = lambda v: -10 * np.log10(max(float(np.mean(v)), 1e-12))

per, store = {}, {}
for s in V2I:
    g, e = collect(load(s))
    store[s] = (g, e)
    ue = np.unique(e); pt = dB(np.concatenate(g)); bs = []
    for _ in range(B):
        drawn = rng.choice(ue, len(ue), replace=True)
        bs.append(dB(np.concatenate([np.concatenate([g[k] for k in np.nonzero(e == d)[0]])
                                     for d in drawn])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    per[f's{s}'] = dict(floor=pt, ci=[float(lo), float(hi)], n_events=int(len(ue)), n_win=len(g))
    print(f'  s{s}: floor {pt:.4f} dB   95% CI [{lo:.4f}, {hi:.4f}]   '
          f'{len(ue)} events, {len(g)} windows')

# The paper's 0.039 is the MEAN OF THE FOUR PER-DEPLOYMENT FLOORS, because the budget in
# Sec 3.3 subtracts L_irr per deployment and then averages the four budgets.  The interval
# must be built for that same estimator: resample events within each deployment, recompute
# each deployment's floor, average the four.
pt = float(np.mean([per[f's{s}']['floor'] for s in V2I]))
bs = []
for _ in range(B):
    reps = []
    for s in V2I:
        g, e = store[s]; ue = np.unique(e)
        drawn = rng.choice(ue, len(ue), replace=True)
        reps.append(dB(np.concatenate([np.concatenate([g[k] for k in np.nonzero(e == d)[0]])
                                       for d in drawn])))
    bs.append(float(np.mean(reps)))
lo, hi = np.percentile(bs, [2.5, 97.5])
nev = sum(per[f's{s}']['n_events'] for s in V2I)
print(f'\nmean of four deployment floors: {pt:.4f} dB   95% CI [{lo:.4f}, {hi:.4f}]   '
      f'({nev} events total)')

# what the naive per-frame bootstrap, ignoring the event clustering, would have claimed
nb = []
for _ in range(B):
    reps = []
    for s in V2I:
        g, _e = store[s]; flat = np.concatenate(g)
        reps.append(dB(rng.choice(flat, len(flat), replace=True)))
    nb.append(float(np.mean(reps)))
nlo, nhi = np.percentile(nb, [2.5, 97.5])
print(f'ignoring event clustering would report [{nlo:.4f}, {nhi:.4f}], '
      f'{(hi-lo)/(nhi-nlo):.1f}x too narrow')

# does the qualitative claim survive the interval?  floor as a share of the 0.088 dB total
print(f'\nfloor share of the 0.088 dB surveyed short-horizon total: '
      f'{100*pt/0.088:.0f}%  (CI {100*lo/0.088:.0f}-{100*hi/0.088:.0f}%)')

per['mean_of_deployments'] = dict(floor=pt, ci=[float(lo), float(hi)], n_events=nev,
                                  naive_ci=[float(nlo), float(nhi)])
json.dump(per, open('results/floor_bootstrap_ci.json', 'w'), indent=1)
print('\nsaved -> results/floor_bootstrap_ci.json')
