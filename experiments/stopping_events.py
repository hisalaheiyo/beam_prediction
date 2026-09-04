"""Confidence intervals over the correct unit of analysis: the stopping event.

master_table.py bootstrapped over seq_index, calling it "the independent unit".  For V2V that
is wrong: seq_index there is a fixed 200-frame chunking of one continuous recording (verified:
199/200/200 lengths, all consecutive ids back-to-back), so those intervals are anticonservative.
The correct unit for a ceiling measured on frozen windows is the distinct STOPPING EVENT: one
vehicle, stopped once.  Windows inside one event are repeated looks at the same geometry.

Reports, per scenario, the jackknife-corrected ceiling with a cluster bootstrap over events,
and then asks the question the headline depends on: given these intervals, can the scenarios'
ceilings actually be told apart, or is "the ceiling is stable across deployments" unfalsifiable?
"""
import numpy as np, json, itertools

W, DMAX, NB = 11, 0.5, 4000
V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]


def load(s, v2v=False):
    d = np.load(f'data/cache/{"v2v_s" if v2v else "s"}{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return dict(beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
                y=d['y'].astype(float), good=np.asarray(g, bool), n=len(d['beam']), K=P.shape[1])


def jk(cnt, W_):
    pl = cnt.max() / W_
    m = 0.0
    for b in np.nonzero(cnt)[0]:
        c = cnt.copy(); c[b] -= 1
        m += (cnt[b] / W_) * (c.max() / (W_ - 1))
    return W_ * pl - (W_ - 1) * m


def collect(D):
    """(ceiling estimate, event id) for every non-overlapping frozen window."""
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    inw = np.zeros(n, bool)
    for st in range(n - W):
        i = np.arange(st, st + W)
        if good[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX:
            inw[i] = True
    ev = np.zeros(n, int); e = 0; prev = False
    for t in range(n):
        if inw[t] and not prev:
            e += 1
        ev[t] = e if inw[t] else 0; prev = inw[t]
    vals, eid, st = [], [], 0
    while st <= n - W:
        i = np.arange(st, st + W)
        if (inw[i].all() and ev[i[0]] == ev[i[-1]]
                and np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX):
            vals.append(jk(np.bincount(D['beam'][i], minlength=D['K']), W))
            eid.append(ev[i[0]]); st += W
        else:
            st += 1
    return np.array(vals), np.array(eid)


rng = np.random.default_rng(7)
res = {}
print('jackknife-corrected ceiling, cluster bootstrap over STOPPING EVENTS')
print(f'{"scn":>7} {"events":>7} {"wins":>5} | {"C_jk":>6} {"95% CI (events)":>18} '
      f'{"width":>6} | {"95% CI (windows, wrong)":>24}')
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v)
    val, eid = collect(D)
    if len(val) == 0:
        continue
    ue = np.unique(eid)
    groups = [val[eid == e] for e in ue]
    # cluster bootstrap: resample events, keep all windows within a drawn event
    bs = np.empty(NB)
    for t in range(NB):
        pick = rng.integers(0, len(groups), len(groups))
        bs[t] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # the naive version master_table used: resample windows independently
    bs2 = np.array([val[rng.integers(0, len(val), len(val))].mean() for _ in range(1000)])
    lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
    nm = f'v2v{s}' if v else f's{s}'
    res[nm] = dict(C_jk=float(val.mean()), ci_events=[float(lo), float(hi)],
                   ci_windows=[float(lo2), float(hi2)], n_events=int(len(ue)),
                   n_windows=int(len(val)))
    print(f'{nm:>7} {len(ue):7d} {len(val):5d} | {val.mean():6.3f} '
          f'[{lo:.3f},{hi:.3f}]{"":>4} {hi-lo:6.3f} | [{lo2:.3f},{hi2:.3f}]{"":>10}')

print()
print('can the scenarios be told apart?  pairwise CI overlap')
ks = list(res)
ov = 0; tot = 0
for a, b in itertools.combinations(ks, 2):
    A, B = res[a]['ci_events'], res[b]['ci_events']
    o = not (A[1] < B[0] or B[1] < A[0]); ov += o; tot += 1
print(f'  {ov}/{tot} scenario pairs have overlapping 95% CIs')
lo_all = min(r['ci_events'][0] for r in res.values())
hi_all = max(r['ci_events'][1] for r in res.values())
pts = [r['C_jk'] for r in res.values()]
print(f'  point estimates span [{min(pts):.3f}, {max(pts):.3f}]  '
      f'-> CI union spans [{lo_all:.3f}, {hi_all:.3f}]')
json.dump(res, open('results/stopping_events.json', 'w'), indent=1)
print('\nsaved -> results/stopping_events.json')
