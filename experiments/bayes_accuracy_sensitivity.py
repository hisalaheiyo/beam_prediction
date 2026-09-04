"""Is the paper's central number sensitive to the two constants that define it?

Everything now rests on the ceiling, and the ceiling is defined by two arbitrary choices:
W = 11 frames per frozen window, and DMAX = 0.5 m of allowed drift.  An earlier robustness
sweep exists but predates the jackknife correction, the drift-check fix and the event-clustered
intervals, so it no longer describes the estimator in the paper.  Re-run over a grid.

If the ceiling moves materially with W or DMAX, it is a property of the window definition
rather than of the deployment, and the paper cannot claim it measures anything physical.
"""
import numpy as np, json

V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]
WS, DS = [7, 9, 11, 15, 21], [0.25, 0.5, 1.0]


def load(s, v2v=False):
    d = np.load(f'data/cache/{"v2v_s" if v2v else "s"}{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return dict(beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
                y=d['y'].astype(float), good=np.asarray(g, bool), n=len(d['beam']), K=P.shape[1])


def jk(cnt, W):
    pl = cnt.max() / W
    m = 0.0
    for b in np.nonzero(cnt)[0]:
        c = cnt.copy(); c[b] -= 1
        m += (cnt[b] / W) * (c.max() / (W - 1))
    return W * pl - (W - 1) * m


def ceiling(D, W, DMAX):
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    inw = np.zeros(n, bool)
    for st in range(n - W):
        i = np.arange(st, st + W)
        if good[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX:
            inw[i] = True
    vals, st = [], 0
    while st <= n - W:
        i = np.arange(st, st + W)
        if (inw[i].all() and seq[i[0]] == seq[i[-1]]
                and np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX):
            vals.append(jk(np.bincount(D['beam'][i], minlength=D['K']), W)); st += W
        else:
            st += 1
    return (float(np.mean(vals)) if len(vals) >= 5 else np.nan), len(vals)


res, tbl = {}, []
print('jackknife-corrected ceiling over the window definition')
print(f'{"scen":>7} |' + ''.join(f'  W={w} d={d}' for d in DS for w in WS))
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v); nm = f'v2v{s}' if v else f's{s}'
    row, cells = {}, []
    for d in DS:
        for w in WS:
            c, nwin = ceiling(D, w, d)
            row[f'W{w}_d{d}'] = dict(C=None if np.isnan(c) else c, n=nwin)
            cells.append(f'{c:9.3f}' if np.isfinite(c) else '      n/a')
    res[nm] = row
    fin = [v2['C'] for v2 in row.values() if v2['C'] is not None]
    tbl.append((nm, min(fin), max(fin), max(fin) - min(fin)))
    print(f'{nm:>7} |' + ''.join(cells), flush=True)

print()
print(f'{"scen":>7} {"min":>7} {"max":>7} {"spread":>8}   (spread over 15 window definitions)')
for nm, lo, hi, sp in tbl:
    print(f'{nm:>7} {lo:7.3f} {hi:7.3f} {sp:8.3f}')
sp = [t[3] for t in tbl]
print(f'\nmean spread {np.mean(sp):.3f}, max {max(sp):.3f}')
print('the ceiling differences the paper reports across deployments are '
      f'{max(t[2] for t in tbl) - min(t[1] for t in tbl):.3f} wide, '
      'so a spread comparable to that would make the claim definitional rather than physical')
json.dump(res, open('results/bayes_accuracy_sensitivity.json', 'w'), indent=1)
print('saved -> results/bayes_accuracy_sensitivity.json')
