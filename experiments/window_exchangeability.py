"""Are the W frames inside a frozen window independent draws from p(b | geometry)?

The whole ceiling estimator assumes they are: max_b n_b / W, its jackknife correction, and the
unbiased estimator of sum p^2 all treat a window as W independent draws.  The earlier evidence
for that was a lag-1 autocorrelation of -0.06..-0.12 reported as "temporally white".  The
review points out that a mean-corrected i.i.d. series of length W has E[rho_1] ~ -1/(W-1) =
-0.10 for W = 11, so that measurement was consistent with independence but had no power to
detect dependence either -- it would have returned the same answer for a true rho of +0.1.

Tested properly here, against the null the data itself defines:

  statistic   the number of RUNS of identical consecutive beams inside a window
              (a direct, distribution-free measure of temporal clumping that does not
              depend on the beam alphabet or on p being uniform)
  null        the same window's beams randomly permuted, 2000 times, which preserves the
              window's beam composition exactly and destroys only the time order
  reading     observed runs FEWER than the null  =>  beams clump in time  =>  the frames are
              NOT independent and the effective sample size is below W
"""
import numpy as np, json

V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]
W, DMAX, NPERM = 11, 0.5, 2000


def load(s, v2v=False):
    d = np.load(f'data/cache/{"v2v_s" if v2v else "s"}{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return dict(beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
                y=d['y'].astype(float), good=np.asarray(g, bool), n=len(d['beam']), K=P.shape[1])


def windows(D):
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    inw = np.zeros(n, bool)
    for st in range(n - W):
        i = np.arange(st, st + W)
        if good[i].all() and seq[i[0]] == seq[i[-1]] and \
           np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX:
            inw[i] = True
    out, st = [], 0
    while st <= n - W:
        i = np.arange(st, st + W)
        if (inw[i].all() and seq[i[0]] == seq[i[-1]]
                and np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX):
            out.append(i); st += W
        else:
            st += 1
    return out


def runs(b):
    return 1 + int((b[1:] != b[:-1]).sum())


rng = np.random.default_rng(0)
print('permutation test for temporal independence inside frozen windows')
print(f'{"scen":>7} {"wins":>5} | {"obs runs":>9} {"null runs":>10} {"z":>7} {"p":>8} | '
      f'{"effective W":>12}')
out = {}
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v); ws = windows(D)
    ws = [i for i in ws if len(np.unique(D['beam'][i])) > 1]     # constant windows carry no info
    if len(ws) < 8:
        continue
    obs = np.array([runs(D['beam'][i]) for i in ws], float)
    null = np.empty((NPERM, len(ws)))
    for t in range(NPERM):
        for j, i in enumerate(ws):
            b = D['beam'][i].copy(); rng.shuffle(b)
            null[t, j] = runs(b)
    nm_ = null.mean(1)
    o = obs.mean()
    z = (o - nm_.mean()) / (nm_.std() + 1e-12)
    p = float((np.abs(nm_ - nm_.mean()) >= abs(o - nm_.mean())).mean())
    # effective sample size: a first-order Markov chain with the observed run rate has
    # n_eff = W * (1 - r) / (1 + r) where r is the lag-1 repeat probability in excess of chance
    rep_obs = 1 - (o - 1) / (W - 1)
    rep_nul = 1 - (nm_.mean() - 1) / (W - 1)
    rho = (rep_obs - rep_nul) / max(1 - rep_nul, 1e-9)
    neff = W * (1 - rho) / (1 + rho) if rho > -1 else W
    nm = f'v2v{s}' if v else f's{s}'
    out[nm] = dict(n_win=len(ws), obs_runs=o, null_runs=float(nm_.mean()), z=float(z),
                   p=p, rho_excess=float(rho), W_eff=float(neff))
    print(f'{nm:>7} {len(ws):5d} | {o:9.3f} {nm_.mean():10.3f} {z:7.2f} {p:8.4f} | '
          f'{neff:12.2f}', flush=True)

print()
dep = [k for k, v2 in out.items() if v2['p'] < 0.05 and v2['obs_runs'] < v2['null_runs']]
print(f'windows show temporal clumping in {len(dep)}/{len(out)} scenarios: {dep}')
we = [v2['W_eff'] for v2 in out.values()]
print(f'effective sample size per window: {min(we):.2f} - {max(we):.2f} against a nominal W=11')
print('if W_eff << 11 the ceiling estimator is optimistic and the correction has to be')
print('recomputed with the effective count rather than W.')
json.dump(out, open('results/window_exchangeability.json', 'w'), indent=1)
print('saved -> results/window_exchangeability.json')
