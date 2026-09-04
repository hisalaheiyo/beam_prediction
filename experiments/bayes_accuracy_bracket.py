"""An unbiased estimator for the collision probability, and the units of analysis.

C4 says max_b count_b / W is upward-biased as an estimate of max_b p_b, and that the bias
runs in the direction that makes "no model exceeds the ceiling" easier to satisfy.  That is
correct.  Two fixes:

(1) Sum of squares has an EXACTLY unbiased estimator from W draws:
        S2_hat = sum_b n_b (n_b - 1) / (W (W - 1))     E[S2_hat] = sum_b p_b^2
    and the elementary bracket sum p^2 <= max p <= sqrt(sum p^2) then gives an interval for
    the ceiling whose INPUT is unbiased.  No extrapolation, no nugget, no circularity: this
    runs on the frozen windows themselves.

(2) For the point estimate, jackknife the plug-in.

Also recorded here, because the review is right that "120,178 frames" misstates the evidence:
    - the number of DISTINCT stopping events behind the V2I ceiling
    - whether V2V seq_index is a real independent unit or a fixed chunking
"""
import numpy as np, json

W, DMAX = 11, 0.5
V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]


def load(s, v2v=False):
    f = f'data/cache/v2v_s{s}.npz' if v2v else f'data/cache/s{s}.npz'
    d = np.load(f, allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return dict(beam=d['beam'].astype(int), seq=d['seq'].astype(int),
                x=d['x'].astype(float), y=d['y'].astype(float),
                good=np.asarray(g, bool), n=len(d['beam']), K=P.shape[1])


def windows(D):
    """Non-overlapping frozen windows + the contiguous stopping EVENT each belongs to."""
    seq, good, x, y, n = D['seq'], D['good'], D['x'], D['y'], D['n']
    inw = np.zeros(n, bool)
    for st in range(0, n - W):
        i = np.arange(st, st + W)
        if not good[i].all() or seq[i[0]] != seq[i[-1]]:
            continue
        if np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() > DMAX:
            continue
        inw[i] = True
    ev = np.zeros(n, int); e = 0; prev = False
    for t in range(n):
        if inw[t] and not prev:
            e += 1
        ev[t] = e if inw[t] else 0
        prev = inw[t]
    wins = []                       # non-overlapping, tagged by event
    st = 0
    while st <= n - W:
        i = np.arange(st, st + W)
        if (inw[i].all() and ev[i[0]] == ev[i[-1]]
                and np.hypot(x[i] - x[i[0]], y[i] - y[i[0]]).max() <= DMAX):
            wins.append((i, ev[i[0]])); st += W
        else:
            st += 1
    return inw, wins, e


def est(cnt, W_):
    """plug-in, jackknife-corrected plug-in, and the unbiased sum of squares."""
    pl = cnt.max() / W_
    jk = []
    for b in np.nonzero(cnt)[0]:
        c = cnt.copy(); c[b] -= 1
        jk.append(c.max() / (W_ - 1))
    w = cnt[cnt > 0] / W_
    jkm = float(np.sum(w * np.array(jk)))
    s2 = float((cnt * (cnt - 1)).sum() / (W_ * (W_ - 1)))
    return pl, W_ * pl - (W_ - 1) * jkm, s2


# ---- simulate the bias so the correction can be reported, not asserted -------------------
rng = np.random.default_rng(0)
print('bias check on synthetic p (10000 draws of W=11)')
print(f'{"true max p":>11} | {"plug-in":>9} {"bias":>7} | {"jackknife":>10} {"bias":>7} | '
      f'{"E[S2_hat]":>10} {"true S2":>8}')
for lab, p in [('0.50/2', np.array([.5, .5])), ('0.33/3', np.ones(3) / 3),
               ('0.25/4', np.ones(4) / 4), ('0.60+', np.array([.6, .2, .1, .1])),
               ('0.80+', np.array([.8, .1, .1]))]:
    A = B = S = 0.0
    for _ in range(10000):
        c = np.bincount(rng.choice(len(p), W, p=p), minlength=len(p))
        a, b, s2 = est(c, W)
        A += a; B += b; S += s2
    A /= 10000; B /= 10000; S /= 10000
    print(f'{p.max():11.3f} | {A:9.3f} {A-p.max():+7.3f} | {B:10.3f} {B-p.max():+7.3f} | '
          f'{S:10.3f} {(p**2).sum():8.3f}')

# ---- real data ---------------------------------------------------------------------------
out = {}
print()
print('real scenarios: ceiling estimators and the TRUE evidence base')
print(f'{"scn":>7} {"wins":>5} {"events":>7} {"drives":>7} | {"plug-in":>8} {"jackknife":>10} '
      f'| {"unbiased bracket":>19} {"ok":>3}')
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v)
    inw, wins, nev = windows(D)
    if not wins:
        continue
    pl, jk, s2 = [], [], []
    for i, _ in wins:
        c = np.bincount(D['beam'][i], minlength=D['K'])
        a, b, q = est(c, W)
        pl.append(a); jk.append(b); s2.append(q)
    PL, JK, S2 = float(np.mean(pl)), float(np.mean(jk)), float(np.mean(s2))
    lo, hi = S2, np.sqrt(max(S2, 0))
    ok = bool(lo - 1e-9 <= JK <= hi + 1e-9)
    drives = len(np.unique(D['seq'][inw]))
    nm = f'v2v{s}' if v else f's{s}'
    out[nm] = dict(n_windows=len(wins), n_events=int(nev), n_drives=int(drives),
                   plugin=PL, jackknife=JK, S2_unbiased=S2,
                   bracket=[lo, hi], jk_in_bracket=ok)
    print(f'{nm:>7} {len(wins):5d} {nev:7d} {drives:7d} | {PL:8.3f} {JK:10.3f} '
          f'| [{lo:.3f},{hi:.3f}]{"":>5} {"Y" if ok else "N":>3}')

# ---- is V2V seq_index a real unit? --------------------------------------------------------
print()
print('is V2V seq_index an independent unit, or a fixed chunking of one recording?')
for s in V2V:
    D = load(s, True)
    u, ln = np.unique(D['seq'], return_counts=True)
    contig = sum(1 for a, b in zip(u[:-1], u[1:])
                 if np.where(D['seq'] == a)[0].max() + 1 == np.where(D['seq'] == b)[0].min())
    print(f'  v2v{s}: {len(u):4d} ids   len min/med/max = {ln.min()}/{int(np.median(ln))}/{ln.max()}'
          f'   back-to-back {contig}/{len(u)-1}')

json.dump(out, open('results/bayes_accuracy_bracket.json', 'w'), indent=1)
print('\nsaved -> results/bayes_accuracy_bracket.json')
