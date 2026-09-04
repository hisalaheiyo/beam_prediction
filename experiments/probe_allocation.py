"""Final probe-allocation numbers, with a closed-form check on the uniform arm and BOTH splits.

Self-check: a random subset of size n_hi upgrades each frame with probability n_hi/n, so its
expected power is  mean(P_lo) + (n_hi/n) * mean(d)  exactly.  The Monte-Carlo arm must match
that to numerical precision, otherwise the comparator is wrong and every conclusion below is.
"""
import numpy as np, json
_src = open('experiments/_probe_common.py').read()
exec(compile(_src[:_src.index('rows = []')], 'defs', 'exec'))
HIST, KMAP = 3, 15
V2I = [31, 32, 33, 34]
PAIRS = [(1, 2), (1, 4), (2, 4), (2, 8)]

def seqsplit(D, rng):
    """hold out whole driving sequences: the surveyed regime, matching Sec 3.5's other numbers"""
    us = np.unique(D['seq']); p = rng.permutation(us)
    a, b = int(.5 * len(us)), int(.7 * len(us))
    return (np.isin(D['seq'], p[:a]), np.isin(D['seq'], p[a:b]), np.isin(D['seq'], p[b:]))

worst = 0.0; out = {}
for tag, sp in [('surveyed', seqsplit), ('unsurveyed', lambda D, r: split(D, r))]:
    rows = []
    for s in V2I:
        D = load(s)
        tr, va, te = sp(D, np.random.default_rng(20260901 + s))
        for k in [8, 16, 32]:
            eva, ete = evalidx(D, va, k), evalidx(D, te, k)
            if len(ete) < 200 or len(eva) < 150: continue
            Sv, Cv = lookup(D, tr, eva, k); St, Ct = lookup(D, tr, ete, k)
            Pf = D['Pn'][ete + k]; order = np.argsort(-St, 1); n = len(ete)
            miss = St.argmax(1) != D['beam'][ete + k]
            for lo, hi in PAIRS:
                Plo = np.array([Pf[i, order[i, :lo]].max() for i in range(n)])
                d = np.array([Pf[i, order[i, :hi]].max() for i in range(n)]) - Plo
                for q in [0.1, 0.25, 0.5]:
                    tau = float(np.quantile(Cv, q)); up = Ct < tau; nhi = int(up.sum())
                    if nhi < 5 or nhi > n - 5: continue
                    rng = np.random.default_rng(1)
                    mc = float(np.mean([np.mean(Plo) + d[rng.choice(n, nhi, False)].sum() / n
                                        for _ in range(400)]))
                    cf = float(np.mean(Plo) + (nhi / n) * np.mean(d))   # closed form
                    worst = max(worst, abs(mc - cf))
                    f = lambda p: float(-10 * np.log10(max(p, 1e-12)))
                    rows.append(dict(scen=s, k=k, lo=lo, hi=hi, q=q, nhi=nhi, n=n,
                                     budget=(nhi*hi+(n-nhi)*lo)/n, uniform=f(cf),
                                     chat=f(np.mean(Plo) + d[up].sum()/n),
                                     miss=f(np.mean(Plo) + d[np.argsort(-miss.astype(float)
                                          + rng.random(n)*1e-9)[:nhi]].sum()/n),
                                     best=f(np.mean(Plo) + np.sort(d)[::-1][:nhi].sum()/n)))
    M = lambda key: float(np.mean([r[key] for r in rows]))
    W = lambda key: sum(r[key] < r['uniform'] - 1e-9 for r in rows)
    print(f'=== {tag}: {len(rows)} cells ===')
    print(f'{"allocator":>28} {"dB":>8} {"vs uniform":>11} {"wins":>11}')
    print(f'{"uniform":>28} {M("uniform"):>8.4f}')
    for key, nm in [('chat','local-ceiling threshold'), ('miss','oracle: frames that miss'),
                    ('best','oracle: best two-level')]:
        print(f'{nm:>28} {M(key):>8.4f} {M("uniform")-M(key):>+11.4f} {W(key):>8}/{len(rows)}')
    out[tag] = dict(n=len(rows), uniform=M('uniform'), chat=M('chat'), miss=M('miss'),
                    best=M('best'), chat_wins=W('chat'), miss_wins=W('miss'), best_wins=W('best'))
    print()

print(f'SELF-CHECK  max |monte-carlo - closed form| over all cells = {worst:.2e}  '
      f'-> {"uniform arm is exact" if worst < 1e-4 else "COMPARATOR IS WRONG"}')
json.dump(out, open('results/probe_allocation.json','w'), indent=1)
print('saved -> results/probe_allocation.json')
