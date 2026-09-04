"""Window geometry and how often a geometry-only predictor crosses the bound.s sitting below it.

The reviewer is right that an experiment in which nothing approaches the ceiling cannot separate
"the ceiling binds" from "the models are undertrained".  But the argument does not have to be
empirical at all, and restructuring it removes the objection:

  Inside a frozen window the geometry is constant to within DMAX.  A predictor that is a
  function of geometry alone therefore emits (near-)identical scores at all W frames, so it
  commits to ONE beam b and scores count_b / W <= max_b count_b / W = the ceiling.
  For geometry-only predictors the ceiling is an ANALYTIC bound, not an empirical finding.

  What is genuinely empirical is whether a predictor with EXTRA sensing (camera, radar) can
  exceed it -- it can only do so by resolving something the geometry does not, e.g. a blocking
  vehicle.  That is the question the TransFuser experiment actually answers.

Two things measured here:
  T1  how constant is the geometry inside a window, in the units a predictor sees?
  T2  does a trained geometry-only model in fact commit to one beam per window, and does its
      per-window accuracy respect the bound?  (if it ever exceeds, the analytic claim is wrong)
"""
import numpy as np, json, torch, torch.nn as nn

W, DMAX = 11, 0.5
V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]
torch.manual_seed(0)


def load(s, v2v=False):
    d = np.load(f'data/cache/{"v2v_s" if v2v else "s"}{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return dict(P=P, beam=d['beam'].astype(int), seq=d['seq'].astype(int),
                x=d['x'].astype(float), y=d['y'].astype(float),
                good=np.asarray(g, bool), n=len(d['beam']), K=P.shape[1])


def wins(D):
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
                and np.hypot(D['x'][i] - D['x'][i[0]],
                             D['y'][i] - D['y'][i[0]]).max() <= DMAX):
            out.append(i); st += W
        else:
            st += 1
    return out


print('T1: how constant is the geometry a predictor sees, inside a frozen window?')
print(f'{"scn":>7} | {"max drift [m]":>14} {"median":>8} | {"max heading swing [deg]":>24}')
geo = {}
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v); ws = wins(D)
    if not ws:
        continue
    dr, hd = [], []
    for i in ws:
        x, y = D['x'][i], D['y'][i]
        dr.append(np.hypot(x - x[0], y - y[0]).max())
        a = np.degrees(np.arctan2(y - y.mean(), x - x.mean()))
        hd.append(float(np.ptp(np.unwrap(np.radians(a)) * 180 / np.pi)))
    nm = f'v2v{s}' if v else f's{s}'
    geo[nm] = dict(drift_max=float(np.max(dr)), drift_med=float(np.median(dr)))
    print(f'{nm:>7} | {np.max(dr):14.3f} {np.median(dr):8.3f} | {np.median(hd):24.1f}')

print()
print('T2: a trained geometry-only model -- does it commit to one beam per window,')
print('    and does its per-window accuracy respect the analytic bound?')
print(f'{"scn":>7} | {"ceiling":>8} {"model acc":>10} | {"1 beam/window":>14} {"exceeds":>9}')
res = {}
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v); ws = wins(D)
    if len(ws) < 8:
        continue
    n, good, beam, K = D['n'], D['good'], D['beam'], D['K']
    xs = np.stack([D['x'], D['y']], 1)
    mu, sd = np.nanmean(xs[good], 0), np.nanstd(xs[good], 0) + 1e-9
    F = (xs - mu) / sd
    tr = good.copy()
    for i in ws:
        tr[i] = False                              # train off the frozen windows
    Xtr = torch.tensor(F[tr], dtype=torch.float32)
    ytr = torch.tensor(beam[tr], dtype=torch.long)
    net = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, K))
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150)
    for ep in range(150):                          # trained to convergence, not 20 epochs
        p = torch.randperm(len(Xtr))
        for a in range(0, len(p), 4096):
            b = p[a:a + 4096]
            opt.zero_grad(); loss = nn.functional.cross_entropy(net(Xtr[b]), ytr[b])
            loss.backward(); opt.step()
        sch.step()
    net.eval()
    accs, ceils, one = [], [], []
    with torch.no_grad():
        for i in ws:
            pr = net(torch.tensor(F[i], dtype=torch.float32)).argmax(1).numpy()
            accs.append((pr == beam[i]).mean())
            ceils.append(np.bincount(beam[i], minlength=K).max() / W)
            one.append(len(np.unique(pr)) == 1)
    accs, ceils = np.array(accs), np.array(ceils)
    exc = int((accs > ceils + 1e-9).sum())
    nm = f'v2v{s}' if v else f's{s}'
    res[nm] = dict(ceiling=float(ceils.mean()), acc=float(accs.mean()),
                   frac_single_beam=float(np.mean(one)), n_exceed=exc, n_win=len(ws))
    print(f'{nm:>7} | {ceils.mean():8.3f} {accs.mean():10.3f} | {np.mean(one):13.1%} '
          f'{exc:>4}/{len(ws):<4}')

tot_e = sum(r['n_exceed'] for r in res.values())
tot_w = sum(r['n_win'] for r in res.values())
print(f'\ngeometry-only model exceeds the per-window bound in {tot_e}/{tot_w} windows')
print('(the analytic claim predicts 0 whenever the model commits to a single beam)')
json.dump(dict(geometry=geo, models=res), open('results/window_geometry.json', 'w'), indent=1)
print('saved -> results/window_geometry.json')
