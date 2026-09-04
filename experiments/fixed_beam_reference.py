"""Which fixed beam is 'the best a geometry-only predictor can do'?  My two claims disagree.

The irreducible term (0.030 dB) uses the POWER-optimal fixed beam: argmax over b of the mean
normalised power across the window.  The error-cost claim (median 0.050-0.060 dB) uses the
MODAL beam: the one most often the per-frame argmax.  These are different predictors -- one
maximises delivered power, the other maximises top-1 accuracy -- and a paper that argues top-1
and power disagree cannot quietly switch between their optima.

Measured here on the same frozen windows:
  how often do the two differ; what each costs in dB; what top-1 each achieves; and the error
  distribution under BOTH, so the paper can quote one reference consistently.
"""
import numpy as np, json

V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]
W, DMAX = 11, 0.5


def load(s, v2v=False):
    d = np.load(f'data/cache/{"v2v_s" if v2v else "s"}{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
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


def dbl(v):
    return float(-10 * np.log10(max(float(np.mean(v)), 1e-12)))


print('the two candidate references for "best geometry-only predictor"')
print(f'{"scen":>7} {"differ":>7} | {"MODAL: dB":>10} {"top1":>6} {"med err":>8} {"p90":>7} '
      f'| {"POWER: dB":>10} {"top1":>6} {"med err":>8} {"p90":>7}')
out = {}
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    D = load(s, v); ws = windows(D)
    if len(ws) < 8:
        continue
    idx = np.concatenate(ws)
    Pn = D['Pn'][idx]
    mod = np.concatenate([np.full(W, np.bincount(D['beam'][i], minlength=D['K']).argmax())
                          for i in ws])
    pwr = np.concatenate([np.full(W, int(np.nanmean(D['Pn'][i], 0).argmax())) for i in ws])
    diff = float(np.mean(mod != pwr))
    r = {}
    for nmp, pred in [('modal', mod), ('power', pwr)]:
        got = Pn[np.arange(len(idx)), pred]
        err = pred != D['beam'][idx]
        loss = -10 * np.log10(np.maximum(got[err], 1e-12)) if err.any() else np.array([0.0])
        r[nmp] = dict(dB=dbl(got), top1=float((~err).mean()),
                      med=float(np.median(loss)), p90=float(np.percentile(loss, 90)),
                      err_rate=float(err.mean()))
    nm = f'v2v{s}' if v else f's{s}'
    out[nm] = dict(frac_differ=diff, **r)
    print(f'{nm:>7} {diff:7.1%} | {r["modal"]["dB"]:10.3f} {r["modal"]["top1"]:6.1%} '
          f'{r["modal"]["med"]:8.3f} {r["modal"]["p90"]:7.3f} '
          f'| {r["power"]["dB"]:10.3f} {r["power"]["top1"]:6.1%} '
          f'{r["power"]["med"]:8.3f} {r["power"]["p90"]:7.3f}')

a = [out[k] for k in out if not k.startswith('v2v')]
print()
print('V2I summary')
print(f'  the two references pick a different beam in '
      f'{min(x["frac_differ"] for x in a):.1%}-{max(x["frac_differ"] for x in a):.1%} of frames')
print(f'  delivered-power loss: modal {min(x["modal"]["dB"] for x in a):.3f}-'
      f'{max(x["modal"]["dB"] for x in a):.3f} dB   '
      f'power-optimal {min(x["power"]["dB"] for x in a):.3f}-'
      f'{max(x["power"]["dB"] for x in a):.3f} dB')
print(f'  top-1 accuracy:       modal {min(x["modal"]["top1"] for x in a):.1%}-'
      f'{max(x["modal"]["top1"] for x in a):.1%}   '
      f'power-optimal {min(x["power"]["top1"] for x in a):.1%}-'
      f'{max(x["power"]["top1"] for x in a):.1%}')
print(f'  median cost of an error: modal {min(x["modal"]["med"] for x in a):.3f}-'
      f'{max(x["modal"]["med"] for x in a):.3f}   '
      f'power-optimal {min(x["power"]["med"] for x in a):.3f}-'
      f'{max(x["power"]["med"] for x in a):.3f}')
print()
print('CONSEQUENCE for the paper: the irreducible term uses the power-optimal beam, so the')
print('error-cost claim must use it too.  If the two references differ materially, that gap is')
print('itself the cleanest statement of "top-1 and delivered power disagree".')
json.dump(out, open('results/fixed_beam_reference.json', 'w'), indent=1)
print('saved -> results/fixed_beam_reference.json')
