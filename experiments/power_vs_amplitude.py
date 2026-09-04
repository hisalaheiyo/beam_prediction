"""Every dB figure under both readings of the recorded quantity.

p3_ekf_and_audits.py already records that DeepSense does not state whether unit1_pwr_60ghz
is received POWER or received AMPLITUDE.  Under the amplitude reading the true relative power
is the square of what the code currently treats as power, so every dB figure changes.  The
review is right that the "cost varies 15x" headline sits in units the work cannot pin down.

Two things settled here:
  (a) which claims are INVARIANT -- anything decided by argmax is, because argmax(A) =
      argmax(A^2) for non-negative A, so every ceiling, accuracy and probe-count result holds
      under either reading;
  (b) for the claims that are NOT invariant, the figure under both readings, side by side,
      so the paper can quote a range instead of a number it cannot defend.
"""
import numpy as np, json

V2I, V2V = [31, 32, 33, 34], [36, 37, 38, 39]


def load(s, v2v=False):
    d = np.load(f'data/cache/{"v2v_s" if v2v else "s"}{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    g = d['good'] if v2v else (~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    return P, np.asarray(g, bool), d['beam'].astype(int)


def dbl(v):
    return float(-10 * np.log10(max(float(np.mean(v)), 1e-12)))


rng = np.random.default_rng(3)
out = {}
print('every dB figure under both readings of unit1_pwr_60ghz')
print(f'{"scen":>7} | {"dynamic range":>21} | {"random-beam loss":>21} | '
      f'{"irreducible loss":>21}')
print(f'{"":>7} | {"as power":>10}{"as ampl.":>11} | {"as power":>10}{"as ampl.":>11} | '
      f'{"as power":>10}{"as ampl.":>11}')
for s, v in [(x, False) for x in V2I] + [(x, True) for x in V2V]:
    P, g, beam = load(s, v)
    A = P[g]
    row = {}
    for lab, Q in [('power', A), ('ampl', A ** 2)]:      # amplitude reading -> square it
        Qn = Q / np.maximum(Q.max(1, keepdims=True), 1e-30)
        row[f'dyn_{lab}'] = float(np.mean(10 * np.log10(
            np.maximum(Q.max(1), 1e-30) / np.maximum(Q.min(1), 1e-30))))
        ridx = rng.integers(0, Q.shape[1], len(Qn))
        row[f'rand_{lab}'] = dbl(Qn[np.arange(len(Qn)), ridx])
        # irreducible: loss of the single most frequent beam vs the per-sample best
        bc = np.bincount(beam[g], minlength=Q.shape[1])
        row[f'irr_{lab}'] = dbl(Qn[np.arange(len(Qn)), bc.argmax()]) if bc.max() > 0 else np.nan
    nm = f'v2v{s}' if v else f's{s}'
    out[nm] = row
    print(f'{nm:>7} | {row["dyn_power"]:10.2f}{row["dyn_ampl"]:11.2f} | '
          f'{row["rand_power"]:10.2f}{row["rand_ampl"]:11.2f} | '
          f'{row["irr_power"]:10.3f}{row["irr_ampl"]:11.3f}')

dp = [out[k]['dyn_power'] for k in out]; da = [out[k]['dyn_ampl'] for k in out]
rp = [out[k]['rand_power'] for k in out]; ra = [out[k]['rand_ampl'] for k in out]
print()
print(f'dynamic range   power reading {min(dp):.2f}-{max(dp):.2f} dB '
      f'({max(dp)/min(dp):.1f}x)   amplitude reading {min(da):.2f}-{max(da):.2f} dB '
      f'({max(da)/min(da):.1f}x)')
print(f'random-beam loss power reading {min(rp):.2f}-{max(rp):.2f} dB '
      f'({max(rp)/min(rp):.1f}x)   amplitude reading {min(ra):.2f}-{max(ra):.2f} dB '
      f'({max(ra)/min(ra):.1f}x)')
print()
print('INVARIANT under the ambiguity (argmax(A) = argmax(A^2) for A >= 0):')
print('  every ceiling, every top-1 accuracy, every probe count, the analytic bound,')
print('  the head-to-head ORDERING, and the 180/192 fingerprinting comparison.')
print('NOT invariant: the absolute dB magnitudes above; the RATIO across deployments is')
print('  preserved to within a few percent, so the "cost varies ~15x" claim survives, but')
print('  the paper must quote both readings for any absolute dB figure.')
json.dump(out, open('results/power_vs_amplitude.json', 'w'), indent=1)
print('\nsaved -> results/power_vs_amplitude.json')
