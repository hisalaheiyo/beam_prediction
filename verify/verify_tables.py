"""Parse Tables 1-3 out of the .tex and recompute every cell from the result files.

verify_numbers checks the numbers the prose quotes.  The tables carry dozens more that nothing
else checks -- they were typed by hand once and never re-derived after the leak fix, the
LOO-floor change or the mask corrections.
"""
import json, glob, re, numpy as np

R = 'results'; P = 'paper'
bad = []


def chk(claim, stated, computed, tol):
    ok = abs(stated - computed) <= tol
    if not ok: bad.append(f'{claim}: table {stated} vs computed {computed:.4f}')
    print(f'  {"OK " if ok else "BAD"}  {claim:<44} table {stated:<8} computed {computed:.4f}')


def cells(path, ncol):
    out = []
    for ln in open(f'{P}/{path}').read().split('\n'):
        ln = ln.strip()
        if ln.startswith('\\') or not ln.endswith('\\\\'): continue
        c = [x.strip() for x in ln[:-2].split('&')]
        if len(c) == ncol: out.append(c)
    return out


num = lambda s: float(re.sub(r'\\textbf\{|\}|\\,|\s', '', s).split('(')[0])
se_of = lambda s: (int(re.search(r'\((\d+)\)', s).group(1)) if '(' in s else None)

# ---------------------------------------------------------------- Table 1
M = ['persist', 'linext_map', 'geo_mlp', 'transfuser']
def col(pat):
    dd = {}
    for f in glob.glob(f'{R}/{pat}'):
        for k, v in json.load(open(f)).items(): dd.setdefault(int(k[1:]), []).append(v)
    return dd
sq = col('headtohead_s*_seq_h8-16-32_s0.json'); sp = col('headtohead_s*_spatial_h8-16-32_s*.json')
rows = [r for r in cells('tab1_headtohead.tex', 7)
        if re.fullmatch(r'\d\.\d+', r[1])]        # drop the two header rows
print(f'--- Table 1: {len(rows)} data rows x 4 predictors ---')
last = None
for r in rows:
    regime = 'surveyed' if r[0] == 'surveyed' else ('unsurveyed' if r[0] == 'unsurveyed' else None)
    if regime is None:                       # continuation row, inherit from the last labelled one
        regime = last
    last = regime
    k = int(round(float(r[1]) / 0.092))
    src = sq[k] if regime == 'surveyed' else sp[k]
    for j, m in enumerate(M):
        v = [x[m]['dB'] for x in src]
        chk(f'T1 {regime} h={r[1]} {m}', num(r[2 + j]), float(np.mean(v)), 0.0015)
        s = se_of(r[2 + j])
        if s is not None:
            comp = float(np.std(v, ddof=1) / np.sqrt(len(v)))
            chk(f'T1 {regime} h={r[1]} {m} s.e.', s, round(comp * 1000), 1)
    if regime == 'unsurveyed':
        p = json.load(open(f'{R}/pooled_transfuser_h{k}.json'))
        chk(f'T1 pooled h={r[1]}', num(r[6]),
            float(np.mean([p[s]['pooled_transfuser']['dB'] for s in p])), 0.0015)

# ---------------------------------------------------------------- Table 2
ci, ub, ref, t = (json.load(open(f'{R}/{f}')) for f in
                  ['stopping_events.json', 'bayes_accuracy_bracket.json', 'fixed_beam_reference.json',
                   'floor_out_of_sample.json'])
rows = [r for r in cells('tab2_deployments.tex', 7) if re.match(r'V2I', r[0])]
print(f'\n--- Table 2: {len(rows)} deployments x 6 quantities ---')
for r in rows:
    s = re.search(r'V2I.*?(\d\d)', r[0]).group(1); key = f's{s}'
    chk(f'T2 {key} stop events', int(r[1]), ci[key]['n_events'], 0)
    lo, hi = [float(x) for x in re.findall(r'\d\.\d+', r[3])]
    chk(f'T2 {key} bracket low', lo, ub[key]['bracket'][0], 0.006)
    chk(f'T2 {key} bracket high', hi, ub[key]['bracket'][1], 0.006)
    chk(f'T2 {key} residual rate', float(r[4].rstrip('\\%')),
        100 * t['per_site'][key]['err_loo'], 0.6)
    chk(f'T2 {key} median cost', float(r[5]), ref[key]['power']['med'], 0.0015)
    chk(f'T2 {key} floor', float(r[6]), t['per_site'][key]['loo'], 0.0006)

# ---------------------------------------------------------------- Table 3
w3 = json.load(open(f'{R}/probe_allocation.json'))
ROW = {'a random subset':'uniform', 'lowest local ceiling':'chat',
       'frames the top-1 misses':'miss', 'largest actual gain':'best'}
rows = [r for r in cells('tab3_allocation.tex', 5) if r[0] in ROW]
print(f'\n--- Table 3: {len(rows)} allocators x 2 regimes ---')
for r in rows:
    key = ROW[r[0]]
    for j, tag in ((1, 'surveyed'), (3, 'unsurveyed')):
        chk(f'T3 {tag} {r[0]}', num(r[j]), w3[tag][key], 0.0015)
    if key != 'uniform':
        for j, tag in ((2, 'surveyed'), (4, 'unsurveyed')):
            chk(f'T3 {tag} {r[0]} wins', int(r[j].split('/')[0]), w3[tag][key + '_wins'], 0)
            chk(f'T3 {tag} {r[0]} cells', int(r[j].split('/')[1]), w3[tag]['n'], 0)

print('\n' + ('ALL TABLE CELLS VERIFIED' if not bad
              else f'{len(bad)} MISMATCH:\n    ' + '\n    '.join(bad)))
