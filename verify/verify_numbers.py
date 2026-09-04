"""Verify every number in the manuscript against the authoritative corrected result files."""
import json, glob, numpy as np
R='results'; J=lambda f: json.load(open(f'{R}/{f}'))
bad=[]
def chk(claim, stated, computed, tol):
    ok=abs(stated-computed)<=tol
    if not ok: bad.append(claim)
    print(f'  {"OK " if ok else "BAD"}  {claim:<50} paper {stated:<9} computed {computed:.4f}')

print('--- surveyed budget: LOO floor (Fig 1c, Sec 3.3, intro, conclusion) ---')
d=J('loss_budget.json')
g=lambda k,key: float(np.mean([r[key] for r in d if r['split']=='seq' and r['k']==k]))
for k,tot,tr in [(8,0.088,0.003),(16,0.141,0.056),(32,0.563,0.478)]:
    chk(f'surveyed h={k*0.092:.2f} total', tot, g(k,'total'), 0.0006)
    chk(f'surveyed h={k*0.092:.2f} tracking', tr, g(k,'track'), 0.0006)
mp=[g(k,'maperr') for k in (8,16,32)]
chk('map term low 0.046', 0.046, min(mp), 0.0006)
chk('map term high 0.047', 0.047, max(mp), 0.0006)
chk('floor 0.039 dB', 0.039, g(8,'irred'), 0.0006)
chk('floor share 44%', 44, 100*g(8,'irred')/g(8,'total'), 0.6)
chk('remainder 0.049 dB', 0.049, g(8,'total')-g(8,'irred'), 0.0006)
chk('remainder map part 0.047', 0.047, g(8,'maperr'), 0.0006)
chk('tracking share at 2.94 s is 85%', 85, 100*g(32,'track')/g(32,'total'), 0.6)

print('--- out-of-sample floor (Table 2, Sec 3.2, method, conclusion) ---')
t=J('floor_out_of_sample.json'); ps=t['per_site']
loo=[ps[s]['loo'] for s in ps]; ins=[ps[s]['ins'] for s in ps]
for s,want in [('s31',0.028),('s32',0.030),('s33',0.031),('s34',0.066)]:
    chk(f'{s} floor', want, ps[s]['loo'], 0.0006)
chk('pooled LOO floor 0.039', 0.039, g(8,'irred'), 0.0006)
chk('in-sample floor 0.030', 0.030, float(np.mean(ins)), 0.0006)
chk('in-sample understates by 31%', 31, 100*(np.mean(loo)/np.mean(ins)-1), 1.0)
chk('LOO error rate min 26%', 26, 100*min(ps[s]['err_loo'] for s in ps), 0.6)
chk('LOO error rate max 50%', 50, 100*max(ps[s]['err_loo'] for s in ps), 0.6)
sv=[v for row in t['sensitivity'].values() for v in [float(np.mean(row))]]
chk('floor sensitivity min 0.031', 0.031, min(sv), 0.0006)
chk('floor sensitivity max 0.050', 0.050, max(sv), 0.0006)
chk('sensitivity combinations 10', 10, len(t['sensitivity']), 0)

print('--- bootstrap interval on the floor (Sec 3.2, 3.3, method) ---')
u=J('floor_bootstrap_ci.json'); md=u['mean_of_deployments']
chk('floor point matches the budget 0.039', 0.039, md['floor'], 0.0006)
chk('floor CI low 0.024', 0.024, md['ci'][0], 0.0015)
chk('floor CI high 0.068', 0.068, md['ci'][1], 0.0015)
chk('47 events in the bootstrap', 47, md['n_events'], 0)
chk('clustering widens the interval 3.4x', 3.4,
    (md['ci'][1]-md['ci'][0])/(md['naive_ci'][1]-md['naive_ci'][0]), 0.06)
chk('floor share low 27%', 27, 100*md['ci'][0]/g(8,'total'), 0.6)
chk('floor share high 77%', 77, 100*md['ci'][1]/g(8,'total'), 0.6)


print('--- residual is not untracked structure (Sec 3.2) ---')
vv=J('transfer_bound_revisits.json')
for s,(fx,hp) in [('s31',(0.028,0.051)),('s32',(0.030,0.047)),('s33',(0.031,0.068))]:
    chk(f'{s} fixed-beam LOO', fx, vv[s]['fixed_loo'], 0.0006)
    chk(f'{s} hold-previous-beam', hp, vv[s]['hold_prev'], 0.0006)
chk('s34 hold-previous ties the fixed beam', 1,
    int(abs(vv['s34']['hold_prev']-vv['s34']['fixed_loo'])<0.002), 0)


print('--- stops-to-motion transfer bound (Sec 3.2, intro, conclusion) ---')
rv=J('transfer_bound_revisits.json'); sw=J('transfer_bound_controls.json')
import numpy as _np
chk('revisit floor 0.083', 0.083, _np.mean([rv[f's{s}']['revisit_floor'] for s in (31,32,33,34)]), 0.0015)
chk('ratio 2.1x', 2.1, sw['sweep']['R0.5_H180']['ratio'], 0.06)
chk('heading-controlled ratio still 2.1', 2.1, sw['sweep']['R0.5_H20']['ratio'], 0.06)
chk('radius 0.25 m ratio 2.3', 2.3, sw['sweep']['R0.25_H20']['ratio'], 0.06)
chk('median speed min 4.0 m/s', 4.0, min(rv[f's{s}']['med_speed'] for s in (31,32,33,34)), 0.06)
chk('median speed max 6.1 m/s', 6.1, max(rv[f's{s}']['med_speed'] for s in (31,32,33,34)), 0.06)
chk('site-map at the upper end 0.002', 0.002,
    g(8,'total')-g(8,'track')-_np.mean([rv[f's{s}']['revisit_floor'] for s in (31,32,33,34)]), 0.0015)

print('--- distance curve, pooled and by regime (Fig 1b, Sec 3.4) ---')
r=J('sitemap_vs_distance.json')
for bi,want in [(0,0.062),(1,0.086),(2,0.120),(3,0.208),(4,0.454),(5,1.091)]:
    chk(f'distance band {bi} pooled', want, float(np.mean([x['loss'] for x in r if x['bin']==bi])), 0.0015)
sp=[max(v)-min(v) for bi in range(4)
    for v in [[np.mean([x['loss'] for x in r if x['bin']==bi and x['k']==k]) for k in (8,16,32)]]]
chk('horizon spread <= 0.02 dB', 0.02, max(sp), 0.0011)
sq0=float(np.mean([x['loss'] for x in r if x['bin']==0 and x['split']=='seq']))
sa0=float(np.mean([x['loss'] for x in r if x['bin']==0 and x['split']=='spatial']))
chk('band 0 surveyed 0.076', 0.076, sq0, 0.0015)
chk('band 0 unsurveyed 0.031', 0.031, sa0, 0.0015)
ok=all(np.mean([x['loss'] for x in r if x['bin']==b and x['split']=='seq'])>
       np.mean([x['loss'] for x in r if x['bin']==b and x['split']=='spatial']) for b in range(3))
chk('surveyed harder below 1 m (bands 0-2)', 1, int(ok), 0)
for sp,vals in [('seq',[0.076,0.104]),('spatial',[0.031,0.064,0.096])]:
    for bi,want in enumerate(vals):
        chk(f'design envelope {sp} band {bi}', want,
            float(np.mean([x['loss'] for x in r if x['bin']==bi and x['split']==sp])), 0.0015)
rev=np.mean([x['loss'] for x in r if x['bin']==3 and x['split']=='seq'])< \
    np.mean([x['loss'] for x in r if x['bin']==3 and x['split']=='spatial'])
chk('ordering reverses beyond 1 m', 1, int(rev), 0)

print('--- contamination and permutation spread (Sec 3.1, intro) ---')
rows=[l.split() for l in open(f'{R}/permutation_spread.log').read().split('\n') if l.split()[:1] in (['seq'],['spatial'])]
rows=[x for x in rows if len(x)==7 and x[6].endswith('x')]
spr=[float(x[6][:-1]) for x in rows if x[0]=='spatial']; sqr=[float(x[6][:-1]) for x in rows if x[0]=='seq']
chk('unsurveyed permutation spread min 15', 15, min(spr), 0.2)
chk('unsurveyed permutation spread max 26', 26, max(spr), 0.5)
chk('surveyed permutation spread max 3.3', 3.3, max(sqr), 0.1)


print('--- target-frame leak (Sec 3.1, intro, abstract) ---')
lk=J('target_frame_leak.json')
for k,want in [('8',28.3),('16',47.3),('32',67.1)]:
    chk(f'contamination at k={k}', want, 100*lk[k]['frac'], 0.06)
chk('cells with median target-to-train distance 0', 6, lk['n_zero_median_cells_total'], 0)
chk('cells total', 12, lk['n_cells_total'], 0)
chk('median target->train at longest horizon 0 m', 0.0, lk['32']['med_target'], 1e-9)

print('--- head to head (Table 1, Sec 3.6, intro) ---')
M=['persist','linext_map','geo_mlp','transfuser']
def col(pat):
    dd={}
    for f in glob.glob(f'{R}/{pat}'):
        for k,v in json.load(open(f)).items(): dd.setdefault(int(k[1:]),[]).append(v)
    return dd
sq2=col('headtohead_s*_seq_h8-16-32_s0.json'); sp2=col('headtohead_s*_spatial_h8-16-32_s*.json')
nb=sum(1 for dd in (sq2,sp2) for k in dd for x in dd[k] if x['transfuser']['dB']<=min(x[m]['dB'] for m in M))
chk('TransFuser best in 0 runs', 0, nb, 0)
chk('total runs 43', 43, sum(len(dd[k]) for dd in (sq2,sp2) for k in dd), 0)
chk('surveyed 0.74 table 0.092', 0.092, float(np.mean([x['linext_map']['dB'] for x in sq2[8]])), 0.0015)
chk('unsurveyed 0.74 table', 0.520, float(np.mean([x['linext_map']['dB'] for x in sp2[8]])), 0.0015)
chk('unsurveyed 0.74 hold-beam', 0.524, float(np.mean([x['persist']['dB'] for x in sp2[8]])), 0.0015)
chk('hold-beam within 0.004 of table', 0.004,
    float(np.mean([x['persist']['dB'] for x in sp2[8]])-np.mean([x['linext_map']['dB'] for x in sp2[8]])), 0.0015)
nr=[len(sp2[k]) for k in (8,16,32)]
for k,want in zip((8,16,32),(12,11,8)): chk(f'unsurveyed runs at k={k}', want, len(sp2[k]), 0)
hs=set(); ks=set()
for f in glob.glob(f'{R}/headtohead_s*_h8-16-32*.json'):
    for _,x in json.load(open(f)).items(): hs.add(x['linext_map']['HIST']); ks.add(x['linext_map']['KMAP'])
chk('table k grid min 1', 1, min(ks), 0); chk('table k grid max 25', 25, max(ks), 0)
chk('table history span min 3', 3, min(hs), 0); chk('table history span max 8', 8, max(hs), 0)


print('--- capacity result is not an artifact of the horizon (Sec 3.6) ---')
import glob as _g
_k0={}
for f in _g.glob(f'{R}/headtohead_s*_seq_h0-2-4.json'):
    _k0[f.split('_s')[1].split('_')[0]]=json.load(open(f))['k0']
_t=[v['transfuser']['top1'] for v in _k0.values()]
chk('k=0 deployments', 4, len(_k0), 0)
chk('k=0 TransFuser best top1 47.2%', 47.2, max(_t), 0.06)
_mt=[v['linext_map']['top1'] for v in _k0.values()]
chk('k=0 table top1 on that same cell 49.5%', 49.5, max(_mt), 0.06)
chk('k=0 table mean top1 42.9%', 42.9, float(np.mean(_mt)), 0.06)
chk('k=0 TransFuser mean top1 33.3%', 33.3, float(np.mean(_t)), 0.06)
chk('k=0 table beats TransFuser on power in 4/4', 4,
    sum(v['linext_map']['dB'] < v['transfuser']['dB'] for v in _k0.values()), 0)


print('--- surveyed 2.94 s ordering, bootstrapped over held-out sequences (Sec 3.6) ---')
xb=J('surveyed_sequence_bootstrap.json')
chk('MLP ahead by 0.26 dB', 0.26, xb['gap'], 0.006)
chk('interval low 0.15', 0.15, xb['ci'][0], 0.006)
chk('interval high 0.38', 0.38, xb['ci'][1], 0.006)
chk('MLP ahead in every resample', 1.0, xb['frac_mlp_better'], 1e-9)

print('--- pooled (Sec 3.6, conclusion) ---')
for k,want in [(8,1.150),(16,1.462),(32,1.562)]:
    p=J(f'pooled_transfuser_h{k}.json')
    chk(f'pooled TransFuser h={k*0.092:.2f}', want,
        float(np.mean([p[s]['pooled_transfuser']['dB'] for s in p])), 0.0015)
    chk(f'map wins {k}', 4, sum(1 for s in p if p[s]['site_map']['dB']<p[s]['pooled_transfuser']['dB']), 0)

print('--- instrument (Table 2, Sec 3.2, intro) ---')
ci,ub,c6,ref,un=J('stopping_events.json'),J('bayes_accuracy_bracket.json'),J('window_geometry.json'),J('fixed_beam_reference.json'),J('power_vs_amplitude.json')
v2i=[k for k in ci if not k.startswith('v2v')]
chk('47 stopping events', 47, sum(ci[k]['n_events'] for k in v2i), 0)
chk('bracket low 0.42', 0.42, min(ub[k]['bracket'][0] for k in v2i), 0.006)
chk('bracket high 0.79', 0.79, max(ub[k]['bracket'][1] for k in v2i), 0.006)
m2={k:x for k,x in c6['models'].items() if not k.startswith('v2v')}
chk('crossed 1 of 156 (num)', 1, sum(x['n_exceed'] for x in m2.values()), 0)
chk('crossed 1 of 156 (den)', 156, sum(x['n_win'] for x in m2.values()), 0)
a=[ref[k]['power'] for k in ref if not k.startswith('v2v')]
chk('median error cost min', 0.045, min(x['med'] for x in a), 0.0015)
chk('median error cost max', 0.055, max(x['med'] for x in a), 0.0015)
rv=[un[k]['rand_power'] for k in un if not k.startswith('v2v')]
chk('random beam 1.2 dB', 1.2, min(rv), 0.05); chk('random beam 2.7 dB', 2.7, max(rv), 0.05)

print('--- worth and probing (Sec 3.5) ---')
q=J('rate_cost_and_probing.json')
w=lambda k,key: float(np.mean([x[key] for x in q if x['split']=='seq' and x['k']==k]))
for c,st,key,tol in [('SE 1.5% @0dB',1.5,('serel0',8),0.06),('SE 0.45% @20dB',0.45,('serel20',8),0.02),
                     ('SE 9.6% @0dB h=2.94',9.6,('serel0',32),0.06),('abs SE 0.015',0.015,('se0',8),0.0006),
                     ('abs SE 0.216',0.216,('se20',32),0.0015),('probe2 0.042',0.042,('probe2',8),0.0015),
                     ('probe8 0.005',0.005,('probe8',8),0.0015)]:
    chk(c, st, w(key[1],key[0]), tol)
chk('outage 0.7%', 0.7, 100*w(8,'out1'), 0.06); chk('outage 20.8%', 20.8, 100*w(32,'out1'), 0.06)

print('--- negatives: allocation and survey planning (Sec 3.4, 3.5, conclusion) ---')
w3=J('probe_allocation.json')
for tag,(unif,best,miss,chat,nb,nm,nc,ncell) in {
        'surveyed':  (0.196,0.140,0.184,0.189,144,144,137,144),
        'unsurveyed':(0.471,0.372,0.454,0.458,108,104, 54,108)}.items():
    r=w3[tag]
    chk(f'{tag} uniform', unif, r['uniform'], 0.0015)
    chk(f'{tag} best two-level oracle', best, r['best'], 0.0015)
    chk(f'{tag} miss oracle', miss, r['miss'], 0.0015)
    chk(f'{tag} deployable c_hat', chat, r['chat'], 0.0015)
    chk(f'{tag} best wins in all cells', ncell, r['best_wins'], 0)
    chk(f'{tag} miss wins', nm, r['miss_wins'], 0)
    chk(f'{tag} c_hat wins', nc, r['chat_wins'], 0)
    chk(f'{tag} cells', ncell, r['n'], 0)
chk('oracle gain surveyed 0.056', 0.056, w3['surveyed']['uniform']-w3['surveyed']['best'], 0.0015)
chk('oracle gain unsurveyed 0.099', 0.099, w3['unsurveyed']['uniform']-w3['unsurveyed']['best'], 0.0015)
chk('miss-oracle gain surveyed 0.012', 0.012, w3['surveyed']['uniform']-w3['surveyed']['miss'], 0.0015)
chk('miss-oracle gain unsurveyed 0.017', 0.017, w3['unsurveyed']['uniform']-w3['unsurveyed']['miss'], 0.0015)
chk('c_hat recovers about an eighth', 0.125,
    (w3['surveyed']['uniform']-w3['surveyed']['chat'])/(w3['surveyed']['uniform']-w3['surveyed']['best']), 0.02)
s=J('survey_allocation.json')
dl=lambda pol:[np.median([x['loss'] for x in s if x['frac']==f and x['policy']==pol])-
               np.median([x['loss'] for x in s if x['frac']==f and x['policy']=='uniform']) for f in [0.25,0.12,0.06]]
p_,o_=dl('gradient'),dl('oracle-grad')
chk('pilot survey loses min 0.025', 0.025, min(p_), 0.0015)
chk('pilot survey loses max 0.073', 0.073, max(p_), 0.0015)
chk('oracle survey loses min 0.015', 0.015, min(o_), 0.0015)
chk('oracle survey loses max 0.167', 0.167, max(o_), 0.0015)

print('--- unit ambiguity factor (Sec 3.1) ---')
rr=[un[k][f'{w}_ampl']/un[k][f'{w}_power'] for k in un if not k.startswith('v2v') for w in ('dyn','rand','irr')]
chk('amplitude reading inflates: min 1.7', 1.7, min(rr), 0.02)
chk('amplitude reading inflates: max 2.0', 2.0, max(rr), 0.02)

print('\n' + ('ALL VERIFIED' if not bad else f'{len(bad)} MISMATCH:\n    '+'\n    '.join(bad)))
