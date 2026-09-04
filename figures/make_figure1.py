"""Figure 1. Drawn at final size so nothing is rescaled.

(a) the instrument: within one stopping window the optimal beam moves between frames, and the
    best fixed beam is wrong on a fraction of them; the bracket delimits the Bayes accuracy.
(b) the site-map term against distance to the nearest surveyed frame, with the two survey
    regimes plotted separately and the three horizons overlaid, which is what the data support:
    invariance to horizon at fixed distance, not to the regime label.
(c) the loss budget on a surveyed route, where the block partition is stable.

Style follows the venue: 17.8 cm double-column, 9 pt body text so figure text is 10 pt, no
titles, a light-to-dark ramp that survives greyscale printing, no overlapping elements.
"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R='results'; CM=1/2.54
C_IRR, C_MAP, C_TRK = '#8C3B22', '#6E9BC0', '#D6DDE3'
C_A, C_B = '#8C3B22', '#2E5E8C'
W, DMAX = 11, 0.5

plt.rcParams.update({
    # ICASSP allows Times-Roman or Computer Modern; no Times clone is installed here, so use
    # Computer Modern, which also matches the document's own maths.  fonttype 42 embeds
    # TrueType instead of Type 3 -- IEEE PDF eXpress rejects Type 3.
    'pdf.fonttype':42, 'ps.fonttype':42, 'mathtext.fontset':'cm',
    'font.family':'serif','font.serif':['cmr10','DejaVu Serif'],
    'axes.unicode_minus':False,
    'font.size':10,'axes.labelsize':10,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
    'legend.fontsize':9.0,'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7,
    'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.frameon':False,
    'axes.spines.top':False,'axes.spines.right':False})
fig, ax = plt.subplots(1, 3, figsize=(17.8*CM, 6.0*CM))
plt.subplots_adjust(left=0.080, right=0.982, bottom=0.225, top=0.905, wspace=0.44)

# ------------------------------------------------------------------ (a) the instrument
d=np.load('data/cache/s32.npz', allow_pickle=True)
P=d['pwr'].astype(np.float64); Pn=P/np.maximum(np.nanmax(P,1,keepdims=True),1e-30)
beam=d['beam'].astype(int); seq=d['seq'].astype(int)
x=d['x'].astype(float); y=d['y'].astype(float)
good=~np.isnan(P).any(1)&~np.isnan(x)&~np.isnan(y)
cand=None
for st in range(len(beam)-W):
    i=np.arange(st,st+W)
    if good[i].all() and seq[i[0]]==seq[i[-1]] and np.hypot(x[i]-x[i[0]],y[i]-y[i[0]]).max()<=DMAX:
        if len(np.unique(beam[i]))>=3: cand=i; break
b=beam[cand]; cnt=np.bincount(b, minlength=64)
S=float((cnt*(cnt-1)).sum()/(W*(W-1)))
bfix=int(np.nanmean(Pn[cand],0).argmax())
a=ax[0]
occ=np.nonzero(cnt)[0]
a.bar(occ, cnt[occ]/W, 0.85, color=C_MAP, zorder=3, label='window histogram')
a.bar([bfix],[cnt[bfix]/W], 0.85, color=C_IRR, zorder=4, label='fixed beam')
a.axhline(S, color='#555', lw=1.0, ls=(0,(4,3)), zorder=5)
a.axhline(np.sqrt(S), color='#555', lw=1.0, ls=(0,(1,2)), zorder=5)
a.text(occ.max()+0.60, S, r'$\hat S$', fontsize=9.5, va='center', ha='left')
a.text(occ.max()+0.60, np.sqrt(S), r'$\sqrt{\hat S}$', fontsize=9.5, va='center', ha='left')
a.set_xlim(occ.min()-0.7, occ.max()+1.75); a.set_ylim(0, 1.02)
a.set_xlabel('beam index'); a.set_ylabel('frequency in window')
a.set_xticks(list(occ))
a.legend(loc='upper left', handlelength=1.0, handleheight=.8, borderpad=.15, labelspacing=.22,
         bbox_to_anchor=(-0.03,1.05), ncol=1)
a.grid(axis='y', color='#E4E8EB', lw=.6, zorder=0); a.set_axisbelow(True)

# ------------------------------------------------------------------ (b) distance, by regime
r=json.load(open(f'{R}/sitemap_vs_distance.json'))
bx=ax[1]
for sp,col,mk,lab in [('seq',C_A,'o','surveyed'),('spatial',C_B,'s','unsurveyed')]:
    pts={}
    for xx in r:
        if xx['split']==sp: pts.setdefault(xx['bin'],[]).append((xx['dist'],xx['loss']))
    bs=sorted(k for k in pts if len(pts[k])>=6)
    bx.plot([np.median([p[0] for p in pts[k]]) for k in bs],
            [np.mean([p[1] for p in pts[k]]) for k in bs],
            mk+'-', color=col, ms=4.2, lw=1.5, mew=0, label=lab, zorder=3)
for k,ls in [(8,(0,(1,2))),(32,(0,(4,3)))]:
    pts={}
    for xx in r:
        if xx['k']==k: pts.setdefault(xx['bin'],[]).append((xx['dist'],xx['loss']))
    bs=sorted(kk for kk in pts if len(pts[kk])>=6)
    bx.plot([np.median([p[0] for p in pts[kk]]) for kk in bs],
            [np.mean([p[1] for p in pts[kk]]) for kk in bs],
            ls=ls, color='#9AA7B1', lw=1.0, zorder=2,
            label=('pooled, 0.74 s' if k==8 else 'pooled, 2.94 s'))
bx.set_xscale('log'); bx.set_yscale('log')
bx.set_xticks([0.1,1,10]); bx.set_xticklabels(['0.1','1','10'])
bx.xaxis.set_minor_formatter(plt.NullFormatter())
bx.set_xlabel('distance to nearest surveyed frame [m]')
bx.set_ylabel('site-map term [dB]')
bx.set_xlim(0.08, 16); bx.set_ylim(0.022, 3.4)
bx.set_yticks([0.03,0.1,0.3,1]); bx.set_yticklabels(['0.03','0.1','0.3','1'])
bx.legend(loc='upper left', handlelength=1.5, borderpad=.15, labelspacing=.25,
          bbox_to_anchor=(-0.03,1.05))
bx.grid(color='#E4E8EB', lw=.6, which='major', zorder=0); bx.set_axisbelow(True)

# ------------------------------------------------------------------ (c) budget, surveyed
dd=json.load(open(f'{R}/loss_budget.json'))
g=lambda k,key: float(np.mean([x[key] for x in dd if x['split']=='seq' and x['k']==k]))
HOR=[8,16,32]; xs=np.arange(3)
irr=np.array([g(k,'irred') for k in HOR]); mp=np.array([max(g(k,'maperr'),0) for k in HOR])
tk=np.array([max(g(k,'track'),0) for k in HOR])
c=ax[2]
c.bar(xs, irr, .62, color=C_IRR, label='irreducible', zorder=3)
c.bar(xs, mp, .62, bottom=irr, color=C_MAP, label='site map', zorder=3)
c.bar(xs, tk, .62, bottom=irr+mp, color=C_TRK, edgecolor='#9AA7B1', lw=.5, label='tracking', zorder=3)
for xx,t in zip(xs, irr+mp+tk):
    c.text(xx, t+0.012, f'{t:.3f}', ha='center', va='bottom', fontsize=9.0)
c.set_xticks(xs); c.set_xticklabels(['0.74','1.47','2.94'])
c.set_xlabel('horizon [s]'); c.set_ylabel('power loss [dB]')
c.set_ylim(0, 0.75); c.set_xlim(-0.55, 2.55)
c.legend(loc='upper left', handlelength=1.0, handleheight=.8, borderpad=.15, labelspacing=.22,
         bbox_to_anchor=(-0.03,1.05))
c.grid(axis='y', color='#E4E8EB', lw=.6, zorder=0); c.set_axisbelow(True)

for lbl,axx in zip('abc', ax):
    axx.text(-0.015, 1.13, f'({lbl})', transform=axx.transAxes, fontsize=10,
             fontweight='bold', va='top', ha='left')
plt.savefig('figures/figure1.pdf', dpi=600); plt.savefig('figures/figure1_preview.png', dpi=220)
print(f'(a) window from s32, beams {sorted(set(b))}, S={S:.3f}, bracket [{S:.2f},{np.sqrt(S):.2f}]')
print('(c) surveyed totals:', ' '.join(f'{v:.3f}' for v in irr+mp+tk))
