"""Retrain the multimodal model on nine pooled deployments and score it per deployment.

The head-to-head trains it on one scenario at a time, so the obvious objection to the capacity
result is that the model is data-limited.  This gives it four to six times more data.

The head-to-head trains the 25.9M model on ONE scenario at a time -- 0.6-2k frames -- while
the DeepSense challenge entries it descends from train across the whole release.  A reviewer
is entitled to say the model was starved.  So: train it on the POOLED training splits of all
nine V2I scenarios (5,6,7,8,9,31,32,33,34), roughly 24k frames, with a learned per-scenario
embedding so it can still specialise per site, and evaluate on exactly the s31-34 test frames
the earlier head-to-head used.

The site map is deliberately NOT given the same benefit: it is built per scenario from that
scenario's own training split, because a position-indexed table cannot transfer across sites.
So the comparison is 24k frames of imagery against 1-3k positions of one site.

If the pooled model now wins, the paper's headline is wrong and must be rewritten.
Usage: python experiments/pooled_transfuser.py <horizon in frames>
"""
import os
import numpy as np, pandas as pd, torch, torch.nn as nn, sys, glob, json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# The TransFuser encoder/fusion modules are NOT redistributed here.  Point TRANSFUSER_SRC at a
# directory providing model.py with ImageEncoder, GPSEncoder and GPTFusion (the architecture of
# Prakash et al., CVPR 2021).  Every other script in this repository runs without it.
_TF = os.environ.get('TRANSFUSER_SRC')
if not _TF:
    raise SystemExit('set TRANSFUSER_SRC to a directory containing model.py '
                     '(ImageEncoder, GPSEncoder, GPTFusion); see README')
sys.path.insert(0, _TF)
from model import ImageEncoder, GPSEncoder, GPTFusion

ALL = [5, 6, 7, 8, 9, 31, 32, 33, 34]
TEST_ON = [31, 32, 33, 34]
SEQ, IMG, HIST = 5, 224, 3
EPOCHS, LR, BS, VERT, HORZ = 30, 1e-4, 24, 8, 8
K = int(sys.argv[1])
torch.manual_seed(0); np.random.seed(0)


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    D = dict(Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
             beam=d['beam'].astype(int), seq=d['seq'].astype(int), x=d['x'].astype(float),
             y=d['y'].astype(float), n=len(d['beam']), K=P.shape[1],
             good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']))
    if 'rgb' in d:                                    # scen5-9 cache carries its own paths
        D['rgb'] = d['rgb']; D['base'] = str(d['base'][0])
    else:                                             # s31-34: paths live in the scenario csv
        b = os.path.join(os.environ.get('DEEPSENSE_ROOT', 'data/deepsense'), f'scenario{s}')
        D['rgb'] = pd.read_csv(glob.glob(f'{b}/scenario{s}*.csv')[0]).unit1_rgb.to_numpy()
        D['base'] = b
    return D


def split(D, rng, nb=12):
    g = D['good']; xy = np.stack([D['x'], D['y']], 1)
    ax = np.linalg.svd(xy[g] - xy[g].mean(0), full_matrices=False)[2][0]
    t = (xy - xy[g].mean(0)) @ ax
    blk = np.clip(np.digitize(t, np.quantile(t[g], np.linspace(0, 1, nb + 1))[1:-1]), 0, nb - 1)
    p = rng.permutation(nb); a, b = int(.5 * nb), int(.7 * nb)
    return np.isin(blk, p[:a]), np.isin(blk, p[a:b]), np.isin(blk, p[b:])


def evalidx(D, m, k):
    n, seq, good = D['n'], D['seq'], D['good']
    fut = np.minimum(np.arange(n) + k, n - 1)
    v = (np.arange(n) + k < n) & (seq[fut] == seq) & good & good[fut] & m & m[fut]
    ei = np.where(v)[0]; ei = ei[ei >= max(SEQ, HIST)]
    ei = ei[seq[ei - HIST + 1] == seq[ei]]
    ei = ei[seq[ei - SEQ + 1] == seq[ei]]
    return ei[good[ei - HIST + 1]]


def dbl(v):
    return float(-10 * np.log10(max(float(np.mean(v)), 1e-12)))


class Frames(Dataset):
    def __init__(s, items):
        s.items = items                    # (D, sid, index)
        s.mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
        s.std = torch.tensor([.229, .224, .225]).view(3, 1, 1)

    def __len__(s):
        return len(s.items)

    def __getitem__(s, j):
        D, sid, i = s.items[j]
        ims = []
        for t in range(SEQ - 1, -1, -1):
            p = os.path.join(D['base'], str(D['rgb'][i - t]).lstrip('./'))
            a = np.array(Image.open(p).convert('RGB').resize((IMG, IMG), Image.BILINEAR))
            v = torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255.
            ims.append((v - s.mean) / s.std)
        gp = torch.tensor([[D['zx'][i - 1], D['zy'][i - 1]],
                           [D['zx'][i], D['zy'][i]]], dtype=torch.float32)
        return torch.stack(ims), gp, sid, int(D['beam'][i + K])


class Pooled(nn.Module):
    """TransFuser plus a per-scenario embedding, so one model can serve nine sites."""
    def __init__(s, nb=64, nsc=9):
        super().__init__()
        s.img = ImageEncoder(); s.gps = GPSEncoder(out_dims=[64, 128, 256, 512])
        s.emb = nn.Embedding(nsc, 512)
        s.fuse = nn.ModuleList([GPTFusion(d, 4, 1, VERT * HORZ + 1)
                                for d in [64, 128, 256, 512]])
        s.pool = nn.ModuleList([nn.AdaptiveAvgPool2d((VERT, HORZ)) for _ in range(4)])
        s.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(.3),
                               nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, nb))

    def forward(s, ims, gps, sid):
        gf = s.gps(gps); acc = 0
        for t in range(ims.shape[1]):
            fs = s.img(ims[:, t])
            for i in range(4):
                p = s.pool[i](fs[i])
                tok = torch.cat([p.flatten(2).transpose(1, 2), gf[i][:, None, :]], 1)
                o = s.fuse[i](tok)
                if i == 3:
                    acc = acc + o.mean(1)
        return s.head(acc / ims.shape[1] + s.emb(sid))


def sitemap(D, tr, ei, KMAP=15):
    m = tr & D['good']
    mx, my, mP = D['x'][m], D['y'][m], D['Pn'][m]
    sc = K / (HIST - 1)
    qx = D['x'][ei] + (D['x'][ei] - D['x'][ei - HIST + 1]) * sc
    qy = D['y'][ei] + (D['y'][ei] - D['y'][ei - HIST + 1]) * sc
    out = np.empty((len(ei), D['K']))
    for a in range(0, len(ei), 256):
        sl = slice(a, min(a + 256, len(ei)))
        d2 = (qx[sl, None] - mx[None]) ** 2 + (qy[sl, None] - my[None]) ** 2
        kk = min(KMAP, d2.shape[1] - 1)
        nb = np.argpartition(d2, kk, axis=1)[:, :kk]
        dd = np.take_along_axis(d2, nb, 1)
        w = 1.0 / (np.sqrt(dd) + 0.5); w /= w.sum(1, keepdims=True)
        out[sl] = (mP[nb] * w[:, :, None]).sum(1)
    return out


dev = 'cuda' if torch.cuda.is_available() else 'cpu'
Ds, tr_items, te_sets = {}, [], {}
for j, s in enumerate(ALL):
    D = load(s)
    g = D['good']
    D['zx'] = (D['x'] - D['x'][g].mean()) / (D['x'][g].std() + 1e-6)
    D['zy'] = (D['y'] - D['y'][g].mean()) / (D['y'][g].std() + 1e-6)
    tr, va, te = split(D, np.random.default_rng(20260901 + s))
    D['tr'] = tr
    Ds[s] = D
    itr = evalidx(D, tr, K)
    tr_items += [(D, j, int(i)) for i in itr]
    if s in TEST_ON:
        te_sets[s] = (j, evalidx(D, te, K))
    print(f'  s{s}: train {len(itr)}  test {len(te_sets.get(s,(0,[]))[1]) if s in te_sets else 0}',
          flush=True)
print(f'pooled training frames: {len(tr_items)}', flush=True)

m = Pooled(64, len(ALL)).to(dev)
opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
dl = DataLoader(Frames(tr_items), BS, shuffle=True, num_workers=8, drop_last=True)
for ep in range(EPOCHS):
    m.train(); tot = 0.0; nb_ = 0
    for im, gp, sid, yb in dl:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(im.to(dev), gp.to(dev), sid.to(dev)), yb.to(dev))
        loss.backward(); opt.step(); tot += float(loss); nb_ += 1
    sch.step()
    print(f'  epoch {ep+1}/{EPOCHS} loss {tot/max(nb_,1):.4f}', flush=True)

out = {}
m.eval()
for s, (sid, ei) in te_sets.items():
    D = Ds[s]
    pr = []
    with torch.no_grad():
        for im, gp, sd_, _ in DataLoader(Frames([(D, sid, int(i)) for i in ei]), 48,
                                         num_workers=8):
            pr.append(m(im.to(dev), gp.to(dev), sd_.to(dev)).argmax(1).cpu().numpy())
    pr = np.concatenate(pr)
    Pf = D['Pn'][ei + K]; idx = np.arange(len(ei))
    smap = sitemap(D, D['tr'], ei).argmax(1)
    out[f's{s}'] = dict(
        pooled_transfuser=dict(top1=float((pr == D['beam'][ei + K]).mean()) * 100,
                               dB=dbl(Pf[idx, pr])),
        site_map=dict(top1=float((smap == D['beam'][ei + K]).mean()) * 100,
                      dB=dbl(Pf[idx, smap])),
        persist=dict(dB=dbl(Pf[idx, D['beam'][ei]])), n=int(len(ei)))
    r = out[f's{s}']
    print(f's{s} k={K}: pooled-TransFuser {r["pooled_transfuser"]["dB"]:.3f} dB  '
          f'site map {r["site_map"]["dB"]:.3f} dB  persist {r["persist"]["dB"]:.3f} dB  '
          f'n={r["n"]}', flush=True)
json.dump(out, open(f'results/pooled_transfuser_h{K}.json', 'w'), indent=1)
print(f'saved -> results/pooled_transfuser_h{K}.json')
