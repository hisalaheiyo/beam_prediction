"""Four predictors compared at a matched operating point.

The review's objection is exact: TransFuser was scored in top-1 on frozen windows while the
proposed pipeline was scored in dB loss at horizon k, so the "574x cheaper" claim had no
matched accuracy point behind it.  A cost ratio without one is not a contribution.

Here every method predicts THE SAME THING -- the best beam k frames ahead -- on THE SAME test
frames, and is scored on the same three metrics.  Selection of any hyperparameter happens on
the validation split only (C2), and the split is available in both geometries (C7):

    persist        hold the beam that is best now                   0 params
    linext+map     linear extrapolation into a non-parametric map   0 params (1.3 MB table)
    geo-MLP        (x,y,v) -> beam, trained to convergence          ~70k params
    TransFuser     5 RGB frames + GPS -> beam at t+k                ~26M params

Usage: python fix_headtohead_headtohead.py <scenario> [seq|spatial]
"""
import os
import numpy as np, pandas as pd, torch, torch.nn as nn, sys, glob, json, itertools
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

ROOT = os.environ.get('DEEPSENSE_ROOT', 'data/deepsense')
SEQ, IMG = 5, 224
HOR = [int(v) for v in os.environ.get("HOR", "8,16,32").split(",")]
SEED = int(os.environ.get('SEED', '0'))
TAG = 'k' + '-'.join(str(h) for h in HOR) + f'_s{SEED}'
EPOCHS, LR, BS = 30, 1e-4, 16
VERT = HORZ = 8
KMAP_G, HIST_G = [1, 5, 15, 25], [3, 5, 8]


def load(s):
    d = np.load(f'data/cache/s{s}.npz', allow_pickle=True)
    P = d['pwr'].astype(np.float64)
    return dict(P=P, Pn=P / np.maximum(np.nanmax(P, 1, keepdims=True), 1e-30),
                beam=d['beam'].astype(int), seq=d['seq'].astype(int),
                x=d['x'].astype(np.float32), y=d['y'].astype(np.float32),
                good=~np.isnan(P).any(1) & ~np.isnan(d['x']) & ~np.isnan(d['y']),
                n=len(d['beam']), K=P.shape[1])


def dbl(v):
    return float(-10 * np.log10(max(float(np.mean(v)), 1e-12)))


def splits(D, mode, rng):
    if mode == 'seq':
        u = np.unique(D['seq'][D['good']]); p = rng.permutation(len(u))
        a, b = int(.5 * len(u)), int(.7 * len(u))
        return [np.isin(D['seq'], u[p[:a]]), np.isin(D['seq'], u[p[a:b]]),
                np.isin(D['seq'], u[p[b:]])]
    g = D['good']; xy = np.stack([D['x'], D['y']], 1)
    c = xy[g] - xy[g].mean(0)
    ax = np.linalg.svd(c, full_matrices=False)[2][0]
    t = (xy - xy[g].mean(0)) @ ax
    nb = 12
    qs = np.quantile(t[g], np.linspace(0, 1, nb + 1))
    blk = np.clip(np.digitize(t, qs[1:-1]), 0, nb - 1)
    p = rng.permutation(nb); a, b = int(.5 * nb), int(.7 * nb)
    return [np.isin(blk, p[:a]), np.isin(blk, p[a:b]), np.isin(blk, p[b:])]


def eval_idx(D, mask, k, hist=SEQ):
    """the SHARED evaluation set: every method is scored on exactly these frames"""
    n, seq, good = D['n'], D['seq'], D['good']
    fut = np.minimum(np.arange(n) + k, n - 1)
    v = (np.arange(n) + k < n) & (seq[fut] == seq) & good & good[fut] & mask & mask[fut]
    ei = np.where(v)[0]
    ei = ei[ei >= hist]
    return ei[seq[ei - hist + 1] == seq[ei]]


def score(D, ei, k, pred):
    Pf = D['Pn'][ei + k]; m = np.arange(len(ei))
    return dict(top1=float((pred == D['beam'][ei + k]).mean()) * 100,
                dB=dbl(Pf[m, pred]), n=int(len(ei)))


def sitemap(D, tr, qx, qy, KMAP):
    mm = tr & D['good']
    mx, my, mP = D['x'][mm], D['y'][mm], D['Pn'][mm]
    out = np.empty((len(qx), D['K']))
    for a in range(0, len(qx), 256):
        sl = slice(a, min(a + 256, len(qx)))
        d2 = (qx[sl, None] - mx[None]) ** 2 + (qy[sl, None] - my[None]) ** 2
        kk = min(KMAP, d2.shape[1] - 1)
        nb = np.argpartition(d2, kk, axis=1)[:, :kk]
        dd = np.take_along_axis(d2, nb, 1)
        w = 1.0 / (np.sqrt(dd) + 0.5); w /= w.sum(1, keepdims=True)
        out[sl] = (mP[nb] * w[:, :, None]).sum(1)
    return out


class Frames(Dataset):
    def __init__(s, base, paths, idx, D, k):
        s.base, s.paths, s.idx, s.D, s.k = base, paths, idx, D, k
        s.mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
        s.std = torch.tensor([.229, .224, .225]).view(3, 1, 1)

    def __len__(s):
        return len(s.idx)

    def __getitem__(s, j):
        i = s.idx[j]
        ims = []
        for t in range(SEQ - 1, -1, -1):
            p = os.path.join(s.base, str(s.paths[i - t]).lstrip('./'))
            a = np.array(Image.open(p).convert('RGB').resize((IMG, IMG), Image.BILINEAR))
            v = torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255.
            ims.append((v - s.mean) / s.std)
        gps = torch.tensor([[s.D['x'][i - 1], s.D['y'][i - 1]],
                            [s.D['x'][i], s.D['y'][i]]], dtype=torch.float32)
        return torch.stack(ims), gps, int(s.D['beam'][i + s.k])   # <- label at t+k


class TF(nn.Module):
    def __init__(s, K=64):
        super().__init__()
        s.img = ImageEncoder(); s.gps = GPSEncoder(out_dims=[64, 128, 256, 512])
        s.fuse = nn.ModuleList([GPTFusion(d, 4, 1, VERT * HORZ + 1)
                                for d in [64, 128, 256, 512]])
        s.pool = nn.ModuleList([nn.AdaptiveAvgPool2d((VERT, HORZ)) for _ in range(4)])
        s.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(.3),
                               nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, K))

    def forward(s, ims, gps):
        gf = s.gps(gps); acc = 0
        for t in range(ims.shape[1]):
            fs = s.img(ims[:, t])
            for i in range(4):
                p = s.pool[i](fs[i])
                tok = torch.cat([p.flatten(2).transpose(1, 2), gf[i][:, None, :]], 1)
                o = s.fuse[i](tok)
                if i == 3:
                    acc = acc + o.mean(1)
        return s.head(acc / ims.shape[1])


def main():
    s = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else 'seq'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = load(s)
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, va, te = splits(D, mode, np.random.default_rng(20260901 + s + 1000 * SEED))
    base = f'{ROOT}/scenario{s}'
    paths = pd.read_csv(glob.glob(f'{base}/scenario{s}*.csv')[0]).unit1_rgb.to_numpy()
    out = {}
    for k in HOR:
        ei_va, ei_te = eval_idx(D, va, k), eval_idx(D, te, k)
        if len(ei_te) < 150 or len(ei_va) < 100:
            print(f'k={k}: too few frames ({len(ei_va)}/{len(ei_te)}), skip', flush=True)
            continue
        row = {}
        # ---- persist ----------------------------------------------------------------
        row['persist'] = score(D, ei_te, k, D['beam'][ei_te])
        # ---- linext + site map, KMAP/HIST chosen on VAL only -------------------------
        best, cfg = None, None
        for KM, HI in itertools.product(KMAP_G, HIST_G):
            e = eval_idx(D, va, k, HI)
            if len(e) < 100:
                continue
            sc = k / (HI - 1)
            q = sitemap(D, tr, D['x'][e] + (D['x'][e] - D['x'][e - HI + 1]) * sc,
                        D['y'][e] + (D['y'][e] - D['y'][e - HI + 1]) * sc, KM)
            v = score(D, e, k, q.argmax(1))['dB']
            if best is None or v < best:
                best, cfg = v, (KM, HI)
        KM, HI = cfg
        e = eval_idx(D, te, k, HI); sc = k / (HI - 1)
        q = sitemap(D, tr, D['x'][e] + (D['x'][e] - D['x'][e - HI + 1]) * sc,
                    D['y'][e] + (D['y'][e] - D['y'][e - HI + 1]) * sc, KM)
        row['linext_map'] = dict(**score(D, e, k, q.argmax(1)), KMAP=KM, HIST=HI, params=0)
        # ---- geometry MLP -----------------------------------------------------------
        def feat(idx):
            return np.stack([D['x'][idx], D['y'][idx],
                             D['x'][idx] - D['x'][idx - SEQ + 1],
                             D['y'][idx] - D['y'][idx - SEQ + 1]], 1).astype(np.float32)
        itr = eval_idx(D, tr, k)
        mu, sd = feat(itr).mean(0), feat(itr).std(0) + 1e-6
        net = nn.Sequential(nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, D['K'])).to(dev)
        opt = torch.optim.Adam(net.parameters(), 3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
        Xt = torch.tensor((feat(itr) - mu) / sd).to(dev)
        yt = torch.tensor(D['beam'][itr + k]).long().to(dev)
        for ep in range(200):
            p = torch.randperm(len(Xt), device=dev)
            for a in range(0, len(p), 4096):
                b = p[a:a + 4096]
                opt.zero_grad()
                nn.functional.cross_entropy(net(Xt[b]), yt[b]).backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            pr = net(torch.tensor((feat(ei_te) - mu) / sd).to(dev)).argmax(1).cpu().numpy()
        row['geo_mlp'] = dict(**score(D, ei_te, k, pr),
                              params=sum(p.numel() for p in net.parameters()))
        # ---- TransFuser -------------------------------------------------------------
        m = TF(D['K']).to(dev)
        opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        dl = DataLoader(Frames(base, paths, itr, D, k), BS, shuffle=True, num_workers=4)
        for ep in range(EPOCHS):
            m.train()
            for im, gp, yb in dl:
                opt.zero_grad()
                nn.functional.cross_entropy(m(im.to(dev), gp.to(dev)), yb.to(dev)).backward()
                opt.step()
            sch.step()
            print(f'  s{s} k={k} tf epoch {ep+1}/{EPOCHS}', flush=True)
        m.eval(); pr = []
        with torch.no_grad():
            for im, gp, _ in DataLoader(Frames(base, paths, ei_te, D, k), 32,
                                        num_workers=4):
                pr.append(m(im.to(dev), gp.to(dev)).argmax(1).cpu().numpy())
        row['transfuser'] = dict(**score(D, ei_te, k, np.concatenate(pr)),
                                 params=sum(p.numel() for p in m.parameters()))
        out[f'k{k}'] = row
        print(f'\ns{s} {mode} k={k} ({k*0.092:.2f}s)  n_te={len(ei_te)}')
        for nm, r in row.items():
            print(f'    {nm:>12}  top1 {r["top1"]:6.2f}%   {r["dB"]:6.3f} dB   '
                  f'params {r.get("params",0):,}', flush=True)
        json.dump(out, open(f'results/headtohead_s{s}_{mode}_{TAG}.json', 'w'), indent=1)
    print(f'saved -> results/headtohead_s{s}_{mode}_{TAG}.json')


main()
