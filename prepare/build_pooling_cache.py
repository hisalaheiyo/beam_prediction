"""Cache scenarios 5-9 with image paths, so TransFuser can be trained across sites.

The head-to-head so far trains the 25.9M model on one scenario at a time -- 0.6-2k frames --
while the DeepSense challenge entries it descends from train across the whole release.  That
is the one caveat that can still overturn the paper's strongest claim, so the model has to be
given the data budget it was designed for.  These five scenarios add 15,512 frames of imagery
to the 18,479 already cached.

Self-checks, and the cache is not written if they fail:
  the power vector must have 64 entries;  the recorded best beam must equal argmax(power);
  every referenced image must exist on disk;  positions must parse to plausible metres.
"""
import os
import numpy as np, pandas as pd, glob, os, sys

R = os.path.join(os.environ.get('DEEPSENSE_ROOT', 'data/deepsense'), '_gen')
def latlon_to_m(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111320.0 * np.cos(np.radians(lat0)),
            (lat - lat0) * 110540.0)


def readvec(base, rel):
    p = os.path.join(base, str(rel).lstrip('./'))
    with open(p) as f:
        return np.fromstring(f.read().replace(',', ' '), sep=' ')


def readloc(base, rel):
    p = os.path.join(base, str(rel).lstrip('./'))
    with open(p) as f:
        v = np.fromstring(f.read().replace(',', ' '), sep=' ')
    return (v[0], v[1]) if len(v) >= 2 else (np.nan, np.nan)


for s in [5, 6, 7, 8, 9]:
    cs = glob.glob(f'{R}/s{s}/**/scenario{s}.csv', recursive=True)
    if not cs:
        print(f's{s}: no csv'); continue
    csv = cs[0]; base = os.path.dirname(csv)
    df = pd.read_csv(csv)
    n = len(df)
    P = np.full((n, 64), np.nan, np.float32)
    lat = np.full(n, np.nan); lon = np.full(n, np.nan)
    bad_len = miss_img = 0
    for i in range(n):
        try:
            v = readvec(base, df.unit1_pwr_60ghz.iloc[i])
            if len(v) == 64:
                P[i] = v
            else:
                bad_len += 1
        except Exception:
            bad_len += 1
        try:
            lat[i], lon[i] = readloc(base, df.unit2_loc.iloc[i])
        except Exception:
            pass
        if not os.path.exists(os.path.join(base, str(df.unit1_rgb.iloc[i]).lstrip('./'))):
            miss_img += 1
    ok = ~np.isnan(P).any(1)
    lat0, lon0 = np.nanmedian(lat[ok]), np.nanmedian(lon[ok])
    x, y = latlon_to_m(lat, lon, lat0, lon0)
    beam = np.where(ok, np.nanargmax(np.nan_to_num(P, nan=-1e30), 1), -1).astype(np.int16)
    rec = df.unit1_beam_index.to_numpy() if 'unit1_beam_index' in df else None
    c_beam = float(np.mean(beam[ok] == (rec[ok] - rec[ok].min()))) if rec is not None else np.nan
    good = ok & np.isfinite(x) & np.isfinite(y)
    span = (np.nanmax(x[good]) - np.nanmin(x[good]), np.nanmax(y[good]) - np.nanmin(y[good]))
    passed = (bad_len == 0 and miss_img == 0 and good.sum() > 500
              and 1 < span[0] < 5000 and 1 < span[1] < 5000)
    print(f's{s}: n={n} good={good.sum()} bad_pwr={bad_len} missing_img={miss_img} '
          f'span={span[0]:.0f}x{span[1]:.0f}m  argmax==recorded={c_beam:.3f}  '
          f'{"WRITE" if passed else "REFUSED"}', flush=True)
    if passed:
        os.makedirs('data/cache', exist_ok=True)
        np.savez_compressed(f'data/cache/s{s}.npz', pwr=P, beam=beam,
                            seq=df.seq_index.to_numpy().astype(np.int32),
                            x=x.astype(np.float32), y=y.astype(np.float32),
                            rgb=df.unit1_rgb.to_numpy().astype('U'), base=np.array([base]))
