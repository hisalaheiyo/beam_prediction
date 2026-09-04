"""Cache the DeepSense-V2V scenarios (36-39) with SELF-CHECKS at every step.

Differences from V2I that must be handled correctly:
  * both ends move, so the geometry is unit2's position RELATIVE to unit1, expressed in
    unit1's heading frame -- absolute positions would be meaningless here;
  * four phased arrays, so the codebook is the 256-beam concatenation pwr1|pwr2|pwr3|pwr4,
    and the label decomposes into "which array" (4) x "which beam in it" (64);
  * unit2 GPS is missing for a small fraction of rows in s37/s38/s39.

Self-checks performed and reported (the script refuses to write a cache that fails them):
  C1  argmax of the concatenated 256-vector == unit1_overall-beam        (must be ~100%)
  C2  per-array argmax == unit1_pwrA_best-beam                           (must be ~100%)
  C3  overall-beam == 64 * best_array + best_beam_of_that_array          (must be ~100%)
  C4  heading estimation: report the fraction of frames where unit1 moves too little for a
      reliable heading (those are marked invalid rather than silently given a wrong frame)
  C5  relative range distribution is physically plausible (a few m to a few hundred m)
"""
import os
import numpy as np, pandas as pd, os, json

ROOT = os.path.join(os.environ.get('DEEPSENSE_ROOT', 'data/deepsense'), '_v2v')
OUT = 'data/cache'
SC = [36, 37, 38, 39]
HEAD_LAG = 5           # frames used to estimate unit1's heading
MIN_MOVE = 0.30        # m over HEAD_LAG frames; below this the heading is not identifiable


def local_xy(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111320.0 * np.cos(np.deg2rad(lat0)),
            (lat - lat0) * 111320.0)


def build(s):
    base = f'{ROOT}/scenario{s}'
    df = pd.read_csv(f'{base}/scenario{s}.csv')
    n = len(df)
    lat1 = pd.to_numeric(df['unit1_gps1_lat'], errors='coerce').to_numpy()
    lon1 = pd.to_numeric(df['unit1_gps1_lon'], errors='coerce').to_numpy()
    lat2 = pd.to_numeric(df['unit2_gps1_lat'], errors='coerce').to_numpy()
    lon2 = pd.to_numeric(df['unit2_gps1_lon'], errors='coerce').to_numpy()
    seq = df['seq_index'].to_numpy().astype(int)
    ov = pd.to_numeric(df['unit1_overall-beam'], errors='coerce').to_numpy()
    bb = np.stack([pd.to_numeric(df[f'unit1_pwr{a}_best-beam'], errors='coerce').to_numpy()
                   for a in [1, 2, 3, 4]], 1)

    lat0 = np.nanmean(lat1); lon0 = np.nanmean(lon1)
    x1, y1 = local_xy(lat1, lon1, lat0, lon0)
    x2, y2 = local_xy(lat2, lon2, lat0, lon0)

    # --- power vectors: 4 x 64 concatenated
    P = np.full((n, 256), np.nan, np.float32)
    okp = np.zeros(n, bool)
    for i in range(n):
        v = []
        for a in [1, 2, 3, 4]:
            p = os.path.join(base, f'scenario{s}', str(df[f'unit1_pwr{a}'].iloc[i]).lstrip('./'))
            # np.loadtxt is far too slow for ~450k tiny files; parse directly.
            try:
                with open(p) as fh:
                    w = np.fromstring(fh.read(), sep='\n')
            except Exception:
                w = None
            if w is None or w.shape != (64,):
                v = None; break
            v.append(w)
        if v is not None:
            P[i] = np.concatenate(v); okp[i] = True

    # --- C1..C3
    m = okp & np.isfinite(ov)
    c1 = float((P[m].argmax(1) == ov[m]).mean())
    per = np.stack([P[m][:, a * 64:(a + 1) * 64].argmax(1) for a in range(4)], 1)
    c2 = float((per == bb[m]).all(1).mean())
    mx = np.stack([P[m][:, a * 64:(a + 1) * 64].max(1) for a in range(4)], 1)
    ba = mx.argmax(1)
    c3 = float((64 * ba + per[np.arange(len(ba)), ba] == ov[m]).mean())

    # --- unit1 heading, and the relative position in unit1's frame
    hx = np.full(n, np.nan); hy = np.full(n, np.nan)
    for i in range(n):
        j = i - HEAD_LAG
        if j < 0 or seq[j] != seq[i]:
            continue
        dx, dy = x1[i] - x1[j], y1[i] - y1[j]
        d = np.hypot(dx, dy)
        if np.isfinite(d) and d >= MIN_MOVE:
            hx[i], hy[i] = dx / d, dy / d
    dx, dy = x2 - x1, y2 - y1
    fwd = dx * hx + dy * hy               # along unit1's heading
    lft = -dx * hy + dy * hx              # left of unit1's heading
    rng = np.hypot(dx, dy)

    good = (okp & np.isfinite(ov) & np.isfinite(fwd) & np.isfinite(lft)
            & np.isfinite(rng) & (rng > 0.5) & (rng < 1000))
    c4_bad_head = float(np.mean(~np.isfinite(hx)))
    c5 = (float(np.nanpercentile(rng[good], 1)), float(np.nanmedian(rng[good])),
          float(np.nanpercentile(rng[good], 99))) if good.sum() else (np.nan,) * 3

    report = dict(scenario=s, n=n, valid=int(good.sum()),
                  C1_concat_argmax_eq_overall=c1, C2_perarray=c2, C3_encoding=c3,
                  C4_frac_heading_unidentifiable=c4_bad_head,
                  C5_range_p1_med_p99=c5,
                  frac_missing_pwr=float(np.mean(~okp)),
                  frac_missing_u2gps=float(np.mean(~np.isfinite(lat2))),
                  n_seq=int(len(np.unique(seq))))
    passed = (c1 > 0.99 and c2 > 0.99 and c3 > 0.99 and good.sum() > 1000)
    if passed:
        np.savez_compressed(
            f'{OUT}/v2v_s{s}.npz',
            pwr=P.astype(np.float32), beam=ov.astype(np.int16), seq=seq.astype(np.int32),
            x=fwd.astype(np.float32), y=lft.astype(np.float32),
            rng=rng.astype(np.float32), good=good,
            best_arr=np.where(np.isfinite(ov), np.floor_divide(ov, 64), -1).astype(np.int8),
            beam_in_arr=np.where(np.isfinite(ov), np.mod(ov, 64), -1).astype(np.int16))
    return report, passed


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    allrep = []
    for s in SC:
        d4 = [f'{ROOT}/scenario{s}/scenario{s}/unit1/pwr{a}' for a in [1, 2, 3, 4]]
        if not all(os.path.isdir(d) for d in d4):
            print(f's{s}: power data not extracted yet, skipping'); continue
        # Guard against the mistake made on the first run: the cache was started while the
        # archives were still being extracted, so most power files did not exist yet and the
        # self-checks came out as NaN.  Require the extracted file count to match the CSV.
        nfiles = min(len(os.listdir(d)) for d in d4)
        df_n = len(pd.read_csv(f'{ROOT}/scenario{s}/scenario{s}.csv'))
        if nfiles < 0.98 * df_n:
            print(f's{s}: extraction incomplete ({nfiles} files vs {df_n} rows), skipping')
            continue
        rep, ok = build(s)
        allrep.append(rep)
        print(f"s{s}: n={rep['n']} valid={rep['valid']} seq={rep['n_seq']} | "
              f"C1={rep['C1_concat_argmax_eq_overall']:.4f} C2={rep['C2_perarray']:.4f} "
              f"C3={rep['C3_encoding']:.4f} | no-heading={rep['C4_frac_heading_unidentifiable']:.3f} "
              f"| range p1/med/p99={tuple(round(v,1) for v in rep['C5_range_p1_med_p99'])} "
              f"| {'CACHED' if ok else 'REJECTED'}", flush=True)
    json.dump(allrep, open('results/v2v_cache_report.json', 'w'), indent=1)
    print('report -> results/v2v_cache_report.json')
