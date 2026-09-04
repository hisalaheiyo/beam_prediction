"""Cache DeepSense scen31-34: power vectors, beams, GPS geometry, speed, seq ids."""
import os
import numpy as np, pandas as pd, os, glob

ROOT = os.environ.get('DEEPSENSE_ROOT', 'data/deepsense')
OUT  = 'data/cache'
os.makedirs(OUT, exist_ok=True)

def load_txt(p):
    try:  return np.loadtxt(p)
    except Exception: return None

for s in [31, 32, 33, 34]:
    base = f'{ROOT}/scenario{s}'
    csv  = glob.glob(f'{base}/scenario{s}*.csv')[0]
    df   = pd.read_csv(csv)

    # base-station (unit1) reference position
    bs = load_txt(os.path.join(base, str(df.unit1_loc.iloc[0]).lstrip('./')))
    bs_lat, bs_lon = float(bs[0]), float(bs[1])

    n = len(df)
    P   = np.full((n, 64), np.nan)
    UE  = np.full((n, 2),  np.nan)   # lat, lon
    ok  = np.zeros(n, bool)
    for i in range(n):
        pv = load_txt(os.path.join(base, str(df.unit1_pwr_60ghz.iloc[i]).lstrip('./')))
        if pv is None or pv.shape != (64,):  continue
        gp = load_txt(os.path.join(base, str(df.unit2_loc.iloc[i]).lstrip('./')))
        if gp is None or gp.size < 2:        continue
        P[i], UE[i], ok[i] = pv, gp[:2], True

    # local ENU metres relative to BS
    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.deg2rad(bs_lat))
    x = (UE[:, 1] - bs_lon) * m_lon          # east
    y = (UE[:, 0] - bs_lat) * m_lat          # north
    rng = np.hypot(x, y)
    ang = np.degrees(np.arctan2(y, x))       # geometric azimuth from BS

    out = dict(
        pwr=P.astype(np.float32),
        beam=df.unit1_beam.to_numpy().astype(np.int16) - 1,          # -> 0..63
        seq=df.seq_index.to_numpy().astype(np.int32),
        spd=pd.to_numeric(df.unit2_spd_over_grnd_kmph, errors='coerce').to_numpy().astype(np.float32),
        x=x.astype(np.float32), y=y.astype(np.float32),
        rng=rng.astype(np.float32), ang=ang.astype(np.float32),
        ok=ok, ts=df.time_stamp.astype(str).to_numpy(),
    )
    np.savez_compressed(f'{OUT}/s{s}.npz', **out)
    agree = (P[ok].argmax(1) == out['beam'][ok]).mean()
    print(f's{s}: n={n} valid={ok.sum()} argmax==label:{agree:.4f} '
          f'range {rng[ok].min():.0f}-{rng[ok].max():.0f}m  ang {ang[ok].min():.0f}..{ang[ok].max():.0f}deg')
