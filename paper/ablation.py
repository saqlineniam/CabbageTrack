"""
Ablation and baselines of the frozen model on all four flights, scored on the human answers (reported after the
held-out run; nothing was selected with it).

  removal ablation   the frozen model with ONE component removed or replaced
  baselines          the camera map replaced (constant velocity per flight leg, drone GNSS fitted to the images, drone
                     GNSS from its own metadata), odometry only (no image verification, no registration), ordinal
                     reversal (detection order only), the Field A method (v1)

Blind plants: exact McNemar test against the frozen model (pooled over the four flights and per flight).

  python paper/ablation.py         -> outputs/ablation/ablation.csv, ablation_pairs.csv, ablation.md
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (ROOT, os.path.join(ROOT, 'groundtruth'), os.path.join(ROOT, 'paper', 'development'), os.path.join(ROOT, 'paper', 'figures')):
    sys.path.insert(0, p)
import revisit_exp as RX  # noqa: E402
from cabbagetrack.gnss import read_srt, umeyama  # noqa: E402
from mosaic_panel import legs_from_map  # noqa: E402
from score import score, wilson  # noqa: E402
from steps import FROZEN, load, pairs  # noqa: E402

OUT = 'outputs/ablation'
FLIGHTS = ('fieldA', 'fieldB', 'fieldC', 'heldout')
HFOV_DEG = 70.3      # Mavic 3E wide camera: 12.29 mm lens on a 17.3 mm wide 4/3 sensor, full width in 16:9 video
INF = float('inf')

VARIANTS = [         # name, settings changed from the frozen model, map ('cv', 'gnss_fit', 'gnss_meta') or None
    ('Frozen model', {}, None),
    ('- search by image overlap (fixed 600 px radius)', dict(overlap_search=False), None),
    ('- drift correction', dict(drift_feedback=False), None),
    ('- rotation correction', dict(rot_aware=False), None),
    ('- skipping fast-turn frames', dict(turn_aware=False), None),
    ('- overlap-only alignment (whole frames)', dict(crop_verify=False), None),
    ('- registration (plants moved by the predicted shift)', dict(register=False), None),
    ('- odometry clamp', dict(max_frame_disp=INF), None),
    ('- patience (new identity at once)', dict(patience=1), None),
    ('- exclusion of visible identities', dict(exclude_live=False), None),
    ('Greedy instead of Hungarian assignment', dict(assign='greedy'), None),
    ('Odometry only (no verification, no registration)', dict(register=False, min_response=0.0, shift_tol=INF), None),
    ('Constant-velocity map per flight leg', {}, 'cv'),
    ('Constant-velocity map, no verification or registration', dict(register=False, min_response=0.0, shift_tol=INF), 'cv'),
    ('Drone GNSS map, fitted to the images', {}, 'gnss_fit'),
    ('Drone GNSS map, drone metadata only', {}, 'gnss_meta'),
    ('Drone GNSS map, metadata only, no verification or registration', dict(register=False, min_response=0.0, shift_tol=INF), 'gnss_meta'),
    ('Field A method (v1)', 'v1', None),
    ('Ordinal reversal (detection order only)', 'ordinal', None),
]


def cv_map(motion, clamp=25.0):
    """Constant-velocity dead reckoning: within every flight leg / transit segment of the odometry map the camera moves
    at that segment's mean velocity (an ideal flight plan with the true turn times)."""
    cx, cy = RX.odometry_map(motion, clamp)
    m = pd.DataFrame({'frame': np.arange(len(cx)), 'cam_x': cx, 'cam_y': cy})
    lg = pd.Series(legs_from_map(m))
    seg = (lg != lg.shift()).cumsum().values
    vx, vy = np.zeros(len(cx)), np.zeros(len(cx))
    for k in np.unique(seg):
        i = np.flatnonzero(seg == k)
        i0, i1 = max(i[0] - 1, 0), i[-1]
        vx[i] = (cx[i1] - cx[i0]) / max(i1 - i0, 1)
        vy[i] = (cy[i1] - cy[i0]) / max(i1 - i0, 1)
    return np.c_[np.cumsum(vx), np.cumsum(vy)]


def gnss_fit_map(flight, motion, clamp=25.0):
    """Drone GNSS track in image pixels with the similarity that best fits the odometry (best-case calibration)."""
    g = read_srt(f'data/gnss/{flight}.SRT')
    cx, cy = RX.odometry_map(motion, clamp)
    n = min(len(cx), len(g))
    s, R, t = umeyama(np.c_[cx[:n], -cy[:n]], g[['E', 'N']].values[:n])
    px = ((R.T @ (g[['E', 'N']].values - t).T).T) / s
    out = np.c_[px[:, 0], -px[:, 1]]
    if len(out) < len(cx):
        out = np.vstack([out, np.repeat(out[-1:], len(cx) - len(out), 0)])
    return out[:len(cx)]


def gnss_meta_map(flight, n_frames):
    """Drone GNSS track in image pixels from the drone's own metadata only: ground sampling distance from the logged
    height above take-off and the lens field of view, orientation from the logged gimbal yaw."""
    g = read_srt(f'data/gnss/{flight}.SRT')
    gsd = 2 * g.rel_alt.median() * np.tan(np.radians(HFOV_DEG / 2)) / 1920.0
    psi = np.radians(g.yaw.median())
    E, N = g.E.values, g.N.values
    x = (E * np.cos(psi) - N * np.sin(psi)) / gsd
    y = -(E * np.sin(psi) + N * np.cos(psi)) / gsd
    out = np.c_[x - x[0], y - y[0]]
    if len(out) < n_frames:
        out = np.vstack([out, np.repeat(out[-1:], n_frames - len(out), 0)])
    return out[:n_frames]


def ordinal_labels(dets, motion):
    """Ordinal reversal: on two opposite legs over the same ground, the k-th track of the earlier leg is matched to the
    (N-k)-th track of the later leg, by first-appearance order only."""
    cx, cy = RX.odometry_map(motion, 25.0)
    m = pd.DataFrame({'frame': np.arange(len(cx)), 'cam_x': cx, 'cam_y': cy})
    leg = np.array(legs_from_map(m))
    d = dets[(dets.cls == 1) & (dets.raw_id >= 0)]
    first = d.groupby('raw_id').frame.min()
    tleg = pd.Series(leg[first.values], index=first.index)
    raw2id = {r: k for k, r in enumerate(first.index)}
    legs = [l for l in pd.unique(leg) if l != 'transit']
    xs = {l: np.median(cx[leg == l]) for l in legs}
    used = set()
    for i, la in enumerate(legs):
        for lb in legs[i + 1:]:
            if la in used or lb in used or abs(xs[la] - xs[lb]) > 600 or la.split('_')[-1] == lb.split('_')[-1]:
                continue
            A = first[tleg.index[tleg == la]].sort_values().index.tolist()
            B = first[tleg.index[tleg == lb]].sort_values().index.tolist()
            for k, rb in enumerate(B):
                if k < len(A):
                    raw2id[rb] = raw2id[A[len(A) - 1 - k]]
            used |= {la, lb}
    lab = d[['frame', 'raw_id', 'cx', 'cy']].copy()
    lab['id'] = lab.raw_id.map(raw2id)
    return lab


def run_flight(flight):
    os.chdir(ROOT)
    data = load(flight)
    dets, motion, small, phi = data
    P = pairs(flight)
    rows, per = [], {}
    for name, kw, mp in VARIANTS:
        if kw == 'ordinal':
            lab = ordinal_labels(dets, motion)
        elif name == 'Frozen model':
            lab = pd.read_csv(f'data/runs/{flight}/labels.csv')
        else:
            cfg = {} if kw == 'v1' else dict(FROZEN, **kw)
            map_xy = {'cv': lambda: cv_map(motion), 'gnss_fit': lambda: gnss_fit_map(flight, motion),
                      'gnss_meta': lambda: gnss_meta_map(flight, len(motion)), None: lambda: None}[mp]()
            out, _ = RX.run(dets, motion, small, RX.ExpConfig(**cfg), map_xy=map_xy, phi=phi)
            lab = out['labels']
        o = score(lab[['frame', 'raw_id', 'id', 'cx', 'cy']], P, dets)
        b, a, neg = P.kind == 'blind', P.same & (P.kind != 'blind'), ~P.same
        rows.append({'variant': name, 'flight': flight, 'blind_ok': int((o[b] == 'correct').sum()), 'blind_n': int(b.sum()),
                     'links_ok': int((o[a] == 'correct').sum()), 'links_n': int(a.sum()),
                     'wrong': int((o[neg] == 'linked').sum()), 'wrong_n': int(neg.sum()),
                     'identities': int(lab.id.nunique())})
        per[name] = (o[b] == 'correct').values
        print(flight, rows[-1], flush=True)
    pp = pd.DataFrame({k: v for k, v in per.items()})
    pp.insert(0, 'flight', flight)
    pp.insert(1, 'task', P[P.kind == 'blind'].task.values)
    return rows, pp


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: a = frozen model, b = variant."""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    return n01, n10, (1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(n01, n10), n, 0.5)))


def main():
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(run_flight, FLIGHTS))
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    pp = pd.concat([p for _, p in res], ignore_index=True)
    df.to_csv(f'{OUT}/ablation.csv', index=False)
    pp.to_csv(f'{OUT}/ablation_pairs.csv', index=False)
    lines = ['| variant | Field A | Field B | Field C | held-out | all four (95 % CI) | McNemar (lost / gained, p) |',
             '|---|---|---|---|---|---|---|']
    for name, _, _ in VARIANTS:
        d = df[df.variant == name].set_index('flight')
        k, n = int(d.blind_ok.sum()), int(d.blind_n.sum())
        p, lo, hi = wilson(k, n)
        n01, n10, pv = mcnemar(pp['Frozen model'].values.astype(bool), pp[name].values.astype(bool))
        cells = [f"{d.blind_ok[f]}/{d.blind_n[f]}" for f in FLIGHTS]
        mc = '-' if name == 'Frozen model' else f'{n01} / {n10}, p = {pv:.2g}'
        lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {100 * p:.1f} % ({100 * lo:.1f}-{100 * hi:.1f}) | {mc} |')
    md = '# Ablation and baselines: blind plants re-identified\n\n' + '\n'.join(lines) + '\n'
    open(f'{OUT}/ablation.md', 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    main()
