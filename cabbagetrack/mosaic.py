"""
Mosaic of a field from its own video, placed with the method's camera path (no GNSS, no keypoint stitching).

Camera positions: least squares over the frame-to-frame odometry and every verified revisit link (loop closure), so
the passes line up. Every k-th frame is placed at its position; each mosaic pixel comes from the frame in which it is
most central. A second image marks every plant identity (revisited across passes / seen on one pass only).

Outputs (<out>/): mosaic.jpg, mosaic_identities.jpg, mosaic.json, mosaic_positions.csv
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

BLUE, ORANGE = (94, 70, 26), (49, 119, 242)          # house palette (BGR): ocean blue, orange
W, H = 1920, 1080


def refine(run, extract, n, clamp=25.0):
    """Camera positions that agree with the odometry and with every revisit link's measured image shift.
    Returns refined x, y and the median link residual before / after (px)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    mo = pd.read_csv(os.path.join(extract, 'motion.csv'))
    tx, ty = mo.tx.values.copy(), mo.ty.values.copy()
    bad = np.hypot(tx, ty) > clamp
    tx[bad] = ty[bad] = 0.0
    L = pd.read_csv(os.path.join(run, 'links.csv'))
    rows, cols, vals, bx, by = [], [], [], [], []
    k = 0
    for t in range(1, n):                                          # p_t - p_{t-1} = camera motion
        rows += [k, k]; cols += [t, t - 1]; vals += [1.0, -1.0]; bx.append(tx[t]); by.append(ty[t]); k += 1
    for r in L.itertuples():                                       # p_t - p_s = -(measured content shift)
        rows += [k, k]; cols += [int(r.frame), int(r.s)]; vals += [1.0, -1.0]; bx.append(-r.dx); by.append(-r.dy); k += 1
    rows.append(k); cols.append(0); vals.append(1.0); bx.append(0.0); by.append(0.0); k += 1   # anchor
    A = coo_matrix((vals, (rows, cols)), shape=(k, n)).tocsr()
    x = lsqr(A, np.array(bx), atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    y = lsqr(A, np.array(by), atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    cx, cy = np.cumsum(tx), np.cumsum(ty)
    res = lambda X, Y: np.hypot(X[L.frame] - X[L.s] + L.dx, Y[L.frame] - Y[L.s] + L.dy)  # noqa: E731
    return x, y, float(np.median(res(cx, cy))), float(np.median(res(x, y)))


def extract_dir_of(run):
    p = os.path.join(run, 'extract')
    if os.path.exists(os.path.join(p, 'motion.csv')):
        return p
    s = json.load(open(os.path.join(run, 'summary.json')))
    return s['provenance']['extract']


def source_frames(extract):
    """Video frame number of every pipeline frame (differs from the frame index only when --segments was used)."""
    mo = pd.read_csv(os.path.join(extract, 'motion.csv'))
    return mo.src_frame.values if 'src_frame' in mo else mo.frame.values


def plant_positions(run, cx, cy, gap=300):
    """Median map position of every identity and whether it was seen again more than `gap` frames later."""
    lab = pd.read_csv(os.path.join(run, 'labels.csv'))
    lab = lab[lab.id >= 0]
    lab = lab.assign(mx=cx[lab.frame] + lab.cx, my=cy[lab.frame] + lab.cy)
    pos = lab.groupby('id')[['mx', 'my']].median()
    pos['revisited'] = lab.groupby('id').frame.apply(lambda f: bool((np.diff(np.sort(f.unique())) > gap).any()))
    return pos


def build(video, run, out=None, extract=None, step=4, scale=0.5, refine_path=True):
    out = out or run
    os.makedirs(out, exist_ok=True)
    extract = extract or extract_dir_of(run)
    mp = pd.read_csv(os.path.join(run, 'map.csv'))
    n = len(mp)
    cx, cy = mp.cam_x.values, mp.cam_y.values
    info = {}
    if refine_path:
        cx, cy, r0, r1 = refine(run, extract, n)
        info['link_residual_median_px'] = {'odometry': round(r0, 1), 'after_loop_closure': round(r1, 1)}
    pd.DataFrame({'frame': np.arange(n), 'x': cx, 'y': cy}).to_csv(os.path.join(out, 'mosaic_positions.csv'), index=False)
    s = scale
    x0, y0 = cx.min(), cy.min()
    cw, ch = int(np.ceil((cx.max() - x0 + W) * s)) + 2, int(np.ceil((cy.max() - y0 + H) * s)) + 2
    acc = np.zeros((ch, cw, 3), np.float32)
    wsum = np.zeros((ch, cw), np.float32)
    fw, fh = int(W * s), int(H * s)
    feather = (np.outer(1 - np.abs(np.linspace(-1, 1, fh)), 1 - np.abs(np.linspace(-1, 1, fw))) + 1e-3).astype(np.float32)
    want = {}
    for t, f in enumerate(source_frames(extract)[:n]):
        if t % step == 0:
            want.setdefault(int(f), []).append(t)
    cap = cv2.VideoCapture(video)
    last = max(want)
    for f in range(last + 1):                             # sequential read (seeking is slow on long videos)
        if f not in want:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (fw, fh), interpolation=cv2.INTER_AREA).astype(np.float32)
        for t in want[f]:
            px, py = int(round((cx[t] - x0) * s)), int(round((cy[t] - y0) * s))
            win = feather > wsum[py:py + fh, px:px + fw]      # each pixel from the frame where it is most central
            acc[py:py + fh, px:px + fw][win] = fr[win]
            wsum[py:py + fh, px:px + fw][win] = feather[win]
    cap.release()
    mosaic = acc.clip(0, 255).astype(np.uint8)
    mosaic[wsum < 1e-6] = 255
    cv2.imwrite(os.path.join(out, 'mosaic.jpg'), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
    pos = plant_positions(run, cx, cy)
    ov = mosaic.copy()
    r = max(4, int(18 * s))
    for _, p in pos.iterrows():
        cv2.circle(ov, (int((p.mx - x0) * s), int((p.my - y0) * s)), r, BLUE if p.revisited else ORANGE,
                   max(2, int(5 * s)), cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, 'mosaic_identities.jpg'), ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
    info.update({'frames_used': sum(len(v) for v in want.values()), 'scale': s, 'size_px': [cw, ch], 'origin': [x0, y0],
                 'identities': int(len(pos)), 'revisited': int(pos.revisited.sum())})
    json.dump(info, open(os.path.join(out, 'mosaic.json'), 'w'), indent=1)
    print('mosaic:', info, '->', os.path.abspath(out))
    return info
