"""
Annotation tool for a CabbageTrack run (any flight pattern): a self-contained web page (index.html + tasks.js + img/)
that one or more annotators open in a browser, offline. Answers are exported as CSV from the page.

Part A (verification): every re-identification across passes (gap > --min-gap frames) is shown as a pair:
the plant on the later pass | the plant the method says it is on an earlier pass. About 10 % hidden decoys
(the earlier partner replaced by a neighbour) measure annotator vigilance.

Part B (blind): N plants sampled from first sightings, independent of the method's links. The annotator sees the
plant circled in the earlier photo and clicks the same plant in a later-pass photo of the same ground. The later
photo is chosen by whole-image alignment only (phase correlation, smallest shift with a clear peak, plant well
inside the frame); plant identities are never used. Plants whose ground is never flown over again are not eligible.

Output: <out>/index.html + tasks.js + img/ ; answer key <run>/gt_answer_key_rawid_to_gid.csv
Usage: python groundtruth/make_tool.py runs/heldout --extract runs/heldout/extract --video data/videos/heldout.mp4            --out tools/heldout_check --tag heldout
Answers -> position pairs: groundtruth/score.py tool_pairs(); scoring: groundtruth/score.py score(), link_check()
"""
import argparse
import json
import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd

CENTER = np.array([960.0, 540.0])
W, H = 1920, 1080
VIEW_W = 960
MARGIN = 150          # px: the plant must be at least this far inside the later photo
MIN_RESP = 0.08       # phase-correlation peak needed to trust an alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--extract', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='gt_tool_field2')
    ap.add_argument('--tag', default='field2', help='separate save slot in the browser')
    ap.add_argument('--min-gap', type=int, default=300, help='frames between passes')
    ap.add_argument('--decoy-frac', type=float, default=0.10)
    ap.add_argument('--n-blind', type=int, default=100)
    ap.add_argument('--half', type=int, default=220, help='verification crop half-size (px); field-2 plants are large')
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--only-raw', default='', help='CSV with a raw_id column: Part A shows only these later tracks '
                                                  '(e.g. links that a previous verification did not cover)')
    ap.add_argument('--blind-select', choices=['phasecorr', 'map'], default='phasecorr',
                    help="how Part B picks the later photo: whole-image alignment (default) or odometry map position")
    ap.add_argument('--map-margin', type=int, default=250, help="with --blind-select map: px the plant must be inside")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    lab = pd.read_csv(os.path.join(a.run, 'labels.csv'))            # frame, raw_id, id, cx, cy
    ev = pd.read_csv(os.path.join(a.run, 'events.csv'))
    mp = pd.read_csv(os.path.join(a.run, 'map.csv')).set_index('frame')
    dets = pd.read_csv(os.path.join(a.extract, 'detections.csv.gz'))
    dets = dets[dets.cls == 1]
    small = np.load(os.path.join(a.extract, 'frames_small.npy'), mmap_mode='r')
    scale = W / small.shape[2]
    hann = cv2.createHanningWindow((small.shape[2], small.shape[1]), cv2.CV_32F)
    lab['dc'] = np.hypot(lab.cx - CENTER[0], lab.cy - CENTER[1])
    best_view = lab.loc[lab.groupby('raw_id').dc.idxmin()].set_index('raw_id')    # best-centred view per track

    crops = {}

    def want(frame, x, y, name):
        crops.setdefault(int(frame), []).append((name, float(x), float(y)))

    # ---------- Part A: verification of every cross-pass re-identification ----------
    r = ev[(ev.kind == 'reid') & ev.gap.notna()].copy()
    r['gap'] = r.gap.astype(int)
    rv = r[r.gap > a.min_gap]
    if a.only_raw:
        rv = rv[rv.raw_id.isin(set(pd.read_csv(a.only_raw).raw_id))]
    verify = []
    for e in rv.itertuples():
        q = best_view.loc[e.raw_id]
        earlier = lab[(lab.id == e.id) & (lab.frame < e.frame) & (lab.raw_id != e.raw_id)]
        if not len(earlier):
            continue
        t = earlier.loc[earlier.dc.idxmin()]
        want(q.frame, q.cx, q.cy, f'v_back_{e.raw_id}')
        want(t.frame, t.cx, t.cy, f'v_out_{e.raw_id}')
        verify.append({'id': f'A{e.raw_id}', 'gid': int(e.id), 'decoy': False,
                       'back': f'v_back_{e.raw_id}.jpg', 'out': f'v_out_{e.raw_id}.jpg'})
    n_decoy = int(round(a.decoy_frac * len(verify)))
    for v in rng.sample(list(verify), n_decoy):
        raw = int(v['id'][1:])
        t_name = v['out'][:-4]
        tf, tx, ty = next((f, x, y) for f, lst in crops.items() for n, x, y in lst if n == t_name)
        fr = lab[lab.frame == tf].drop_duplicates('id')
        nb = fr[(fr.id != v['gid']) & np.hypot(fr.cx - tx, fr.cy - ty).between(100, 300)]
        if len(nb):
            x = nb.iloc[rng.randrange(len(nb))]
            want(tf, x.cx, x.cy, f'v_decoy_{raw}')
            verify.append({'id': f'D{raw}', 'gid': v['gid'], 'decoy': True, 'decoy_gid': int(x.id),
                           'back': v['back'], 'out': f'v_decoy_{raw}.jpg'})
    rng.shuffle(verify)

    # ---------- Part B: blind, sampled from first sightings ----------
    first_raw = lab.groupby('id').apply(lambda g: g.loc[g.frame.idxmin(), 'raw_id'], include_groups=False)
    track_end = lab.groupby('raw_id').frame.max()
    cam = mp[['cam_x', 'cam_y']].to_numpy()
    eligible = {}
    for gid, raw in first_raw.items():
        t = best_view.loc[raw]
        f0 = int(t.frame)
        cand = np.arange(int(track_end[raw]) + a.min_gap, len(cam), 3)
        if not len(cand):
            continue
        best = None
        if a.blind_select == 'map':
            # later photo where the plant's map position is closest to the image centre (passes overlapping too
            # little for whole-image alignment); uses the odometry map only, never the method's plant matches
            x, y = t.cx + cam[f0, 0] - cam[cand, 0], t.cy + cam[f0, 1] - cam[cand, 1]
            inside = (x >= a.map_margin) & (x <= W - a.map_margin) & (y >= a.map_margin) & (y <= H - a.map_margin)
            if inside.any():
                d = np.hypot(x - CENTER[0], y - CENTER[1])
                d[~inside] = np.inf
                eligible[int(gid)] = (raw, f0, float(t.cx), float(t.cy), int(cand[int(d.argmin())]))
            continue
        near = cand[np.hypot(*(cam[cand] - cam[f0]).T) < 900]           # rough map pre-filter only
        ref = small[f0].astype(np.float32)
        for f in near:
            (sx, sy), resp = cv2.phaseCorrelate(ref, small[int(f)].astype(np.float32), hann)
            x, y = t.cx + scale * sx, t.cy + scale * sy                    # where the plant is in the later photo
            if resp < MIN_RESP or not (MARGIN <= x <= W - MARGIN and MARGIN <= y <= H - MARGIN):
                continue
            score = np.hypot(sx, sy)
            if best is None or score < best[0]:
                best = (score, int(f))
        if best:
            eligible[int(gid)] = (raw, f0, float(t.cx), float(t.cy), best[1])
    pool = sorted(eligible)
    chosen = sorted(rng.sample(pool, min(a.n_blind, len(pool))))
    blind, full_frames = [], {}
    sc = VIEW_W / W
    for gid in chosen:
        raw, f0, cx, cy, f2 = eligible[gid]
        dd = dets[dets.frame == f2]
        blind.append({'id': f'B{gid}', 'gid': gid, 'out_frame': f0, 'back_frame': f2,
                      'out': f'b_out_{gid}.jpg', 'back': f'b_back_{gid}.jpg',
                      'target': [round(cx * sc, 1), round(cy * sc, 1)],
                      'dets': [[round(d.cx * sc, 1), round(d.cy * sc, 1), int(d.raw_id)] for d in dd.itertuples()]})
        full_frames.setdefault(f0, []).append(f'b_out_{gid}')
        full_frames.setdefault(f2, []).append(f'b_back_{gid}')

    # ---------- render images (one sequential pass over the video) ----------
    img_dir = os.path.join(a.out, 'img')
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(img_dir)
    cap, pos, HALF = cv2.VideoCapture(a.video), -1, a.half
    for fi in sorted(set(crops) | set(full_frames)):
        if fi != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        pos = fi
        if not ok:
            continue
        pad = cv2.copyMakeBorder(frame, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        for name, x, y in crops.get(fi, []):
            xi, yi = int(round(x)), int(round(y))
            c = pad[yi:yi + 2 * HALF, xi:xi + 2 * HALF].copy()
            for (pp, qq) in [((HALF, 6), (HALF, 30)), ((HALF, 2 * HALF - 30), (HALF, 2 * HALF - 6)),
                             ((6, HALF), (30, HALF)), ((2 * HALF - 30, HALF), (2 * HALF - 6, HALF))]:
                cv2.line(c, pp, qq, (0, 230, 255), 4)
            cv2.imwrite(os.path.join(img_dir, name + '.jpg'), c, [cv2.IMWRITE_JPEG_QUALITY, 88])
        for name in full_frames.get(fi, []):
            s = cv2.resize(frame, (VIEW_W, int(H * VIEW_W / W)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(img_dir, name + '.jpg'), s, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()

    meta = {'run': os.path.abspath(a.run), 'seed': a.seed, 'tag': a.tag, 'n_verify': sum(not v['decoy'] for v in verify),
            'n_decoy': sum(v['decoy'] for v in verify), 'n_blind': len(blind), 'blind_eligible': len(pool),
            'identities': int(lab.id.nunique())}
    with open(os.path.join(a.out, 'tasks.js'), 'w') as f:
        f.write('window.TASKS = ' + json.dumps({'meta': meta, 'verify': verify, 'blind': blind}) + ';\n')
    # answer key OUTSIDE the tool folder: every track keeps one identity for life
    lab.drop_duplicates('raw_id')[['raw_id', 'id']].rename(columns={'id': 'gid'}).to_csv(
        os.path.join(a.run, 'gt_answer_key_rawid_to_gid.csv'), index=False)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'tool_index.html'), os.path.join(a.out, 'index.html'))
    print(json.dumps(meta, indent=1), '\n->', os.path.abspath(a.out))


if __name__ == '__main__':
    main()
