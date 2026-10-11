"""
The first annotation tool (Field A, built for the method of the submitted paper; used for gt_tool, gt_tool_blind2
and gt_tool_rowA). Kept as used; new tools: groundtruth/make_tool.py.

Part A (verification): every Phase-1 plant that received a Phase-2 identity is shown
as a pair (way-back view | claimed way-out partner), shuffled, with ~10% decoy pairs
(partner replaced by a neighbour) to measure annotator vigilance.
Part B (blind): N randomly sampled Phase-1 plants; the annotator sees the plant circled
in the way-out photo and clicks the same plant in the way-back photo (all detections
shown, no identities). The computer's answer is never displayed.

Output: <out>/index.html + tasks.js + img/   (open index.html in a browser)
Usage: python groundtruth/make_tool_original.py <two-phase run> --extract <stage-1 folder> \
           --video <video> --out <tool folder>
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
HALF = 170            # verification crop half-size (px, full resolution)
VIEW_W = 960          # blind-task frame width in the tool


def source_frame_map(extract_dir):
    """virtual frame index -> frame index in the source video (joined-segment extracts)."""
    m = pd.read_csv(os.path.join(extract_dir, 'motion.csv'))
    if 'src_frame' in m.columns:
        return dict(zip(m.frame.astype(int), m.src_frame.astype(int)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--extract', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='gt_tool')
    ap.add_argument('--decoy-frac', type=float, default=0.10)
    ap.add_argument('--n-blind', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--blind-only', action='store_true', help='build only Part B (no verification pairs)')
    ap.add_argument('--exclude', nargs='*', default=[], help='result CSVs whose blind plants are excluded')
    ap.add_argument('--tag', default='v1', help='separate save slot in the browser')
    a = ap.parse_args()
    rng = random.Random(a.seed)

    s = json.load(open(os.path.join(a.run, 'summary.json')))
    n1, lf = s['p1_plants'], s['loop_frame']
    labels = pd.read_csv(os.path.join(a.run, 'labels.csv'))
    frames = pd.read_csv(os.path.join(a.run, 'frames.csv'))
    dets = pd.read_csv(os.path.join(a.extract, 'detections.csv.gz'))
    dets = dets[dets.cls == 1]

    lab = labels.copy()
    lab['dc'] = np.hypot(lab.cx - CENTER[0], lab.cy - CENTER[1])
    p1 = lab[lab.phase == 1]
    p2 = lab[(lab.phase == 2) & (lab.gid < n1)]
    p1_best = p1.loc[p1.groupby('gid').dc.idxmin()].set_index('gid')
    p2_best = p2.loc[p2.groupby('gid').dc.idxmin()].set_index('gid')
    p1_by_frame = {f: g.drop_duplicates('gid') for f, g in p1.groupby('frame')}

    # ---------- Part A: verification pairs + decoys ----------
    gids = sorted(set(p2_best.index) & set(p1_best.index))
    n_decoy = int(round(a.decoy_frac * len(gids)))
    decoy_src = set(rng.sample(gids, n_decoy))
    crops = {}                      # (frame) -> list of (name, x, y)

    def want(frame, x, y, name):
        crops.setdefault(int(frame), []).append((name, float(x), float(y)))

    verify = []
    for g in gids:
        q, t = p2_best.loc[g], p1_best.loc[g]
        want(q.frame, q.cx, q.cy, f'v_back_{g}')
        want(t.frame, t.cx, t.cy, f'v_out_{g}')
        verify.append({'id': f'A{g}', 'gid': int(g), 'decoy': False, 'back': f'v_back_{g}.jpg', 'out': f'v_out_{g}.jpg'})
        if g in decoy_src:
            fr = p1_by_frame[int(t.frame)]
            nb = fr[(fr.gid != g) & (np.hypot(fr.cx - t.cx, fr.cy - t.cy).between(80, 220))]
            if len(nb):
                r = nb.iloc[rng.randrange(len(nb))]
                want(t.frame, r.cx, r.cy, f'v_decoy_{g}')
                verify.append({'id': f'D{g}', 'gid': int(g), 'decoy': True, 'decoy_gid': int(r.gid),
                               'back': f'v_back_{g}.jpg', 'out': f'v_decoy_{g}.jpg'})
    rng.shuffle(verify)
    if a.blind_only:
        verify, crops = [], {}

    # ---------- Part B: blind linking ----------
    p2f = frames[(frames.phase == 2)].dropna(subset=['p1_frame_est'])
    small = np.load(os.path.join(a.extract, 'frames_small.npy'), mmap_mode='r')
    hann = cv2.createHanningWindow((small.shape[2], small.shape[1]), cv2.CV_32F)
    blind = []
    full_frames = {}
    done = set()
    for path in a.exclude:
        e = pd.read_csv(path)
        done |= set(e[e.part == 'B'].gid.astype(int))
    pool = [g for g in p1_best.index if g not in done]
    for g in sorted(rng.sample(pool, min(a.n_blind, len(pool)))):
        t = p1_best.loc[g]
        # way-back photo of the same spot, chosen by IMAGE alignment only (what a person scrolling would do):
        # among Phase-2 frames roughly at this place, take the one whose whole picture lines up best with
        # the way-out photo (smallest along-track offset, good correlation). No plant-level matches are used.
        near = p2f[(p2f.p1_frame_est - t.frame).abs() <= 150]
        if len(near) == 0:
            near = p2f.iloc[[int((p2f.p1_frame_est - t.frame).abs().values.argmin())]]
        ref = small[int(t.frame)].astype(np.float32)
        best = None
        for f in near.frame.values[::2]:
            (sx, sy), resp = cv2.phaseCorrelate(ref, small[int(f)].astype(np.float32), hann)
            score = abs(sy) + (1000 if resp < 0.08 else 0)
            if best is None or score < best[0]:
                best = (score, int(f))
        f2 = best[1]
        dd = dets[dets.frame == f2]
        sc = VIEW_W / 1920.0
        blind.append({'id': f'B{g}', 'gid': int(g), 'out_frame': int(t.frame), 'back_frame': f2,
                      'out': f'b_out_{g}.jpg', 'back': f'b_back_{g}.jpg',
                      'target': [round(t.cx * sc, 1), round(t.cy * sc, 1)],
                      'dets': [[round(r.cx * sc, 1), round(r.cy * sc, 1), int(r.raw_id)] for r in dd.itertuples()]})
        full_frames.setdefault(int(t.frame), []).append(f'b_out_{g}')
        full_frames.setdefault(f2, []).append(f'b_back_{g}')

    # ---------- render images (one sequential pass over the video) ----------
    img_dir = os.path.join(a.out, 'img')
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(img_dir)
    need = sorted(set(crops) | set(full_frames))
    cap = cv2.VideoCapture(a.video)
    vmap = source_frame_map(a.extract) if getattr(a, 'extract', None) else None
    def frames_in_order(need_frames):
        """yield (virtual_frame, image) for every needed virtual frame, reading the source video in order."""
        pairs = sorted((vmap[f] if vmap else f, f) for f in need_frames)
        pos = -1
        for src, vf in pairs:
            if src != pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src)
            ok, img = cap.read()
            pos = src
            if ok:
                yield vf, img
    for fi, frame in frames_in_order(list(set(crops) | set(full_frames))):
        pad = cv2.copyMakeBorder(frame, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        for name, x, y in crops.get(fi, []):
            xi, yi = int(round(x)), int(round(y))
            c = pad[yi:yi + 2 * HALF, xi:xi + 2 * HALF].copy()
            for (pp, qq) in [((HALF, 6), (HALF, 30)), ((HALF, 2 * HALF - 30), (HALF, 2 * HALF - 6)),
                             ((6, HALF), (30, HALF)), ((2 * HALF - 30, HALF), (2 * HALF - 6, HALF))]:
                cv2.line(c, pp, qq, (0, 230, 255), 4)
            cv2.imwrite(os.path.join(img_dir, name + '.jpg'), c, [cv2.IMWRITE_JPEG_QUALITY, 88])
        for name in full_frames.get(fi, []):
            small = cv2.resize(frame, (VIEW_W, int(1080 * VIEW_W / 1920)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(img_dir, name + '.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()

    meta = {'run': os.path.abspath(a.run), 'seed': a.seed, 'tag': a.tag, 'n_verify': sum(not v['decoy'] for v in verify),
            'n_decoy': sum(v['decoy'] for v in verify), 'n_blind': len(blind)}
    with open(os.path.join(a.out, 'tasks.js'), 'w') as f:
        f.write('window.TASKS = ' + json.dumps({'meta': meta, 'verify': verify, 'blind': blind}) + ';\n')
    # key with the computer's answers is kept OUTSIDE the tool folder, so the annotator never sees it
    lab_p2 = labels[labels.phase == 2].drop_duplicates('raw_id', keep='last')[['raw_id', 'gid']]
    lab_p2.to_csv(os.path.join(a.run, 'gt_answer_key_rawid_to_gid.csv'), index=False)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'tool_index_original.html'), os.path.join(a.out, 'index.html'))
    print(json.dumps(meta, indent=1), '\n->', os.path.abspath(a.out))


if __name__ == '__main__':
    main()
