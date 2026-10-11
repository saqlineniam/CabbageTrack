"""
Demo video of a CabbageTrack run: the flight with every plant segmented and numbered, a live map of the camera path
and the plants found so far, and a feed of plants recognised on a later pass (earlier view | now).

  left    the video frame: plants (green) with their identity number, aisles / dividers / drains (outlines);
          a plant recognised from an earlier pass is outlined in crimson for 2 s
  right   live map (camera path, current view, plants: orange = seen on one pass so far, blue = recognised again),
          and the latest re-identified plants: earlier pass | now, with the seconds between the two views

The segmentation masks are recomputed with the run's detector and matched to the run's own detections (box IoU > 0.5),
so every number shown is the identity the pipeline assigned. Output: H.264 MP4 (yuv420p; YouTube, browsers).
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from .mosaic import extract_dir_of, source_frames

HEX = {'dark_red': '#8F020A', 'crimson': '#C01331', 'orange': '#F27731', 'leaf_green': '#56A854', 'deep_teal': '#1C694B',
       'aqua_marine': '#4EA9A2', 'steel_blue': '#77A7BF', 'slate_blue': '#44728C', 'ocean_blue': '#1A465E',
       'midnight_navy': '#02203A', 'charcoal_gray': '#5D6161'}          # house palette
C = {k: tuple(int(v[i:i + 2], 16) for i in (5, 3, 1)) for k, v in HEX.items()}   # BGR
CLASS_COL = {0: C['steel_blue'], 1: C['leaf_green'], 2: C['orange'], 3: C['aqua_marine']}
FONT = cv2.FONT_HERSHEY_SIMPLEX
OUT_W, OUT_H = 1920, 1080
TOP = 60                                     # header height
VID_W, VID_H = 1440, 810                     # video panel
SIDE_X = VID_W + 14                          # right column
SIDE_W = OUT_W - SIDE_X - 14
MAP_BOX = (SIDE_X, TOP + 36, SIDE_W, 400)
CROSS_PASS = 300                             # frames: a re-identification across passes (10 s)
SHOW_S = 2.0                                 # seconds a re-identification stays highlighted


def text(img, s, org, scale=0.6, col=(255, 255, 255), th=1, bg=None, pad=4):
    if bg is not None:
        (w, h), b = cv2.getTextSize(s, FONT, scale, th)
        cv2.rectangle(img, (org[0] - pad, org[1] - h - pad), (org[0] + w + pad, org[1] + b + pad // 2), bg, -1)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def box_iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def crop(frame, x, y, w=360, h=270):
    x0 = int(np.clip(x - w / 2, 0, frame.shape[1] - w))
    y0 = int(np.clip(y - h / 2, 0, frame.shape[0] - h))
    c = frame[y0:y0 + h, x0:x0 + w].copy()
    cv2.circle(c, (int(x - x0), int(y - y0)), 70, C['crimson'], 3, cv2.LINE_AA)
    return c


class MapPanel:
    """Live map: camera path so far, current footprint, plants found so far."""

    def __init__(self, cx, cy, pos, first_seen, revisit_at, box):
        self.box = box
        x0, y0 = cx.min() - 60, cy.min() - 60
        x1, y1 = cx.max() + 1920 + 60, cy.max() + 1080 + 60
        self.s = min(box[2] / (x1 - x0), box[3] / (y1 - y0))
        self.off = (box[0] + (box[2] - (x1 - x0) * self.s) / 2 - x0 * self.s,
                    box[1] + (box[3] - (y1 - y0) * self.s) / 2 - y0 * self.s)
        self.cx, self.cy = cx, cy
        self.ids = pos.index.values
        self.px = np.c_[self.off[0] + pos.mx.values * self.s, self.off[1] + pos.my.values * self.s].astype(int)
        self.first = first_seen.reindex(self.ids).values
        self.rev = np.array([revisit_at.get(i, 1 << 30) for i in self.ids])
        self.r = max(2, int(round(45 * self.s)))

    def p(self, x, y):
        return int(self.off[0] + x * self.s), int(self.off[1] + y * self.s)

    def draw(self, img, t):
        x, y, w, h = self.box
        cv2.rectangle(img, (x, y), (x + w, y + h), (248, 248, 248), -1)
        pts = np.array([self.p(self.cx[k] + 960, self.cy[k] + 540) for k in range(0, t + 1, 3)], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, C['charcoal_gray'], 1, cv2.LINE_AA)
        for k in np.flatnonzero(self.first <= t):
            cv2.circle(img, tuple(self.px[k]), self.r, C['ocean_blue'] if self.rev[k] <= t else C['orange'], -1,
                       cv2.LINE_AA)
        cv2.rectangle(img, self.p(self.cx[t], self.cy[t]), self.p(self.cx[t] + 1920, self.cy[t] + 1080), C['crimson'], 2,
                      cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), C['charcoal_gray'], 1)


def render(video, run, weights, out_path, start=0, end=None, step=1, conf=0.5, title='CabbageTrack', fps_out=None):
    """Render the demo for pipeline frames start..end (every `step`-th frame)."""
    from ultralytics import YOLO
    import imageio_ffmpeg
    extract = extract_dir_of(run)
    lab = pd.read_csv(os.path.join(run, 'labels.csv'))
    lab = lab[lab.id >= 0]
    ev = pd.read_csv(os.path.join(run, 'events.csv'))
    mp = pd.read_csv(os.path.join(run, 'map.csv'))
    dets = pd.read_csv(os.path.join(extract, 'detections.csv.gz'), usecols=['frame', 'raw_id', 'cls', 'x1', 'y1', 'x2', 'y2'])
    meta = json.load(open(os.path.join(extract, 'meta.json')))
    fps = float(meta.get('fps') or 29.97)
    n = len(mp)
    end = min(end or n, n)
    src = source_frames(extract)[:n]
    cx, cy = mp.cam_x.values, mp.cam_y.values

    raw2id = lab.drop_duplicates(['frame', 'raw_id']).set_index(['frame', 'raw_id']).id
    lab = lab.assign(mx=cx[lab.frame] + lab.cx, my=cy[lab.frame] + lab.cy)
    pos = lab.groupby('id')[['mx', 'my']].median()
    first_seen = lab.groupby('id').frame.min()
    xr = ev[(ev.kind != 'new') & (ev.gap > CROSS_PASS)].sort_values('frame')
    revisit_at = xr.drop_duplicates('id').set_index('id').frame.to_dict()
    # every cross-pass re-identification: the earlier-pass view and the later view (within 3 s) in which the plant
    # is most central; the pair enters the feed at the later view
    by_id = {i: g for i, g in lab.groupby('id')}
    jobs, need, push = {}, {}, {}
    central = lambda g: int(np.argmin(np.hypot(g.cx.values - 960, g.cy.values - 540)))  # noqa: E731
    for r in xr.itertuples():
        g = by_id.get(r.id)
        if g is None:
            continue
        e = g[g.frame < r.frame - CROSS_PASS]
        l_ = g[(g.frame >= r.frame) & (g.frame <= r.frame + 3 * fps)]
        if not len(e) or not len(l_):
            continue
        k, j = central(e), central(l_)
        fe, fl, key = int(e.frame.iloc[k]), int(l_.frame.iloc[j]), (int(r.id), int(r.frame))
        jobs.setdefault(int(r.frame), []).append((key, fe))
        need.setdefault(fe, []).append((key, 'a', float(e.cx.iloc[k]), float(e.cy.iloc[k])))
        need.setdefault(fl, []).append((key, 'b', float(l_.cx.iloc[j]), float(l_.cy.iloc[j])))
        push.setdefault(fl, []).append((key, (fl - fe) / fps))
    crops, feed, hot = {}, [], {}

    det_by_f = {f: g for f, g in dets[(dets.raw_id >= 0) & (dets.cls == 1)].groupby('frame')}
    model = YOLO(weights)
    mpanel = MapPanel(cx, cy, pos, first_seen, revisit_at, MAP_BOX)
    writer = imageio_ffmpeg.write_frames(out_path, (OUT_W, OUT_H), fps=fps_out or fps / step, codec='libx264',
                                         quality=None, output_params=['-crf', '20', '-preset', 'medium'],
                                         macro_block_size=8)
    writer.send(None)
    cap = cv2.VideoCapture(video)
    want = {int(src[t]): t for t in range(end)}
    last_src = int(src[end - 1])
    n_ids, f = 0, 0
    while f <= last_src:
        if f not in want:
            if not cap.grab():
                break
            f += 1
            continue
        ok, frame = cap.read()
        t = want[f]
        f += 1
        if not ok:
            break
        for key, which, x, y in need.get(t, []):              # views for the feed (read on every frame)
            crops[key + (which,)] = crop(frame, x, y)
        for key, gap_s in push.get(t, []):
            a_, b_ = crops.pop(key + ('a',), None), crops.pop(key + ('b',), None)
            if a_ is not None and b_ is not None:
                feed.insert(0, (key[0], a_, b_, gap_s))
                del feed[2:]
        for key, fe in jobs.get(t, []):                       # re-identified at this frame
            hot[key[0]] = (t + int(SHOW_S * fps), (key[1] - fe) / fps)
        if t < start or (t - start) % step:
            continue
        res = model.predict(frame, imgsz=640, conf=conf, retina_masks=True, verbose=False)[0]
        vis = frame.copy()
        here = {}
        items = []
        if res.masks is not None and len(res.boxes):
            mk = res.masks.data.cpu().numpy() > 0.5
            cls = res.boxes.cls.cpu().numpy().astype(int)
            bx = res.boxes.xyxy.cpu().numpy()
            d = det_by_f.get(t)
            J = box_iou(bx, d[['x1', 'y1', 'x2', 'y2']].values) if d is not None and len(d) else None
            over = vis.copy()
            for i in range(len(cls)):
                pid = None
                if cls[i] == 1:
                    over[mk[i]] = CLASS_COL[1]
                    if J is not None and J[i].max() > 0.5:
                        pid = raw2id.get((t, int(d.raw_id.values[int(J[i].argmax())])))
                items.append((mk[i], cls[i], pid, (bx[i, 0] + bx[i, 2]) / 2, (bx[i, 1] + bx[i, 3]) / 2))
            vis = cv2.addWeighted(over, 0.35, vis, 0.65, 0)
        for m_, c_, pid, x, y in items:
            on = pid is not None and pid in hot and hot[pid][0] >= t
            cnts, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, C['crimson'] if on else CLASS_COL.get(c_, C['charcoal_gray']),
                             7 if on else (2 if c_ == 1 else 4), cv2.LINE_AA)
            if pid is not None:
                here[pid] = (x, y, on)
        for pid, (x, y, on) in here.items():
            s = f'{pid}'
            sc = 1.0 if on else 0.85
            (w, h), _ = cv2.getTextSize(s, FONT, sc, 2)
            text(vis, s, (int(x - w / 2), int(y + h / 2)), sc, (255, 255, 255), 2,
                 bg=C['crimson'] if on else C['ocean_blue'], pad=5)

        canvas = np.full((OUT_H, OUT_W, 3), 255, np.uint8)
        canvas[TOP:TOP + VID_H, :VID_W] = cv2.resize(vis, (VID_W, VID_H), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (0, 0), (OUT_W, TOP - 4), C['midnight_navy'], -1)
        text(canvas, title, (20, 40), 1.0, (255, 255, 255), 2)
        n_ids = int((first_seen <= t).sum())
        n_re = int(sum(1 for v in revisit_at.values() if v <= t))
        s = f't = {t / fps:5.1f} s    plants found {n_ids}    recognised on a later pass {n_re}'
        text(canvas, s, (OUT_W - 20 - cv2.getTextSize(s, FONT, 0.75, 1)[0][0], 40), 0.75, (255, 255, 255), 1)
        text(canvas, 'camera path and plants (video only, no GNSS)', (SIDE_X, TOP + 24), 0.55, C['midnight_navy'], 1)
        mpanel.draw(canvas, t)
        ly = MAP_BOX[1] + MAP_BOX[3] + 28
        for col, lbl in ((C['orange'], 'plant seen on one pass so far'), (C['ocean_blue'], 'plant recognised on a later pass'),
                         (C['crimson'], 'current camera view')):
            cv2.circle(canvas, (SIDE_X + 8, ly - 6), 7, col, -1, cv2.LINE_AA)
            text(canvas, lbl, (SIDE_X + 24, ly), 0.52, C['midnight_navy'], 1)
            ly += 24
        fy = ly + 22
        text(canvas, 'recognised again:  earlier pass  |  now', (SIDE_X, fy), 0.55, C['midnight_navy'], 1)
        fy += 12
        cw_ = (SIDE_W - 10) // 2
        chh = int(cw_ * 0.75)
        for pid, a, b, gap_s in feed:
            if fy + chh > OUT_H - 10:
                break
            canvas[fy:fy + chh, SIDE_X:SIDE_X + cw_] = cv2.resize(a, (cw_, chh), interpolation=cv2.INTER_AREA)
            canvas[fy:fy + chh, SIDE_X + cw_ + 10:SIDE_X + 2 * cw_ + 10] = cv2.resize(b, (cw_, chh), interpolation=cv2.INTER_AREA)
            text(canvas, f'plant {pid}, {gap_s:.0f} s later', (SIDE_X + 6, fy + chh - 8), 0.5, (255, 255, 255), 1,
                 bg=C['crimson'], pad=3)
            fy += chh + 12
        lx, ly2 = 20, TOP + VID_H + 44
        for c_, nm in ((1, 'plant + identity number'), (0, 'aisle'), (2, 'divider'), (3, 'drain')):
            cv2.rectangle(canvas, (lx, ly2 - 15), (lx + 24, ly2 + 3), CLASS_COL[c_], -1)
            text(canvas, nm, (lx + 32, ly2), 0.62, C['midnight_navy'], 1)
            lx += 70 + cv2.getTextSize(nm, FONT, 0.62, 1)[0][0]
        cv2.rectangle(canvas, (lx, ly2 - 15), (lx + 24, ly2 + 3), C['crimson'], 3)
        text(canvas, 'recognised from an earlier pass', (lx + 32, ly2), 0.62, C['midnight_navy'], 1)
        text(canvas, 'Plants are tracked within a pass and recognised again on later passes from the video alone '
             '(camera odometry + image alignment).', (20, ly2 + 50), 0.55, C['charcoal_gray'], 1)
        text(canvas, f'Numbers are the identities assigned by the pipeline.   detector: {os.path.basename(weights)}',
             (20, ly2 + 80), 0.55, C['charcoal_gray'], 1)
        writer.send(np.ascontiguousarray(canvas[:, :, ::-1]))
        if t % 300 == 0:
            print(f'demo: frame {t}/{end}', flush=True)
        if t >= end - 1:
            break
    writer.close()
    cap.release()
    print(f'demo video -> {os.path.abspath(out_path)}  (plants {n_ids}, plants recognised on a later pass {len(revisit_at)})')
    return out_path
