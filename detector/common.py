"""Shared helpers for the detector datasets: classes, paths, tracing every annotated image to its video frame,
writing YOLO-seg labels and COCO ground truth."""
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd

RAW = 'data/detector/cabbage/roboflow'
OUT = 'data/detector/cabbage/split'
CLASSES = ['Aisle', 'Cabbage', 'Divider', 'Drain']        # same ids as the Roboflow export
VIDEOS = {'seq_1': 'data/videos/fieldA.mp4',            # source names as in split_manifest.csv: seq_1 = Field A,
          'DJI_0004': 'data/videos/heldout.mp4'}          # DJI_0004 = the held-out flight
WEIGHTS = 'data/weights/cabbage.pt'                       # start of the fine-tuned detectors
FRAME_W, FRAME_H = 1920, 1080


def small(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32)


def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def trace_sources(man):
    caps = {k: cv2.VideoCapture(v) for k, v in VIDEOS.items() if os.path.exists(v)}
    n = {k: int(c.get(7)) for k, c in caps.items()}
    src, frame, sim = [], [], []
    for r in man.itertuples():
        img = cv2.imread(os.path.join(RAW, 'images', f'{r.id}.jpg'))
        f = int(''.join(ch for ch in os.path.splitext(r.name)[0] if ch.isdigit()) or -1)
        q, best = small(img), ('unknown', 0.0)
        for k, c in caps.items():
            if 0 <= f < n[k]:
                c.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = c.read()
                if ok:
                    s = ncc(q, small(fr))
                    if s > best[1]:
                        best = (k, s)
        src.append(best[0] if best[1] > 0.9 else 'unknown')
        frame.append(f)
        sim.append(round(best[1], 3))
    return src, frame, sim


def write_yolo(rows, split):
    idir, ldir = os.path.join(OUT, split, 'images'), os.path.join(OUT, split, 'labels')
    os.makedirs(idir, exist_ok=True)
    os.makedirs(ldir, exist_ok=True)
    coco = {'images': [], 'annotations': [], 'categories': [{'id': i, 'name': c} for i, c in enumerate(CLASSES)]}
    aid = 1
    for k, r in enumerate(rows.itertuples(), 1):
        stem = f'{r.source}_{r.frame:06d}_{r.id[:6]}'
        shutil.copy(os.path.join(RAW, 'images', f'{r.id}.jpg'), os.path.join(idir, stem + '.jpg'))
        ann = json.load(open(os.path.join(RAW, 'ann', f'{r.id}.json'))).get('annotation', {})
        W, H = float(ann['width']), float(ann['height'])
        coco['images'].append({'id': k, 'file_name': stem + '.jpg', 'width': int(W), 'height': int(H)})
        lines = []
        for b in ann.get('boxes', []):
            if b['label'] not in CLASSES:
                continue
            c = CLASSES.index(b['label'])
            if b.get('points'):
                pts = np.array(b['points'], dtype=float)
            else:                                   # rectangle -> 4-point polygon
                x, y, w, h = (float(b[t]) for t in ('x', 'y', 'width', 'height'))
                pts = np.array([[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]])
            pts[:, 0] = pts[:, 0].clip(0, W)
            pts[:, 1] = pts[:, 1].clip(0, H)
            if len(pts) < 3 or np.ptp(pts[:, 0]) < 3 or np.ptp(pts[:, 1]) < 3:
                continue                            # accidental mis-clicks (3 objects in the project)
            lines.append(f'{c} ' + ' '.join(f'{x / W:.6f} {y / H:.6f}' for x, y in pts))
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            area = float(cv2.contourArea(pts.astype(np.float32)))
            coco['annotations'].append({'id': aid, 'image_id': k, 'category_id': c, 'iscrowd': 0, 'area': area,
                                        'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                        'segmentation': [pts.flatten().round(2).tolist()]})
            aid += 1
        open(os.path.join(ldir, stem + '.txt'), 'w').write('\n'.join(lines) + ('\n' if lines else ''))
    json.dump(coco, open(os.path.join(OUT, f'{split}_coco.json'), 'w'))
    return coco
