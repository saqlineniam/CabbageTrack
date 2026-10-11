"""
Detector adaptation for young cabbage (field 1, 14 Dec 2025), same recipe as build_cauliflower_v2.py.

Source: Roboflow project niam-q7thy/cabbage-young, fetched read-only at full 1920x1080 with original annotations
(detector/fetch_roboflow.py --project cabbage-young --include-unassigned -> data/detector/young_cabbage/roboflow).
NB: all 63 images are frames of DJI_20251214150231_0001 (the outbound clip of the early-stage loop), checked by
whole-image correlation (data/detector/young_cabbage/source_check.csv): the loop is therefore NOT held out for the detector.
Checks: 2 unannotated images skipped; labels Aisle/Cabbage/Divider/Drain map 1:1 onto the detector schema; any other
label stops the build; polygons < 3 points dropped. Correction (raw data unchanged): in frame 240 the 29 plant-sized
outlines labelled 'Drain' (long side <= 136 px = 1.5 x the 95th percentile of Cabbage outlines) are plants -> Cabbage;
single small edge pieces of drains/aisles/dividers elsewhere were checked by eye and kept.
Split: contiguous blocks (15 blocks of the frame-ordered images; blocks 1, 6, 11 -> val, 3, 8, 13 -> test). Train x3
augmentation (detector/augment.py). Fine-tune from yolov9c-seg_s2 (imgsz 640, 200 epochs, patience 50, batch 8, seed 2).
Usage: python detector/build_young.py [--train] [--evaluate]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from augment import augment_split  # noqa: E402

RAW = 'data/detector/young_cabbage/roboflow'
DST = 'data/detector/young_cabbage/split'
CLASSES = ['Aisle', 'Cabbage', 'Divider', 'Drain']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
FIX_FRAME, FIX_MAX_SIDE = 240, 136
N_BLOCKS, VAL_BLOCKS, TEST_BLOCKS, FAR = 15, (1, 6, 11), (3, 8, 13), 30
WEIGHTS = 'data/weights/cabbage.pt'          # the cabbage detector (Table 1, seed 2)
PROJECT, RUN = 'outputs/detector/young', 'yolov9c-seg_s2_young'


def frame_of(p):
    return int(re.search(r'frame_(\d+)', p).group(1))


def build():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    m = pd.read_csv(f'{RAW}/manifest.csv')
    m = m[m.width > 0].copy()                                   # skip unannotated images
    m['frame'] = m.name.map(frame_of)
    assert not m.frame.duplicated().any()
    m = m.sort_values('frame').reset_index(drop=True)
    block = np.array_split(np.arange(len(m)), N_BLOCKS)
    m['split'] = 'train'
    for b in VAL_BLOCKS:
        m.loc[block[b], 'split'] = 'val'
    for b in TEST_BLOCKS:
        m.loc[block[b], 'split'] = 'test'
    tr = m[m.split == 'train'].frame.values
    m['to_train'] = [int(np.abs(tr - f).min()) for f in m.frame]
    m[['frame', 'split', 'to_train', 'id']].to_csv(f'{RAW}/split.csv', index=False)
    stats, dropped, fixed = {}, [], 0
    for r in m.itertuples():
        stem = f'young_frame_{r.frame:06d}'
        for sub in ('images', 'labels'):
            os.makedirs(f'{DST}/{r.split}/{sub}', exist_ok=True)
        ann = json.load(open(f'{RAW}/ann/{r.id}.json'))['annotation']
        im = cv2.imread(f'{RAW}/images/{r.id}.jpg')
        H, W = im.shape[:2]
        assert (W, H) == (int(ann['width']), int(ann['height'])) == (1920, 1080), r.name
        lines = []
        for b in ann['boxes']:
            pts = json.loads(b['points']) if isinstance(b.get('points'), str) else b.get('points')
            if b.get('type') != 'polygon' or not pts or len(pts) < 3:
                dropped.append(f"frame {r.frame}: {b.get('label')}")
                continue
            lab = b['label']
            if lab not in LABEL_MAP:
                raise ValueError(f'unexpected label {lab!r} in frame {r.frame}')
            P = np.array(pts, float)
            if r.frame == FIX_FRAME and lab == 'Drain' and np.ptp(P, 0).max() <= FIX_MAX_SIDE:
                lab, fixed = 'Cabbage', fixed + 1
            P = np.clip(P, 0, [W, H]) / [W, H]
            lines.append(f'{LABEL_MAP[lab]} ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in P))
            stats[(r.split, lab)] = stats.get((r.split, lab), 0) + 1
        shutil.copy(f'{RAW}/images/{r.id}.jpg', f'{DST}/{r.split}/images/{stem}.jpg')
        open(f'{DST}/{r.split}/labels/{stem}.txt', 'w').write('\n'.join(lines) + '\n')
    augment_split(DST, 'train', CLASSES, copies=2, seed=0)
    names = ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
    open(f'{DST}/data.yaml', 'w').write(f'path: {os.path.abspath(DST)}\ntrain: train/images\nval: val/images\n'
                                        f'test: test/images\nnames:\n' + names)
    far = m[(m.split == 'test') & (m.to_train >= FAR)]
    open(f'{DST}/test_far.txt', 'w').write(''.join(f'{os.path.abspath(DST)}/test/images/young_frame_{f:06d}.jpg\n'
                                                   for f in far.frame))
    open(f'{DST}/test_far.yaml', 'w').write(f'path: {os.path.abspath(DST)}\ntrain: train/images\nval: test_far.txt\n'
                                            f'names:\n' + names)
    for s in ('train', 'val', 'test'):
        print(s, len(glob.glob(f'{DST}/{s}/images/*.jpg')), 'images' + (' (incl. augmented)' if s == 'train' else ''),
              {k[1]: v for k, v in stats.items() if k[0] == s})
    for s in ('val', 'test'):
        g = m[m.split == s]
        print(f'{s} frames:', g.frame.tolist(), '| to nearest training frame:', g.to_train.tolist())
    print(f'test frames >= {FAR} frames from training: {len(far)} | relabelled Drain->Cabbage in frame {FIX_FRAME}: {fixed}'
          f' | dropped: {dropped}')


def train():
    from ultralytics import YOLO
    YOLO(WEIGHTS).train(data=f'{DST}/data.yaml', epochs=200, patience=50, imgsz=640, batch=8, seed=2, deterministic=True,
                        workers=2, project=os.path.abspath(PROJECT), name=RUN, exist_ok=True, plots=True, verbose=False)


def evaluate():
    from ultralytics import YOLO
    model = YOLO(f'{PROJECT}/{RUN}/weights/best.pt')
    rows = []
    for name, data, split in [('val', 'data.yaml', 'val'), ('test (all)', 'data.yaml', 'test'),
                              (f'test (>= {FAR} frames from training)', 'test_far.yaml', 'val')]:
        r = model.val(data=f'{DST}/{data}', split=split, imgsz=640, batch=4, workers=0, verbose=False, plots=False,
                      project=PROJECT, name=f'{RUN}_eval', exist_ok=True)
        idx = list(r.box.ap_class_index)
        for i in idx:
            k = idx.index(i)
            rows.append({'split': name, 'class': r.names[i], 'box_P': r.box.p[k], 'box_R': r.box.r[k],
                         'box_mAP50': r.box.ap50[k], 'box_mAP50_95': r.box.ap[k], 'mask_mAP50': r.seg.ap50[k],
                         'mask_mAP50_95': r.seg.ap[k]})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(f'{PROJECT}/{RUN}_metrics.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    a = ap.parse_args()
    if not a.evaluate:
        build()
    if a.train:
        train()
    if a.evaluate:
        evaluate()
