"""
Detector adaptation for the cauliflower field (Field C): 72 annotated frames of a sister flight over the same crop.

Source: every image of Roboflow project niam-q7thy/cauliflower-annotation-for-cabba with its ORIGINAL annotation, fetched
read-only at full 1920x1080 (detector/fetch_roboflow.py --project cauliflower-annotation-for-cabba
--include-unassigned -> data/detector/cauliflower/roboflow), so no stretching. Frames 0-2080 (every 20th) of a sister flight of the
field; none matches a frame of the Field C video (data/detector/cauliflower/source_check.csv, max correlation 0.39).
Checks: only polygon annotations with >= 3 points are kept (1 empty Drain annotation dropped); labels mapped onto the
detector's schema (Aisle 0, plant 1, Divider 2, Drain 3); any other label stops the build.
Split in CONTIGUOUS BLOCKS along the flight (as for the cabbage benchmark), so validation and test frames are not
interleaved with training frames: the 72 frames (in frame order) form 15 blocks of ~5; blocks 1, 6, 11 -> val
(epoch selection), blocks 3, 8, 13 -> test (reported once, never used for any choice), the other 9 -> train.
No frame is dropped (authors' choice), so the frames at a block boundary are 20 frames (0.67 s) from a training frame;
test is therefore reported for all test frames AND for those >= 1 s from every training frame.
Train augmented x3 with the cabbage recipe (detector/augment.py). Fine-tune from the pipeline's cabbage detector
(yolov9c-seg seed 2): imgsz 640, 200 epochs, patience 50, batch 8, seed 2.
(An earlier round with 42 frames and an interleaved split was superseded and is not released.)
Usage: python detector/build_cauliflower.py [--train] [--evaluate]
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
from common import WEIGHTS  # noqa: E402

CLASSES = ['Aisle', 'Cauliflower', 'Divider', 'Drain']

RAW = 'data/detector/cauliflower/roboflow'
DST = 'data/detector/cauliflower/split'
LABEL_MAP = {'Aisle': 0, 'Cauliflower': 1, 'Drain': 3}
N_BLOCKS, VAL_BLOCKS, TEST_BLOCKS = 15, (1, 6, 11), (3, 8, 13)
FAR = 30                                   # frames (1 s): test frames at least this far from every training frame
RUN = 'yolov9c-seg_s2_cauli_v2'


def frame_of(p):
    return int(re.search(r'frame_(\d+)', p).group(1))


def build():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    m = pd.read_csv(f'{RAW}/manifest.csv')
    m['frame'] = m.name.map(frame_of)
    assert not m.frame.duplicated().any(), 'two images with the same frame number'
    m = m.sort_values('frame').reset_index(drop=True)
    block = np.array_split(np.arange(len(m)), N_BLOCKS)
    m['split'] = 'train'
    for b in VAL_BLOCKS:
        m.loc[block[b], 'split'] = 'val'
    for b in TEST_BLOCKS:
        m.loc[block[b], 'split'] = 'test'
    tr = m[m.split == 'train'].frame.values
    m['to_train'] = [int(np.abs(tr - f).min()) for f in m.frame]
    m[['frame', 'split', 'to_train', 'id']].to_csv(f'{RAW}/split_v2.csv', index=False)
    stats, dropped = {}, []
    for r in m.itertuples():
        split = r.split
        stem = f'cauli_frame_{r.frame:06d}'
        for sub in ('images', 'labels'):
            os.makedirs(f'{DST}/{split}/{sub}', exist_ok=True)
        ann = json.load(open(f'{RAW}/ann/{r.id}.json'))['annotation']
        im = cv2.imread(f'{RAW}/images/{r.id}.jpg')
        H, W = im.shape[:2]
        assert (W, H) == (int(ann['width']), int(ann['height'])) == (1920, 1080), f'size mismatch {r.name}'
        lines = []
        for b in ann['boxes']:
            pts = b.get('points')
            pts = json.loads(pts) if isinstance(pts, str) else pts
            if b.get('type') != 'polygon' or not pts or len(pts) < 3:
                dropped.append(f"frame {r.frame}: {b.get('label')} ({b.get('type')})")
                continue
            if b['label'] not in LABEL_MAP:
                raise ValueError(f"unexpected label {b['label']!r} in frame {r.frame}")
            P = np.clip(np.array(pts, float), 0, [W, H]) / [W, H]
            lines.append(f"{LABEL_MAP[b['label']]} " + ' '.join(f'{x:.6f} {y:.6f}' for x, y in P))
            stats[(split, b['label'])] = stats.get((split, b['label']), 0) + 1
        shutil.copy(f'{RAW}/images/{r.id}.jpg', f'{DST}/{split}/images/{stem}.jpg')
        open(f'{DST}/{split}/labels/{stem}.txt', 'w').write('\n'.join(lines) + '\n')
    augment_split(DST, 'train', CLASSES, copies=2, seed=0)
    names = ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
    open(f'{DST}/data.yaml', 'w').write(f'path: {os.path.abspath(DST)}\ntrain: train/images\nval: val/images\n'
                                        f'test: test/images\nnames:\n' + names)
    far = m[(m.split == 'test') & (m.to_train >= FAR)]
    open(f'{DST}/test_far.txt', 'w').write(''.join(f'{os.path.abspath(DST)}/test/images/cauli_frame_{f:06d}.jpg\n'
                                                   for f in far.frame))
    open(f'{DST}/test_far.yaml', 'w').write(f'path: {os.path.abspath(DST)}\ntrain: train/images\nval: test_far.txt\n'
                                            f'names:\n' + names)
    for s in ('train', 'val', 'test'):
        print(s, len(glob.glob(f'{DST}/{s}/images/*.jpg')), 'images (incl. augmented)' if s == 'train' else 'images',
              {k[1]: v for k, v in stats.items() if k[0] == s})
    for s in ('val', 'test'):
        g = m[m.split == s]
        print(f'{s} frames:', g.frame.tolist(), '| frames to nearest training frame:', g.to_train.tolist())
    print(f'test frames >= {FAR} frames from training: {len(far)}')
    print('dropped:', dropped)


def train():
    from ultralytics import YOLO
    YOLO(WEIGHTS).train(data=f'{DST}/data.yaml', epochs=200, patience=50, imgsz=640, batch=8, seed=2, deterministic=True,
                        workers=2, project=os.path.abspath('outputs/detector/cauliflower'), name=RUN,
                        exist_ok=True, plots=True, verbose=False)


def evaluate():
    """Best epoch (chosen on val) scored once on val, test, and test frames >= 1 s from training."""
    from ultralytics import YOLO
    model = YOLO(f'outputs/detector/cauliflower/{RUN}/weights/best.pt')
    rows = []
    for name, data, split in [('val', 'data.yaml', 'val'), ('test (all)', 'data.yaml', 'test'),
                              (f'test (>= {FAR} frames from training)', 'test_far.yaml', 'val')]:
        r = model.val(data=f'{DST}/{data}', split=split, imgsz=640, batch=4, workers=0, verbose=False, plots=False,
                      project='outputs/detector/cauliflower', name=f'{RUN}_eval', exist_ok=True)
        idx = list(r.box.ap_class_index)
        for i in idx:
            k = idx.index(i)
            rows.append({'split': name, 'images': len(r.speed) and None, 'class': r.names[i], 'box_P': r.box.p[k],
                         'box_R': r.box.r[k], 'box_mAP50': r.box.ap50[k], 'box_mAP50_95': r.box.ap[k],
                         'mask_mAP50': r.seg.ap50[k], 'mask_mAP50_95': r.seg.ap[k]})
    df = pd.DataFrame(rows).drop(columns='images').round(3)
    df.to_csv(f'outputs/detector/cauliflower/{RUN}_metrics.csv', index=False)
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
