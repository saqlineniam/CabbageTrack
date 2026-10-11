"""
Build the detector dataset (train / val / test) from the raw Roboflow annotations.

Both fields are in every split:
  field 1 = seq_1 U-flight (younger plants), field 2 = DJI_0004 congested field (bigger plants).

Split rule (per video, by time, so val/test pictures are not near-copies of training pictures):
  * the annotated frames of each video are sorted in time and cut into 20 consecutive blocks
    with equal numbers of images;
  * 3 evenly spaced blocks -> test, 3 other evenly spaced blocks -> val, the other 14 -> train
    (about 70 / 15 / 15 %, spread over the whole flight);
  * any training image within GAP frames (1 s) of a val/test image is dropped, so no val/test
    picture has a near-identical neighbour in training.
The test set is also written per field (test_field1, test_field2) for separate reporting.
Training images then get offline augmentation (detector/augment.py, the original Roboflow x3 recipe:
each training image + 2 augmented copies); val/test stay un-augmented real pictures.

Outputs data/detector/cabbage/split/: YOLO-seg folders per split, data.yaml (train/val), test*.yaml,
COCO ground truth per split (<split>_coco.json), split_manifest.csv, split_summary.csv.

Usage: python detector/build_cabbage.py [--aug-copies 2]
(needs data/detector/cabbage/roboflow and data/videos/fieldA.mp4, heldout.mp4)
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from augment import augment_split                                                 # noqa: E402
from common import CLASSES, OUT, RAW, trace_sources, write_yolo   # noqa: E402

N_BLOCKS = 20
TEST_BLOCKS = {2, 9, 16}
VAL_BLOCKS = {5, 12, 19}
GAP = 30                    # frames (1 s at 30 fps)
FIELD = {'seq_1': 'field1', 'DJI_0004': 'field2'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug-copies', type=int, default=2, help='augmented copies per training image (0 = none)')
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(RAW, 'manifest.csv'))
    man = man[man.n_objects > 0].copy()
    man['source'], man['frame'], man['match_sim'] = trace_sources(man)
    man['split'] = 'excluded_unknown_source'
    for src in FIELD:
        g = man[man.source == src].sort_values('frame')
        block = (np.arange(len(g)) * N_BLOCKS) // len(g)
        man.loc[g.index, 'split'] = ['test' if b in TEST_BLOCKS else 'val' if b in VAL_BLOCKS else 'train' for b in block]
        held = man.loc[g.index][man.loc[g.index, 'split'] != 'train'].frame.values
        for i in g.index:
            if man.at[i, 'split'] == 'train' and np.abs(held - man.at[i, 'frame']).min() <= GAP:
                man.at[i, 'split'] = 'excluded_gap'
    man['field'] = man.source.map(FIELD).fillna('unknown')

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    sets = {'train': man.split == 'train', 'val': man.split == 'val', 'test': man.split == 'test',
            'test_field1': (man.split == 'test') & (man.field == 'field1'),
            'test_field2': (man.split == 'test') & (man.field == 'field2')}
    summary = []
    for s, sel in sets.items():
        rows = man[sel]
        coco = write_yolo(rows, s)
        if s == 'train' and a.aug_copies > 0:
            coco = augment_split(OUT, 'train', CLASSES, copies=a.aug_copies, seed=0)
        cnt = pd.Series([CLASSES[a['category_id']] for a in coco['annotations']], dtype=object).value_counts()
        summary.append({'split': s, 'images': len(coco['images']), 'source_images': len(rows), 'field1_images': int((rows.field == 'field1').sum()),
                        'field2_images': int((rows.field == 'field2').sum()), **{c: int(cnt.get(c, 0)) for c in CLASSES}})
    man.to_csv(os.path.join(OUT, 'split_manifest.csv'), index=False)
    base = os.path.abspath(OUT).replace('\\', '/')
    names = '\n'.join(f'  {i}: {c}' for i, c in enumerate(CLASSES))
    open(os.path.join(OUT, 'data.yaml'), 'w').write(f'path: {base}\ntrain: train/images\nval: val/images\nnames:\n{names}\n')
    for t in ['test', 'test_field1', 'test_field2']:
        open(os.path.join(OUT, f'{t}.yaml'), 'w').write(f'path: {base}\ntrain: train/images\nval: {t}/images\nnames:\n{names}\n')
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, 'split_summary.csv'), index=False)
    print(pd.crosstab(man.split, man.field).to_string())
    print('\ninstances per split:\n' + sm.to_string(index=False))


if __name__ == '__main__':
    main()
