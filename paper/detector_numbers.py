"""
Detector numbers for the paper, all computed from the files (nothing typed by hand) -> outputs/detectors/README.md
and outputs/detectors/detector_numbers.json.

  1. the three detectors: what each is used on, training data per flight and split, settings, epochs, test scores
     (Table 1 protocol: full-resolution masks, plant P/R at confidence 0.5 and IoU 0.5, AP from pycocotools);
  2. the cabbage detector's test frames split by flight (Field A / held-out);
  3. the held-out flight and the cabbage detector's training images: frame identity check (full-resolution NCC of each
     training image with its own video frame and with frames 2 s either side), share of held-out plant identities that
     appear in a training image, share of the flight's frames within 1 s / 2 s of a training frame.
Usage: python paper/detector_numbers.py
"""
import glob
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
import yaml

OUT = 'outputs/detectors'
FPS = 30
CAB_RUN = 'data/detector/cabbage/benchmark/runs/yolov9c-seg_s2'
YOUNG_RUN = 'data/detector/young_cabbage/training'
CAULI_RUN = 'data/detector/cauliflower/training'
HELDOUT_VIDEO = 'data/videos/heldout.mp4'
HELDOUT_RUN = 'data/runs/heldout'


def epochs(run):
    d = pd.read_csv(os.path.join(run, 'results.csv'))
    d.columns = [c.strip() for c in d.columns]
    fit = (0.1 * d['metrics/mAP50(M)'] + 0.9 * d['metrics/mAP50-95(M)'] + 0.1 * d['metrics/mAP50(B)']
           + 0.9 * d['metrics/mAP50-95(B)'])
    return len(d), int(d.epoch[fit.idxmax()])


def args(run):
    return yaml.safe_load(open(os.path.join(run, 'args.yaml')))


def originals(folder):
    """Non-augmented images of a split folder -> frame numbers."""
    names = [os.path.basename(p) for p in glob.glob(os.path.join(folder, 'images', '*.jpg')) if '_aug' not in p]
    return names, sorted(int(re.findall(r'(\d{6})', n)[-1]) for n in names)


def plant_instances(folder, prefix=''):
    n = 0
    for p in glob.glob(os.path.join(folder, 'labels', prefix + '*.txt')):
        if '_aug' not in p:
            n += sum(1 for line in open(p) if line.split()[:1] == ['1'])
    return n


def ncc(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(float); b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(float)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    os.makedirs(OUT, exist_ok=True)
    J = {}
    # ------------------------------------------------------------ cabbage detector
    man = pd.read_csv('data/detector/cabbage/split/split_manifest.csv')
    cnt = pd.crosstab(man.source, man.split)
    seed = lambda f: pd.read_csv(f).query("run == 'yolov9c-seg_s2'").iloc[0]
    cab_test = {'all': seed('data/detector/cabbage/benchmark/per_seed_test.csv'),
                'Field A': seed('data/detector/cabbage/benchmark/per_seed_test_field1.csv'),
                'held-out flight': seed('data/detector/cabbage/benchmark/per_seed_test_field2.csv')}
    cab_inst = {'all': plant_instances('data/detector/cabbage/split/test'), 'Field A': plant_instances('data/detector/cabbage/split/test', 'seq_1_'),
                'held-out flight': plant_instances('data/detector/cabbage/split/test', 'DJI_0004_')}
    # ------------------------------------------------------------ adapted detectors
    ad = pd.read_csv('data/detector/adapted_detectors_table1_protocol.csv')
    young = {s: originals(f'data/detector/young_cabbage/split/{s}')[1] for s in ('train', 'val', 'test')}
    cauli = {s: originals(f'data/detector/cauliflower/split/{s}')[1] for s in ('train', 'val', 'test')}
    ysrc = pd.read_csv('data/detector/young_cabbage/source_check.csv')
    csrc = pd.read_csv('data/detector/cauliflower/source_check.csv')

    # ------------------------------------------------------------ held-out flight vs cabbage training images
    h = man[man.source == 'DJI_0004'].sort_values('frame')
    cap = cv2.VideoCapture(HELDOUT_VIDEO)

    def frame(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, x = cap.read()
        return x if ok else None
    own, side = [], []
    for r in h[h.split != 'excluded_gap'].itertuples():
        p = glob.glob(f'data/detector/cabbage/split/{r.split}/images/DJI_0004_{r.frame:06d}_*.jpg')
        p = [q for q in p if '_aug' not in q]
        if not p:
            continue
        im = cv2.imread(p[0])
        own.append(ncc(im, frame(r.frame)))
        side += [ncc(im, x) for x in (frame(r.frame - 60), frame(r.frame + 60)) if x is not None]
    tr, trv = set(h[h.split == 'train'].frame), set(h[h.split.isin(['train', 'val'])].frame)
    lab = pd.read_csv(os.path.join(HELDOUT_RUN, 'labels.csv'))
    lab = lab[lab.id >= 0]
    n_id = lab.id.nunique()
    fr = np.array(sorted(lab.frame.unique()))
    dist = np.min(np.abs(fr[:, None] - np.array(sorted(tr))[None]), axis=1)
    J['heldout_vs_cabbage_training'] = {
        'images_by_split': {k: int(v) for k, v in cnt.loc['DJI_0004'].items()},
        'ncc_own_frame_min_max': [round(min(own), 3), round(max(own), 3)], 'ncc_pm2s_max': round(max(side), 3),
        'images_checked': len(own), 'identities': int(n_id),
        'identities_in_training_image': int(lab[lab.frame.isin(tr)].id.nunique()),
        'identities_in_training_or_val_image': int(lab[lab.frame.isin(trv)].id.nunique()),
        'frames_within_1s_of_training_frame': round(float((dist <= FPS).mean()), 3),
        'frames_within_2s_of_training_frame': round(float((dist <= 2 * FPS).mean()), 3)}
    H = J['heldout_vs_cabbage_training']

    # ------------------------------------------------------------ summary table
    rows = []
    for name, run, used, train, val, test, start in [
        ('Cabbage', CAB_RUN, 'Field A, held-out flight',
         f"{cnt.loc['seq_1', 'train']} Field A + {cnt.loc['DJI_0004', 'train']} held-out flight",
         f"{cnt.loc['seq_1', 'val']} + {cnt.loc['DJI_0004', 'val']}", f"{cnt.loc['seq_1', 'test']} + {cnt.loc['DJI_0004', 'test']}",
         'COCO (yolov9c-seg.pt)'),
        ('Young cabbage', YOUNG_RUN, 'Field B', f"{len(young['train'])} Field B (first clip)", str(len(young['val'])),
         str(len(young['test'])), 'cabbage detector'),
        ('Cauliflower', CAULI_RUN, 'Field C', f"{len(cauli['train'])} sister flight of Field C", str(len(cauli['val'])),
         str(len(cauli['test'])), 'cabbage detector')]:
        a, (n_ep, best) = args(run), epochs(run)
        if name == 'Cabbage':
            t = cab_test['all']
            m = dict(P=t['mask_cab_p@0.5'], R=t['mask_cab_r@0.5'], AP50=t['mask_cab_ap50'], AP=t['mask_cab_ap'],
                     bP=t['box_cab_p@0.5'], bR=t['box_cab_r@0.5'], bAP50=t['box_cab_ap50'], bAP=t['box_cab_ap'], n=cab_inst['all'])
        else:
            t = ad[(ad.detector == {'Young cabbage': 'yolov9c-seg_s2_young', 'Cauliflower': 'yolov9c-seg_s2_cauli_v2'}[name]) & (ad.split == 'test')].iloc[0]
            m = dict(P=t['mask_P@0.5'], R=t['mask_R@0.5'], AP50=t['mask_AP50'], AP=t['mask_AP50_95'], bP=t['box_P@0.5'],
                     bR=t['box_R@0.5'], bAP50=t['box_AP50'], bAP=t['box_AP50_95'], n=int(t.plant_instances))
        rows.append(dict(name=name, used=used, start=start, train=train, val=val, test=test, epochs=a['epochs'], run=n_ep,
                         best=best, imgsz=a['imgsz'], batch=a['batch'], patience=a['patience'], seed=a['seed'], **m))
    J['detectors'] = [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in rows]
    pc = lambda x: f'{100 * x:.1f}'

    L = ['# Detectors - numbers for the paper',
         '',
         'Made by `python paper/detector_numbers.py` from the run folders, split folders and evaluation files. YOLOv9c-seg; '
         'plant = class 1 (Cabbage / Cauliflower); test scores with the Table 1 protocol (`detector/evaluate.py`, '
         '`detector/evaluate_adapted.py`): full-resolution masks, plant precision / recall at confidence 0.5 and IoU 0.5, '
         'AP from pycocotools. Test frames were never used for any choice.',
         '',
         '## 1. The three detectors',
         '',
         '| detector | used on | starts from | training frames | val | test | epochs run (best) | plant P | plant R | '
         'mask AP50 | mask AP50-95 | box AP50 | box AP50-95 | test plants |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['name']} | {r['used']} | {r['start']} | {r['train']} | {r['val']} | {r['test']} | {r['run']} ({r['best']}) | "
                 f"{pc(r['P'])} | {pc(r['R'])} | {pc(r['AP50'])} | {pc(r['AP'])} | {pc(r['bAP50'])} | {pc(r['bAP'])} | {r['n']} |")
    a0 = rows[0]
    L += ['',
          f"P and R = mask precision / recall (%). Same settings for all three: input {a0['imgsz']} px, batch {a0['batch']}, "
          f"optimizer auto (AdamW), seed {a0['seed']}, early stopping patience {a0['patience']}, best epoch by validation "
          f"fitness; training images augmented x3 (`detector/augment.py`) plus the Ultralytics online augmentation. "
          f"Epoch limit {rows[0]['epochs']} (cabbage) / {rows[1]['epochs']} (fine-tuning), never reached. "
          'Validation and test frames come in contiguous time blocks, not interleaved with training frames. '
          'The cabbage detector is the seed-2 run of the Table 1 benchmark (Table 1 reports the mean over three seeds).',
          '',
          '## 2. Cabbage detector, test frames by flight',
          '',
          '| test frames | images | plants | plant P | plant R | mask AP50 | mask AP50-95 |', '|---|---|---|---|---|---|---|']
    for k, t in cab_test.items():
        L.append(f"| {k} | {int(t.images)} | {cab_inst[k]} | {pc(t['mask_cab_p@0.5'])} | {pc(t['mask_cab_r@0.5'])} | "
                 f"{pc(t['mask_cab_ap50'])} | {pc(t['mask_cab_ap'])} |")
    L += ['',
          '## 3. Adapted detectors, all test frames and those >= 1 s (30 frames) from every training frame',
          '',
          '| detector | test frames | images | plants | plant P | plant R | mask AP50 | mask AP50-95 |',
          '|---|---|---|---|---|---|---|---|']
    for _, r in ad[ad.split != 'val'].iterrows():
        nm = 'Young cabbage' if 'young' in r.detector else 'Cauliflower'
        L.append(f"| {nm} | {'all' if r.split == 'test' else '>= 1 s from training'} | {r.images} | {r.plant_instances} | "
                 f"{pc(r['mask_P@0.5'])} | {pc(r['mask_R@0.5'])} | {pc(r.mask_AP50)} | {pc(r.mask_AP50_95)} |")
    L += ['',
          'Only the plant class is reported for the adapted detectors: the young-cabbage test frames hold 3 Divider outlines, '
          'and the cauliflower data has no Divider class.',
          '',
          f"Training sources: young cabbage = {len(ysrc[ysrc.annotated])} annotated frames, all from Field B's first clip "
          f"(whole-image correlation with the clip {ysrc[ysrc.annotated].ncc_loop.min():.3f}-{ysrc[ysrc.annotated].ncc_loop.max():.3f}; "
          '`data/detector/young_cabbage/source_check.csv`), so Field B is not independent of its detector. '
          f"Cauliflower = {len(csrc) } frames (frames {int(csrc.frame.min())}-{int(csrc.frame.max())}) of a sister flight over "
          f"the Field C crop; none is a Field C frame (highest correlation with any Field C frame "
          f"{csrc.max_ncc_vs_test_flight.max():.2f}; `data/detector/cauliflower/source_check.csv`).",
          '',
          '## 4. Held-out flight and the cabbage detector',
          '',
          f"The cabbage detector's annotated images include {sum(v for k, v in H['images_by_split'].items())} frames of the "
          f"held-out flight: {H['images_by_split'].get('train', 0)} training, {H['images_by_split'].get('val', 0)} validation, "
          f"{H['images_by_split'].get('test', 0)} test, {H['images_by_split'].get('excluded_gap', 0)} unused (gap frames). "
          f"They are frames of the held-out video itself: full-resolution correlation with their own video frame "
          f"{H['ncc_own_frame_min_max'][0]:.2f}-{H['ncc_own_frame_min_max'][1]:.2f}, with the frames 2 s either side at most "
          f"{H['ncc_pm2s_max']:.2f} ({H['images_checked']} images checked).",
          '',
          '| | value |', '|---|---|',
          f"| plant identities on the held-out flight (frozen model) | {H['identities']} |",
          f"| ... seen in at least one detector training image | {H['identities_in_training_image']} "
          f"({100 * H['identities_in_training_image'] / H['identities']:.0f} %) |",
          f"| ... seen in a training or validation image | {H['identities_in_training_or_val_image']} "
          f"({100 * H['identities_in_training_or_val_image'] / H['identities']:.0f} %) |",
          f"| flight frames within 1 s of a training frame | {100 * H['frames_within_1s_of_training_frame']:.0f} % |",
          f"| flight frames within 2 s of a training frame | {100 * H['frames_within_2s_of_training_frame']:.0f} % |",
          '',
          'So the held-out flight is held out for re-identification (the model and its settings were fixed on Fields A-C), '
          'not for detection.',
          '',
          '## Files',
          '`weights.md` (paths, SHA-1), `table1.md` (benchmark), `adapted_detectors_table1_protocol.csv`, '
          '`yolov9c-seg_s2_young_metrics.csv` / `yolov9c-seg_s2_cauli_v2_metrics.csv` (Ultralytics val, per class; not the '
          'Table 1 protocol), `detector_numbers.json` (everything above).']
    open(os.path.join(OUT, 'README.md'), 'w', encoding='utf8').write('\n'.join(L) + '\n')
    json.dump(J, open(os.path.join(OUT, 'detector_numbers.json'), 'w'), indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
