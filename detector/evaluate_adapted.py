"""Score the adapted detectors (young cabbage, cauliflower) with the Table 1 protocol of detector/evaluate.py
(full-resolution masks; plant P/R at confidence 0.5, IoU 0.5; AP from pycocotools).
Usage: python detector/evaluate_adapted.py outputs/detector/adapted_detectors_table1_protocol.csv          (published weights)
       python detector/evaluate_adapted.py out.csv --runs     (your retrained runs in outputs/detector/young|cauliflower)"""
import os, sys, glob, json, cv2, numpy as np, pandas as pd
sys.path.insert(0, 'detector')
import evaluate as E
from pycocotools.coco import COCO

def yolo_to_coco(img_paths):
    coco = {'images': [], 'annotations': [], 'categories': [{'id': i, 'name': c} for i, c in enumerate(E.CLASSES)]}
    aid = 1
    for k, p in enumerate(img_paths, 1):
        H, W = cv2.imread(p).shape[:2]
        coco['images'].append({'id': k, 'file_name': p, 'width': W, 'height': H})
        lab = p.replace(chr(92), '/').replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        for line in open(lab).read().split('\n'):
            v = line.split()
            if len(v) < 7:
                continue
            xy = np.array(v[1:], float).reshape(-1, 2) * [W, H]
            x0, y0 = xy.min(0); x1, y1 = xy.max(0)
            area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
            coco['annotations'].append({'id': aid, 'image_id': k, 'category_id': int(v[0]), 'iscrowd': 0, 'area': float(area),
                                        'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                        'segmentation': [xy.ravel().tolist()]})
            aid += 1
    gt = COCO(); gt.dataset = coco; gt.createIndex()
    return gt

def published(name, weights):
    """Predictor for a published weights file (same settings as for a training run)."""
    from ultralytics import YOLO
    from ultralytics.utils import ops
    ops.process_mask_native = E._process_mask_native_chunked
    p = object.__new__(E.Predictor)
    p.name, p.kind, p.has_masks = name, 'yolo', True
    p.model = YOLO(weights)
    p.params = sum(x.numel() for x in p.model.model.parameters())
    return p


RETRAINED = '--runs' in sys.argv
rows = []
for run, weights, data in [('outputs/detector/young/yolov9c-seg_s2_young', 'data/weights/young_cabbage.pt', 'data/detector/young_cabbage/split'),
                           ('outputs/detector/cauliflower/yolov9c-seg_s2_cauli_v2', 'data/weights/cauliflower.pt', 'data/detector/cauliflower/split')]:
    pred = E.Predictor(run) if RETRAINED else published(os.path.basename(run), weights)
    far = [f"{data}/test/images/{os.path.basename(l.strip().replace(chr(92), '/'))}" for l in open(f'{data}/test_far.txt') if l.strip()]
    for sp, imgs in [('val', sorted(glob.glob(f'{data}/val/images/*.jpg'))), ('test', sorted(glob.glob(f'{data}/test/images/*.jpg'))),
                     ('test_far (>=30 frames from training)', far)]:
        gt = yolo_to_coco(imgs)
        dets, by = [], {}
        for iid in sorted(gt.getImgIds()):
            out, _ = pred(cv2.imread(gt.loadImgs(iid)[0]['file_name']), conf=0.001)
            by[iid] = out
            dets += [{'image_id': iid, 'category_id': c, 'score': s, 'bbox': bb, 'segmentation': r} for c, s, bb, r in out]
        mb = E.coco_eval(gt, [{x: y for x, y in d.items() if x != 'segmentation'} for d in dets], 'bbox')
        mm = E.coco_eval(gt, dets, 'segm')
        pb, pm = E.prf(gt, by, 0.5, False), E.prf(gt, by, 0.5, True)
        n_pl = len(gt.getAnnIds(catIds=[1]))
        rows.append({'detector': os.path.basename(run), 'split': sp, 'images': len(imgs), 'plant_instances': n_pl,
                     'box_P@0.5': pb['cab_p'], 'box_R@0.5': pb['cab_r'], 'box_AP50': mb['cab_ap50'], 'box_AP50_95': mb['cab_ap'],
                     'mask_P@0.5': pm['cab_p'], 'mask_R@0.5': pm['cab_r'], 'mask_AP50': mm['cab_ap50'], 'mask_AP50_95': mm['cab_ap']})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).round(4).to_csv(sys.argv[1], index=False)
