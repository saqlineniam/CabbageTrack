"""
Common evaluation protocol for every detector run (Ultralytics YOLO-seg / RT-DETR-L, detectron2 Mask R-CNN).

For each run x split (val, test, test_field1, test_field2), on the full-resolution 1920x1080 images:
  * COCO mAP@50 and mAP@50-95 (pycocotools, maxDets 300, all detections with score >= 0.001)
    for boxes and masks, macro-averaged over the 4 classes and for the Cabbage class alone;
  * precision / recall / F1 at confidence 0.5 and 0.7, IoU >= 0.5, greedy one-to-one matching,
    box and mask, Cabbage and macro;
  * end-to-end inference time per image (batch 1, FP32, 1080p BGR input -> boxes + masks, each
    framework's standard predict call at confidence 0.5, median over the test images after 3 warm-up
    images) and parameter count.
Each model uses its own native input size (Ultralytics imgsz 640; detectron2 short edge 800 / max 1333).
All masks are produced and compared with the ground truth at full 1920x1080 resolution (Ultralytics
retina_masks mode, computed in chunks; detectron2 pastes its masks at full resolution by default).
Results: outputs/detector/eval/metrics_long.csv (one row per run x split).

Usage: python detector/evaluate.py [--runs outputs/detector/runs] [--only yolov9c-seg_s0 ...]
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SPLIT = 'data/detector/cabbage/split'
CLASSES = ['Aisle', 'Cabbage', 'Divider', 'Drain']
CAB = 1
IMGSZ = 640
D2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'detectron2'))


def _process_mask_native_chunked(protos, masks_in, bboxes, shape, chunk=24):
    """Same maths as ultralytics.utils.ops.process_mask_native (retina masks), computed in chunks of
    detections so that 300 full-resolution masks fit in 8 GB of GPU memory."""
    from ultralytics.utils import ops
    c, mh, mw = protos.shape
    P = protos.float().view(c, -1)
    out = torch.zeros((len(masks_in), *shape), dtype=torch.uint8, device=protos.device)
    for i in range(0, len(masks_in), chunk):
        m = (masks_in[i:i + chunk] @ P).view(-1, mh, mw)
        m = ops.scale_masks(m[None], shape)[0]
        m = ops.crop_mask(m, bboxes[i:i + chunk])
        out[i:i + chunk] = m.gt_(0.0).byte()
    return out


def rle_encode(masks):
    """masks: (N, H, W) bool/uint8 numpy -> list of COCO RLE dicts."""
    if len(masks) == 0:
        return []
    rles = mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))
    for r in rles:
        r['counts'] = r['counts'].decode()
    return rles


class Predictor:
    def __init__(self, run_dir):
        self.name = os.path.basename(run_dir)
        model = self.name.rsplit('_s', 1)[0]
        if model.startswith('maskrcnn'):
            self.kind = 'detectron2'
            sys.path.insert(0, D2_ROOT)
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            cfg = get_cfg()
            cfg.merge_from_file(os.path.join(run_dir, 'config.yaml'))
            cfg.MODEL.WEIGHTS = os.path.join(run_dir, 'model_final.pth')
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
            cfg.TEST.DETECTIONS_PER_IMAGE = 300
            self.pred = DefaultPredictor(cfg)
            self.params = sum(p.numel() for p in self.pred.model.parameters())
        else:
            from ultralytics import YOLO, RTDETR
            from ultralytics.utils import ops
            ops.process_mask_native = _process_mask_native_chunked
            self.kind = 'rtdetr' if model.startswith('rtdetr') else 'yolo'
            w = os.path.join(run_dir, 'weights', 'best.pt')
            self.model = RTDETR(w) if self.kind == 'rtdetr' else YOLO(w)
            self.params = sum(p.numel() for p in self.model.model.parameters())
        self.has_masks = self.kind != 'rtdetr'

    @torch.no_grad()
    def __call__(self, img_bgr, conf=0.001, encode=True):
        """Return list of (cls, score, [x0,y0,w,h], rle_or_None) and the end-to-end inference ms
        (each framework's predict call, masks at full image resolution)."""
        H, W = img_bgr.shape[:2]
        t0 = time.perf_counter()
        if self.kind == 'detectron2':
            self.pred.model.roi_heads.box_predictor.test_score_thresh = conf
            ins = self.pred(img_bgr)['instances']
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            cls, sc = ins.pred_classes.cpu().numpy(), ins.scores.cpu().numpy()
            xyxy = ins.pred_boxes.tensor.cpu().numpy()
            masks = ins.pred_masks                                   # (N, H, W) bool, full resolution
        else:
            res = self.model.predict(img_bgr, imgsz=IMGSZ, conf=conf, iou=0.7, max_det=300, retina_masks=True,
                                     verbose=False)[0]
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            cls, sc = res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()
            xyxy = res.boxes.xyxy.cpu().numpy()
            masks = res.masks.data if (self.has_masks and res.masks is not None) else None   # (N, H, W) full res
        if not encode:
            return None, ms
        rles = self.full_res_rles(masks, H, W) if masks is not None else [None] * len(sc)
        out = [(int(cls[k]), float(sc[k]), [float(xyxy[k, 0]), float(xyxy[k, 1]), float(xyxy[k, 2] - xyxy[k, 0]),
                                            float(xyxy[k, 3] - xyxy[k, 1])], rles[k]) for k in range(len(sc))]
        return out, ms

    @staticmethod
    def full_res_rles(masks, H, W, chunk=32):
        """Masks -> full-resolution RLEs, in chunks to bound CPU memory (masks that are not yet at full
        resolution are un-padded, bilinearly resized and thresholded at 0.5)."""
        from ultralytics.utils import ops
        rles = []
        for i in range(0, len(masks), chunk):
            m = masks[i:i + chunk]
            if tuple(m.shape[-2:]) != (H, W):
                m = ops.scale_masks(m[None].float(), (H, W))[0] > 0.5
            rles += rle_encode(m.cpu().numpy())
        return rles


def coco_eval(gt, dets, iou_type):
    if not dets:
        return {'all_ap': 0.0, 'all_ap50': 0.0, 'cab_ap': 0.0, 'cab_ap50': 0.0}
    ev = COCOeval(gt, gt.loadRes(dets), iou_type)
    ev.params.maxDets = [1, 10, 300]
    ev.evaluate()
    ev.accumulate()
    # read AP from the precision array at maxDets=300 (ev.stats[0] would look for maxDets=100 and return -1)
    p = ev.eval['precision'][:, :, :, 0, -1]                 # [IoU, recall, class] at area 'all'
    k = ev.params.catIds.index(CAB)

    def mean(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float('nan')
    return {'all_ap': mean(p), 'all_ap50': mean(p[0]), 'cab_ap': mean(p[:, :, k]), 'cab_ap50': mean(p[0, :, k])}


def prf(gt, preds_by_img, thr, use_mask):
    """Greedy one-to-one matching at IoU >= 0.5 per image and class."""
    tp = np.zeros(len(CLASSES)); fp = np.zeros(len(CLASSES)); fn = np.zeros(len(CLASSES))
    for iid in gt.getImgIds():
        anns = gt.loadAnns(gt.getAnnIds(imgIds=iid))
        for c in range(len(CLASSES)):
            g = [a for a in anns if a['category_id'] == c]
            d = sorted([p for p in preds_by_img.get(iid, []) if p[0] == c and p[1] >= thr], key=lambda x: -x[1])
            if not g and not d:
                continue
            if g and d:
                if use_mask:
                    iou = mask_utils.iou([p[3] for p in d], [gt.annToRLE(a) for a in g], [0] * len(g))
                else:
                    iou = mask_utils.iou([p[2] for p in d], [a['bbox'] for a in g], [0] * len(g))
                iou = np.array(iou).reshape(len(d), len(g))
                used = np.zeros(len(g), bool)
                for i in range(len(d)):
                    cand = np.where(used, -1.0, iou[i])
                    j = int(np.argmax(cand))
                    if cand[j] >= 0.5:
                        used[j] = True; tp[c] += 1
                    else:
                        fp[c] += 1
                fn[c] += len(g) - used.sum()
            else:
                fp[c] += len(d); fn[c] += len(g)
    P = tp / np.maximum(tp + fp, 1); R = tp / np.maximum(tp + fn, 1)
    present = (tp + fn) > 0
    return {'cab_p': float(P[CAB]), 'cab_r': float(R[CAB]), 'cab_f1': float(2 * P[CAB] * R[CAB] / max(P[CAB] + R[CAB], 1e-9)),
            'macro_p': float(P[present].mean()), 'macro_r': float(R[present].mean())}


def timing(pred, n_warm=3):
    """End-to-end ms/image at the deployment threshold (conf 0.5), median over the test images."""
    files = sorted(glob.glob(os.path.join(SPLIT, 'test', 'images', '*.jpg')))
    times = []
    for k, f in enumerate(files):
        _, ms = pred(cv2.imread(f), conf=0.5, encode=False)
        if k >= n_warm:
            times.append(ms)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='outputs/detector/runs')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--splits', nargs='*', default=['val', 'test', 'test_field1', 'test_field2'])
    ap.add_argument('--out', default='outputs/detector/eval')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    long_path = os.path.join(a.out, 'metrics_long.csv')
    done = pd.read_csv(long_path) if os.path.exists(long_path) else pd.DataFrame()
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, '*')) if os.path.exists(os.path.join(d, 'DONE.json')))
    if a.only:
        runs = [r for r in runs if os.path.basename(r) in a.only]
    for rd in runs:
        name = os.path.basename(rd)
        if len(done) and (done.run == name).sum() >= len(a.splits) and not a.only:
            continue
        pred = Predictor(rd)
        ms_deploy = timing(pred)
        for sp in a.splits:
            gt = COCO(os.path.join(SPLIT, f'{sp}_coco.json'))
            dets, by_img = [], {}
            for iid in sorted(gt.getImgIds()):
                info = gt.loadImgs(iid)[0]
                out, _ = pred(cv2.imread(os.path.join(SPLIT, sp, 'images', info['file_name'])), conf=0.001)
                by_img[iid] = out
                for c, s, bb, rle in out:
                    d = {'image_id': iid, 'category_id': c, 'score': s, 'bbox': bb}
                    if rle is not None:
                        d['segmentation'] = rle
                    dets.append(d)
            row = {'run': name, 'model': name.rsplit('_s', 1)[0], 'seed': int(name.rsplit('_s', 1)[1]), 'split': sp,
                   'images': len(gt.getImgIds()), 'instances': len(gt.getAnnIds()), 'params_M': pred.params / 1e6,
                   'ms_per_img': ms_deploy}
            for k, v in coco_eval(gt, [{x: y for x, y in d.items() if x != 'segmentation'} for d in dets], 'bbox').items():
                row[f'box_{k}'] = v
            if pred.has_masks:
                for k, v in coco_eval(gt, dets, 'segm').items():
                    row[f'mask_{k}'] = v
            for thr in (0.5, 0.7):
                for k, v in prf(gt, by_img, thr, False).items():
                    row[f'box_{k}@{thr}'] = v
                if pred.has_masks:
                    for k, v in prf(gt, by_img, thr, True).items():
                        row[f'mask_{k}@{thr}'] = v
            done = pd.concat([done[~((done.run == name) & (done.split == sp))] if len(done) else done, pd.DataFrame([row])])
            done.to_csv(long_path, index=False)
            print(f"{name:24} {sp:12} box mAP50 {row['box_all_ap50']:.3f} mAP {row['box_all_ap']:.3f} | "
                  f"mask mAP50 {row.get('mask_all_ap50', float('nan')):.3f} mAP {row.get('mask_all_ap', float('nan')):.3f} | "
                  f"cab P/R@0.5 {row['box_cab_p@0.5']:.3f}/{row['box_cab_r@0.5']:.3f} | {ms_deploy:.1f} ms", flush=True)
        del pred
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
