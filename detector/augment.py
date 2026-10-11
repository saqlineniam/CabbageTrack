"""
Offline augmentation of the TRAIN split only, reproducing the Roboflow recipe used for the original
dataset (drone-shot v11, "x3"): every training image is kept and gets N augmented copies with
  * horizontal flip (p 0.5) and vertical flip (p 0.5)
  * crop 0-20 % (random window, resized back to full frame)
  * rotation -15..+15 deg (black border)
  * brightness -20..+20 %
  * blur 0-2 px (Gaussian sigma)
Polygons are transformed with the image; objects cut by the frame are clipped and kept if at least
25 % of the object (and >= 20 px) remains visible. val/test are never augmented.
After augmentation <split>_coco.json is rebuilt from the YOLO labels (used by detectron2).
"""
import glob
import json
import os

import cv2
import numpy as np

MIN_VISIBLE = 0.25
MIN_AREA = 20


def _matrix(rng, W, H):
    M = np.eye(3)
    if rng.random() < 0.5:
        M = np.array([[-1, 0, W - 1], [0, 1, 0], [0, 0, 1]], float) @ M
    if rng.random() < 0.5:
        M = np.array([[1, 0, 0], [0, -1, H - 1], [0, 0, 1]], float) @ M
    R = np.vstack([cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-15, 15), 1.0), [0, 0, 1]])
    M = R @ M
    k = 1.0 - rng.uniform(0, 0.2)                        # keep fraction of the frame (crop 0-20 %)
    x0, y0 = rng.uniform(0, W * (1 - k)), rng.uniform(0, H * (1 - k))
    C = np.array([[1 / k, 0, -x0 / k], [0, 1 / k, -y0 / k], [0, 0, 1]])
    return C @ M, k


def _augment_one(img, objs, rng):
    H, W = img.shape[:2]
    M, k = _matrix(rng, W, H)
    A = M[:2]
    out = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    sigma = rng.uniform(0, 2.0)
    if sigma > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    new = []
    for c, pts in objs:
        full = abs(cv2.contourArea(pts.astype(np.float32))) / (k * k)
        if full < 1:
            continue
        tp = cv2.transform(pts[None].astype(np.float64), A)[0]
        x0, y0 = np.floor(tp.min(0)).astype(int) - 1
        x1, y1 = np.ceil(tp.max(0)).astype(int) + 1
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        if x0 >= 0 and y0 >= 0 and x1 < W and y1 < H:      # fully inside: keep the exact polygon
            new.append((c, tp))
            continue
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        m = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
        cv2.fillPoly(m, [np.round(tp - [cx0, cy0]).astype(np.int32)], 1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area < MIN_VISIBLE * full or len(cnt) < 3:
            continue
        cnt = cv2.approxPolyDP(cnt, 1.0, True)
        if len(cnt) < 3:
            continue
        new.append((c, cnt[:, 0].astype(float) + [cx0, cy0]))
    return out, new


def read_yolo(path, W, H):
    objs = []
    for line in open(path).read().split('\n'):
        v = line.split()
        if len(v) >= 7:
            objs.append((int(v[0]), np.array(v[1:], float).reshape(-1, 2) * [W, H]))
    return objs


def write_yolo_label(path, objs, W, H):
    lines = [f'{c} ' + ' '.join(f'{x / W:.6f} {y / H:.6f}' for x, y in np.clip(p, 0, [W, H])) for c, p in objs]
    open(path, 'w').write('\n'.join(lines) + ('\n' if lines else ''))


def coco_from_yolo(split_dir, classes, out_json):
    coco = {'images': [], 'annotations': [], 'categories': [{'id': i, 'name': c} for i, c in enumerate(classes)]}
    aid = 1
    for k, ip in enumerate(sorted(glob.glob(os.path.join(split_dir, 'images', '*.jpg'))), 1):
        H, W = cv2.imread(ip, cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[:2]
        H, W = H * 8, W * 8
        stem = os.path.splitext(os.path.basename(ip))[0]
        coco['images'].append({'id': k, 'file_name': stem + '.jpg', 'width': W, 'height': H})
        for c, p in read_yolo(os.path.join(split_dir, 'labels', stem + '.txt'), W, H):
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            coco['annotations'].append({'id': aid, 'image_id': k, 'category_id': c, 'iscrowd': 0,
                                        'area': float(abs(cv2.contourArea(p.astype(np.float32)))),
                                        'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                        'segmentation': [p.flatten().round(2).tolist()]})
            aid += 1
    json.dump(coco, open(out_json, 'w'))
    return coco


def augment_split(root, split, classes, copies=2, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, split)
    originals = sorted(p for p in glob.glob(os.path.join(d, 'images', '*.jpg')) if '_aug' not in p)
    for ip in originals:
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        objs = read_yolo(os.path.join(d, 'labels', stem + '.txt'), W, H)
        for j in range(1, copies + 1):
            out, new = _augment_one(img, objs, rng)
            cv2.imwrite(os.path.join(d, 'images', f'{stem}_aug{j}.jpg'), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo_label(os.path.join(d, 'labels', f'{stem}_aug{j}.txt'), new, W, H)
    return coco_from_yolo(d, classes, os.path.join(root, f'{split}_coco.json'))
