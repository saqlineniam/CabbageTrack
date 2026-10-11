"""
Train the six Table-1 detector configurations with 3 seeds on the leakage-free split (resumable).

Hyper-parameters are copied from the original training notebook (Files/cabbage_yolo_model.ipynb)
and args.yaml files:
  * Ultralytics (YOLOv8s-seg, YOLO11s-seg, YOLOv9c-seg, RT-DETR-L): imgsz 640, epochs 400,
    patience 50, optimizer 'auto', batch 16/16/8/4, default online augmentation.
    RT-DETR-L is the model the submitted manuscript calls "RF-DETR".
  * detectron2 Mask R-CNN R50-FPN and R101-FPN (COCO-pretrained 3x model-zoo configs), solver
    settings as in the notebook (see detector/train_detectron2.py), now with all 4 classes.
Differences by design: no Roboflow offline x3 augmentation (the split is made on raw images),
seeds 0, 1, 2.

Usage: python detector/train_queue.py [--only yolov9c-seg ...] [--seeds 0 1 2] [--root outputs/detector/runs]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

MODELS = {   # name: (weights or model-zoo config, batch, framework)
    'yolov8s-seg': ('yolov8s-seg.pt', 16, 'ultralytics'),
    'yolo11s-seg': ('yolo11s-seg.pt', 16, 'ultralytics'),
    'yolov9c-seg': ('yolov9c-seg.pt', 8, 'ultralytics'),
    'rtdetr-l': ('rtdetr-l.pt', 4, 'ultralytics'),
    'maskrcnn-r50-fpn': ('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 4, 'detectron2'),
    'maskrcnn-r101-fpn': ('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml', 4, 'detectron2'),
}
DATA = 'data/detector/cabbage/split/data.yaml'
ROOT = 'outputs/detector/runs'
LOG = 'outputs/detector/train_queue.log'


def log(msg):
    os.makedirs(os.path.dirname(LOG) or '.', exist_ok=True)
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {msg}'
    print(line, flush=True)
    open(LOG, 'a').write(line + '\n')


def train_ultralytics(name, weights, batch, seed, epochs, patience, root, retries=60):
    """Train, or resume an interrupted run from weights/last.pt. On Windows a checkpoint write can be
    refused (PermissionError) while the file is open in Excel or being scanned; then wait and resume."""
    from ultralytics import YOLO, RTDETR
    cls = RTDETR if name.startswith('rtdetr') else YOLO
    last = os.path.join(root, f'{name}_s{seed}', 'weights', 'last.pt')
    for attempt in range(retries + 1):
        try:
            if os.path.exists(last):
                log(f'  resuming {name} seed {seed} from {last}')
                cls(last).train(resume=True, workers=2)
            else:
                cls(weights).train(data=DATA, epochs=epochs, patience=patience, imgsz=640, batch=batch, seed=seed,
                                   deterministic=True, workers=2, project=os.path.abspath(root), name=f'{name}_s{seed}',
                                   exist_ok=True, plots=True, verbose=False)
            return
        except AssertionError as e:
            if 'nothing to resume' in str(e):           # last.pt is from a finished run
                return
            raise
        except PermissionError:
            log(f'  PermissionError (attempt {attempt + 1}), close any file of this run that is open in Excel; resuming in 60 s:\n{traceback.format_exc()}')
            time.sleep(60)
    raise RuntimeError(f'{name} seed {seed}: gave up after {retries} PermissionErrors')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--max-iter-scale', type=float, default=1.0, help='detectron2 only; <1 for smoke tests')
    ap.add_argument('--root', default=ROOT)
    a = ap.parse_args()
    jobs = [(m, s) for s in a.seeds for m in MODELS if a.only is None or m in a.only]
    log(f'queue: {len(jobs)} jobs {jobs}')
    for name, seed in jobs:
        out = os.path.join(a.root, f'{name}_s{seed}')
        done_flag = os.path.join(out, 'DONE.json')
        if os.path.exists(done_flag):
            log(f'skip {name} seed {seed} (done)')
            continue
        weights, batch, framework = MODELS[name]
        t0 = time.time()
        log(f'start {name} seed {seed}')
        try:
            if framework == 'detectron2':
                subprocess.run([sys.executable, 'detector/train_detectron2.py', '--config', weights, '--seed', str(seed),
                                '--out', out, '--batch', str(batch), '--max-iter-scale', str(a.max_iter_scale)], check=True)
            else:
                train_ultralytics(name, weights, batch, seed, a.epochs, a.patience, a.root)
        except Exception as e:
            log(f'FAILED {name} seed {seed}: {e!r}\n{traceback.format_exc()}')
            continue
        json.dump({'model': name, 'seed': seed, 'minutes': round((time.time() - t0) / 60, 1)}, open(done_flag, 'w'))
        log(f'done {name} seed {seed} in {(time.time() - t0) / 60:.1f} min')
    log('queue finished')


if __name__ == '__main__':
    main()
