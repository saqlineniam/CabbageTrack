"""
Mask R-CNN R50-FPN / R101-FPN (detectron2 model zoo, COCO-pretrained 3x) on the leakage-free split.

Solver settings are the ones in the original training notebook (Files/cabbage_yolo_model.ipynb,
cells 14 and 19); the only change is NUM_CLASSES = 4 (Aisle, Cabbage, Divider, Drain), the same
four classes the YOLO models learn (the original runs merged everything into one class).
  R50-FPN : IMS_PER_BATCH 4, BASE_LR 0.00025, MAX_ITER 1000, no LR decay (default 1000-iter warm-up)
  R101-FPN: IMS_PER_BATCH 4, BASE_LR 0.00025, MAX_ITER 3000, WARMUP_ITERS 200, STEPS (2000, 2500), GAMMA 0.1
  both    : ROI_HEADS.BATCH_SIZE_PER_IMAGE 128, default input sizes (short edge 640-800, max 1333),
            random horizontal flip, no validation during training, the final checkpoint is evaluated.
detectron2 is used from source (third_party/detectron2); its C++ extension is not needed for
Mask R-CNN (ROIAlign and NMS come from torchvision).

Usage: python detector/train_detectron2.py --config COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --seed 0 --out <dir>
"""
import argparse
import json
import os
import sys

D2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'detectron2'))
sys.path.insert(0, D2_ROOT)

from detectron2 import model_zoo                                   # noqa: E402
from detectron2.config import get_cfg                              # noqa: E402
from detectron2.data.datasets import register_coco_instances       # noqa: E402
from detectron2.engine import DefaultTrainer                       # noqa: E402
from detectron2.utils.env import seed_all_rng                      # noqa: E402

SPLIT = 'data/detector/cabbage/split'
NC = 4
SOLVER = {   # from the original notebook
    'mask_rcnn_R_50_FPN_3x': dict(MAX_ITER=1000, STEPS=(), WARMUP_ITERS=1000, GAMMA=0.1),
    'mask_rcnn_R_101_FPN_3x': dict(MAX_ITER=3000, STEPS=(2000, 2500), WARMUP_ITERS=200, GAMMA=0.1),
}


def register():
    for s in ('train', 'val', 'test', 'test_field1', 'test_field2'):
        name = f'cab_{s}'
        if os.path.exists(os.path.join(SPLIT, f'{s}_coco.json')):
            register_coco_instances(name, {}, os.path.join(SPLIT, f'{s}_coco.json'), os.path.join(SPLIT, s, 'images'))


def build_cfg(config, seed=0, batch=4, out='.', max_iter_scale=1.0):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(D2_ROOT, 'configs', config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NC
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.DATASETS.TRAIN = ('cab_train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    s = SOLVER[os.path.splitext(os.path.basename(config))[0]]
    cfg.SOLVER.IMS_PER_BATCH = batch
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max(1, int(round(s['MAX_ITER'] * max_iter_scale)))
    cfg.SOLVER.STEPS = tuple(int(round(x * max_iter_scale)) for x in s['STEPS'])
    cfg.SOLVER.WARMUP_ITERS = max(1, int(round(s['WARMUP_ITERS'] * max_iter_scale)))
    cfg.SOLVER.GAMMA = s['GAMMA']
    cfg.SEED = seed
    cfg.CUDNN_BENCHMARK = False
    cfg.OUTPUT_DIR = out
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max-iter-scale', type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    register()
    seed_all_rng(a.seed)
    cfg = build_cfg(a.config, a.seed, a.batch, a.out, a.max_iter_scale)
    open(os.path.join(a.out, 'config.yaml'), 'w').write(cfg.dump())
    json.dump(vars(a) | {'detectron2_root': D2_ROOT}, open(os.path.join(a.out, 'args.json'), 'w'), indent=2)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    main()
