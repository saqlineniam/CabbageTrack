"""
Summarise the detector benchmark (after detector/train_queue.py and detector/evaluate.py).

Outputs in outputs/detector/:
  table1_<split>.csv / table1.md  mean +- SD over seeds, paper Table-1 layout (all-class macro and Cabbage only)
  per_seed_<split>.csv            every run
  protocol.csv / protocol.md      what was trained and how (read back from each run's own config/log)
  environment.json                software / hardware versions
  split_summary.csv, split_manifest.csv   copied from data/detector/cabbage/split (which is not versioned)
"""
import glob
import json
import os
import platform
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

OUT = 'outputs/detector'
RUNS = os.path.join(OUT, 'runs')
QUEUE_OUT = 'outputs/detector_queue.out'
SPLITS = {'test': 'Test: both fields',
          'test_field1': 'Test: field 1 only (younger plants)',
          'test_field2': 'Test: field 2 only (bigger plants, congested rows)',
          'val': 'Validation: both fields (model selection only)'}
ORDER = ['yolov8s-seg', 'yolo11s-seg', 'yolov9c-seg', 'rtdetr-l', 'maskrcnn-r50-fpn', 'maskrcnn-r101-fpn']
LABEL = {'yolov8s-seg': 'YOLOv8s-seg', 'yolo11s-seg': 'YOLO11s-seg', 'yolov9c-seg': 'YOLOv9c-seg',
         'rtdetr-l': 'RT-DETR-L', 'maskrcnn-r50-fpn': 'Mask R-CNN R50-FPN', 'maskrcnn-r101-fpn': 'Mask R-CNN R101-FPN'}
SUPPLEMENTARY = ['maskrcnn-r50-fpn']    # original 1,000-iteration schedule undertrains the rare classes (see notes)
MAIN = [m for m in ORDER if m not in SUPPLEMENTARY]
N_TRAIN = len(os.listdir('data/detector/cabbage/split/train/images')) if os.path.isdir('data/detector/cabbage/split/train/images') else 128


def pm(x, scale=100.0, nd=1):
    x = np.asarray(x, float) * scale
    x = x[~np.isnan(x)]
    if not len(x):
        return '–'
    return f'{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}' if len(x) > 1 else f'{x.mean():.{nd}f}'


def table(df, split, models=None):
    rows = []
    d = df[df.split == split]
    for m in (models or MAIN):
        g = d[d.model == m]
        if not len(g):
            continue
        for out in ('box', 'mask'):
            if f'{out}_all_ap' not in g or g[f'{out}_all_ap'].isna().all():
                continue
            rows.append({'Model': LABEL[m], 'Output': out.capitalize(), 'Seeds': len(g),
                         'P (%)': pm(g[f'{out}_macro_p@0.5']), 'R (%)': pm(g[f'{out}_macro_r@0.5']),
                         'mAP@50 (%)': pm(g[f'{out}_all_ap50']), 'mAP@50–95 (%)': pm(g[f'{out}_all_ap']),
                         'Cabbage P (%)': pm(g[f'{out}_cab_p@0.5']), 'Cabbage R (%)': pm(g[f'{out}_cab_r@0.5']),
                         'Cabbage AP@50 (%)': pm(g[f'{out}_cab_ap50']), 'Cabbage AP@50–95 (%)': pm(g[f'{out}_cab_ap']),
                         'Cabbage P@0.7 (%)': pm(g[f'{out}_cab_p@0.7']), 'Cabbage R@0.7 (%)': pm(g[f'{out}_cab_r@0.7']),
                         'Params (M)': f"{g.params_M.mean():.1f}", 'ms / image': pm(g.ms_per_img, 1.0, 1)})
    return pd.DataFrame(rows)


def md(t, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for _, r in t.iterrows()]
    return '\n'.join(lines)


def optimizer_lines():
    """Map (model, seed) -> the optimizer line Ultralytics printed (optimizer='auto' picks it)."""
    found, cur = {}, None
    if not os.path.exists(QUEUE_OUT):
        return found
    for line in open(QUEUE_OUT, encoding='utf8', errors='ignore'):
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)          # strip terminal colour codes
        m = re.search(r'start (\S+) seed (\d+)', line)
        if m:
            cur = (m.group(1), int(m.group(2)))
        m = re.search(r'optimizer: (\w+\(lr=[^)]*\))', line)
        if m and cur and cur not in found:
            found[cur] = m.group(1)
    return found


def protocol():
    opt = optimizer_lines()
    rows = []
    for m in ORDER:
        runs = sorted(glob.glob(os.path.join(RUNS, f'{m}_s*')))
        runs = [r for r in runs if os.path.exists(os.path.join(r, 'DONE.json'))]
        if not runs:
            continue
        mins = [json.load(open(os.path.join(r, 'DONE.json'))).get('minutes', np.nan) for r in runs]
        seeds = [int(r.rsplit('_s', 1)[1]) for r in runs]
        if m.startswith('maskrcnn'):
            c = yaml.safe_load(open(os.path.join(runs[0], 'config.yaml')))
            s = c['SOLVER']
            it, b = s['MAX_ITER'], s['IMS_PER_BATCH']
            rows.append({'Model': LABEL[m], 'Framework': 'detectron2 (model zoo)',
                         'Initialisation': 'COCO ' + os.path.basename(os.path.dirname(os.path.dirname(c['MODEL']['WEIGHTS']))),
                         'Input size': f"short edge {c['INPUT']['MIN_SIZE_TRAIN']} (train) / {c['INPUT']['MIN_SIZE_TEST']} (test), max {c['INPUT']['MAX_SIZE_TRAIN']}",
                         'Schedule': f"{it} iterations (≈{it * b / N_TRAIN:.0f} epochs), warm-up {s['WARMUP_ITERS']}, "
                                     f"LR steps {list(s['STEPS']) or 'none'} ×{s['GAMMA']}",
                         'Batch': b, 'Optimiser': f"SGD lr {s['BASE_LR']}, momentum {s['MOMENTUM']}, wd {s['WEIGHT_DECAY']}",
                         'Augmentation': 'random horizontal flip, multi-scale short edge',
                         'Model selection': 'final checkpoint (no validation during training, as in the original runs)',
                         'Epochs run': f'{it * b / N_TRAIN:.0f} (fixed)', 'Seeds': ', '.join(map(str, seeds)),
                         'Train time (min)': pm(mins, 1.0, 0)})
        else:
            a = yaml.safe_load(open(os.path.join(runs[0], 'args.yaml')))
            ep_run, best = [], []
            for r in runs:
                res = pd.read_csv(os.path.join(r, 'results.csv'))
                res.columns = [c.strip() for c in res.columns]
                fit = 0.1 * res['metrics/mAP50(B)'] + 0.9 * res['metrics/mAP50-95(B)']
                if 'metrics/mAP50(M)' in res:
                    fit = fit + 0.1 * res['metrics/mAP50(M)'] + 0.9 * res['metrics/mAP50-95(M)']
                ep_run.append(len(res))
                best.append(int(res.epoch.iloc[int(fit.values.argmax())]))
            augs = (f"mosaic {a['mosaic']} (off last {a['close_mosaic']} ep), flip-LR {a['fliplr']}, flip-UD {a['flipud']}, "
                    f"HSV {a['hsv_h']}/{a['hsv_s']}/{a['hsv_v']}, translate {a['translate']}, scale {a['scale']}, "
                    f"degrees {a['degrees']}, mixup {a.get('mixup', 0)}, copy-paste {a.get('copy_paste', 0)}")
            o = sorted({opt.get((m, s), '') for s in seeds} - {''})
            rows.append({'Model': LABEL[m], 'Framework': 'Ultralytics', 'Initialisation': f"COCO {a['model']}",
                         'Input size': f"{a['imgsz']} px (long side)",
                         'Schedule': f"up to {a['epochs']} epochs, early stopping patience {a['patience']}",
                         'Batch': a['batch'], 'Optimiser': f"optimizer={a['optimizer']} → {'; '.join(o) if o else 'see log'}",
                         'Augmentation': augs, 'Model selection': 'best validation fitness (0.1·mAP50 + 0.9·mAP50–95)',
                         'Epochs run': ' / '.join(f'{e} (best {b})' for e, b in zip(ep_run, best)),
                         'Seeds': ', '.join(map(str, seeds)), 'Train time (min)': pm(mins, 1.0, 0)})
    return pd.DataFrame(rows)


def environment():
    import torch
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'os': platform.platform()}
    try:
        import ultralytics
        env['ultralytics'] = ultralytics.__version__
    except Exception:
        pass
    d2 = os.path.abspath(os.path.join('third_party', 'detectron2'))
    try:
        env['detectron2_commit'] = subprocess.check_output(['git', '-C', d2, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        pass
    try:
        env['nvidia_driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                                       text=True).strip()
    except Exception:
        pass
    return env


def main():
    df = pd.read_csv(os.path.join(OUT, 'eval', 'metrics_long.csv'))
    blocks = ['# Detector benchmark (both fields; val/test kept 1 s away from training frames; mean ± SD over seeds)\n',
              'P / R at confidence 0.5 and IoU 0.5; mAP from pycocotools (maxDets 300); all classes = macro average '
              'over Aisle, Cabbage, Divider, Drain. Speed: end-to-end ms per 1920×1080 image, batch 1, FP32, '
              f"{environment().get('gpu')}.\n"]
    for sp, title in SPLITS.items():
        t = table(df, sp)
        if not len(t):
            continue
        t.to_csv(os.path.join(OUT, f'table1_{sp}.csv'), index=False)
        df[df.split == sp].sort_values(['model', 'seed']).to_csv(os.path.join(OUT, f'per_seed_{sp}.csv'), index=False)
        n_img, n_inst = int(df[df.split == sp].images.iloc[0]), int(df[df.split == sp].instances.iloc[0])
        blocks.append(f'\n## {title} — {n_img} images, {n_inst} instances\n')
        blocks.append('All classes:\n\n' + md(t, ['Model', 'Output', 'Seeds', 'P (%)', 'R (%)', 'mAP@50 (%)', 'mAP@50–95 (%)',
                                                   'Params (M)', 'ms / image']) + '\n')
        blocks.append('Cabbage only:\n\n' + md(t, ['Model', 'Output', 'Cabbage P (%)', 'Cabbage R (%)', 'Cabbage AP@50 (%)',
                                                    'Cabbage AP@50–95 (%)', 'Cabbage P@0.7 (%)', 'Cabbage R@0.7 (%)']) + '\n')
    # manuscript-ready Table 1: test set (both fields), all classes, with model size and speed
    t = table(df, 'test')
    d = df[df.split == 'test']
    fps = {LABEL[m]: 1000 / d[d.model == m].ms_per_img.mean() for m in MAIN if (d.model == m).any()}
    t['FPS'] = t.Model.map(lambda x: f'{fps[x]:.0f}')
    cols = ['Model', 'Output', 'P (%)', 'R (%)', 'mAP@50 (%)', 'mAP@50–95 (%)', 'Params (M)', 'ms / image', 'FPS']
    t[cols].to_csv(os.path.join(OUT, 'table1_manuscript.csv'), index=False)
    open(os.path.join(OUT, 'table1_manuscript.md'), 'w', encoding='utf8').write(
        '# Table 1 (manuscript): detector benchmark on the held-out test set (34 images, both fields), mean ± SD over 3 seeds\n\n'
        + md(t, cols) + '\n\nP / R at confidence 0.5, IoU 0.5; mAP macro-averaged over Aisle, Cabbage, Divider, Drain (pycocotools, '
        f"maxDets 300). Speed: end-to-end inference per 1920×1080 image (pre-processing, network, NMS, full-resolution masks), "
        f"batch 1, FP32, {environment().get('gpu')}; FPS = 1000 / mean ms. RT-DETR-L predicts boxes only. "
        'Mask R-CNN R50-FPN: Supplementary Table S1.\n')
    blocks.append('\n## Supplementary Table S1 — Mask R-CNN R50-FPN (original schedule), not part of the main comparison\n')
    for sp in SPLITS:
        t = table(df, sp, SUPPLEMENTARY)
        if len(t):
            t.to_csv(os.path.join(OUT, f'tableS1_{sp}.csv'), index=False)
            blocks.append(f'{SPLITS[sp]}:\n\n' + md(t, ['Model', 'Output', 'Seeds', 'P (%)', 'R (%)', 'mAP@50 (%)', 'mAP@50–95 (%)',
                                                      'Cabbage AP@50 (%)', 'Cabbage AP@50–95 (%)', 'ms / image']) + '\n')
    blocks.append('\n## Notes for the manuscript (Table 1 footnotes)\n')
    blocks.append('- Mask R-CNN R50-FPN (Supplementary Table S1) was trained with the schedule of the original submission (1,000 '
                  'iterations at batch 4, ≈10 epochs, with a 1,000-iteration linear warm-up, so the learning rate never reached its base '
                  'value). This short schedule is sufficient for the abundant Cabbage class (Cabbage AP@50 ≈ 98 %) but not for the rare '
                  'non-crop classes: Aisle and Drain scores mostly stay below the 0.5 confidence threshold (Divider partly). Because the '
                  'comparison would reflect the schedule rather than the architecture, it is excluded from Table 1 and from the statistical '
                  'tests; Mask R-CNN is represented by R101-FPN (3,000 iterations), which does not show this.\n'
                  '- "RT-DETR-L" is the model reported as "RF-DETR" in the original submission (Ultralytics rtdetr-l.pt); '
                  'it predicts boxes only.\n'
                  '- Box and mask precision/recall at IoU 0.5 are nearly identical because a correctly detected plant passes the '
                  '0.5 overlap test with both its box and its mask; the outputs differ at stricter overlaps (mAP@50–95).\n'
                  '- All-class field-2 values rest on few non-cabbage objects (6 aisles, 3 dividers, 21 drains in 7 test images) '
                  'and are therefore noisy; Cabbage-only values are the reliable comparison there.\n')
    open(os.path.join(OUT, 'table1.md'), 'w', encoding='utf8').write('\n'.join(blocks))
    p = protocol()
    p.to_csv(os.path.join(OUT, 'protocol.csv'), index=False)
    open(os.path.join(OUT, 'protocol.md'), 'w', encoding='utf8').write('# Training protocol\n\n' + md(p, list(p.columns)) + '\n')
    json.dump(environment(), open(os.path.join(OUT, 'environment.json'), 'w'), indent=2)
    for f in ('split_summary.csv', 'split_manifest.csv'):
        if os.path.exists(os.path.join('data/detector/cabbage/split', f)):
            shutil.copy(os.path.join('data/detector/cabbage/split', f), os.path.join(OUT, f))
    print(open(os.path.join(OUT, 'table1.md'), encoding='utf8').read())


if __name__ == '__main__':
    main()
