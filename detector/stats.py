"""
Statistical comparison of the detector benchmark (run after evaluate.py).

1. Seed variability: for every model and metric, mean, SD, range and coefficient of variation over the
   seeds; one-way ANOVA across models on the seed-level test values (F, p, eta^2 = share of the variance
   explained by the choice of model rather than the seed), for all models and among the three YOLO-seg
   models only. Models = the Table-1 main comparison (summarize.MAIN; Mask R-CNN R50-FPN is supplementary).
2. Pairwise model differences, hierarchical paired bootstrap (B draws): test images are resampled with
   replacement within each field (keeps the field-1 / field-2 mix) and seeds are resampled with
   replacement within each model; COCO AP is recomputed exactly for every draw (pycocotools matching,
   accumulation re-implemented so it can run on a resampled image set). Reported: observed difference of
   the seed means, 95 % percentile CI, two-sided bootstrap p, Holm-adjusted p (within each metric).
3. Friedman test over models on per-image Cabbage F1 (box, confidence 0.5, IoU 0.5, mean over seeds),
   Wilcoxon signed-rank post-hoc tests with Holm correction.

Detections of every run on the test images are cached in outputs/detector/dets/ (regenerable).
Outputs: outputs/detector/stats/*.csv and outputs/detector/stats/stats.md
Usage: python detector/stats.py [--B 2000]
"""
import argparse
import glob
import gzip
import itertools
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import stats as st

sys.path.insert(0, os.path.dirname(__file__))
from summarize import LABEL, MAIN   # noqa: E402   (Table-1 main comparison; Mask R-CNN R50-FPN is supplementary)

SPLIT = 'data/detector/cabbage/split'
RUNS = 'outputs/detector/runs'
DETS = 'outputs/detector/dets'
OUT = 'outputs/detector/stats'
CAB = 1
METRICS = {   # name: (iou type, quantity)
    'Box mAP@50–95 (all classes)': ('bbox', 'all_ap'),
    'Box mAP@50 (all classes)': ('bbox', 'all_ap50'),
    'Mask mAP@50–95 (all classes)': ('segm', 'all_ap'),
    'Mask mAP@50 (all classes)': ('segm', 'all_ap50'),
    'Cabbage box AP@50–95': ('bbox', 'cab_ap'),
    'Cabbage mask AP@50–95': ('segm', 'cab_ap'),
    'Cabbage F1 @0.5 (box)': ('bbox', 'cab_f1'),
}
SPLITS = {'test': 'both fields', 'test_field1': 'field 1', 'test_field2': 'field 2'}
ANOVA_SETS = {'all models': None,
              'YOLO-seg only': ['yolov8s-seg', 'yolo11s-seg', 'yolov9c-seg']}


# ---------------------------------------------------------------- detections
def predict_all(runs, gt):
    import torch
    import evaluate as E
    os.makedirs(DETS, exist_ok=True)
    for rd in runs:
        name = os.path.basename(rd)
        path = os.path.join(DETS, f'{name}_test.json.gz')
        if os.path.exists(path):
            continue
        pred, dets = E.Predictor(rd), []
        for iid in sorted(gt.getImgIds()):
            out, _ = pred(cv2.imread(os.path.join(SPLIT, 'test', 'images', gt.loadImgs(iid)[0]['file_name'])), conf=0.001)
            for c, s, bb, rle in out:
                d = {'image_id': iid, 'category_id': c, 'score': s, 'bbox': bb}
                if rle is not None:
                    d['segmentation'] = rle
                dets.append(d)
        json.dump(dets, gzip.open(path, 'wt'))
        print(f'  detections saved: {name} ({len(dets)})', flush=True)
        del pred
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- COCO per-image evaluation + own accumulation
def eval_imgs(gt, dets, iou_type):
    """pycocotools per-image matching (area 'all', maxDets 300) -> evalImgs[k][i] (None when empty)."""
    if iou_type == 'bbox':
        dets = [{k: v for k, v in d.items() if k != 'segmentation'} for d in dets]
    else:
        dets = [dict(d) for d in dets if 'segmentation' in d]
    if not dets:
        return None
    ev = COCOeval(gt, gt.loadRes(dets), iou_type)
    ev.params.maxDets = [1, 10, 300]
    ev.evaluate()
    p = ev.params
    I, A = len(p.imgIds), len(p.areaRng)
    return [[ev.evalImgs[k * A * I + i] for i in range(I)] for k in range(len(p.catIds))], p.iouThrs, p.recThrs


def precision(entries, iou_thrs, rec_thrs):
    """COCO interpolated precision [T, R] for one category over a multiset of per-image entries (None = no GT)."""
    es = [e for e in entries if e is not None]
    if not es:
        return None
    gt_ig = np.concatenate([e['gtIgnore'] for e in es])
    npig = np.count_nonzero(gt_ig == 0)
    if npig == 0:
        return None
    sc = np.concatenate([e['dtScores'] for e in es])
    order = np.argsort(-sc, kind='mergesort')
    dtm = np.concatenate([e['dtMatches'] for e in es], axis=1)[:, order]
    dt_ig = np.concatenate([e['dtIgnore'] for e in es], axis=1)[:, order]
    tp = np.cumsum(np.logical_and(dtm, ~dt_ig.astype(bool)), 1).astype(float)
    fp = np.cumsum(np.logical_and(~dtm.astype(bool), ~dt_ig.astype(bool)), 1).astype(float)
    out = np.zeros((len(iou_thrs), len(rec_thrs)))
    for t in range(len(iou_thrs)):
        rc = tp[t] / npig
        pr = tp[t] / (fp[t] + tp[t] + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, rec_thrs, side='left')
        ok = idx < len(pr)
        out[t, ok] = pr[idx[ok]]
    return out


def counts_at(e, thr=0.5):
    """Cabbage TP / FP / FN of one image at confidence thr, IoU 0.5 (COCO greedy matching)."""
    if e is None:
        return 0, 0, 0
    keep = np.asarray(e['dtScores']) >= thr
    m, ig = e['dtMatches'][0][keep], e['dtIgnore'][0][keep].astype(bool)
    tp = int(np.count_nonzero((m > 0) & ~ig))
    fp = int(np.count_nonzero((m == 0) & ~ig))
    return tp, fp, int(np.count_nonzero(np.asarray(e['gtIgnore']) == 0)) - tp


def run_metrics(R, sel):
    """All metrics of one run on a multiset of image indices `sel`."""
    out = {}
    for typ in ('bbox', 'segm'):
        if R[typ] is None:
            continue
        E, it, rt = R[typ]
        per = [precision([E[k][i] for i in sel], it, rt) for k in range(len(E))]
        valid = [p for p in per if p is not None]
        out[f'{typ}_all_ap'] = float(np.mean([p.mean() for p in valid]))
        out[f'{typ}_all_ap50'] = float(np.mean([p[0].mean() for p in valid]))
        out[f'{typ}_cab_ap'] = float(per[CAB].mean()) if per[CAB] is not None else np.nan
    tp, fp, fn = np.sum([R['counts'][i] for i in sel], axis=0)
    out['bbox_cab_f1'] = 2 * tp / max(2 * tp + fp + fn, 1)
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - r) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    gt = COCO(os.path.join(SPLIT, 'test_coco.json'))
    img_ids = sorted(gt.getImgIds())
    field = np.array(['field2' if gt.loadImgs(i)[0]['file_name'].startswith('DJI_0004') else 'field1' for i in img_ids])
    runs = sorted(d for d in glob.glob(os.path.join(RUNS, '*')) if os.path.exists(os.path.join(d, 'DONE.json'))
                  and os.path.basename(d).rsplit('_s', 1)[0] in MAIN)
    predict_all(runs, gt)

    # per-run evaluation entries
    R, models = {}, {}
    for rd in runs:
        name = os.path.basename(rd)
        dets = json.load(gzip.open(os.path.join(DETS, f'{name}_test.json.gz'), 'rt'))
        R[name] = {'bbox': eval_imgs(gt, dets, 'bbox'), 'segm': eval_imgs(gt, dets, 'segm')}
        R[name]['counts'] = [counts_at(R[name]['bbox'][0][CAB][i]) for i in range(len(img_ids))]
        models.setdefault(name.rsplit('_s', 1)[0], []).append(name)
    models = {m: models[m] for m in MAIN if m in models}

    # sanity check against evaluate.py
    ml = pd.read_csv('outputs/detector/eval/metrics_long.csv')
    full = np.arange(len(img_ids))
    chk = []
    for name in R:
        mine = run_metrics(R[name], full)
        ref = ml[(ml.run == name) & (ml.split == 'test')].iloc[0]
        chk.append(abs(mine['bbox_all_ap'] - ref.box_all_ap))
        if 'segm_all_ap' in mine:
            chk.append(abs(mine['segm_all_ap'] - ref.mask_all_ap))
    print(f'accumulation check vs evaluate.py: max |difference| = {max(chk):.2e}')

    md = ['# Detector benchmark — statistical comparison\n',
          f'Test images: {len(img_ids)} ({(field == "field1").sum()} field 1, {(field == "field2").sum()} field 2); '
          f'{len(models)} models × {min(len(v) for v in models.values())} seeds. Bootstrap draws: {a.B}.\n']

    # ---------------- 1. seed variability (seed-level test values) ----------------
    rows, anova, seed_vals = [], [], {}
    for mname, (typ, q) in METRICS.items():
        for m, rs in models.items():
            vals = [run_metrics(R[r], full).get(f'{typ}_{q}', np.nan) for r in rs]
            if np.isnan(vals).all():
                continue
            v = np.array(vals) * 100
            seed_vals[(mname, m)] = v
            rows.append({'metric': mname, 'model': LABEL[m], 'mean': v.mean(), 'SD': v.std(ddof=1), 'min': v.min(),
                         'max': v.max(), 'range': v.max() - v.min(), 'CV %': 100 * v.std(ddof=1) / v.mean(),
                         'seeds': ', '.join(f'{x:.1f}' for x in v)})
        for set_name, members in ANOVA_SETS.items():
            groups = [seed_vals[(mname, m)] for m in models if (mname, m) in seed_vals and (members is None or m in members)]
            if len(groups) < 2:
                continue
            F, p = st.f_oneway(*groups)
            allv = np.concatenate(groups)
            ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
            anova.append({'metric': mname, 'models compared': set_name, 'n models': len(groups), 'F': F, 'p': p,
                          'eta2 (share of variance due to model)': ss_b / ((allv - allv.mean()) ** 2).sum(),
                          'mean seed SD (pp)': np.mean([g.std(ddof=1) for g in groups]),
                          'SD between model means (pp)': np.std([g.mean() for g in groups], ddof=1)})
    sv, an = pd.DataFrame(rows), pd.DataFrame(anova)
    sv.round(2).to_csv(os.path.join(OUT, 'seed_variability.csv'), index=False)
    an.to_csv(os.path.join(OUT, 'anova_model_vs_seed.csv'), index=False)
    md.append('## 1. Seed variability (test, both fields; percentage points)\n')
    md.append('Spread over seeds per model (SD, pp):\n')
    md.append(sv.pivot(index='model', columns='metric', values='SD').reindex([LABEL[m] for m in models]).round(2)
              .to_markdown() + '\n')
    md.append('One-way ANOVA, model vs seed (eta² = share of the variance due to the choice of model):\n')
    md.append(an.assign(F=an.F.round(1), p=an.p.map(lambda x: f'{x:.1e}')).round(2).to_markdown(index=False) + '\n')
    md.append('Per-model seed values: `seed_variability.csv`.\n')

    # ---------------- 2. bootstrap pairwise differences ----------------
    md.append('## 2. Pairwise differences (hierarchical bootstrap over test images and seeds)\n')
    md.append('A model is "tied with the best" when its Holm-adjusted p against the best model is ≥ 0.05.\n')
    for sp, desc in SPLITS.items():
        idx = {f: np.where(field == f)[0] for f in ('field1', 'field2')}
        if sp == 'test_field1':
            idx = {'field1': idx['field1']}
        elif sp == 'test_field2':
            idx = {'field2': idx['field2']}
        base = np.concatenate(list(idx.values()))
        point = {m: {} for m in models}
        for m, rs in models.items():
            vals = [run_metrics(R[r], base) for r in rs]
            for k in vals[0]:
                point[m][k] = np.mean([v[k] for v in vals])
        draws = {m: [] for m in models}
        for b in range(a.B):
            sel = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in idx.values()])
            cache = {r: run_metrics(R[r], sel) for r in R}
            for m, rs in models.items():
                pick = rng.choice(rs, size=len(rs), replace=True)
                draws[m].append({k: np.mean([cache[r][k] for r in pick]) for k in cache[pick[0]]})
            if (b + 1) % 500 == 0:
                print(f'  {sp}: bootstrap {b + 1}/{a.B}', flush=True)
        pair_rows, summary = [], []
        for mname, (typ, q) in METRICS.items():
            k = f'{typ}_{q}'
            ms = [m for m in models if k in point[m] and not np.isnan(point[m][k])]
            res = []
            for m1, m2 in itertools.combinations(ms, 2):
                d = np.array([x[k] for x in draws[m1]]) - np.array([x[k] for x in draws[m2]])
                d = d[~np.isnan(d)]
                p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                res.append({'metric': mname, 'model A': LABEL[m1], 'model B': LABEL[m2],
                            'A − B (pp)': 100 * (point[m1][k] - point[m2][k]),
                            'CI low': 100 * np.percentile(d, 2.5), 'CI high': 100 * np.percentile(d, 97.5),
                            'p boot': max(p, 1 / a.B)})
            if not res:
                continue
            adj = holm([r['p boot'] for r in res])
            for r, pa in zip(res, adj):
                r['p Holm'] = pa
                r['significant (0.05)'] = pa < 0.05
            pair_rows += res
            best = max(ms, key=lambda m: point[m][k])
            tied = [LABEL[m] for m in ms if m != best and any(
                ({r['model A'], r['model B']} == {LABEL[m], LABEL[best]}) and not r['significant (0.05)'] for r in res)]
            ranking = ', '.join(f'{LABEL[m]} {100 * point[m][k]:.1f}' for m in sorted(ms, key=lambda m: -point[m][k]))
            worse = [LABEL[m] for m in ms if m != best and LABEL[m] not in tied]
            summary.append({'metric': mname, 'best': LABEL[best], 'statistically tied with best': ', '.join(tied) or '—',
                            'significantly worse than best': ', '.join(worse) or '—', 'ranking (mean %)': ranking})
        pd.DataFrame(pair_rows).round(4).to_csv(os.path.join(OUT, f'pairwise_{sp}.csv'), index=False)
        md.append(f'\n### {desc} ({len(base)} images) — full table: `pairwise_{sp}.csv`\n')
        md.append(pd.DataFrame(summary).to_markdown(index=False) + '\n')

    # ---------------- 3. Friedman on per-image Cabbage F1 ----------------
    f1 = {}
    for m, rs in models.items():
        per_img = []
        for i in range(len(img_ids)):
            vals = []
            for r in rs:
                tp, fp, fn = R[r]['counts'][i]
                vals.append(2 * tp / max(2 * tp + fp + fn, 1))
            per_img.append(np.mean(vals))
        f1[LABEL[m]] = per_img
    F1 = pd.DataFrame(f1)
    chi, p = st.friedmanchisquare(*[F1[c] for c in F1])
    post = []
    for c1, c2 in itertools.combinations(F1.columns, 2):
        diff = F1[c1] - F1[c2]
        pw = st.wilcoxon(F1[c1], F1[c2]).pvalue if (diff != 0).any() else 1.0
        post.append({'model A': c1, 'model B': c2, 'median per-image F1 A': F1[c1].median(),
                     'median per-image F1 B': F1[c2].median(), 'mean A − B (pp)': 100 * diff.mean(), 'p Wilcoxon': pw})
    post = pd.DataFrame(post)
    post['p Holm'] = holm(post['p Wilcoxon'])
    post.round(4).to_csv(os.path.join(OUT, 'friedman_posthoc_cabbage_f1.csv'), index=False)
    F1.assign(image=[gt.loadImgs(i)[0]['file_name'] for i in img_ids]).to_csv(os.path.join(OUT, 'per_image_cabbage_f1.csv'), index=False)
    md.append('\n## 3. Friedman test, per-image Cabbage F1 (box, conf 0.5, mean over seeds), test both fields\n')
    md.append(f'χ² = {chi:.1f}, p = {p:.1e} ({len(img_ids)} images, {F1.shape[1]} models). Mean rank (1 = best): '
              + ', '.join(f'{c} {r:.2f}' for c, r in F1.rank(axis=1, ascending=False).mean().sort_values().items()) + '\n')
    md.append('Post-hoc Wilcoxon (Holm) significant pairs: ' +
              (', '.join(f'{r["model A"]} vs {r["model B"]} (Δ {r["mean A − B (pp)"]:+.1f} pp)' for _, r in post[post['p Holm'] < 0.05].iterrows())
               or 'none') + '\n')
    open(os.path.join(OUT, 'stats.md'), 'w', encoding='utf8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
