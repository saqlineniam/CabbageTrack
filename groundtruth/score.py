"""
Scoring a CabbageTrack run against the human ground truth.

The human answers (annotation tools, docs/05_ground_truth.md) are stored as pairs of IMAGE POSITIONS, independent of
any run:  earlier sighting (frame_a, xa, ya)  <->  later sighting (frame_b, xb, yb),  same = True / False.
kind: blind (the annotator found the plant on the earlier pass without seeing the method's answer), partA / v2_link
(a link the method made, judged same / different), decoy / v2_decoy (a wrong partner shown on purpose).

Any run (labels.csv + its stage-1 detections) can be scored on the same pairs:
  * score(): outcome per pair (correct / missed / other_identity / not_tracked for same-plant pairs; ok / linked for
    different-plant pairs);
  * link_check(): every cross-pass link of the run (re-identification > 300 frames later) -> correct / wrong / not
    covered by any answer;
  * duplicates(): identity pairs closer than 20 cm on the map that are never in the same frame (one plant, two numbers).
"""
import json
from math import sqrt

import numpy as np
import pandas as pd

CENTER = np.array([960.0, 540.0])
TOOL_SCALE = 1920 / 960.0          # tool images are 960 px wide


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f'{k}/{n} = {p:.1%} (95% CI {lo:.1%}-{hi:.1%})'


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def tasks(path):
    """tasks.js of an annotation tool -> dict (meta, verify, blind)."""
    return json.loads(open(path, encoding='utf8').read().split('=', 1)[1].rstrip().rstrip(';'))


def answers(*paths):
    """Exported answer CSVs (one per annotator) -> task_id -> (answer, clicked_raw_id) where all annotators agree."""
    dfs = [pd.read_csv(p, dtype=str, keep_default_na=False).set_index('task_id') for p in paths]
    out = {}
    for tid in dfs[0].index:
        rows = [d.loc[tid] for d in dfs if tid in d.index]
        if len(rows) < len(dfs):
            continue
        a = {(r.answer, r.clicked_raw_id) for r in rows}
        out[tid] = a.pop() if len(a) == 1 else None
    return out


def blind_pairs(T, ans, seq, frame_map=int, adjudicated=None):
    """Blind items answered with a click -> same-plant position pairs."""
    rows = []
    for b in T['blind']:
        a = ans.get(b['id'])
        raw = None
        if a is not None and a[0] == 'clicked':
            raw = int(a[1])
        elif adjudicated and (seq, b['id']) in adjudicated:
            raw = adjudicated[(seq, b['id'])]
        if raw is None:
            continue
        d = next(d for d in b['dets'] if d[2] == raw)
        rows.append({'seq': seq, 'kind': 'blind', 'task': b['id'], 'same': True,
                     'frame_a': frame_map(b['out_frame']), 'xa': b['target'][0] * TOOL_SCALE, 'ya': b['target'][1] * TOOL_SCALE,
                     'frame_b': frame_map(b['back_frame']), 'xb': d[0] * TOOL_SCALE, 'yb': d[1] * TOOL_SCALE})
    return rows


def tool_pairs(tool, run, answer_csvs, seq):
    """Answers of a link-verification tool (groundtruth/make_tool.py) -> position pairs, using the run the tool was
    made from: Part A 'same' -> same-plant pair, 'different' -> different-plant pair, decoys answered 'different' ->
    different-plant pair, blind clicks -> same-plant pairs; unsure / not visible are skipped."""
    T = tasks(f'{tool}/tasks.js')
    ans = answers(*answer_csvs)
    lab = pd.read_csv(f'{run}/labels.csv')
    ev = pd.read_csv(f'{run}/events.csv')
    lab['dc'] = np.hypot(lab.cx - CENTER[0], lab.cy - CENTER[1])
    best = lab.loc[lab.groupby('raw_id').dc.idxmin()].set_index('raw_id')
    reid_frame = ev[ev.kind == 'reid'].groupby('raw_id').frame.min()
    rows = []
    for v in T['verify']:
        a = ans.get(v['id'])
        if a is None or a[0] not in ('same', 'different') or (v['decoy'] and a[0] == 'same'):
            continue
        raw = int(v['id'][1:])
        q = best.loc[raw]
        earlier = lab[(lab.id == v['gid']) & (lab.frame < reid_frame[raw]) & (lab.raw_id != raw)]
        t = earlier.loc[earlier.dc.idxmin()]
        if v['decoy']:
            t = lab[(lab.frame == t.frame) & (lab.id == v['decoy_gid'])].iloc[0]
        rows.append({'seq': seq, 'kind': 'decoy' if v['decoy'] else 'partA', 'task': v['id'],
                     'same': (not v['decoy']) and a[0] == 'same', 'frame_a': int(t.frame), 'xa': t.cx, 'ya': t.cy,
                     'frame_b': int(q.frame), 'xb': q.cx, 'yb': q.cy})
    rows += blind_pairs(T, ans, seq)
    return pd.DataFrame(rows)


def _tracks(dets):
    d = dets[(dets.cls == 1) & (dets.raw_id >= 0)]
    return {f: g for f, g in d.groupby('frame')}


def score(labels, pairs, dets=None, radius=40.0):
    """Outcome per pair. A position is resolved to the run's tracker track (via its detections) and then to that
    track's identity, so frames before the track received its identity count too.
    same-plant pairs: correct | missed (the later sighting started a new identity) | other_identity (linked to a
                      different, already existing identity) | not_tracked (no identity at one of the positions)
    different-plant pairs: ok | linked (the two plants share an identity = wrong link)"""
    first = labels.groupby('id').frame.min()
    raw2id = labels.drop_duplicates('raw_id').set_index('raw_id').id
    by_frame = _tracks(dets) if dets is not None else {f: g for f, g in labels.groupby('frame')}

    def ident(f, x, y):
        g = by_frame.get(int(f))
        if g is None:
            return None
        d = np.hypot(g.cx.values - x, g.cy.values - y)
        k = int(d.argmin())
        if d[k] > radius:
            return None
        if dets is None:
            return int(g.id.values[k])
        i = raw2id.get(int(g.raw_id.values[k]))
        return None if i is None else int(i)

    out = []
    for p in pairs.itertuples():
        ia, ib = ident(p.frame_a, p.xa, p.ya), ident(p.frame_b, p.xb, p.yb)
        if not p.same:
            out.append('linked' if (ia is not None and ia == ib) else 'ok')
        elif ia is None or ib is None:
            out.append('not_tracked')
        elif ia == ib:
            out.append('correct')
        elif first[ib] > p.frame_a:
            out.append('missed')
        else:
            out.append('other_identity')
    return pd.Series(out, index=pairs.index)


def link_check(labels, events, dets, pairs, gap=300, radius=40.0):
    """Every cross-pass re-identification of the run (gap > `gap` frames): correct when a same-plant answer joins its
    track to an earlier track of the same identity, wrong when a different-plant answer does, else not covered."""
    by_frame = _tracks(dets)

    def track(f, x, y):
        g = by_frame.get(int(f))
        if g is None:
            return None
        k = np.hypot(g.cx.values - x, g.cy.values - y)
        return int(g.raw_id.values[k.argmin()]) if k.min() <= radius else None
    pos, neg = {}, {}
    for p in pairs[pairs.kind != 'blind'].itertuples():
        a, b = track(p.frame_a, p.xa, p.ya), track(p.frame_b, p.xb, p.yb)
        if a is not None and b is not None:
            (pos if p.same else neg).setdefault(b, set()).add(a)
    r = events[(events.kind == 'reid') & (events.gap > gap)]
    out = {'links': len(r), 'correct': 0, 'wrong': 0, 'not_covered': 0}
    for e in r.itertuples():
        earlier = set(labels[(labels.id == e.id) & (labels.frame < e.frame) & (labels.raw_id != e.raw_id)].raw_id)
        if neg.get(e.raw_id, set()) & earlier:
            out['wrong'] += 1
        elif pos.get(e.raw_id, set()) & earlier:
            out['correct'] += 1
        else:
            out['not_covered'] += 1
    return out


def duplicates(labels, cam, gsd_mm, dist_m=0.2):
    """Identity pairs whose median map positions are closer than dist_m and that never appear in the same frame."""
    from scipy.spatial import cKDTree
    lab = labels.assign(gx=cam.cam_x.values[labels.frame] + labels.cx, gy=cam.cam_y.values[labels.frame] + labels.cy)
    pos = lab.groupby('id')[['gx', 'gy']].median()
    fr = lab.groupby('id').frame.apply(set)
    ids = pos.index.values
    pairs = cKDTree(pos.values).query_pairs(dist_m * 1000 / gsd_mm)
    return sum(1 for a, b in pairs if not (fr[ids[a]] & fr[ids[b]])), len(pos)
