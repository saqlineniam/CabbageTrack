"""
Diagnose a model version on a DEVELOPMENT flight with the human ground truth: for every human-confirmed same-plant pair
(blind clicks + verified links) that the version does not link, find where the chain broke while the later track was
waiting for an identity:
  not_searched      no frame of the plant's earlier sightings was among the looked-up earlier frames
  tried_other_frames earlier ground looked up, but the frames tried for alignment were not those showing this plant
  weak_alignment    looked up, but every alignment with those frames had a weak phase-correlation peak (< 0.08)
  position_mismatch looked up, peak clear, but the measured shift disagreed with the position estimate (> shift_tol)
  not_remembered    a link was accepted, but the plant was not among the remembered plants (outside the window / excluded)
  off_target        remembered and registered, but it landed further from the new track than the gate
  wrong_plant       the later track joined a different, older identity
Usage: python paper/development/diagnose.py <flight> '<json config>' [tag]    flight: fieldA | fieldB | fieldC
  e.g. python paper/development/diagnose.py fieldB "{\"multi_visit\": true, \"overlap_search\": true}" step2
Output: outputs/development/diagnosis_<flight>_<tag>.csv (+ printed summary)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revisit_exp as RX  # noqa: E402
from steps import ROOT, load, pairs as gt  # noqa: E402


def track_at(by_frame, f, x, y, R=40.0):
    g = by_frame.get(int(f))
    if g is None:
        return None
    d = np.hypot(g.cx.values - x, g.cy.values - y)
    return int(g.raw_id.values[d.argmin()]) if d.min() <= R else None


def diagnose(flight, kw, tag):
    dets, motion, small, phi = load(flight)
    c = RX.ExpConfig(**kw)
    tr = {}
    out, _ = RX.run(dets, motion, small, c, phi=phi, trace=tr)
    lab = out['labels']
    raw2id = lab.drop_duplicates('raw_id').set_index('raw_id').id
    raw_first_label = lab.groupby('raw_id').frame.min()
    id_frames = lab.groupby('id').frame.apply(lambda f: np.unique(f.values))
    first_seen = lab.groupby('id').frame.min()
    d = dets[(dets.cls == 1) & (dets.raw_id >= 0)]
    by_frame = {f: g for f, g in d.groupby('frame')}
    track_frames = d.groupby('raw_id').frame.apply(lambda f: np.sort(f.values))
    look = {}
    for t, near in tr.get('lookups', []):
        look[t] = near
    att = pd.DataFrame(tr.get('attempts', []), columns=['t', 's', 'peak', 'resid', 'ok', 'rot'])
    att_by_t = {t: g for t, g in att.groupby('t')} if len(att) else {}
    reg = tr.get('registered', {})
    P = gt(flight)
    P = P[P.same & P.kind.isin(['blind', 'partA'])]
    rows = []
    for p in P.itertuples():
        a, b = track_at(by_frame, p.frame_a, p.xa, p.ya), track_at(by_frame, p.frame_b, p.xb, p.yb)
        if a is None or b is None or a not in raw2id or b not in raw2id:
            rows.append({'kind': p.kind, 'task': p.task, 'outcome': 'not_tracked'})
            continue
        ia, ib = int(raw2id[a]), int(raw2id[b])
        if ia == ib:
            rows.append({'kind': p.kind, 'task': p.task, 'outcome': 'correct'})
            continue
        if first_seen[ib] <= p.frame_a:
            rows.append({'kind': p.kind, 'task': p.task, 'outcome': 'wrong_plant'})
            continue
        fb = track_frames[b]
        start = fb[min(len(fb) - 1, c.min_track_frames - 1)]
        end = int(raw_first_label[b])
        window = [t for t in fb if start <= t <= end]
        earlier = id_frames[ia]
        earlier = earlier[earlier < start]
        searched, attempts, best_dist = False, [], None
        for t in window:
            near = look.get(t)
            if near is not None and len(np.intersect1d(near, earlier)):
                searched = True
            g = att_by_t.get(t)
            if g is not None:
                gg = g[g.s.apply(lambda s: bool(len(earlier)) and np.abs(earlier - s).min() <= c.window)]
                attempts.append(gg)
            if (t, ia) in reg:
                x, y = reg[(t, ia)]
                q = by_frame[t][by_frame[t].raw_id == b]
                if len(q):
                    dd = float(np.hypot(q.cx.values[0] - x, q.cy.values[0] - y))
                    best_dist = dd if best_dist is None else min(best_dist, dd)
        A = pd.concat(attempts) if attempts else pd.DataFrame(columns=att.columns)
        if not searched and not len(A):
            outcome = 'not_searched'
        elif not len(A):
            outcome = 'tried_other_frames'        # earlier ground found, but the frames tried did not show this plant
        elif not A.ok.any():
            outcome = 'weak_alignment' if (A.peak < c.min_response).all() else 'position_mismatch'
        elif best_dist is None:
            outcome = 'not_remembered'
        else:
            outcome = 'off_target'
        rows.append({'kind': p.kind, 'task': p.task, 'outcome': outcome, 'frame_a': p.frame_a, 'frame_b': p.frame_b,
                     'best_peak': float(A.peak.max()) if len(A) else None,
                     'min_resid_px': float(A.resid.min()) if len(A) else None, 'best_registered_dist_px': best_dist})
    df = pd.DataFrame(rows)
    os.makedirs('outputs/development', exist_ok=True)
    df.to_csv(f'outputs/development/diagnosis_{flight}_{tag}.csv', index=False)
    print(f'{flight} [{tag}]')
    print(pd.crosstab(df.outcome, df.kind, margins=True).to_string())
    bad = df[~df.outcome.isin(['correct'])]
    if len(bad):
        print(bad.groupby('outcome')[['best_peak', 'min_resid_px', 'best_registered_dist_px']].median().round(2).to_string())
    return df


if __name__ == '__main__':
    os.chdir(ROOT)
    diagnose(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}, sys.argv[3] if len(sys.argv) > 3 else 'v1')
