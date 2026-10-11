"""
How the model was developed (paper Section on development, Figure 5) and the settings check (Figure 12), replayed on
the published stage-1 outputs of the development flights (Fields A, B, C) and scored with the human ground truth.

  python paper/development/steps.py steps fieldA fieldB fieldC        every development step -> dev_steps_<flight>.csv
  python paper/development/steps.py settings fieldA fieldB fieldC     one setting at a time -> settings_<flight>.csv
  python paper/development/steps.py check fieldA heldout              the last step == the frozen model (cabbagetrack/)
  python paper/development/steps.py v1 heldout                        the Field A method (v1) -> runs/heldout_v1

Each step adds one component (paper/development/revisit_exp.py, one switch per component); the frozen model is the
last step. Outputs in outputs/development/.
"""
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'groundtruth'))
import revisit_exp as RX  # noqa: E402
from score import duplicates, score  # noqa: E402

RUNS = os.path.join('data', 'runs')
OUT = os.path.join('outputs', 'development')
GSD = {'fieldA': 4.15, 'fieldB': 4.16, 'fieldC': 2.74, 'heldout': 2.96}   # mm/px for the 20-cm duplicate check
PAIRS = 'data/groundtruth/pairs'

STEPS = {}
STEPS['1 v1 (Field A method)'] = {}
STEPS['2 + search by image overlap'] = dict(multi_visit=True, overlap_search=True)
STEPS['3 + drift correction'] = dict(STEPS['2 + search by image overlap'], drift_feedback=True)
STEPS['4 + rotation correction'] = dict(STEPS['3 + drift correction'], rot_aware=True)
STEPS['5 + skip fast-turn frames'] = dict(STEPS['4 + rotation correction'], turn_aware=True)
STEPS['6 + overlap-only alignment'] = dict(STEPS['5 + skip fast-turn frames'], crop_verify=True)
STEPS['7 + drift only from clear links'] = dict(STEPS['6 + overlap-only alignment'], drift_min_peak=0.2)
STEPS['8 settings check: patience 60 (frozen)'] = dict(STEPS['7 + drift only from clear links'], patience=60)
FROZEN = STEPS['8 settings check: patience 60 (frozen)']

BASE = dict(STEPS['7 + drift only from clear links'])          # the settings check starts from step 7
GRID = [('base', {})] + [(f'{k}={v}', {k: v}) for k, vs in [
    ('gate', [40, 80]), ('min_overlap', [0.15, 0.35]), ('shift_tol', [300, 600]), ('drift_min_peak', [0.15, 0.3]),
    ('turn_rate', [0.2, 0.5]), ('rot_min_deg', [0.5, 2.0]), ('patience', [15, 60])] for v in vs]


def pairs(flight):
    if flight == 'fieldA':
        P = pd.read_csv(f'{PAIRS}/gt_pairs.csv')
        return P[P.seq.isin(['f1_rowB', 'f1_rowA'])]
    return pd.read_csv(f'{PAIRS}/' + {'fieldB': 'gt_pairs_early_loop.csv', 'fieldC': 'gt_pairs_cauli.csv',
                                      'heldout': 'heldout_table_pairs.csv'}[flight])


def load(flight):
    ex = os.path.join(RUNS, flight, 'extract')
    phi = pd.read_csv(os.path.join(RUNS, flight, 'rotation.csv')).phi_deg.values
    return (pd.read_csv(f'{ex}/detections.csv.gz'), pd.read_csv(f'{ex}/motion.csv'),
            np.load(f'{ex}/frames_small.npy', mmap_mode='r'), phi)


def run(flight, kw, data):
    dets, motion, small, phi = data
    return RX.run(dets, motion, small, RX.ExpConfig(**kw), phi=phi)


def score_run(flight, kw, data):
    out, s = run(flight, kw, data)
    lab = out['labels']
    dup, nid = duplicates(lab, out['map'], GSD[flight])
    P = pairs(flight)
    o = score(lab[['frame', 'raw_id', 'id', 'cx', 'cy']], P, data[0])
    b, a, neg = P.kind == 'blind', P.same & (P.kind == 'partA'), ~P.same
    return {'blind_ok': int((o[b] == 'correct').sum()), 'blind_n': int(b.sum()),
            'links_ok': int((o[a] == 'correct').sum()), 'links_n': int(a.sum()),
            'wrong': int((o[neg] == 'linked').sum()), 'wrong_n': int(neg.sum()),
            'identities': nid, 'dup_pairs': dup, 'cross_pass_links': s['reid_gap_gt_300']}


def steps(flight):
    data, rows = load(flight), []
    for name, kw in STEPS.items():
        r = score_run(flight, kw, data)
        rows.append(dict(step=name, flight=flight, blind_ok=r['blind_ok'], blind_n=r['blind_n'], links_ok=r['links_ok'],
                         links_n=r['links_n'], wrong=r['wrong'], wrong_n=r['wrong_n'], identities=r['identities'],
                         duplicate_pairs=r['dup_pairs'], cross_pass_links=r['cross_pass_links']))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'dev_steps_{flight}.csv'), index=False)


def settings(flight):
    data, rows = load(flight), []
    for name, kw in GRID:
        s = score_run(flight, dict(BASE, **kw), data)
        rows.append({'setting': name, 'flight': flight, 'correct': s['blind_ok'] + s['links_ok'], 'wrong': s['wrong'],
                     'blind': f"{s['blind_ok']}/{s['blind_n']}", 'links': f"{s['links_ok']}/{s['links_n']}",
                     'cross_pass_links': s['cross_pass_links']})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'settings_{flight}.csv'), index=False)


def check(flight):
    """The last development step must give exactly the published (frozen-model) labels."""
    out, _ = run(flight, FROZEN, load(flight))
    ref = pd.read_csv(os.path.join(RUNS, flight, 'labels.csv'))
    buf = io.StringIO()                                   # compare as written to CSV (as the published file)
    out['labels'].to_csv(buf, index=False)
    buf.seek(0)
    same = pd.read_csv(buf).equals(ref)
    print(f'{flight}: development step 8 {"==" if same else "!="} frozen model (labels.csv)')
    return same


def v1(flight):
    """The Field A method (all switches off) -> runs/<flight>_v1, compared with data/runs/<flight>_v1 if published."""
    out, s = run(flight, {}, load(flight))
    d = os.path.join('runs', f'{flight}_v1')
    os.makedirs(d, exist_ok=True)
    for k in ('labels', 'events', 'links', 'map'):
        out[k].to_csv(os.path.join(d, f'{k}.csv'), index=False)
    ref = os.path.join(RUNS, f'{flight}_v1', 'labels.csv')
    if os.path.exists(ref):
        same = pd.read_csv(os.path.join(d, 'labels.csv')).equals(pd.read_csv(ref))
        print(f'{flight}: Field A method (v1) {"==" if same else "!="} published run')


if __name__ == '__main__':
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    what, flights = sys.argv[1], sys.argv[2:] or ['fieldA', 'fieldB', 'fieldC']
    for f in flights:
        {'steps': steps, 'settings': settings, 'check': check, 'v1': v1}[what](f)
