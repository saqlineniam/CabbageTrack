"""
Rebuild every published position-pair file (data/groundtruth/pairs/) from the annotation tools, the exported answers
and the run each tool was made from (data/groundtruth/tool_runs/), and check them against the published files.

  python groundtruth/build_pairs.py            -> outputs/pairs/*.csv + identical / differs per file

Link items: the later sighting = the later track where it is most central; the earlier sighting = the identity's
most central sighting before the re-identification (for a decoy: the neighbour shown instead, same frame).
Blind items: the circled plant (earlier photo) and the clicked detection (later photo).
Only answers on which all annotators of a tool agree are used.
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import CENTER, answers, blind_pairs  # noqa: E402

GT = 'data/groundtruth'
OUT = 'outputs/pairs'
ROWB_OFFSET = 1768                    # Field A row-B tool: run on video frames 1768.. (frame numbers + offset)
ADJUDICATED = {('f2', 'B260'): 1110}  # one held-out blind disagreement, resolved from the image


def tasks(tool):
    with zipfile.ZipFile(f'{GT}/tools/{tool}.zip') as z:
        return json.loads(z.read(f'{tool}/tasks.js').decode('utf8').split('=', 1)[1].rstrip().rstrip(';'))


def ans(flight, tool, *names):
    return answers(*[f'{GT}/answers/{flight}/{tool}__{n}' for n in names])


def run(name):
    return f'{GT}/tool_runs/{name}'


def link_pairs(T, a, run_dir, seq, kinds=('partA', 'decoy'), keep_rejected=True):
    """keep_rejected=False: links answered 'different' are not kept (the first held-out round was built so)."""
    lab = pd.read_csv(f'{run_dir}/labels.csv')
    ev = pd.read_csv(f'{run_dir}/events.csv')
    lab['dc'] = np.hypot(lab.cx - CENTER[0], lab.cy - CENTER[1])
    best = lab.loc[lab.groupby('raw_id').dc.idxmin()].set_index('raw_id')
    reid_frame = ev[ev.kind == 'reid'].groupby('raw_id').frame.min()
    rows = []
    for v in T['verify']:
        x = a.get(v['id'])
        if x is None or x[0] not in ('same', 'different') or (v['decoy'] and x[0] == 'same'):
            continue
        if not keep_rejected and not v['decoy'] and x[0] != 'same':
            continue
        raw = int(v['id'][1:])
        q = best.loc[raw]
        earlier = lab[(lab.id == v['gid']) & (lab.frame < reid_frame[raw]) & (lab.raw_id != raw)]
        t = earlier.loc[earlier.dc.idxmin()]
        if v['decoy']:
            t = lab[(lab.frame == t.frame) & (lab.id == v['decoy_gid'])].iloc[0]
        rows.append({'seq': seq, 'kind': kinds[1] if v['decoy'] else kinds[0], 'task': v['id'],
                     'same': (not v['decoy']) and x[0] == 'same', 'frame_a': int(t.frame), 'xa': t.cx, 'ya': t.cy,
                     'frame_b': int(q.frame), 'xb': q.cx, 'yb': q.cy})
    return rows


def rowB_links(T, a):
    """Field A row B: the first tool was made from the Field A method's two-phase run (phase 1 = way out)."""
    r = run('run_reg')
    lab = pd.read_csv(f'{r}/labels.csv')
    n1 = json.load(open(f'{r}/summary.json'))['p1_plants']
    lab['dc'] = np.hypot(lab.cx - CENTER[0], lab.cy - CENTER[1])
    p1, p2 = lab[lab.phase == 1], lab[(lab.phase == 2) & (lab.gid < n1)]
    p1b = p1.loc[p1.groupby('gid').dc.idxmin()].set_index('gid')
    p2b = p2.loc[p2.groupby('gid').dc.idxmin()].set_index('gid')
    rows = []
    for v in T['verify']:
        x = a.get(v['id'])
        if x is None or x[0] != ('different' if v['decoy'] else 'same'):
            continue
        q = p2b.loc[v['gid']]
        t = p1b.loc[v['gid']]
        if v['decoy']:
            t = p1[(p1.frame == int(t.frame)) & (p1.gid == v['decoy_gid'])].iloc[0]
        rows.append({'seq': 'f1_rowB', 'kind': 'decoy' if v['decoy'] else 'partA', 'task': v['id'], 'same': not v['decoy'],
                     'frame_a': int(t.frame) + ROWB_OFFSET, 'xa': t.cx, 'ya': t.cy,
                     'frame_b': int(q.frame) + ROWB_OFFSET, 'xb': q.cx, 'yb': q.cy})
    return rows


def build():
    P = {}
    # Field A + held-out first round -> gt_pairs.csv
    T1, a1 = tasks('gt_tool'), ans('fieldA', 'gt_tool', 'gt_results_Saklain.csv', 'gt_results_Mashrup.csv')
    T2, a2 = tasks('gt_tool_blind2'), ans('fieldA', 'gt_tool_blind2', 'gt_results_blind2_Saklain.csv', 'gt_results_blind2_Mashrup.csv')
    rows = rowB_links(T1, a1)
    rows += blind_pairs(T1, a1, 'f1_rowB', lambda f: int(f) + ROWB_OFFSET)
    rows += [dict(r, task='b2_' + r['task']) for r in blind_pairs(T2, a2, 'f1_rowB', lambda f: int(f) + ROWB_OFFSET)]
    vm = pd.read_csv(f"{run('run_rowA_reg')}/extract_motion.csv")             # joined row-A segments -> video frames
    vmap = dict(zip(vm.frame.astype(int), vm.src_frame.astype(int)))
    TA = tasks('gt_tool_rowA')
    aA = ans('fieldA', 'gt_tool_rowA', 'gt_results_rowA_Saklain.csv', 'gt_results_rowA_Mashrup.csv')
    rows += blind_pairs(TA, aA, 'f1_rowA', lambda f: vmap[int(f)])
    TF, aF = tasks('gt_tool_field2'), ans('heldout', 'gt_tool_field2', 'gt_results_field2_Saklain.csv', 'gt_results_field2_Mashrup.csv')
    rows += link_pairs(TF, aF, run('revisit_dji0004_newdet'), 'f2', keep_rejected=False)
    rows += blind_pairs(TF, aF, 'f2', int, ADJUDICATED)
    df = pd.DataFrame(rows)
    df['field'] = np.where(df.seq == 'f2', 'field2', 'field1')
    P['gt_pairs.csv'] = df
    # held-out second and third rounds, follow-up
    v2 = pd.DataFrame(link_pairs(tasks('gt_tool_field2_v2'), ans('heldout', 'gt_tool_field2_v2', 'gt_results_field2v2_saklain.csv'),
                                 run('revisit_v2__field2'), 'f2', ('v2_link', 'v2_decoy')))
    v2['field'] = 'field2'
    P['gt_pairs_f2v2.csv'] = v2
    v3 = pd.DataFrame(link_pairs(tasks('gt_tool_field2_v3'), ans('heldout', 'gt_tool_field2_v3', 'gt_results_field2v3_Saklain.csv'),
                                 run('revisit_v3__field2'), 'f2'))
    P['heldout_table_pairs.csv'] = pd.concat([df[df.seq == 'f2'], v2, v3], ignore_index=True)
    P['heldout_followup_pairs.csv'] = pd.DataFrame(link_pairs(
        tasks('gt_tool_heldout_DJI0004_followup'),
        ans('heldout', 'gt_tool_heldout_DJI0004_followup', 'gt_results_heldout_followup_saklain.csv'),
        run('heldout_DJI0004__run'), 'f2'))
    # Field B, Field C
    TB = tasks('gt_tool_early_loop')
    rows = link_pairs(TB, ans('fieldB', 'gt_tool_early_loop', 'gt_results_early_loop_Saklain.csv'), run('revisit_v2__early_loop'), 'early')
    rows += blind_pairs(TB, ans('fieldB', 'gt_tool_early_loop', 'gt_results_early_loop_Saklain.csv'), 'early')
    TB2, aB2 = tasks('gt_tool_early_loop_v2rdt'), ans('fieldB', 'gt_tool_early_loop_v2rdt', 'gt_results_early_v2rdt_Saklain.csv')
    rows += link_pairs(TB2, aB2, run('revisit_v2rdt__early_loop'), 'early') + blind_pairs(TB2, aB2, 'early')
    P['gt_pairs_early_loop.csv'] = pd.DataFrame(rows)
    TC, aC = tasks('gt_tool_cauliflower_v3'), ans('fieldC', 'gt_tool_cauliflower_v3', 'gt_results_cauli_v3_Saklain.csv')
    P['gt_pairs_cauli.csv'] = pd.DataFrame(link_pairs(TC, aC, run('revisit_v3__heldout'), 'cauli') + blind_pairs(TC, aC, 'cauli'))
    return P


def same_pairs(a, b):
    cols = ['seq', 'kind', 'task', 'same', 'frame_a', 'xa', 'ya', 'frame_b', 'xb', 'yb']
    key = ['kind', 'task']
    a = a[cols].sort_values(key).reset_index(drop=True)
    b = b[cols].sort_values(key).reset_index(drop=True)
    if a.shape != b.shape:
        return False, f'{len(a)} vs {len(b)} pairs'
    num = ['frame_a', 'xa', 'ya', 'frame_b', 'xb', 'yb']
    ok = a[['seq', 'kind', 'task', 'same']].astype(str).equals(b[['seq', 'kind', 'task', 'same']].astype(str)) and \
        np.allclose(a[num].values.astype(float), b[num].values.astype(float), atol=1e-6)
    return ok, f'{len(a)} pairs'


if __name__ == '__main__':
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    for name, df in build().items():
        df.to_csv(f'{OUT}/{name}', index=False)
        ok, note = same_pairs(df, pd.read_csv(f'{GT}/pairs/{name}'))
        print(f'{name:30s} {note:12s} {"identical to the published pairs" if ok else "DIFFERS from the published pairs"}')
