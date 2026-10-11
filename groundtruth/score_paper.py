"""
Score the paper's runs against the human ground truth and check the numbers reported in the paper.

  python groundtruth/score_paper.py                      the published runs (data/runs/)
  python groundtruth/score_paper.py --runs runs           your own reruns (tools/run_flights.py writes runs/<flight>/)

Writes <out>/gt_scores.md and prints whether every reported number is reproduced.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import duplicates, fmt, link_check, score  # noqa: E402

PAIRS = 'data/groundtruth/pairs'
GSD = {'fieldA': 4.15, 'fieldB': 4.07, 'fieldC': 2.74, 'heldout': 2.96}     # mm per px (camera path vs GNSS)
# numbers reported in the paper (frozen model; held-out also the Field A method v1)
EXPECTED = {
    ('fieldA', 'blind'): (299, 299), ('fieldA', 'links'): (372, 372), ('fieldA', 'wrong'): (0, 32),
    ('fieldB', 'blind'): (80, 85), ('fieldB', 'links'): (687, 690), ('fieldB', 'wrong'): (1, 66),
    ('fieldC', 'blind'): (43, 74), ('fieldC', 'links'): (64, 82), ('fieldC', 'wrong'): (13, 63),
    ('heldout', 'blind'): (90, 91), ('heldout', 'links'): (606, 637), ('heldout', 'wrong'): (11, 72),
    ('heldout', 'identities'): 374, ('heldout', 'duplicates'): 2, ('heldout', 'link_precision'): (601, 611),
    ('heldout_v1', 'blind'): (73, 91), ('heldout_v1', 'links'): (406, 637), ('heldout_v1', 'identities'): 592,
    ('heldout_v1', 'duplicates'): 142,
}


def pairs_for(flight):
    if flight == 'fieldA':
        P = pd.read_csv(f'{PAIRS}/gt_pairs.csv')
        return P[P.seq.isin(['f1_rowB', 'f1_rowA'])]
    if flight == 'fieldB':
        return pd.read_csv(f'{PAIRS}/gt_pairs_early_loop.csv')
    if flight == 'fieldC':
        return pd.read_csv(f'{PAIRS}/gt_pairs_cauli.csv')
    return pd.read_csv(f'{PAIRS}/heldout_table_pairs.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='data/runs')
    ap.add_argument('--out', default='outputs/scores')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    got, rows = {}, []
    for flight in ('fieldA', 'fieldB', 'fieldC', 'heldout', 'heldout_v1'):
        run = os.path.join(a.runs, flight)
        if not os.path.exists(os.path.join(run, 'labels.csv')):
            print(f'(skipping {flight}: no run at {run})')
            continue
        base = 'heldout' if flight == 'heldout_v1' else flight
        ex = os.path.join(a.runs, base, 'extract')
        if not os.path.exists(ex):
            ex = os.path.join('data', 'runs', base, 'extract')
        lab = pd.read_csv(f'{run}/labels.csv')
        ev = pd.read_csv(f'{run}/events.csv')
        cam = pd.read_csv(f'{run}/map.csv')
        dets = pd.read_csv(f'{ex}/detections.csv.gz')
        P = pairs_for(base)
        o = score(lab[['frame', 'raw_id', 'id', 'cx', 'cy']], P, dets)
        b, l_, neg = P.kind == 'blind', P.same & (P.kind != 'blind'), ~P.same
        dup, nid = duplicates(lab, cam, GSD[base])
        got[(flight, 'blind')] = (int((o[b] == 'correct').sum()), int(b.sum()))
        got[(flight, 'links')] = (int((o[l_] == 'correct').sum()), int(l_.sum()))
        got[(flight, 'wrong')] = (int((o[neg] == 'linked').sum()), int(neg.sum()))
        got[(flight, 'identities')], got[(flight, 'duplicates')] = nid, dup
        row = {'run': flight, 'blind plants re-identified': fmt(*got[(flight, 'blind')]),
               'blind: new identity': int((o[b] == 'missed').sum()), 'blind: wrong plant': int((o[b] == 'other_identity').sum()),
               'checked same-plant links made': fmt(*got[(flight, 'links')]),
               'wrong links made (of different-plant answers)': '{}/{}'.format(*got[(flight, 'wrong')]),
               'identities': nid, 'duplicate pairs (< 20 cm)': dup,
               'cross-pass links': int(((ev.kind == 'reid') & (ev.gap > 300)).sum())}
        if flight == 'heldout':
            allp = pd.concat([P, pd.read_csv(f'{PAIRS}/heldout_followup_pairs.csv')], ignore_index=True)
            lc = link_check(lab, ev, dets, allp)
            got[(flight, 'link_precision')] = (lc['correct'], lc['links'])
            row['every cross-pass link checked (incl. follow-up)'] = (fmt(lc['correct'], lc['links'])
                                                                       + f", not covered {lc['not_covered']}")
        rows.append(row)
    df = pd.DataFrame(rows).set_index('run').T.fillna('')
    md = '# Human ground truth on the CabbageTrack runs\n\n' + df.to_markdown() + '\n'
    open(os.path.join(a.out, 'gt_scores.md'), 'w', encoding='utf8').write(md)
    print(md)
    bad = [(k, v, got.get(k)) for k, v in EXPECTED.items() if k in got and got[k] != v]
    checked = sum(k in got for k in EXPECTED)
    print(f'{checked - len(bad)}/{checked} numbers reported in the paper reproduced' + ('' if not bad else f'; differ: {bad}'))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
