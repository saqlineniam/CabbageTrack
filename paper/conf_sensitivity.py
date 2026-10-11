"""
Sensitivity of the frozen model to the detector confidence threshold (0.3, 0.5 = used, 0.7) on all four flights:
stage 1 is re-run at each threshold, everything else unchanged, and every run is scored on the same human answers.
Reported as a sensitivity analysis after the held-out run; nothing was selected with it.

  python paper/conf_sensitivity.py                -> outputs/conf_sensitivity/<flight>_c<conf>/ + conf_sensitivity.csv
"""
import os
import subprocess
import sys

import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'groundtruth'))
sys.path.insert(0, os.path.join(ROOT, 'paper', 'development'))
from score import score  # noqa: E402
from steps import pairs  # noqa: E402

OUT = 'outputs/conf_sensitivity'
CONFS = (0.3, 0.7)


def evaluate(flight, run, ex):
    lab = pd.read_csv(f'{run}/labels.csv')
    ev = pd.read_csv(f'{run}/events.csv')
    dets = pd.read_csv(f'{ex}/detections.csv.gz')
    P = pairs(flight)
    o = score(lab[['frame', 'raw_id', 'id', 'cx', 'cy']], P, dets)
    b, a, neg = P.kind == 'blind', P.same & (P.kind != 'blind'), ~P.same
    return {'blind_ok': int((o[b] == 'correct').sum()), 'blind_n': int(b.sum()),
            'links_ok': int((o[a] == 'correct').sum()), 'links_n': int(a.sum()),
            'wrong': int((o[neg] == 'linked').sum()), 'wrong_n': int(neg.sum()),
            'identities': int(lab.id.nunique()), 'plant_detections_per_frame': round(len(dets[dets.cls == 1]) / dets.frame.nunique(), 1),
            'cross_pass_links': int(((ev.kind == 'reid') & (ev.gap > 300)).sum())}


def main():
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    F = yaml.safe_load(open('flights.yaml'))
    rows = []
    for k, f in F.items():
        rows.append(dict(flight=k, conf=0.5, **evaluate(k, f'data/runs/{k}', f'data/runs/{k}/extract')))
        print(rows[-1], flush=True)
        for c in CONFS:
            out = os.path.join(OUT, f'{k}_c{c}')
            if not os.path.exists(os.path.join(out, 'labels.csv')):
                cmd = [sys.executable, '-m', 'cabbagetrack', 'run', '--video', f['video'], '--weights', f['weights'],
                       '--conf', str(c), '--out', out] + (['--segments', f['segments']] if f.get('segments') else [])
                subprocess.run(cmd, check=True, cwd=ROOT)
            rows.append(dict(flight=k, conf=c, **evaluate(k, out, os.path.join(out, 'extract'))))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, 'conf_sensitivity.csv'), index=False)


if __name__ == '__main__':
    main()
