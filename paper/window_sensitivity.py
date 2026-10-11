"""
Sensitivity of the frozen model to the memory window w (frames around the linked earlier frame whose plants are
registered; frozen value 60) on all four flights, scored on the human answers. Run after the held-out test; nothing
was selected with it.

  python paper/window_sensitivity.py      -> outputs/window_sensitivity.csv
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'groundtruth'))
sys.path.insert(0, os.path.join(ROOT, 'paper', 'development'))
from steps import FROZEN, load, score_run  # noqa: E402

if __name__ == '__main__':
    os.chdir(ROOT)
    rows = []
    for f in ('fieldA', 'fieldB', 'fieldC', 'heldout'):
        data = load(f)
        for w in (30, 60, 90):
            r = score_run(f, dict(FROZEN, window=w), data)
            rows.append(dict(flight=f, window=w, **r))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('outputs/window_sensitivity.csv', index=False)
