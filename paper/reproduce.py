"""
Every table and figure of the paper from the data record (data/), in order:

  gnss         camera path vs drone GNSS for the four flights          -> outputs/gnss/<flight>/
  mosaic       mosaics from the videos                                 -> outputs/mosaic/<flight>/
  pairs        human answers -> position pairs (checked)               -> outputs/pairs/
  scores       human ground truth on the published runs (19 numbers)   -> outputs/scores/gt_scores.md
  development  every development step + settings check (Figs 5, 12)    -> outputs/development/   (~1-2 h; --quick copies
                                                                          the published tables instead)
  detector     Table 1 + statistics from the published evaluations,    -> outputs/detector/, outputs/detectors/
               detector numbers (training data, test scores, overlap)
  figures      Figures 1-12 and S1-S5                                  -> outputs/figures/

  python paper/reproduce.py                 everything
  python paper/reproduce.py --quick         without replaying the development steps
  python paper/reproduce.py figures         one or more steps

The runs themselves are checked with tools/run_flights.py; the detectors are retrained with detector/ (docs/04).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
FLIGHTS = ('fieldA', 'fieldB', 'fieldC', 'heldout')
PY = sys.executable


def sh(*args):
    print('$', ' '.join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8'))


def gnss():
    from cabbagetrack.gnss import compare
    for k in FLIGHTS:
        compare(f'data/runs/{k}', f'data/gnss/{k}.SRT', f'outputs/gnss/{k}')


def mosaic():
    from cabbagetrack.mosaic import build
    for k in FLIGHTS:
        build(f'data/videos/{k}.mp4', f'data/runs/{k}', f'outputs/mosaic/{k}', extract=f'data/runs/{k}/extract')


def pairs():
    sh('groundtruth/build_pairs.py')


def scores():
    sh('groundtruth/score_paper.py')


def development(quick=False):
    if quick:
        os.makedirs('outputs/development', exist_ok=True)
        for f in glob.glob('data/results/development/tables/*.csv'):
            shutil.copy(f, 'outputs/development')
        print('development: published tables copied (run without --quick to replay every step)')
        return
    sh('paper/development/steps.py', 'steps', 'fieldA', 'fieldB', 'fieldC')
    sh('paper/development/steps.py', 'settings', 'fieldA', 'fieldB', 'fieldC')
    for f in glob.glob('outputs/development/*.csv'):
        ref = os.path.join('data/results/development/tables', os.path.basename(f))
        if os.path.exists(ref):
            same = open(f, 'rb').read().replace(b'\r\n', b'\n') == open(ref, 'rb').read().replace(b'\r\n', b'\n')
            print(f'   {os.path.basename(f)}: {"identical to the published table" if same else "DIFFERS from the published table"}')


def detector():
    B = 'data/detector/cabbage/benchmark'
    os.makedirs('outputs/detector', exist_ok=True)
    for r in glob.glob(f'{B}/runs/*'):                          # run metadata only (no weights needed)
        dst = os.path.join('outputs/detector/runs', os.path.basename(r))
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(r, '*')):
            if os.path.isfile(f) and not f.endswith(('.pt', '.pth')):
                shutil.copy(f, dst)
    for sub in ('eval', 'dets'):
        shutil.copytree(f'{B}/{sub}', f'outputs/detector/{sub}', dirs_exist_ok=True)
    shutil.copy(f'{B}/detector_queue.out', 'outputs/detector_queue.out')
    sh('detector/summarize.py')
    sh('detector/stats.py')
    for f in ('table1.md', 'table1_test.csv'):
        same = open(f'outputs/detector/{f}', 'rb').read() == open(f'{B}/{f}', 'rb').read()
        print(f'   {f}: {"identical to the published file" if same else "differs from the published file"}')
    sh('paper/detector_numbers.py')


def figures():
    sh('paper/figures/figures.py')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('steps', nargs='*', default=['gnss', 'mosaic', 'pairs', 'scores', 'development', 'detector', 'figures'])
    ap.add_argument('--quick', action='store_true', help='copy the published development tables instead of replaying')
    a = ap.parse_args()
    os.chdir(ROOT)
    for step in a.steps:
        print(f'== {step}', flush=True)
        development(a.quick) if step == 'development' else globals()[step]()
