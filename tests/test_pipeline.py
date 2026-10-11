"""
Smoke tests (python -m pytest tests -q). They use the data record (data/) and are skipped without it.

  * stages 2-3 on the published held-out stage 1 give the published labels (no GPU needed, ~2 min);
  * stage 1 on the first 60 frames of the held-out video gives the published detections and odometry (GPU, ~30 s);
  * the human ground truth reproduces the paper's held-out numbers.
"""
import io
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
H = os.path.join(ROOT, 'data', 'runs', 'heldout')
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(H, 'extract', 'motion.csv')), reason='no data record in data/')


def as_published(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def test_reidentification_reproduces_published_run(tmp_path):
    from cabbagetrack.pipeline import run
    run(str(tmp_path), extract_dir=os.path.join(H, 'extract'))
    for k in ('labels', 'events', 'links', 'map'):
        assert pd.read_csv(tmp_path / f'{k}.csv').equals(pd.read_csv(os.path.join(H, f'{k}.csv'))), k


def test_stage1_reproduces_published_detections(tmp_path):
    pytest.importorskip('torch')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('stage 1 check needs the GPU used for the paper (CPU results can differ slightly)')
    subprocess.run([sys.executable, '-m', 'cabbagetrack.extract', '--video', os.path.join(ROOT, 'data', 'videos', 'heldout.mp4'),
                    '--weights', os.path.join(ROOT, 'data', 'weights', 'cabbage.pt'), '--out', str(tmp_path), '--max-frames', '60'],
                   check=True, cwd=ROOT)
    for k in ('detections.csv.gz', 'motion.csv'):
        a = pd.read_csv(tmp_path / k)
        b = pd.read_csv(os.path.join(H, 'extract', k))
        b = b[b.frame < 60].reset_index(drop=True)
        cols = [c for c in a.columns if not c.startswith('t_')]
        pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-6)


def test_ground_truth_heldout_numbers():
    sys.path.insert(0, os.path.join(ROOT, 'groundtruth'))
    from score import link_check, score
    P = pd.read_csv(os.path.join(ROOT, 'data', 'groundtruth', 'pairs', 'heldout_table_pairs.csv'))
    F = pd.read_csv(os.path.join(ROOT, 'data', 'groundtruth', 'pairs', 'heldout_followup_pairs.csv'))
    lab, ev = pd.read_csv(os.path.join(H, 'labels.csv')), pd.read_csv(os.path.join(H, 'events.csv'))
    dets = pd.read_csv(os.path.join(H, 'extract', 'detections.csv.gz'))
    o = score(lab, P, dets)
    assert ((o[P.kind == 'blind'] == 'correct').sum(), (P.kind == 'blind').sum()) == (90, 91)
    lc = link_check(lab, ev, dets, pd.concat([P, F], ignore_index=True))
    assert (lc['correct'], lc['links'], lc['not_covered']) == (601, 611, 0)
