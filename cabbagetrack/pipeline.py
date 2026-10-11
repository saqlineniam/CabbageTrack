"""
CabbageTrack pipeline: from a nadir drone video to one identity number per plant, kept across passes.

  stage 1  detection + tracking + camera odometry + 1/4-scale frames  (cabbagetrack.extract)
  stage 2  camera rotation between consecutive frames                (cabbagetrack.rotation.frame_rotation)
  stage 3  plant re-identification across passes                      (cabbagetrack.reid.reidentify)

Outputs (<out>/)
  extract/       stage 1 (detections.csv.gz, motion.csv, frames_small.npy, meta.json), unless an existing one is reused
  rotation.csv   per-frame rotation phi (deg) and whether the frame is inside a fast turn
  labels.csv     every plant detection with its identity (frame, raw_id = tracker track, id, cx, cy)
  events.csv     one row per identity decision (new identity or re-identification, with link frame and gap)
  links.csv      every verified revisit link (frame, earlier frame s, measured / predicted shift, rotation)
  map.csv        camera path in image pixels (odometry)
  summary.json   counts, settings, provenance (model / detector / video SHA-1)
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from .config import Config
from .reid import reidentify, turning_frames
from .rotation import frame_rotation

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = ('config.py', 'rotation.py', 'reid.py')      # the frozen decision logic (byte-identical to the paper)


def model_sha1():
    h = hashlib.sha1()
    for f in MODEL_FILES:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    return h.hexdigest()


def run(out, video=None, weights=None, conf=0.5, segments='', extract_dir=None):
    """Run the whole pipeline. Give video + weights, or extract_dir (a finished stage 1) to skip detection."""
    os.makedirs(out, exist_ok=True)
    ex = extract_dir or os.path.join(out, 'extract')
    if not extract_dir:
        if not (video and weights):
            raise ValueError('give video and weights, or extract_dir')
        print('stage 1: detection + tracking + camera odometry ...', flush=True)
        # own process: importing Ultralytics sets OpenCV to one thread, which would slow stages 2-3 about five-fold
        ab = os.path.abspath
        cmd = [sys.executable, '-m', 'cabbagetrack.extract', '--video', ab(video), '--weights', ab(weights),
               '--conf', str(conf), '--out', ab(ex)] + (['--segments', segments] if segments else [])
        subprocess.run(cmd, check=True, cwd=os.path.dirname(HERE))
    dets = pd.read_csv(os.path.join(ex, 'detections.csv.gz'))
    motion = pd.read_csv(os.path.join(ex, 'motion.csv'))
    small = np.load(os.path.join(ex, 'frames_small.npy'), mmap_mode='r')
    c = Config()
    print('stage 2: camera rotation ...', flush=True)
    phi = frame_rotation(small)
    pd.DataFrame({'frame': np.arange(len(phi)), 'phi_deg': phi,
                  'in_fast_turn': turning_frames(phi, len(phi), c)}).to_csv(os.path.join(out, 'rotation.csv'),
                                                                           index=False)
    print('stage 3: plant re-identification ...', flush=True)
    res, s = reidentify(dets, motion, small, phi, c)
    for k in ('labels', 'events', 'links', 'map'):
        res[k].to_csv(os.path.join(out, f'{k}.csv'), index=False)
    meta_path = os.path.join(ex, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    from . import __version__
    s['provenance'] = {'cabbagetrack': __version__, 'model_sha1': model_sha1(), 'extract': os.path.abspath(ex),
                       'video': meta.get('video'), 'video_sha1': meta.get('video_sha1'),
                       'weights': meta.get('weights'), 'weights_sha1': meta.get('weights_sha1'),
                       'conf': meta.get('conf'), 'segments': meta.get('segments')}
    json.dump(s, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: s[k] for k in ('frames', 'identities', 'reid_events', 'cross_pass_reid', 'fast_turn_frames',
                                        'runtime_s')}), '->', os.path.abspath(out))
    return out
