"""
Run CabbageTrack on the paper's flights (flights.yaml) and check the result against the published runs.

  python tools/run_flights.py                         all four flights: pipeline + mosaic + GNSS comparison
  python tools/run_flights.py heldout --demo          one flight, also render the demo video
  python tools/run_flights.py heldout --reuse-extract use the published stage-1 output (no GPU needed)

Writes runs/<flight>/ and prints whether labels / events / links / map are identical to data/runs/<flight>/.
"""
import argparse
import hashlib
import os
import sys

import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from cabbagetrack.pipeline import run  # noqa: E402


def sha1(path, chunk=1 << 24):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def compare(out, ref):
    if not os.path.exists(ref):
        print(f'   (no published run at {ref} to compare with)')
        return
    for k in ('labels', 'events', 'links', 'map', 'rotation'):
        a, b = pd.read_csv(os.path.join(out, f'{k}.csv')), pd.read_csv(os.path.join(ref, f'{k}.csv'))
        print(f'   {k:9s} {"identical to the published run" if a.equals(b) else "DIFFERS from the published run"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('flights', nargs='*', help='keys of flights.yaml (default: all)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs'))
    ap.add_argument('--reuse-extract', action='store_true', help='start from the published stage 1 (data/runs/<flight>/extract)')
    ap.add_argument('--demo', action='store_true', help='also render the demo video')
    ap.add_argument('--no-mosaic', action='store_true')
    ap.add_argument('--skip-hash', action='store_true', help='do not check the video SHA-1')
    a = ap.parse_args()
    os.chdir(ROOT)
    F = yaml.safe_load(open('flights.yaml'))
    for k in a.flights or list(F):
        f = F[k]
        out = os.path.join(a.out, k)
        print(f"== {f['name']} ({f['role']}) -> {out}")
        if not a.reuse_extract and not a.skip_hash:
            h = sha1(f['video'])
            if h != f['video_sha1']:
                sys.exit(f"{f['video']}: SHA-1 {h} is not the published video ({f['video_sha1']})")
        ex = os.path.join('data', 'runs', k, 'extract') if a.reuse_extract else None
        run(out, f['video'], f['weights'], 0.5, f.get('segments') or '', ex)
        compare(out, os.path.join('data', 'runs', k))
        if not a.no_mosaic:
            from cabbagetrack.mosaic import build
            build(f['video'], out, extract=ex)
        if f.get('srt') and os.path.exists(f['srt']):
            from cabbagetrack.gnss import compare as gnss
            gnss(out, f['srt'])
        if a.demo:
            from cabbagetrack.demo import render
            render(f['video'], out, f['weights'], os.path.join(out, 'demo.mp4'), title=f"CabbageTrack - {f['name']}")


if __name__ == '__main__':
    main()
