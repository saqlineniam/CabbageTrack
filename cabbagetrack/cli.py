"""
Command line:

  python -m cabbagetrack run    --video V.mp4 --weights best.pt --out runs/X [--conf 0.5] [--segments a-b,c-d]
  python -m cabbagetrack run    --extract runs/X/extract --out runs/X2          (reuse a finished stage 1)
  python -m cabbagetrack mosaic --video V.mp4 --run runs/X [--step 4] [--scale 0.5]
  python -m cabbagetrack demo   --video V.mp4 --run runs/X --weights best.pt --out demo.mp4 [--start 0 --end 900]
  python -m cabbagetrack gnss   --run runs/X --srt flight.SRT                   (evaluation only)
  python -m cabbagetrack all    --video V.mp4 --weights best.pt --out runs/X [--srt flight.SRT]   (everything)
"""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog='cabbagetrack', description='Plant identities across passes of a nadir drone video')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('run', help='detection + tracking + odometry, rotation, re-identification')
    p.add_argument('--video')
    p.add_argument('--weights', help='Ultralytics YOLO segmentation weights, plant = class 1')
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--segments', default='', help='two clips of one flight joined: source-frame ranges, e.g. 0-4696,4697-8174')
    p.add_argument('--extract', default='', help='reuse a finished stage-1 folder instead of --video/--weights')
    p.add_argument('--out', required=True)

    p = sub.add_parser('mosaic', help='mosaic of the field from its own video and camera path')
    p.add_argument('--video', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--step', type=int, default=4)
    p.add_argument('--scale', type=float, default=0.5)

    p = sub.add_parser('demo', help='demo video (segmentation, identity numbers, live map, re-identified plants)')
    p.add_argument('--video', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True, help='output .mp4')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=0)
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--title', default='CabbageTrack')

    p = sub.add_parser('gnss', help='compare the camera path with the drone GNSS log (evaluation only)')
    p.add_argument('--run', required=True)
    p.add_argument('--srt', required=True)
    p.add_argument('--out', default='')

    p = sub.add_parser('all', help='run + mosaic + demo (+ gnss when --srt is given)')
    p.add_argument('--video', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--segments', default='')
    p.add_argument('--srt', default='')
    p.add_argument('--title', default='CabbageTrack')

    a = ap.parse_args(argv)
    if a.cmd in ('run', 'all'):
        from .pipeline import run
        if a.cmd == 'run' and not a.extract and not (a.video and a.weights):
            ap.error('give --video and --weights, or --extract')
        run(a.out, a.video, a.weights, a.conf, a.segments, getattr(a, 'extract', '') or None)
    if a.cmd in ('mosaic', 'all'):
        from .mosaic import build
        build(a.video, getattr(a, 'run', None) or a.out, getattr(a, 'out', '') if a.cmd == 'mosaic' and a.out else None,
              step=getattr(a, 'step', 4), scale=getattr(a, 'scale', 0.5))
    if a.cmd in ('gnss',) or (a.cmd == 'all' and a.srt):
        from .gnss import compare
        compare(getattr(a, 'run', None) or a.out, a.srt, (a.out or None) if a.cmd == 'gnss' else None)
    if a.cmd in ('demo', 'all'):
        import os
        from .demo import render
        if a.cmd == 'demo':
            render(a.video, a.run, a.weights, a.out, a.start, a.end or None, a.step, a.conf, a.title)
        else:
            render(a.video, a.out, a.weights, os.path.join(a.out, 'demo.mp4'), title=a.title)


if __name__ == '__main__':
    main()
