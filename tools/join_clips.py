"""
Join two clips of one flight (recording paused while the drone hovered) into one video, without re-encoding, and
join their DJI .SRT logs: clip-2 records are renumbered after clip 1 (sequence number and FrameCnt + frames of clip 1).
Run the pipeline on the joined video with --segments 0-<n1-1>,<n1>-<n1+n2-1> so the tracker restarts at the join.

Usage: python tools/join_clips.py clip1.mp4 clip2.mp4 joined.mp4 [--srt clip1.SRT clip2.SRT joined.SRT]
"""
import argparse
import os
import re
import subprocess
import tempfile

import cv2
import imageio_ffmpeg


def frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def join_video(a, b, out):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for p in (a, b):
            f.write(f"file '{os.path.abspath(p)}'\n")
        lst = f.name
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', out],
                   check=True)
    os.remove(lst)


def join_srt(a, b, out, n1):
    A = open(a, encoding='utf8').read().rstrip('\n') + '\n\n'
    blocks = open(b, encoding='utf8').read().strip('\n').split('\n\n')
    res = []
    for blk in blocks:
        lines = blk.split('\n')
        lines[0] = str(int(lines[0]) + n1)
        blk = '\n'.join(lines)
        res.append(re.sub(r'FrameCnt: (\d+)', lambda m: f'FrameCnt: {int(m.group(1)) + n1}', blk, count=1))
    with open(out, 'w', encoding='utf8') as f:                         # platform line endings (the paper's: Windows)
        f.write(A + '\n\n'.join(res) + '\n\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('clip1'); ap.add_argument('clip2'); ap.add_argument('out')
    ap.add_argument('--srt', nargs=3, metavar=('SRT1', 'SRT2', 'OUT_SRT'))
    a = ap.parse_args()
    n1, n2 = frames(a.clip1), frames(a.clip2)
    join_video(a.clip1, a.clip2, a.out)
    if a.srt:
        join_srt(*a.srt, n1)
    print(f'joined {n1} + {n2} frames -> {a.out}; run with --segments 0-{n1 - 1},{n1}-{n1 + n2 - 1}')
