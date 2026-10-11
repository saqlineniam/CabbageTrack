"""
Reduce a 4K drone video to 1920 x 1080 the way the paper's videos were made: ffmpeg area scaling, H.264, CRF 12
(near-lossless), audio dropped, every frame kept.

Usage: python tools/convert_video.py input_4k.MP4 output_1080p.mp4
"""
import subprocess
import sys

import imageio_ffmpeg


def convert(src, dst):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', src, '-vf', 'scale=1920:1080:flags=area',
           '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', '-an', dst]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
