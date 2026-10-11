"""
Stage 1: detection + tracking + camera odometry, run once per video.

For every frame it records
  * the frame-to-frame camera motion from phase correlation of consecutive full-resolution grayscale frames
    (tx, ty, peak response),
  * every detection with its BoT-SORT track id, class, confidence, box, mask centroid / area and a 5-coefficient
    Fourier shape descriptor,
  * a 1/4-scale grayscale copy of the frame (used for rotation, link verification and the annotation tools),
  * per-stage wall-clock timing.

Outputs (<out>/): detections.csv.gz, motion.csv, frames_small.npy, meta.json

Usage: python -m cabbagetrack.extract --video V.mp4 --weights best.pt --out out/extract [--conf 0.5] [--segments ...]
"""
import argparse
import hashlib
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

FOURIER_DESC_N = 5
SMALL_SCALE = 4          # stored frames are (W/4, H/4) grayscale


def file_sha1(path, chunk=1 << 24):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_fourier_desc(mask_data, n=FOURIER_DESC_N):
    """Centroid-distance FFT magnitudes 1..n of the mask outline, normalised by the first harmonic (None if degenerate)."""
    mask_uint8 = (mask_data > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 12:
        return None
    pts = cnt.squeeze()
    if pts.ndim == 1:
        return None
    M = cv2.moments(cnt)
    if M['m00'] < 1e-6:
        return None
    cx_ = M['m10'] / M['m00']
    cy_ = M['m01'] / M['m00']
    r = np.sqrt((pts[:, 0] - cx_) ** 2 + (pts[:, 1] - cy_) ** 2)
    fft = np.fft.fft(r)
    desc = np.abs(fft[1:n + 1]).astype(np.float32)
    if desc[0] > 1e-6:
        desc = desc / desc[0]
    return desc


def polygon_stats(poly):
    """Centroid and area of a mask polygon given in original image pixels."""
    if poly is None or len(poly) < 3:
        return np.nan, np.nan, np.nan
    M = cv2.moments(poly.astype(np.float32))
    if M['m00'] < 1e-6:
        return np.nan, np.nan, np.nan
    return M['m10'] / M['m00'], M['m01'] / M['m00'], M['m00']


def extract(video, weights, out, conf=0.5, tracker='botsort.yaml', segments='', max_frames=0):
    """Run stage 1 on a video. segments, e.g. "0-4696,4697-8174": process only these source-frame ranges as one
    video (two clips of one flight joined); the tracker restarts at each segment, the camera path continues, and
    track ids of segment k are offset by k*100000."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    os.makedirs(out, exist_ok=True)
    model = YOLO(weights)
    names = model.names

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if segments:
        segs = [tuple(int(v) for v in part.split('-')) for part in segments.split(',')]
        src_frames = [f for a_, b_ in segs for f in range(a_, b_ + 1)]
        seg_of = [k for k, (a_, b_) in enumerate(segs) for _ in range(a_, b_ + 1)]
    else:
        src_frames, seg_of = list(range(n_frames)), [0] * n_frames
    if max_frames:
        src_frames, seg_of = src_frames[:max_frames], seg_of[:max_frames]
    n_frames = len(src_frames)

    small = np.lib.format.open_memmap(os.path.join(out, 'frames_small.npy'), mode='w+', dtype=np.uint8,
                                      shape=(n_frames, height // SMALL_SCALE, width // SMALL_SCALE))

    det_rows, motion_rows = [], []
    prev_gray = None
    frame_idx = 0
    t_start = time.time()
    src_pos = 0
    while frame_idx < n_frames:
        want = src_frames[frame_idx]
        if want != src_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            src_pos = want
        ok, frame = cap.read()
        if not ok:
            break
        src_pos += 1
        seg = seg_of[frame_idx]
        if frame_idx > 0 and seg != seg_of[frame_idx - 1]:
            pred = getattr(model, 'predictor', None)
            for trk in (getattr(pred, 'trackers', None) or []):
                trk.reset()
            print(f'segment {seg} starts at frame {frame_idx} (source frame {want}); tracker reset', flush=True)
        t0 = time.perf_counter()
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small[frame_idx] = cv2.resize(cur_gray, (width // SMALL_SCALE, height // SMALL_SCALE),
                                      interpolation=cv2.INTER_AREA)

        # camera odometry: phase correlation of consecutive full-resolution frames
        t1 = time.perf_counter()
        tx = ty = resp = 0.0
        if prev_gray is not None:
            (sx, sy), resp = cv2.phaseCorrelate(prev_gray.astype(np.float32), cur_gray.astype(np.float32))
            tx, ty = -float(sx), -float(sy)
        t2 = time.perf_counter()

        # detection + BoT-SORT tracking
        results = model.track(frame, persist=True, verbose=False, conf=conf, tracker=tracker, save=False, show=False)
        t3 = time.perf_counter()
        res = results[0]
        boxes = res.boxes
        masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        polys = res.masks.xy if res.masks is not None else None
        if boxes is not None and len(boxes) > 0:
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            cls = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for i in range(len(boxes)):
                fd = compute_fourier_desc(masks[i]) if masks is not None and i < len(masks) else None
                mcx, mcy, marea = polygon_stats(polys[i]) if polys is not None and i < len(polys) else (np.nan,) * 3
                x1, y1, x2, y2 = xyxy[i]
                row = {
                    'frame': frame_idx, 'det_idx': i,
                    'raw_id': int(ids[i]) + seg * 100000 if ids is not None else -1,
                    'cls': int(cls[i]), 'cls_name': names[int(cls[i])], 'conf': float(confs[i]),
                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                    'cx': float((x1 + x2) / 2), 'cy': float((y1 + y2) / 2),
                    'mask_cx': mcx, 'mask_cy': mcy, 'mask_area': marea,
                }
                for k in range(FOURIER_DESC_N):
                    row[f'fd{k}'] = float(fd[k]) if fd is not None else np.nan
                det_rows.append(row)
        t4 = time.perf_counter()

        motion_rows.append({'frame': frame_idx, 'src_frame': want, 'segment': seg,
                            'tx': tx, 'ty': ty, 'pc_response': float(resp),
                            't_read_ms': (t1 - t0) * 1e3, 't_phasecorr_ms': (t2 - t1) * 1e3,
                            't_detect_track_ms': (t3 - t2) * 1e3, 't_descriptors_ms': (t4 - t3) * 1e3})
        prev_gray = cur_gray
        frame_idx += 1
        if frame_idx % 200 == 0:
            el = time.time() - t_start
            print(f'frame {frame_idx}/{n_frames}  {frame_idx / el:.1f} fps  detections so far {len(det_rows)}', flush=True)
    cap.release()

    dets = pd.DataFrame(det_rows)
    motion = pd.DataFrame(motion_rows)
    dets.to_csv(os.path.join(out, 'detections.csv.gz'), index=False)
    motion.to_csv(os.path.join(out, 'motion.csv'), index=False)

    tracker_cfg_path = os.path.join(os.path.dirname(ultralytics.__file__), 'cfg', 'trackers', tracker)
    meta = {
        'video': os.path.abspath(video), 'video_sha1': file_sha1(video),
        'weights': os.path.abspath(weights), 'weights_sha1': file_sha1(weights),
        'conf': conf, 'tracker': tracker, 'segments': segments,
        'tracker_cfg': open(tracker_cfg_path).read() if os.path.exists(tracker_cfg_path) else None,
        'class_names': {int(k): v for k, v in names.items()},
        'frames': frame_idx, 'width': width, 'height': height, 'fps': fps,
        'small_scale': SMALL_SCALE,
        'versions': {'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
                     'opencv': cv2.__version__, 'numpy': np.__version__},
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'wall_time_s': time.time() - t_start,
    }
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'stage 1 done: {frame_idx} frames, {len(dets)} detections, {meta["wall_time_s"]:.0f} s -> {out}')
    return out


def main():
    ap = argparse.ArgumentParser(description='CabbageTrack stage 1: detection + tracking + camera odometry')
    ap.add_argument('--video', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--tracker', default='botsort.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=0, help='debug: stop early')
    ap.add_argument('--segments', default='', help='e.g. "0-4696,4697-8174": join source-frame ranges (two clips)')
    a = ap.parse_args()
    extract(a.video, a.weights, a.out, a.conf, a.tracker, a.segments, a.max_frames)


if __name__ == '__main__':
    main()
