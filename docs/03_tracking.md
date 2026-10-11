# 3. The tracker

CabbageTrack turns a nadir drone video into one identity number per plant, kept across passes. This page describes
what each stage does, every setting, and every output file. The code is in [`cabbagetrack/`](../cabbagetrack).

## Run

```bat
cabbagetrack run --video V.mp4 --weights cabbage.pt --out runs\X            :: stages 1-3
cabbagetrack run --extract runs\X\extract --out runs\X2                    :: stages 2-3 on a finished stage 1 (no GPU)
cabbagetrack mosaic --video V.mp4 --run runs\X                             :: mosaic from the video + plant marks
cabbagetrack demo --video V.mp4 --run runs\X --weights cabbage.pt --out runs\X\demo.mp4
cabbagetrack gnss --run runs\X --srt V.SRT                                 :: camera path vs drone GNSS (evaluation)
cabbagetrack all --video V.mp4 --weights cabbage.pt --out runs\X [--srt V.SRT]
```

`--segments 0-4696,4697-8174` is for one flight recorded as two clips and joined into one video (Field B): the
tracker restarts at the join and the camera path continues, which needs the two clips to overlap at the join (the
drone hovered between them).

From Python:

```python
from cabbagetrack.pipeline import run
run('runs/X', video='V.mp4', weights='cabbage.pt')
```

Requirements on the input: a nadir (straight-down) video at 1920 x 1080 (convert 4K with
`python tools/convert_video.py`), and a segmentation detector whose class 1 is the plant. The settings are in image
pixels for flights at about the paper's height (ground resolution ~3-4 mm per pixel).

## Stage 1 - detection, tracking, camera odometry (`cabbagetrack/extract.py`)

For every frame:
* **Detection + tracking.** Ultralytics YOLO segmentation (confidence 0.5) with the BoT-SORT tracker
  (Ultralytics `botsort.yaml`, unchanged) gives each plant a track id while it stays in view.
* **Camera odometry.** `cv2.phaseCorrelate` of the previous and the current full-resolution grayscale frame gives the
  image shift; its negative is the camera motion. Summed over frames this is the camera path (`map.csv`).
  Steps larger than 25 px per frame are treated as failures and set to 0 (never happened in the paper's flights).
* A 1/4-scale grayscale copy of every frame (480 x 270) is stored for the next stages.

Outputs in `<out>/extract/`: `detections.csv.gz` (one row per detection: frame, track id `raw_id`, class, confidence,
box, mask centroid and area, shape descriptor), `motion.csv` (per frame: shift, peak, timings), `frames_small.npy`,
`meta.json` (video / weights SHA-1, versions, GPU, run time).

## Stage 2 - camera rotation (`cabbagetrack/rotation.py`)

ORB features (1500 per frame) are matched between consecutive 1/4-scale frames and a similarity transform is fitted
with RANSAC (at least 12 inliers); its angle is the frame-to-frame rotation phi. Frames turning faster than
**0.3 deg per frame**, and 15 frames on either side, are marked as *inside a fast turn* (`rotation.csv`). The
odometry is translation-only; the rotation is used in stage 3.

## Stage 3 - re-identification (`cabbagetrack/reid.py`)

A tracker track gets an identity once it is 3 frames old. A new track either continues an identity seen on an earlier
visit of the same ground (re-identification) or starts a new identity:

1. **Have I been here before?** Earlier frames (older than 45 frames) whose view overlaps the current view by at least
   25 % on the camera path are candidates; frames inside a fast turn are skipped. Candidates are grouped into visits
   (gaps over 15 frames separate visits); the 4 closest visits are checked, each with its 3 closest frames.
2. **Verify the link.** The predicted shift between the two views is the odometry shift, corrected by the median
   residual of recent confirmed revisits (links made in the last 60 frames whose earlier frame lies within 300 frames,
   counting only links with a clear peak >= 0.2). The rotation between the two views is measured (ORB, at least 15
   inliers) and, when it is at least 1 deg, the earlier view is rotated by it. The overlapping parts of the two
   1/4-scale views are phase-correlated (Hann window). A link is accepted when the peak is at least 0.08 and the
   measured shift lies within 400 px of the prediction; per visit the strongest link is kept.
3. **Move the remembered plants.** Plants observed within 60 frames of the linked earlier frame (not inside a fast
   turn, not currently in view) are moved into the current image with the measured rotation and shift.
4. **Match.** New tracks are matched to the moved plants with the Hungarian algorithm (gate 60 px). A track over known
   ground that finds no match waits up to 60 frames before it starts a new identity; over new ground it starts one at
   once.

A re-identification more than 300 frames (10 s) after the linked frame is counted as *cross-pass*.

## Settings (`cabbagetrack/config.py`)

| setting | value | meaning |
|---|---|---|
| `plant_class` | 1 | detector class of the plant |
| `min_track_frames` | 3 | track age before it gets an identity |
| `max_frame_disp` | 25 px | larger frame-to-frame steps count as odometry failures |
| `recent_gap` | 45 frames | only frames older than this count as an earlier visit |
| `min_overlap` | 0.25 | share of the image an earlier frame must overlap |
| `visit_gap` | 15 frames | candidate frames further apart belong to different visits |
| `max_visits` | 4 | earlier visits checked |
| `n_link_candidates` | 3 | frames verified per visit |
| `min_response` | 0.08 | minimum alignment peak |
| `shift_tol` | 400 px | allowed difference between measured and predicted shift |
| `rot_min_deg` | 1 deg | rotation correction applied from this angle |
| `drift_recent` / `drift_window` / `drift_min_peak` | 60 / 300 frames / 0.2 | drift correction from recent clear revisits |
| `turn_rate` / `turn_margin` | 0.3 deg per frame / 15 frames | fast-turn frames |
| `window` | 60 frames | plants remembered around the linked frame |
| `gate` | 60 px | maximum distance for a match |
| `patience` | 60 frames | wait before a new identity over known ground |
| `live_gap` | 5 frames | identities seen this recently are not candidates |
| `cross_pass_gap` | 300 frames | re-identifications over this gap are cross-pass |

All values were fixed on the development flights (Fields A-C) and frozen before the held-out flight was run
([06_development_and_heldout.md](06_development_and_heldout.md)). `config.py`, `rotation.py` and `reid.py` are
byte-identical to the frozen model; `summary.json` records their SHA-1 (`model_sha1`).

## Outputs (`--out`)

| file | content |
|---|---|
| `labels.csv` | every plant detection with its identity: `frame, raw_id, id, cx, cy` (box centre, px) |
| `events.csv` | every identity decision: `new` (reason `no_memory` = new ground, `no_match` = after waiting) or `reid` (distance, linked frame `link_s`, `gap`, peak `resp`, frame the plant was remembered from `mem_frame`) |
| `links.csv` | every verified link: frame, earlier frame `s`, measured shift `dx, dy`, peak `resp`, predicted shift `exp_dx, exp_dy`, rotation `rot` |
| `map.csv` | camera path (px, x right, y down) |
| `rotation.csv` | rotation per frame and fast-turn flag |
| `summary.json` | counts, all settings, provenance (model / video / detector SHA-1) |
| `mosaic.jpg`, `mosaic_identities.jpg` | (mosaic) field from the video, camera path refined with every link; plants marked blue (revisited) / orange |
| `gnss_summary.json`, `gnss_per_frame.csv` | (gnss) RMSE after one similarity fit, fitted scale (mm per px) |
| `demo.mp4` | (demo) the video with plants segmented and numbered, live map, re-identified plants |

## Run time (RTX 4060, held-out flight, 2966 frames)

Stage 1 about 4-6 min (11-12 frames per second); stages 2-3 about 1 min.
