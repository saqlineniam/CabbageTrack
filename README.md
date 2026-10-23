# CabbageTrack

**Recognising the same plant again when a drone flies over it a second time, from the video alone (no GNSS).**

A drone films a field from above in several passes. CabbageTrack gives every plant one identity number and keeps that
number when the drone comes back over the same ground minutes later, so measurements of a plant from different passes
can be joined. It uses only the video: detection and tracking within a pass, camera odometry from the images, and an
image-based "have I been here before?" check that links a later pass to an earlier one.

> Paper: *citation once published* (Computers and Electronics in Agriculture, under revision)

## Demo videos

Every plant is segmented and numbered; when the drone flies over the same ground again, the same plant keeps its
number (crimson outline). Right: the camera path and the plants found so far (orange: seen on one pass, blue:
recognised again) and the latest re-identified plants (earlier pass | now). From the video alone, no GNSS.

All four videos: [YouTube playlist](https://www.youtube.com/playlist?list=PLcuS8bFte5Mc).

**Held-out flight** (frozen model, run once) — click to play:

[![CabbageTrack on the held-out flight](https://img.youtube.com/vi/ySEjaB1Dmok/maxresdefault.jpg)](https://youtu.be/ySEjaB1Dmok)

**Development flights:**

| Field A: full loop over two beds | Field B: young plants, sharp turn | Field C: cauliflower, little overlap |
|:---:|:---:|:---:|
| [![Field A](https://img.youtube.com/vi/s5Dj28gXbIk/mqdefault.jpg)](https://youtu.be/s5Dj28gXbIk) | [![Field B](https://img.youtube.com/vi/n2eswgAq9nE/mqdefault.jpg)](https://youtu.be/n2eswgAq9nE) | [![Field C](https://img.youtube.com/vi/S8zuePsv57Y/mqdefault.jpg)](https://youtu.be/S8zuePsv57Y) |

The same videos are rendered by `cabbagetrack demo` (or `python tools\run_flights.py <flight> --demo`) from the
published runs; archived copies: [10.5281/zenodo.23582580](https://doi.org/10.5281/zenodo.23582580).

## Results

| held-out flight (frozen model, run once) | result |
|---|---|
| blind human check: plants re-identified on a later pass | **90 / 91 (98.9 %)** |
| every cross-pass link checked by a person | **601 / 611 correct (98.4 %)** |
| duplicate identities (same plant, two numbers) | 2 |
| camera path vs drone GNSS (comparison only) | 0.11 m RMSE |

The flight is held out for the re-identification model (designed and tuned on three other flights, then frozen); the
detector's training images include frames of it ([docs/04_detector.md](docs/04_detector.md)).

## How it works

```
video ─► 1 detect + track ─► 2 camera rotation ─► 3 re-identify across passes ─► plant identities
         (YOLO segmentation,     (ORB between          (earlier frames that overlap the current view,
          BoT-SORT, image         consecutive           verified by image alignment, remembered plants
          odometry)               frames)               moved into the current view and matched)
```

1. **Detect, track, odometry** - a YOLOv9c-seg model segments plants (and aisles, dividers, drains); BoT-SORT keeps
   a plant's track within a pass; phase correlation of consecutive frames gives the camera path.
2. **Rotation** - the rotation between consecutive frames (ORB features); frames inside a fast turn are marked.
3. **Re-identification** - for every new track, earlier frames whose view overlaps the current one are looked up on
   the camera path; each candidate is verified by aligning the overlapping image parts (after correcting the
   rotation between the two views), the odometry drift is corrected from recent confirmed revisits, and the plants
   remembered around the linked earlier frame are moved into the current image and matched to the new tracks.

All settings are fixed in [`cabbagetrack/config.py`](cabbagetrack/config.py); none is tuned per field.
Details: [docs/03_tracking.md](docs/03_tracking.md).

## Quick start (Windows, Python 3.12, NVIDIA GPU)

```bat
git clone https://github.com/saqlineniam/CabbageTrack.git
cd CabbageTrack
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

Run it on a video (plant = class 1 of the detector):

```bat
cabbagetrack all --video my_flight.mp4 --weights data\weights\cabbage.pt --out runs\my_flight
```

This writes the plant identities (`labels.csv`), every re-identification (`events.csv`, `links.csv`), the camera
path (`map.csv`), a mosaic of the field from the video with every plant marked, and a demo video. Each step can also
be run alone: `cabbagetrack run | mosaic | demo | gnss` (see `cabbagetrack -h`).

Full setup, including CPU-only use: [docs/01_setup_windows.md](docs/01_setup_windows.md).

## Reproduce the paper

The data record (videos, GNSS logs, annotated detector images, detector weights, human ground truth, published
runs) is on Zenodo: [10.5281/zenodo.23582580](https://doi.org/10.5281/zenodo.23582580). Join the split archives and unpack them into `data/` ([docs/02_data.md](docs/02_data.md)), then:

```bat
python tools\run_flights.py                  :: all four flights, checked against the published runs
python groundtruth\build_pairs.py            :: human answers -> position pairs, checked
python groundtruth\score_paper.py            :: the 19 ground-truth numbers of the paper, checked
python paper\reproduce.py                    :: every table and figure of the paper
```

| step | what | guide |
|---|---|---|
| 1 | set up Python and the GPU | [docs/01_setup_windows.md](docs/01_setup_windows.md) |
| 2 | get the data | [docs/02_data.md](docs/02_data.md) |
| 3 | **the tracker** (stages, settings, outputs) | [docs/03_tracking.md](docs/03_tracking.md) |
| 4 | train and evaluate the detectors (Table 1) | [docs/04_detector.md](docs/04_detector.md) |
| 5 | human ground truth: annotation tools and scoring | [docs/05_ground_truth.md](docs/05_ground_truth.md) |
| 6 | how the model was developed, frozen and tested once | [docs/06_development_and_heldout.md](docs/06_development_and_heldout.md) |
| 7 | every table and figure | [docs/07_reproduce_paper.md](docs/07_reproduce_paper.md) |

## Repository

| folder | content |
|---|---|
| `cabbagetrack/` | the tracker (Python package and command line) |
| `tools/` | run the paper's flights, prepare videos (4K to 1080p, joining two clips of one flight) |
| `detector/` | detector datasets, training, evaluation (Table 1) |
| `groundtruth/` | annotation tools for the human checks and their scoring |
| `paper/` | development steps, settings check, held-out scoring, figures |
| `docs/` | step-by-step guides |
| `flights.yaml` | the four flights: video, detector, GNSS log |

## Citation

See [CITATION.cff](CITATION.cff).

## Licence

Code: GNU AGPL-3.0 ([LICENSE](LICENSE); the detector library, Ultralytics, is AGPL-3.0).
Data and videos (Zenodo record): CC BY 4.0.
