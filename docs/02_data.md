# 2. Data

Everything that is not code is in the Zenodo record *DOI* (CC BY 4.0). Download it and unpack it into `data/` in the
repository root, so that `data/videos/heldout.mp4` exists. `data/` is ignored by git.

## 2.1 Layout

```
data/
  videos/        fieldA.mp4  fieldB.mp4  fieldC.mp4  heldout.mp4      1920 x 1080, the exact files the paper used
  gnss/          fieldA.SRT  fieldB.SRT  fieldC.SRT  heldout.SRT      drone logs (one record per frame), evaluation only
                 fieldB_clip1.SRT  fieldB_clip2.SRT                    Field B's two original logs
  weights/       cabbage.pt  young_cabbage.pt  cauliflower.pt         detectors used (YOLOv9c-seg, plant = class 1)
  detector/      cabbage/  young_cabbage/  cauliflower/               annotated images, splits, manifests (docs/04)
  groundtruth/   tools/  answers/  pairs/  tool_runs/                 human checks (docs/05)
  runs/          fieldA/  fieldB/  fieldC/  heldout/  heldout_v1/     published runs, each with its stage-1 extract/
  results/       tables and figures of the paper
  SHA1SUMS.txt   checksum of every file
```

Check the download (Git Bash: `cd data && sha1sum -c SHA1SUMS.txt`).

## 2.2 Flights

| key | name | date | role | frames | detector |
|---|---|---|---|---|---|
| `fieldA` | Field A | 26 Dec 2025 | development: full loop over the cabbage field | 6354 | cabbage |
| `fieldB` | Field B | 14 Dec 2025 | development: same field, young plants, sharp turn, two clips joined | 8175 | young cabbage |
| `fieldC` | Field C | 14 Dec 2025 | development (supplementary): cauliflower, passes share ~25 % of the image | 2862 | cauliflower |
| `heldout` | Held-out flight | 26 Dec 2025 | held-out test, frozen model run once | 2966 | cabbage |

`flights.yaml` lists video, detector, GNSS log, SHA-1 and the Field B join for each flight.

## 2.3 How the videos were prepared

* **4K to 1080p.** Videos recorded in 4K were reduced to 1920 x 1080 with ffmpeg (bundled with `imageio-ffmpeg`):
  `python tools/convert_video.py input_4k.MP4 output_1080p.mp4`.
* **Field B** was recorded as two clips (the drone hovered between them). The clips were joined into one video and the
  pipeline is run with `--segments 0-4696,4697-8174`, so the tracker restarts at the join while the camera path
  continues; their GNSS logs were joined the same way (clip-2 frame numbers + 4697):
  `python tools/join_clips.py clip1.mp4 clip2.mp4 fieldB.mp4 --srt clip1.SRT clip2.SRT fieldB.SRT`.

## 2.4 Published runs (`data/runs/`)

Each folder holds what `cabbagetrack run` writes (labels, events, links, map, rotation, summary) plus `extract/`
(stage 1). With them, everything after detection reruns without a GPU and gives identical files:

```bat
cabbagetrack run --extract data\runs\heldout\extract --out runs\heldout_check
python tools\run_flights.py heldout --reuse-extract
```

`heldout_v1/` is the Field A method (before development, Section 6) on the held-out flight, for comparison.
