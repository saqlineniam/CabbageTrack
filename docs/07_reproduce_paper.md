# 7. Reproduce the paper

With the data record in `data/` (docs/02_data.md):

```bat
python tools\run_flights.py                 :: 1. rerun the tracker on the four flights, check against the published runs
python groundtruth\build_pairs.py           :: 2. human answers -> position pairs (checked against the published pairs)
python groundtruth\score_paper.py           ::    the human ground truth on the runs (19 numbers checked)
python paper\reproduce.py                   :: 3. every table and figure (add --quick to skip replaying the development)
```

Every step compares its result with the published one and says *identical* or *differs*.

| paper | what | command | output |
|---|---|---|---|
| Table 1 (+ S1, statistics) | detector benchmark | `paper\reproduce.py detector` (from the published evaluations); retraining: docs/04 | `outputs/detector/` |
| detector numbers | training data, test scores, held-out overlap | `paper\detector_numbers.py` | `outputs/detectors/README.md` |
| development results | Fields A-C, frozen model | `groundtruth\score_paper.py` | `outputs/scores/gt_scores.md` |
| held-out results | frozen model vs Field A method | `groundtruth\score_paper.py` | same |
| Figure 1, 3 | concept, pipeline (drawings) | `paper\figures\figures.py 01 03` | `outputs/figures/` |
| Figure 2 | study site (from the submission) | `paper\figures\figures.py 02` | |
| Figure 4 | detector output, one frame per flight | `paper\figures\figures.py 04` | |
| Figure 5 | development steps | `paper\development\steps.py steps` then `figures.py 05` | |
| Figure 6 | identity maps, Fields A-C | `figures.py 06` | |
| Figure 7 | mosaics with odometry and GNSS | `paper\reproduce.py gnss mosaic` then `figures.py 07` | |
| Figure 8 | odometry vs GNSS, true scale | `paper\reproduce.py gnss` then `figures.py 08` | |
| Figure 9 | Field A, row A: first and last pass | `figures.py 09` | |
| Figure 10, 11 | held-out: map, mosaic, re-identified plants | `figures.py 10 11` | |
| Figure 12 | settings check | `paper\development\steps.py settings` then `figures.py 12` | |
| Figures S1-S5 | tool, sharp turn, Field B before/after, Fields B and C pass pairs | `figures.py S1 S2 S3 S4 S5` | `outputs/figures/supplementary/` |

## What was checked before release

On the authors' machine (Windows 11, RTX 4060, the versions in `requirements.txt`):

* the tracker from the raw held-out video: detections, odometry, every output file identical to the published run;
* stages 2-3 on all four flights: identical to the published runs; GNSS RMSE 0.22 / 0.71 / 0.16 / 0.11 m;
* the position pairs rebuilt from the raw answers: identical to the published pairs;
* the human ground truth: all 19 numbers the paper reports;
* the last development step equals the frozen model on all four flights; the development steps and the settings
  check reproduce the published tables;
* Table 1 and the detector numbers from the published evaluations; the fine-tuned detectors' scores from the published
  weights;
* the mosaics and all figures: pixel-identical to the paper's.

Detection (stage 1) on another GPU or library version can differ in rare detections; everything after detection is
deterministic.
