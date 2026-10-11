# 6. Development, freeze, held-out run

## 6.1 Development on Fields A, B, C

Starting from the Field A method (v1, the method of the submitted paper), the model was improved one step at a time on
the development flights. Each step: run on Fields A-C, score with the human ground truth, diagnose every pair the
model misses (`paper/development/diagnose.py`), fix the largest cause, check that nothing else gets worse.

| step | largest cause | fix |
|---|---|---|
| 1 | - | Field A method (v1) |
| 2 | Field B: the earlier pass was never searched (outside the fixed search circle) | search earlier frames by image overlap, one link per earlier visit |
| 3 | Field B: clear alignment, but far from the drifted position estimate | drift correction from confirmed revisits |
| 4 | Field B: weak alignment after the sharp turn (views rotated ~3 deg) | rotation correction |
| 5 | Field B: the frames tried were inside the turn | skip frames inside fast turns |
| 6 | Field C: passes share only ~25 % of the image | align the overlapping part only |
| 7 | Field C: noisy links mislead the drift correction | drift correction only from clear links (peak >= 0.2) |
| 8 | settings check (below) | patience 60 frames |

Field A stays at 100 % throughout. Share of human-checked pairs re-identified (blind + links): Field B 73.4 -> 76.8 ->
94.5 -> 98.1 -> 99.4 -> 99.4 -> 99.4 -> 99.0 %; Field C 0 until step 6, then 35.9 -> 65.4 -> 68.6 % (Figure 5).
The log of each step (causes, alternatives tried, decision): `data/results/development/development_log.md`.

Every component is a switch of the development implementation `paper/development/revisit_exp.py`; with the switches
of step 8 it gives exactly the frozen model in `cabbagetrack/`:

```bat
python paper\development\steps.py steps fieldA fieldB fieldC      :: every step, scored -> outputs\development\dev_steps_*.csv
python paper\development\steps.py check fieldA fieldB fieldC heldout   :: step 8 == cabbagetrack/ on every flight
```

## 6.2 Settings check

One setting at a time around the current value, on Fields A-C (`steps.py settings`; Figure 12): gate 40 / 60 / 80 px,
image overlap 0.15 / 0.25 / 0.35, shift tolerance 300 / 400 / 600 px, drift peak 0.15 / 0.2 / 0.3, fast turn
0.2 / 0.3 / 0.5 deg per frame, minimum rotation 0.5 / 1 / 2 deg, patience 15 / 30 / 60 frames. Rule fixed before the
check: keep the current value unless another value gains at least 2 correct human pairs over A + B + C without more
wrong links. Only patience 60 qualified (+2 correct, wrong links 15 -> 14).

## 6.3 Freeze and held-out run

The model was frozen (`cabbagetrack/`; protocol with the code hash written before the run:
`data/results/heldout_DJI0004/PROTOCOL.md`) and run once on the held-out flight with the cabbage detector.

| held-out flight | frozen model | Field A method (v1) |
|---|---|---|
| blind plants re-identified | 90 / 91 (98.9 %, 95 % CI 94.0-99.8) | 73 / 91 |
| every cross-pass link checked | 601 / 611 correct (98.4 %, CI 97.0-99.1) | - |
| identities | 374 | 592 |
| duplicate identity pairs | 2 | 142 |

The held-out flight is held out for the re-identification model; the cabbage detector's training images include
frames of this flight (docs/04_detector.md). How its human checks were collected: docs/05_ground_truth.md.

## 6.4 Where it stops

Field C is the limit: when two passes share only about a quarter of the image, aligning the views is too coarse
(43 / 74 blind plants). Plants seen only inside a fast turn, or only at the image border, are the other failures.
