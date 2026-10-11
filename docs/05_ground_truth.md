# 5. Human ground truth

Every number about re-identification in the paper comes from people checking the method's output in a browser tool.

## 5.1 The annotation tool (`groundtruth/make_tool.py`)

A self-contained web page (`index.html` + `tasks.js` + `img/`) that works offline. Two parts:

* **Part A - links.** Every cross-pass re-identification of a run is shown as two crops: the plant on the later pass
  and the plant the method says it is on an earlier pass. The annotator answers *same / different / unsure*.
  About 10 % of the pairs are hidden **decoys** (the earlier plant replaced by a neighbour) to check that the annotator
  is paying attention.
* **Part B - blind.** A plant is circled in an earlier photo; the annotator clicks the same plant in a later-pass photo
  of the same ground, among all detections. The later photo is chosen by whole-image alignment only; the method's
  identities are never shown. This measures re-identification without showing the method's answer.

Answers are exported from the page as CSV (one file per annotator).

```bat
python groundtruth\make_tool.py runs\heldout --extract runs\heldout\extract --video data\videos\heldout.mp4 ^
       --out tools\heldout_check --tag heldout --n-blind 100
```

## 5.2 The published checks (`data/groundtruth/`)

`tools/index.csv` lists every tool: flight, number of links, decoys and blind plants, random seed, and the run it was
made from. `tools/<tool>.zip` is the tool as the annotators used it; `answers/<flight>/` holds the exported answers.

| flight | tools | annotators |
|---|---|---|
| Field A | `gt_tool` (372 links, 37 decoys, 50 blind), `gt_tool_blind2` (150 blind), `gt_tool_rowA` (100 blind) | two (Saklain, Mashrup) |
| Field B | `gt_tool_early_loop` (561 links, 56 decoys, 100 blind), `gt_tool_early_loop_v2rdt` (131 links, 13 decoys) | one |
| Field C | `gt_tool_cauliflower_v3` (135 links, 14 decoys, 75 blind) | one |
| held-out | `gt_tool_field2` (406 links, 41 decoys, 100 blind), `gt_tool_field2_v2` (222 + 22), `gt_tool_field2_v3` (27 + 3), `gt_tool_heldout_DJI0004_followup` (10 + 2) | two for `gt_tool_field2`, one for the others |

The held-out flight's checks were collected in four rounds: links made by earlier versions of the method on this
flight (`gt_tool_field2`, `_v2`, `_v3`; the run each was made from is in `index.csv`) and a follow-up round for the
frozen model's last 10 links that no earlier answer covered. Blind plants come only from the first round.

## 5.3 From answers to position pairs (`groundtruth/score.py`)

Answers are turned into pairs of **image positions**, independent of any run:
earlier sighting `(frame_a, xa, ya)` <-> later sighting `(frame_b, xb, yb)`, `same` = True / False. Only answers on
which all annotators of a tool agree are used. Pair files (`data/groundtruth/pairs/`):

| file | flight | content |
|---|---|---|
| `gt_pairs.csv` | Field A (`seq` f1_rowB, f1_rowA), held-out (`seq` f2) | first-round tools |
| `gt_pairs_early_loop.csv` | Field B | both Field B tools |
| `gt_pairs_cauli.csv` | Field C | |
| `gt_pairs_f2v2.csv` | held-out | second round |
| `heldout_table_pairs.csv` | held-out | rounds 1-3 (the paper's held-out table) |
| `heldout_followup_pairs.csv` | held-out | round 4 (follow-up) |
| `gt_pairs_heldout_DJI0004_all.csv` | held-out | rounds 1-4 |

`tool_pairs()` makes such pairs from any new tool and its answers. Every published pair file is rebuilt from the
tools, the answers and the run each tool was made from (`data/groundtruth/tool_runs/`) and checked:

```bat
python groundtruth\build_pairs.py      :: -> outputs\pairs\*.csv, identical to data\groundtruth\pairs\
```

In the first held-out round, links that the annotators judged *different* were not kept as pairs (one link, A1292,
a wrong link of both the frozen model and v1); the second round checked it again, so the per-link precision counts
it as wrong. The "wrong links made" column of the held-out table (11 / 72, v1 2 / 72) does not include it.

## 5.4 Scoring a run

Any run is scored on the same pairs, so every method variant gets the same human answers:

* **blind plants re-identified**: the later sighting carries the identity of the earlier one;
* **checked links made**: same-plant answers the run reproduces;
* **wrong links made**: different-plant answers (rejected links and decoys) the run joins;
* **link precision**: every cross-pass link of the run checked against the answers (correct / wrong / not covered);
* **duplicates**: identity pairs closer than 20 cm on the map that never appear in the same frame.

```bat
python groundtruth\score_paper.py                  :: the published runs: 19 numbers of the paper, checked
python groundtruth\score_paper.py --runs runs      :: your reruns (tools\run_flights.py)
```

Rates come with Wilson 95 % confidence intervals.
