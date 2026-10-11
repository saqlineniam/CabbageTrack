"""
Figure helpers: one mosaic panel (mosaic from the video, camera odometry and the drone GNSS track after one similarity
fit, start marker, RMSE, 2 m scale bar) and the leg labels of a camera path.
Needs outputs/mosaic/<flight>/mosaic.jpg and outputs/gnss/<flight>/gnss_summary.json (paper/reproduce.py).
"""
import json
import os

import cv2
import matplotlib.patheffects
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from cabbagetrack.gnss import read_srt, umeyama
from cabbagetrack.mosaic import refine
from palette import C

S = 0.5                                     # mosaic scale (cabbagetrack.mosaic default)


def panel(ax, field, title):
    run, ex = f'data/runs/{field}', f'data/runs/{field}/extract'
    n = len(pd.read_csv(os.path.join(run, 'map.csv')))
    cx, cy = refine(run, ex, n)[:2]
    x0, y0 = cx.min(), cy.min()
    mos = cv2.cvtColor(cv2.imread(f'outputs/mosaic/{field}/mosaic.jpg'), cv2.COLOR_BGR2RGB)
    ox, oy = (cx + 960 - x0) * S, (cy + 540 - y0) * S
    ax.imshow(mos)
    g = read_srt(f'data/gnss/{field}.SRT')
    m = min(n, len(g))
    sc, R, t = umeyama(np.c_[cx[:m], -cy[:m]], g[['E', 'N']].values[:m])
    p = (R.T @ (g[['E', 'N']].values[:m] - t).T).T / sc
    gx, gy = (p[:, 0] + 960 - x0) * S, (-p[:, 1] + 540 - y0) * S
    ax.plot(gx, gy, '-', color=C['slate_blue'], lw=3.0, alpha=0.9, label='drone GNSS (one similarity fit)')
    summ = json.load(open(f'outputs/gnss/{field}/gnss_summary.json'))
    note, gsd_mm = f'RMSE\n{summ["ATE_rmse_m"]:.2f} m', summ['scale_mm_per_px']
    ax.plot(ox, oy, '-', color=C['crimson'], lw=1.0, label='camera odometry (method)')
    ax.plot(ox[0], oy[0], 'o', ms=5, color=C['leaf_green'], label='start')
    ax.text(1.03, 1.0, note, transform=ax.transAxes, ha='left', va='top', fontsize=7, color=C['midnight_navy'])
    gsd = gsd_mm / 1000 / S
    L, hgt = 2.0 / gsd, mos.shape[0]
    pad = 0.03 * mos.shape[1]
    ax.add_patch(Rectangle((pad, hgt - 3.2 * pad), L, 0.6 * pad, fc='white', ec=C['midnight_navy'], lw=0.6))
    ax.text(pad + L / 2, hgt - 3.6 * pad, '2 m', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold',
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.6, foreground=C['midnight_navy'])])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc='left', fontsize=8)
    return gsd


def legs_from_map(m, min_len=150):
    """Label frames by dominant direction of travel; short pieces are 'transit'."""
    vx = m.cam_x.diff().rolling(15, center=True).mean().fillna(0)
    vy = m.cam_y.diff().rolling(15, center=True).mean().fillna(0)
    d = np.where(np.hypot(vx, vy) < 2.0, 'hover',
                 np.where(np.abs(vy) >= np.abs(vx), np.where(vy < 0, 'up', 'down'), 'side'))
    lab, k, start = np.empty(len(m), dtype=object), 0, 0
    for i in range(1, len(m) + 1):
        if i == len(m) or d[i] != d[start]:
            name = f'leg{k}_{d[start]}' if d[start] in ('up', 'down') and i - start >= min_len else 'transit'
            if name != 'transit':
                k += 1
            lab[start:i] = name
            start = i
    return lab
