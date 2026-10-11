"""
Comparison of the camera odometry with the drone's own GNSS log (DJI .SRT subtitle file: one record per video frame).
For evaluation only: the pipeline never reads GNSS.

1. GNSS -> local east / north metres (equirectangular around the first fix).
2. Odometry map (image pixels, x right / y down) -> (x, -y); ONE similarity transform (scale, rotation, translation;
   Umeyama least squares over all frames) aligns it to the GNSS track. A GNSS time lag of -15..+15 frames is searched.
3. Reported: trajectory error after the fit (RMSE, median, 95th percentile, max), fitted scale (mm per pixel),
   relative error over 1 s and 10 s, and the GNSS distance between the two frames of every revisit link.

Outputs (<out>/): gnss_summary.json, gnss_per_frame.csv
"""
import json
import os
import re

import numpy as np
import pandas as pd

W = 1920
PAT = re.compile(r'FrameCnt: (\d+).*?\n(\S+ \S+)\n.*?\[latitude: ([-\d.]+)\] \[longitude: ([-\d.]+)\] '
                 r'\[rel_alt: ([-\d.]+) abs_alt: ([-\d.]+)\] \[gb_yaw: ([-\d.]+) gb_pitch: ([-\d.]+)', re.S)


def read_srt(path):
    d = pd.DataFrame(PAT.findall(open(path, encoding='utf8').read()),
                     columns=['frame', 'time', 'lat', 'lon', 'rel_alt', 'abs_alt', 'yaw', 'pitch'])
    for c in d.columns:
        if c != 'time':
            d[c] = d[c].astype(float)
    d['frame'] = d.frame.astype(int) - 1                       # SRT FrameCnt is 1-based
    lat0, lon0 = d.lat.iloc[0], d.lon.iloc[0]
    d['E'] = (d.lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
    d['N'] = (d.lat - lat0) * 110540.0
    return d.set_index('frame')


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|^2, proper rotation."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def compare(run, srt, out=None):
    """Compare a CabbageTrack run (map.csv, links.csv) with a GNSS log; writes the summary and per-frame errors."""
    out = out or run
    os.makedirs(out, exist_ok=True)
    g = read_srt(srt)
    m = pd.read_csv(os.path.join(run, 'map.csv')).set_index('frame')
    odo_all = np.c_[m.cam_x.values, -m.cam_y.values]           # image y points down -> right-handed frame
    n = min(len(m), len(g))
    best = None
    for lag in range(-15, 16):                                   # GNSS record of frame f describes frame f + lag
        idx = np.arange(max(0, -lag), min(n, n - lag))
        src, dst = odo_all[idx], g[['E', 'N']].values[idx + lag]
        s, R, t = umeyama(src, dst)
        err = np.hypot(*(dst - (s * (R @ src.T).T + t)).T)
        if best is None or np.sqrt((err ** 2).mean()) < best[0]:
            best = (np.sqrt((err ** 2).mean()), lag, idx, s, R, t, err)
    rmse, lag, idx, s, R, t, err = best
    odo = s * (R @ odo_all.T).T + t
    gps = g[['E', 'N']].values
    path = np.hypot(*np.diff(gps[idx + lag], axis=0).T).sum()
    rel = {}
    for win in (30, 300):
        i0 = idx[:-win] if len(idx) > win else idx[:0]
        dg = gps[i0 + win + lag] - gps[i0 + lag]
        do = odo[i0 + win] - odo[i0]
        moved = np.hypot(*dg.T)
        keep = moved > 0.2
        e = np.hypot(*(do - dg).T)[keep]
        rel[f'{win // 30} s'] = {'median_error_m': float(np.median(e)),
                                 'median_error_pct_of_distance': float(np.median(e / moved[keep]) * 100)}
    L = pd.read_csv(os.path.join(run, 'links.csv'))
    L = L[(L.frame + lag < len(g)) & (L.s + lag < len(g)) & (L.frame + lag >= 0) & (L.s + lag >= 0)]
    g_disp = gps[L.frame.values + lag] - gps[L.s.values + lag]
    link_dist = np.hypot(*g_disp.T)
    long_links = L.gap.values > 300
    summary = {'srt': os.path.abspath(srt), 'run': os.path.abspath(run), 'frames_compared': int(len(idx)),
               'gps_lag_frames': int(lag), 'gps_path_m': float(path), 'scale_mm_per_px': float(s * 1000),
               'footprint_width_m': float(s * W), 'rel_alt_m': float(g.rel_alt.median()),
               'ATE_rmse_m': float(rmse), 'ATE_median_m': float(np.median(err)), 'ATE_p95_m': float(np.percentile(err, 95)),
               'ATE_max_m': float(err.max()), 'ATE_rmse_pct_of_path': float(rmse / path * 100), 'relative_error': rel,
               'revisit_links': int(len(L)), 'revisit_links_gap_gt_300': int(long_links.sum()),
               'link_gps_distance_m_median': float(np.median(link_dist[long_links])) if long_links.any() else None}
    json.dump(summary, open(os.path.join(out, 'gnss_summary.json'), 'w'), indent=2)
    pd.DataFrame({'frame': idx, 'gps_E': gps[idx + lag, 0], 'gps_N': gps[idx + lag, 1], 'odo_E': odo[idx, 0],
                  'odo_N': odo[idx, 1], 'error_m': err}).to_csv(os.path.join(out, 'gnss_per_frame.csv'), index=False)
    print(f"odometry vs GNSS: RMSE {rmse:.2f} m over {path:.0f} m, scale {s * 1000:.2f} mm/px, lag {lag} frames")
    return summary
