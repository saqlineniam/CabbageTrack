"""
Plant re-identification of the final CabbageTrack model.

Every tracker track (from detection + tracking) is given a plant identity. A new track either continues an identity the
drone has already seen on an earlier visit of the same ground (re-identification) or starts a new identity.

  1. Map: camera positions from the frame-to-frame image shifts (visual odometry).
  2. Earlier visits: frames older than recent_gap whose image overlaps the current one on the map (>= min_overlap),
     grouped into visits; frames inside a fast turn are not used.
  3. Verification: for the closest frames of each visit, the rotation between the two views is measured (ORB) and the
     earlier view rotated by it; the overlapping parts are phase-correlated; the link is accepted when the peak is clear
     and the measured shift agrees with the prediction (odometry corrected by the residuals of recent revisits).
  4. Registration: plants remembered around the linked earlier frame (not from frames inside a fast turn) are moved into
     the current image with the measured rotation and shift.
  5. Assignment: Hungarian matching of new tracks to registered plants within the gate; unmatched tracks over known
     ground wait `patience` frames, then start a new identity.
"""
import time
from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import Config
from .rotation import PairRotation


def odometry_map(motion, max_frame_disp):
    tx, ty = motion.tx.values.copy(), motion.ty.values.copy()
    bad = np.hypot(tx, ty) > max_frame_disp
    tx[bad] = 0.0
    ty[bad] = 0.0
    return np.cumsum(tx), np.cumsum(ty)


def turning_frames(phi, n, c: Config):
    turning = np.zeros(n, bool)
    for f in np.flatnonzero(np.abs(np.asarray(phi)) > c.turn_rate):
        turning[max(0, f - c.turn_margin):f + c.turn_margin + 1] = True
    return turning


def reidentify(dets, motion, small, phi, c: Config = Config()):
    """dets: detections.csv (frame, det_idx, raw_id, cls, cx, cy, ...); motion: motion.csv (tx, ty per frame);
    small: (n, 270, 480) uint8 frames; phi: per-frame rotation (deg). Returns dict of DataFrames and a summary."""
    t0 = time.perf_counter()
    dets = dets[(dets.cls == c.plant_class) & (dets.raw_id >= 0)].sort_values(['frame', 'det_idx'])
    by_frame = {f: g for f, g in dets.groupby('frame', sort=True)}
    cam_x, cam_y = odometry_map(motion, c.max_frame_disp)
    n = len(motion)
    H_s, W_s = small.shape[1], small.shape[2]
    sc = c.small_scale
    centre = np.array([W_s * sc / 2.0, H_s * sc / 2.0])
    turning = turning_frames(phi, n, c)
    pair_rotation = PairRotation(small, c.rot_min_deg)

    ident_f, obs_by_frame, last_seen = [], {}, []
    track2id, age, pending = {}, {}, {}
    events, links, labels = [], [], []
    link_cache, resid_log = {}, []

    def new_identity(t, raw, reason):
        i = len(ident_f)
        ident_f.append([]); last_seen.append(-10 ** 9)
        track2id[raw] = i
        events.append({'frame': t, 'raw_id': raw, 'id': i, 'kind': 'new', 'reason': reason})
        return i

    def observe(t, i, cx, cy):
        ident_f[i].append(t)
        obs_by_frame.setdefault(t, []).append((i, cx, cy))
        last_seen[i] = t

    def drift_corr(t, s):
        r_ = [(rx, ry) for (tl, sl, rx, ry) in resid_log[-200:] if t - tl <= c.drift_recent and abs(sl - s) <= c.drift_window]
        if not r_:
            return 0.0, 0.0
        r_ = np.array(r_)
        return float(np.median(r_[:, 0])), float(np.median(r_[:, 1]))

    def correlate(s, t, edx, edy, rot):
        """Measured content shift s -> t (full-res px) and peak, on the parts that overlap at the predicted shift;
        the earlier view is first rotated by rot about the image centre."""
        src = small[s]
        if rot:
            src = cv2.warpAffine(np.ascontiguousarray(small[s]), cv2.getRotationMatrix2D((W_s / 2.0, H_s / 2.0), -rot, 1.0),
                                 (W_s, H_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ex, ey = int(round(edx / sc)), int(round(edy / sc))
        u0, u1 = max(0, -ex), min(W_s, W_s - ex)
        v0, v1 = max(0, -ey), min(H_s, H_s - ey)
        if u1 - u0 < 64 or v1 - v0 < 64:
            return 0.0, 0.0, 0.0
        a = src[v0:v1, u0:u1].astype(np.float32)
        b = small[t][v0 + ey:v1 + ey, u0 + ex:u1 + ex].astype(np.float32)
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (rx, ry), r = cv2.phaseCorrelate(a, b, win)
        return (ex + rx) * sc, (ey + ry) * sc, r

    def candidates(t, hi):
        dx, dy = cam_x[:hi] - cam_x[t], cam_y[:hi] - cam_y[t]
        ov = np.clip(1 - np.abs(dx) / (W_s * sc), 0, None) * np.clip(1 - np.abs(dy) / (H_s * sc), 0, None)
        near = np.flatnonzero(ov >= c.min_overlap)
        near = near[~turning[near]]
        return near, np.hypot(dx, dy)

    def verify_best(t, order):
        picks = []
        for s in order:
            if all(abs(s - p) >= 10 for p in picks):
                picks.append(int(s))
            if len(picks) >= c.n_link_candidates:
                break
        best = None
        for s in picks:
            mx, my = -(cam_x[t] - cam_x[s]), -(cam_y[t] - cam_y[s])          # odometry prediction
            cx_, cy_ = drift_corr(t, s)
            edx, edy = mx + cx_, my + cy_                                     # corrected by recent revisits
            rot = pair_rotation(s, t)
            mdx, mdy, r = correlate(s, t, edx, edy, rot)
            if r >= c.min_response and np.hypot(mdx - edx, mdy - edy) <= c.shift_tol and (best is None or r > best[3]):
                best = (s, mdx, mdy, r, edx, edy, rot, mx, my)
        return best

    def find_links(t):
        """One verified link per earlier visit of this ground; second value: is there earlier ground at all."""
        if t in link_cache:
            return link_cache[t]
        hi = t - c.recent_gap
        res = ([], False)
        if hi > 0:
            near, d = candidates(t, hi)
            if near.size:
                runs = np.split(near, np.flatnonzero(np.diff(near) > c.visit_gap) + 1)
                runs = sorted(runs, key=lambda r_: d[r_].min())[:c.max_visits]
                found = []
                for r_ in runs:
                    b = verify_best(t, r_[np.argsort(d[r_])])
                    if b is not None:
                        found.append(b)
                res = (found, True)
        link_cache[t] = res
        return res

    for t in range(n):
        g = by_frame.get(t)
        if g is None:
            continue
        rows = list(g.itertuples(index=False))
        live, unassigned = set(), []
        for r in rows:
            age[r.raw_id] = age.get(r.raw_id, 0) + 1
            if r.raw_id in track2id:
                live.add(track2id[r.raw_id])
        for r in rows:
            if r.raw_id in track2id:
                i = track2id[r.raw_id]
                observe(t, i, r.cx, r.cy)
                labels.append((t, r.raw_id, i, r.cx, r.cy))
            elif age[r.raw_id] >= c.min_track_frames:
                unassigned.append(r)
        if not unassigned:
            continue
        found, has_memory = find_links(t)
        assigned_now = set()
        if found:
            reg = {}                                           # identity -> (x, y, peak, link frame, memory frame)
            for (s, mdx, mdy, resp, edx, edy, rot, mx, my) in found:
                links.append({'frame': t, 's': s, 'gap': t - s, 'dx': mdx, 'dy': mdy, 'resp': resp,
                              'exp_dx': edx, 'exp_dy': edy, 'cam_x': cam_x[t], 'cam_y': cam_y[t], 'rot': rot})
                if t - s > c.cross_pass_gap and resp >= c.drift_min_peak:
                    resid_log.append((t, s, mdx - mx, mdy - my))
                Rm = None
                if rot:
                    a_ = np.radians(rot)
                    Rm = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
                seen = {}
                for f in range(max(0, s - c.window), min(t, s + c.window + 1)):
                    if turning[f]:
                        continue
                    for (i, x, y) in obs_by_frame.get(f, []):
                        if i in live or t - last_seen[i] <= c.live_gap:
                            continue
                        if i not in seen or abs(f - s) < abs(seen[i][0] - s):
                            seen[i] = (f, x, y)
                for i, (f, x, y) in seen.items():
                    ps = np.array([x + (-(cam_x[s] - cam_x[f])), y + (-(cam_y[s] - cam_y[f]))])
                    if Rm is not None:
                        ps = Rm @ (ps - centre) + centre
                    px_, py_ = ps[0] + mdx, ps[1] + mdy
                    if i not in reg or resp > reg[i][2]:
                        reg[i] = (px_, py_, resp, s, f)
            if reg:
                ids = list(reg)
                P = np.array([(reg[i][0], reg[i][1]) for i in ids])
                Q = np.array([(r.cx, r.cy) for r in unassigned])
                D = np.hypot(Q[:, None, 0] - P[None, :, 0], Q[:, None, 1] - P[None, :, 1])
                pairs = list(zip(*linear_sum_assignment(np.where(D <= c.gate, D, 1e9))))
                for a_, b_ in pairs:
                    if D[a_, b_] <= c.gate:
                        r = unassigned[a_]
                        i = ids[b_]
                        track2id[r.raw_id] = i
                        observe(t, i, r.cx, r.cy)
                        labels.append((t, r.raw_id, i, r.cx, r.cy))
                        assigned_now.add(r.raw_id)
                        pending.pop(r.raw_id, None)
                        si = reg[i][3]
                        events.append({'frame': t, 'raw_id': r.raw_id, 'id': i, 'kind': 'reid', 'reason': '',
                                       'dist': float(D[a_, b_]), 'link_s': si, 'gap': t - si, 'resp': reg[i][2],
                                       'mem_frame': reg[i][4]})
        for r in unassigned:
            if r.raw_id in assigned_now:
                continue
            if not has_memory:
                i = new_identity(t, r.raw_id, 'no_memory')
            else:
                pending[r.raw_id] = pending.get(r.raw_id, 0) + 1
                if pending[r.raw_id] < c.patience:
                    continue
                i = new_identity(t, r.raw_id, 'no_match')
                pending.pop(r.raw_id, None)
            observe(t, i, r.cx, r.cy)
            labels.append((t, r.raw_id, i, r.cx, r.cy))

    ev = pd.DataFrame(events)
    reid = ev[ev.kind == 'reid'] if len(ev) else ev
    summary = {'config': asdict(c), 'frames': n, 'identities': len(ident_f), 'tracks_with_identity': len(track2id),
               'reid_events': int(len(reid)),
               'cross_pass_reid': int((reid.gap > c.cross_pass_gap).sum()) if len(reid) else 0,
               'fast_turn_frames': int(turning.sum()), 'runtime_s': round(time.perf_counter() - t0, 1)}
    return {'events': ev, 'links': pd.DataFrame(links),
            'labels': pd.DataFrame(labels, columns=['frame', 'raw_id', 'id', 'cx', 'cy']),
            'map': pd.DataFrame({'frame': np.arange(n), 'cam_x': cam_x, 'cam_y': cam_y})}, summary
