"""
DEVELOPMENT IMPLEMENTATION (paper/development/steps.py uses it for the development steps and the settings check).
Every component added during development is a switch here; with all switches at their defaults it is the Field A
method (v1, the method of the submitted paper). With the switches of steps.FROZEN it gives exactly the frozen
model in cabbagetrack/ (checked by steps.py --check). Kept as used; not needed to run the tracker.

Original header (paths refer to the authors' working folders):
Experimental copy of src/revisit.py with switches for ablation and baselines (src/revisit.py stays unchanged;
with all switches at their defaults this file reproduces it exactly, checked in run_experiments.py).

Extra switches (ExpConfig):
  verify       True : a revisit link must be confirmed by whole-image phase correlation (peak, agreement with odometry);
               False: the nearest remembered frame on the map is taken as the link (odometry only).
  register     True : remembered positions are moved by the image shift MEASURED between the two frames;
               False: they are moved by the shift PREDICTED by the map (odometry / dead reckoning).
  exclude_live True : identities currently visible under another live track cannot be matched again.
  assign       'hungarian' | 'greedy' (nearest pairs first).
  map_xy       optional (n, 2) array replacing the visual-odometry map (constant-velocity or GNSS baselines).
Clamp off: max_frame_disp = inf. Patience off: patience = 1.
"""
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CLS_CABBAGE = 1


@dataclass
class ExpConfig:
    max_frame_disp: float = 25.0
    min_track_frames: int = 3
    recent_gap: int = 45
    search_radius: float = 600.0
    n_link_candidates: int = 3
    shift_tol: float = 400.0
    min_response: float = 0.08
    window: int = 60
    gate: float = 60.0
    patience: int = 30
    live_gap: int = 5
    small_scale: int = 4
    verify: bool = True
    register: bool = True
    exclude_live: bool = True
    assign: str = 'hungarian'
    # v2 options (all off = the frozen method)
    multi_visit: bool = False        # link to the best frame of EVERY earlier visit of this ground, not only the best one
    overlap_search: bool = False     # candidates = frames sharing >= min_overlap of the image, not a fixed radius
    min_overlap: float = 0.25
    crop_verify: bool = False        # phase-correlate only the overlapping parts, placed by the predicted shift
    visit_gap: int = 15              # frames: candidate runs separated by more than this are different visits
    max_visits: int = 4
    # edge rule (off = unchanged): a detection whose box touches the image border shows only part of the plant, so its
    # centre is biased; such views are not used to link across passes (the track waits until the plant is fully in view)
    edge_hold: bool = False          # an unassigned track whose current box is cut waits (no link, no new identity)
    edge_memory: bool = False        # remembered positions come from views whose box was not cut
    edge_margin: float = 2.0         # px from the border that counts as touching it
    # v3 options (off = v2)
    spread_picks: bool = False       # per visit, verify frames spread over its time span and keep every verified link
    gate_rel: float = 0.0            # > 0: gate = gate_rel x median plant size seen so far (sqrt box area, uncut
                                     # boxes); 0.56 reproduces 60 px on field 1, where the gate was set
    # v4 option (off = v2): local appearance refinement of every remembered plant before matching
    # rotation-aware options (off = v2)
    rot_aware: bool = False          # measure the rotation between the two views at a revisit (ORB similarity) and
                                     # rotate the earlier view by it before aligning / transferring plant positions
    rot_min_deg: float = 1.0         # smaller rotations are treated as 0 (field-1 revisit pairs are within +-0.4 deg)
    drift_feedback: bool = False     # predict/check with the residual of recent verified revisits to the same earlier pass
    drift_recent: int = 60           # frames: residuals of links made within this many frames are used
    drift_window: int = 300          # frames: ... and only those whose earlier frame lies within this many frames of s
    drift_min_peak: float = 0.0      # only links with a clear alignment peak (>= this) update the drift correction
    turn_aware: bool = False         # frames inside a fast turn (|frame-to-frame rotation| > turn_rate, +- turn_margin) are
                                     # used neither as earlier views to link to nor as remembered plant positions
    turn_rate: float = 0.3           # deg/frame (field 1 never exceeds 0.1); rotation per frame from <extract>/yaw.csv
    turn_margin: int = 15            # frames
    local_refine: bool = False       # search the plant's own patch (with surroundings) around its registered position
    refine_patch: float = 2.0        # patch side, in plant sizes
    refine_search: float = 1.0       # search radius, in plant sizes
    refine_min_ncc: float = 0.5      # below this normalised cross-correlation the registered position is kept


def odometry_map(motion, max_frame_disp):
    tx, ty = motion.tx.values.copy(), motion.ty.values.copy()
    bad = np.hypot(tx, ty) > max_frame_disp
    tx[bad] = 0.0
    ty[bad] = 0.0
    return np.cumsum(tx), np.cumsum(ty)


def run(dets, motion, small, c: ExpConfig, map_xy=None, phi=None, trace=None):
    """trace: optional dict, filled with what happened at every lookup (diagnosis only, results unchanged)."""
    t0 = time.perf_counter()
    dets = dets[(dets.cls == CLS_CABBAGE) & (dets.raw_id >= 0)].sort_values(['frame', 'det_idx'])
    m_ = c.edge_margin
    dets = dets.assign(cut=(dets.x1 <= m_) | (dets.y1 <= m_) | (dets.x2 >= 1920 - m_) | (dets.y2 >= 1080 - m_))
    by_frame = {f: g for f, g in dets.groupby('frame', sort=True)}
    gate_t = np.full(len(motion), c.gate, float)
    size_t = np.full(len(motion), np.nan)
    if c.gate_rel > 0 or c.local_refine:                 # causal: median plant size over the frames seen so far
        u = dets[~dets.cut]
        size = np.sqrt((u.x2 - u.x1) * (u.y2 - u.y1)).groupby(u.frame).median()
        size_t = size.expanding().median().reindex(range(len(motion))).ffill().values
    if c.gate_rel > 0:
        gate_t = np.where(~np.isnan(size_t), c.gate_rel * size_t, c.gate)
    if map_xy is None:
        cam_x, cam_y = odometry_map(motion, c.max_frame_disp)
    else:
        cam_x, cam_y = np.asarray(map_xy[:, 0], float), np.asarray(map_xy[:, 1], float)
    n = len(motion)
    hann = cv2.createHanningWindow((small.shape[2], small.shape[1]), cv2.CV_32F)

    ident_f = []
    obs_by_frame = {}
    last_seen = []
    track2id, age, pending = {}, {}, {}
    events, links, labels = [], [], []
    link_cache = {}

    def new_identity(t, raw, reason):
        i = len(ident_f)
        ident_f.append([]); last_seen.append(-10 ** 9)
        track2id[raw] = i
        events.append({'frame': t, 'raw_id': raw, 'id': i, 'kind': 'new', 'reason': reason})
        return i

    def observe(t, i, cx, cy, cut=False):
        ident_f[i].append(t)
        obs_by_frame.setdefault(t, []).append((i, cx, cy, cut))
        last_seen[i] = t

    H_s, W_s = small.shape[1], small.shape[2]
    turning = np.zeros(len(motion), bool)
    if c.turn_aware and phi is not None:
        fast = np.flatnonzero(np.abs(np.asarray(phi)) > c.turn_rate)
        for f in fast:
            turning[max(0, f - c.turn_margin):f + c.turn_margin + 1] = True
    rstat = {'tried': 0, 'patch_outside': 0, 'search_outside': 0, 'low_ncc': 0, 'applied': 0, 'ncc': [], 'moved_px': []}

    orb_cache, resid_log = {}, []
    orb = cv2.ORB_create(1500) if c.rot_aware else None
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if c.rot_aware else None
    centre = np.array([W_s * c.small_scale / 2.0, H_s * c.small_scale / 2.0])

    def pair_rotation(s, t):
        """Rotation phi (deg) with content p_t ~ R(phi) p_s + T, from ORB features; 0 if unreliable or < rot_min_deg."""
        for f in (s, t):
            if f not in orb_cache:
                orb_cache[f] = orb.detectAndCompute(np.ascontiguousarray(small[f]), None)
        (ka, da), (kb, db) = orb_cache[s], orb_cache[t]
        if da is None or db is None:
            return 0.0
        mm = bfm.match(da, db)
        if len(mm) < 15:
            return 0.0
        a = np.float32([ka[m.queryIdx].pt for m in mm])
        b = np.float32([kb[m.trainIdx].pt for m in mm])
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl.sum() < 15:
            return 0.0
        phi = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        return phi if abs(phi) >= c.rot_min_deg else 0.0

    def drift_corr(t, s):
        if not c.drift_feedback:
            return 0.0, 0.0
        r_ = [(rx, ry) for (tl, sl, rx, ry) in resid_log[-200:] if t - tl <= c.drift_recent and abs(sl - s) <= c.drift_window]
        if not r_:
            return 0.0, 0.0
        r_ = np.array(r_)
        return float(np.median(r_[:, 0])), float(np.median(r_[:, 1]))

    def correlate(s, t, edx, edy, rot=0.0):
        """Measured content shift s->t (full-res px) and peak; with crop_verify only the overlap is correlated.
        rot != 0: the earlier view is first rotated by rot about the image centre (p -> R(rot)(p - c) + c)."""
        src = small[s]
        if rot:
            src = cv2.warpAffine(np.ascontiguousarray(small[s]), cv2.getRotationMatrix2D((W_s / 2.0, H_s / 2.0), -rot, 1.0),
                                 (W_s, H_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if not c.crop_verify:
            (sx, sy), r = cv2.phaseCorrelate(src.astype(np.float32), small[t].astype(np.float32), hann)
            return sx * c.small_scale, sy * c.small_scale, r
        ex, ey = int(round(edx / c.small_scale)), int(round(edy / c.small_scale))
        u0, u1 = max(0, -ex), min(W_s, W_s - ex)
        v0, v1 = max(0, -ey), min(H_s, H_s - ey)
        if u1 - u0 < 64 or v1 - v0 < 64:
            return 0.0, 0.0, 0.0
        a = src[v0:v1, u0:u1].astype(np.float32)
        b = small[t][v0 + ey:v1 + ey, u0 + ex:u1 + ex].astype(np.float32)
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (rx, ry), r = cv2.phaseCorrelate(a, b, win)
        return (ex + rx) * c.small_scale, (ey + ry) * c.small_scale, r

    def candidates(t, hi):
        dx, dy = cam_x[:hi] - cam_x[t], cam_y[:hi] - cam_y[t]
        if c.overlap_search:
            ov = np.clip(1 - np.abs(dx) / (W_s * c.small_scale), 0, None) * np.clip(1 - np.abs(dy) / (H_s * c.small_scale), 0, None)
            near = np.flatnonzero(ov >= c.min_overlap)
        else:
            near = np.flatnonzero(np.hypot(dx, dy) < c.search_radius)
        if c.turn_aware:
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
            mx, my = -(cam_x[t] - cam_x[s]), -(cam_y[t] - cam_y[s])          # map (odometry) prediction
            cx_, cy_ = drift_corr(t, s)
            edx, edy = mx + cx_, my + cy_                                     # corrected by recent revisits
            rot = pair_rotation(s, t) if c.rot_aware else 0.0
            mdx, mdy, r = correlate(s, t, edx, edy, rot)
            ok = r >= c.min_response and np.hypot(mdx - edx, mdy - edy) <= c.shift_tol
            if trace is not None:
                trace.setdefault('attempts', []).append((t, s, float(r), float(np.hypot(mdx - edx, mdy - edy)), bool(ok), rot))
            if ok and (best is None or r > best[3]):
                best = (s, mdx, mdy, r, edx, edy, rot, mx, my)
        return best

    def refine(t, f, x, y, px, py, plant):
        """v4: position of the remembered plant (seen at x, y in frame f) in frame t, found by matching its own small
        patch (with surroundings) within refine_search plant sizes of the registered position (px, py)."""
        cx_, cy_ = int(round(px / c.small_scale)), int(round(py / c.small_scale))
        if not (0 <= cx_ < W_s and 0 <= cy_ < H_s):
            return px, py                                   # cannot be in the current image: nothing to refine
        rstat['tried'] += 1
        h = max(8, int(round(c.refine_patch * plant / c.small_scale / 2)))
        rr = int(round(c.refine_search * plant / c.small_scale))
        core = max(4, int(round(plant / c.small_scale / 2)))   # the plant itself must be in the patch
        xs, ys = int(round(x / c.small_scale)), int(round(y / c.small_scale))
        # patch clipped to the remembered image (the plant's centre at offset (l, u) inside it)
        l, r_ = min(h, xs), min(h, W_s - xs)
        u, b = min(h, ys), min(h, H_s - ys)
        if min(l, r_, u, b) < core:
            rstat['patch_outside'] += 1
            return px, py
        tpl = small[f][ys - u:ys + b, xs - l:xs + r_]
        x0, y0 = max(0, cx_ - rr - l), max(0, cy_ - rr - u)
        x1, y1 = min(W_s, cx_ + rr + r_), min(H_s, cy_ + rr + b)
        if x1 - x0 < tpl.shape[1] + 2 or y1 - y0 < tpl.shape[0] + 2:
            rstat['search_outside'] += 1
            return px, py
        res = cv2.matchTemplate(small[t][y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (bx, by) = cv2.minMaxLoc(res)
        rstat['ncc'].append(mx)
        if mx < c.refine_min_ncc:
            rstat['low_ncc'] += 1
            return px, py
        rstat['applied'] += 1
        nx = (x0 + bx + l) * c.small_scale + (x - xs * c.small_scale)
        ny = (y0 + by + u) * c.small_scale + (y - ys * c.small_scale)
        rstat['moved_px'].append(float(np.hypot(nx - px, ny - py)))
        return nx, ny

    def find_links(t):
        """v2: one verified link per earlier visit of this ground (runs of candidate frames)."""
        if t in link_cache:
            return link_cache[t]
        hi = t - c.recent_gap
        res = ([], False)
        if hi > 0:
            near, d = candidates(t, hi)
            if trace is not None:
                trace.setdefault('lookups', []).append((t, near.copy()))
            if near.size:
                runs = np.split(near, np.flatnonzero(np.diff(near) > c.visit_gap) + 1)
                runs = sorted(runs, key=lambda r_: d[r_].min())[:c.max_visits]
                found = []
                for r_ in runs:
                    if c.spread_picks:
                        # frames spread over the whole visit (closest frame in each third of it), every verified one
                        # kept: at a turn the earlier pass and the drone's own recent frames form ONE run, and its
                        # closest frames are always the just-passed ones
                        for part in np.array_split(r_, c.n_link_candidates):
                            if part.size:
                                b = verify_best(t, part[np.argsort(d[part])][:1])
                                if b is not None:
                                    found.append(b)
                    else:
                        b = verify_best(t, r_[np.argsort(d[r_])])
                        if b is not None:
                            found.append(b)
                res = (found, True)
        link_cache[t] = res
        return res

    def find_link(t):
        if t in link_cache:
            return link_cache[t]
        hi = t - c.recent_gap
        res = (None, False)
        if hi > 0:
            d = np.hypot(cam_x[:hi] - cam_x[t], cam_y[:hi] - cam_y[t])
            near = np.flatnonzero(d < c.search_radius)
            if trace is not None:
                trace.setdefault('lookups', []).append((t, near.copy()))
            if near.size:
                order = near[np.argsort(d[near])]
                if not c.verify:                                   # odometry only: nearest remembered frame
                    s = int(order[0])
                    edx, edy = -(cam_x[t] - cam_x[s]), -(cam_y[t] - cam_y[s])
                    res = ((s, edx, edy, np.nan, edx, edy), True)
                else:
                    picks = []
                    for s in order:
                        if all(abs(s - p) >= 10 for p in picks):
                            picks.append(int(s))
                        if len(picks) >= c.n_link_candidates:
                            break
                    best = None
                    cur = small[t].astype(np.float32)
                    for s in picks:
                        (sx, sy), r = cv2.phaseCorrelate(small[s].astype(np.float32), cur, hann)
                        mdx, mdy = sx * c.small_scale, sy * c.small_scale
                        edx, edy = -(cam_x[t] - cam_x[s]), -(cam_y[t] - cam_y[s])
                        ok = r >= c.min_response and np.hypot(mdx - edx, mdy - edy) <= c.shift_tol
                        if trace is not None:
                            trace.setdefault('attempts', []).append((t, s, float(r), float(np.hypot(mdx - edx, mdy - edy)), bool(ok), 0.0))
                        if ok and (best is None or r > best[3]):
                            best = (s, mdx, mdy, r, edx, edy)
                    res = (best, True)
        link_cache[t] = res
        return res

    for t in range(n):
        g = by_frame.get(t)
        if g is None:
            continue
        rows = list(g.itertuples(index=False))
        live = set()
        unassigned = []
        for r in rows:
            age[r.raw_id] = age.get(r.raw_id, 0) + 1
            if r.raw_id in track2id:
                live.add(track2id[r.raw_id])
        for r in rows:
            if r.raw_id in track2id:
                i = track2id[r.raw_id]
                observe(t, i, r.cx, r.cy, r.cut)
                labels.append((t, r.raw_id, i, r.cx, r.cy))
            elif age[r.raw_id] >= c.min_track_frames:
                unassigned.append(r)
        if not unassigned:
            continue

        if c.multi_visit:
            found, has_memory = find_links(t)
        else:
            link, has_memory = find_link(t)
            found = [link] if link is not None else []
        if c.edge_hold and has_memory:                  # cut view over known ground: wait for a full view
            unassigned = [r for r in unassigned if not r.cut]
            if not unassigned:
                continue
        assigned_now = set()
        if found:
            reg = {}                                     # identity -> (registered x, y, peak, s, mem frame)
            for lk in found:
                s, mdx, mdy, resp, edx, edy = lk[:6]
                rot = lk[6] if len(lk) > 6 else 0.0
                if not c.register:
                    mdx, mdy = edx, edy
                links.append({'frame': t, 's': s, 'gap': t - s, 'dx': mdx, 'dy': mdy, 'resp': resp,
                              'exp_dx': edx, 'exp_dy': edy, 'cam_x': cam_x[t], 'cam_y': cam_y[t], 'rot': rot})
                if c.drift_feedback and len(lk) > 8 and t - s > 300 and resp >= c.drift_min_peak:
                    resid_log.append((t, s, mdx - lk[7], mdy - lk[8]))
                Rm = None
                if rot:
                    a_ = np.radians(rot)
                    Rm = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
                seen = {}
                for f in range(max(0, s - c.window), min(t, s + c.window + 1)):
                    for (i, x, y, cut) in obs_by_frame.get(f, []):
                        if c.exclude_live and (i in live or t - last_seen[i] <= c.live_gap):
                            continue
                        if c.edge_memory and cut:
                            continue
                        if c.turn_aware and turning[f]:
                            continue
                        if i not in seen or abs(f - s) < abs(seen[i][0] - s):
                            seen[i] = (f, x, y)
                for i, (f, x, y) in seen.items():
                    px_, py_ = x + (-(cam_x[s] - cam_x[f])) + mdx, y + (-(cam_y[s] - cam_y[f])) + mdy
                    if Rm is not None:                       # rotate about the image centre, then the measured shift
                        q = Rm @ (np.array([x + (-(cam_x[s] - cam_x[f])), y + (-(cam_y[s] - cam_y[f]))]) - centre) + centre
                        px_, py_ = q[0] + mdx, q[1] + mdy
                    if i not in reg or resp > reg[i][2]:
                        reg[i] = (px_, py_, resp, s, f, x, y)
            seen = {i: (v[4],) for i, v in reg.items()}
            s, resp = found[0][0], found[0][3]
            if reg:
                ids = list(reg)
                P = np.array([(reg[i][0], reg[i][1]) for i in ids])
                if c.local_refine and not np.isnan(size_t[t]):
                    P = np.array([refine(t, reg[i][4], reg[i][5], reg[i][6], P[k, 0], P[k, 1], size_t[t])
                                  for k, i in enumerate(ids)])
                Q = np.array([(r.cx, r.cy) for r in unassigned])
                D = np.hypot(Q[:, None, 0] - P[None, :, 0], Q[:, None, 1] - P[None, :, 1])
                if trace is not None:
                    reg_t = trace.setdefault('registered', {})
                    for k_, i_ in enumerate(ids):
                        reg_t[(t, i_)] = (float(P[k_, 0]), float(P[k_, 1]))
                g_ = gate_t[t]
                if c.assign == 'hungarian':
                    cost = np.where(D <= g_, D, 1e9)
                    pairs = list(zip(*linear_sum_assignment(cost)))
                else:                                               # greedy: nearest pairs first
                    pairs, ua, ub = [], set(), set()
                    for a_, b_ in zip(*np.unravel_index(np.argsort(D, axis=None), D.shape)):
                        if D[a_, b_] > g_:
                            break
                        if a_ not in ua and b_ not in ub:
                            pairs.append((a_, b_)); ua.add(a_); ub.add(b_)
                for a_, b_ in pairs:
                    if D[a_, b_] <= g_:
                        r = unassigned[a_]
                        i = ids[b_]
                        track2id[r.raw_id] = i
                        observe(t, i, r.cx, r.cy, r.cut)
                        labels.append((t, r.raw_id, i, r.cx, r.cy))
                        assigned_now.add(r.raw_id)
                        pending.pop(r.raw_id, None)
                        si = reg[i][3]
                        events.append({'frame': t, 'raw_id': r.raw_id, 'id': i, 'kind': 'reid', 'reason': '',
                                       'dist': float(D[a_, b_]), 'link_s': si, 'gap': t - si, 'resp': reg[i][2],
                                       'mem_frame': seen[i][0]})
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
            observe(t, i, r.cx, r.cy, r.cut)
            labels.append((t, r.raw_id, i, r.cx, r.cy))

    ev = pd.DataFrame(events)
    reid = ev[ev.kind == 'reid'] if len(ev) else ev
    s = {'config': asdict(c), 'frames': n, 'identities': len(ident_f), 'tracks_with_identity': len(track2id),
         'reid_events': int(len(reid)), 'reid_gap_gt_300': int((reid.gap > 300).sum()) if len(reid) else 0,
         'runtime_s': time.perf_counter() - t0,
         'refine': {k: (v if not isinstance(v, list) else (float(np.median(v)) if v else None)) for k, v in rstat.items()}}
    return {'events': ev, 'links': pd.DataFrame(links),
            'labels': pd.DataFrame(labels, columns=['frame', 'raw_id', 'id', 'cx', 'cy']),
            'map': pd.DataFrame({'frame': np.arange(n), 'cam_x': cam_x, 'cam_y': cam_y})}, s
