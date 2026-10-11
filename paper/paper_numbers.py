"""
Numbers for the paper that are not produced elsewhere: the flights table, agreement between annotators, the census
check (hand-outlined plants carrying an identity), revisit gaps and the held-out failures.

  python paper/paper_numbers.py          -> outputs/numbers/numbers.json, numbers.md   (needs outputs/gnss/, paper/reproduce.py gnss)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'groundtruth'))
from score import CENTER, kappa, score, tasks  # noqa: E402,F401

OUT = 'outputs/numbers'
FOCAL_MM, SENSOR_W_MM, W = 12.29, 17.3, 1920      # Mavic 3E wide camera (4/3 sensor), 16:9 video at full width
GT = 'data/groundtruth'


def flight_table(F):
    rows = {}
    for k, f in F.items():
        m = json.load(open(f'data/runs/{k}/extract/meta.json'))
        g = json.load(open(f'outputs/gnss/{k}/gnss_summary.json'))
        pf = pd.read_csv(f'outputs/gnss/{k}/gnss_per_frame.csv')
        fps = float(m['fps'])
        sp = np.hypot(np.diff(pf.gps_E.values[::30]), np.diff(pf.gps_N.values[::30])) * fps / 30   # m/s over 1 s
        gsd = g['scale_mm_per_px'] / 1000
        ev = pd.read_csv(f'data/runs/{k}/events.csv')
        lab = pd.read_csv(f'data/runs/{k}/labels.csv')
        rot = pd.read_csv(f'data/runs/{k}/rotation.csv')
        xr = ev[(ev.kind == 'reid') & (ev.gap > 300)]
        d = pd.read_csv(f'data/runs/{k}/extract/detections.csv.gz', usecols=['frame', 'cls', 'mask_cx', 'mask_cy'])
        d = d[(d.cls == 1) & d.mask_cx.notna()]
        nn = []
        for fr, gg in d[d.frame % 30 == 0].groupby('frame'):
            if len(gg) > 3:
                P = gg[['mask_cx', 'mask_cy']].values
                D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
                np.fill_diagonal(D, np.inf)
                nn += list(D.min(1))
        rows[k] = {
            'date': str(f['date']), 'frames': int(m['frames']), 'fps': round(fps, 2), 'duration_s': round(m['frames'] / fps, 1),
            'gsd_mm_per_px': round(g['scale_mm_per_px'], 2), 'footprint_m': round(gsd * W, 2),
            'height_above_plants_m': round(gsd * W * FOCAL_MM / SENSOR_W_MM, 2), 'logged_height_above_takeoff_m': round(g['rel_alt_m'], 2),
            'gnss_path_m': round(g['gps_path_m'], 0), 'ground_speed_moving_median_m_s': round(float(np.median(sp[sp > 0.2])), 2),
            'odometry_rmse_m': round(g['ATE_rmse_m'], 2),
            'nearest_plant_spacing_m_median': round(float(np.median(nn)) * gsd, 3),
            'nearest_plant_spacing_m_iqr': [round(float(np.percentile(nn, 25)) * gsd, 3), round(float(np.percentile(nn, 75)) * gsd, 3)],
            'identities': int(lab.id.nunique()), 'identities_seen_again_gt10s': int(xr.id.nunique()),
            'cross_pass_links': int(len(xr)),
            'revisit_gap_s_median': round(float(xr.gap.median()) / fps, 0), 'revisit_gap_s_max': round(float(xr.gap.max()) / fps, 0),
            'fast_turn_frames': int(rot.in_fast_turn.sum()), 'max_rotation_deg_per_frame': round(float(rot.phi_deg.abs().max()), 2),
            'detector': os.path.basename(f['weights'])}
    return rows


def agreement():
    out = {}
    for flight, tool, files in [('fieldA', 'gt_tool', ('gt_results_Saklain.csv', 'gt_results_Mashrup.csv')),
                                ('fieldA', 'gt_tool_blind2', ('gt_results_blind2_Saklain.csv', 'gt_results_blind2_Mashrup.csv')),
                                ('fieldA', 'gt_tool_rowA', ('gt_results_rowA_Saklain.csv', 'gt_results_rowA_Mashrup.csv')),
                                ('heldout', 'gt_tool_field2', ('gt_results_field2_Saklain.csv', 'gt_results_field2_Mashrup.csv'))]:
        a, b = [pd.read_csv(f'{GT}/answers/{flight}/{tool}__{x}', dtype=str, keep_default_na=False).set_index('task_id') for x in files]
        r = {}
        A = [t for t in a.index if a.loc[t, 'part'] == 'A' and t in b.index]
        if A:
            r['partA_items'] = len(A)
            r['partA_kappa'] = round(kappa(a.loc[A, 'answer'], b.loc[A, 'answer']), 3)
            r['partA_raw_agreement'] = round(float((a.loc[A, 'answer'] == b.loc[A, 'answer']).mean()), 3)
            dec = [t for t in A if a.loc[t, 'decoy'].lower() == 'true']
            r['decoys'] = len(dec)
            r['decoys_caught'] = [int((a.loc[dec, 'answer'] == 'different').sum()), int((b.loc[dec, 'answer'] == 'different').sum())]
        B = [t for t in a.index if a.loc[t, 'part'] == 'B' and t in b.index]
        both = [t for t in B if a.loc[t, 'answer'] == 'clicked' and b.loc[t, 'answer'] == 'clicked']
        r['blind_items'] = len(B)
        r['blind_both_clicked'] = len(both)
        r['blind_same_plant'] = int(sum(a.loc[t, 'clicked_raw_id'] == b.loc[t, 'clicked_raw_id'] for t in both))
        r['blind_not_visible'] = [int((a.loc[B, 'answer'] == 'not_visible').sum()), int((b.loc[B, 'answer'] == 'not_visible').sum())]
        out[tool] = r
    return out


def census(F):
    """Hand-outlined plants in the detectors' validation and test frames (never used for training) that carry an
    identity in the published run (a detection inside the outline whose track received an identity), and that were
    detected at all."""
    sets = {'fieldA': ('data/detector/cabbage/split', 'seq_1_'), 'heldout': ('data/detector/cabbage/split', 'DJI_0004_'),
            'fieldB': ('data/detector/young_cabbage/split', 'young_frame_')}
    out = {}
    for k, (split, pre) in sets.items():
        lab = pd.read_csv(f'data/runs/{k}/labels.csv')
        has_id = set(lab.raw_id.unique())                        # tracks that received an identity (at any frame)
        dets = pd.read_csv(f'data/runs/{k}/extract/detections.csv.gz', usecols=['frame', 'raw_id', 'cls', 'cx', 'cy'])
        dets = dets[dets.cls == 1]
        ided = dets[dets.raw_id.isin(has_id)]
        lab_f = {f: g[['cx', 'cy']].values for f, g in ided.groupby('frame')}
        det_f = {f: g[['cx', 'cy']].values for f, g in dets.groupby('frame')}
        n = with_id = detected = frames = 0
        for sp in ('val', 'test'):
            for lp in glob.glob(f'{split}/{sp}/labels/{pre}*.txt'):
                stem = os.path.basename(lp)[:-4]
                fr = int(stem[len(pre):len(pre) + 6])
                frames += 1
                for line in open(lp):
                    v = line.split()
                    if len(v) < 7 or v[0] != '1':
                        continue
                    poly = (np.array(v[1:], float).reshape(-1, 2) * [1920, 1080]).astype(np.float32)
                    n += 1
                    inside = lambda pts: any(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -10 for x, y in pts)  # noqa: E731
                    with_id += inside(lab_f.get(fr, []))
                    detected += inside(det_f.get(fr, []))
        out[k] = {'frames': frames, 'hand_outlined_plants': n, 'detected': detected, 'with_identity': with_id,
                  'with_identity_pct': round(100 * with_id / n, 1), 'detected_pct': round(100 * detected / n, 1)}
    return out


def heldout_failures():
    """The held-out blind failure and the wrong cross-pass links of the frozen model."""
    run, ex = 'data/runs/heldout', 'data/runs/heldout/extract'
    lab, ev = pd.read_csv(f'{run}/labels.csv'), pd.read_csv(f'{run}/events.csv')
    dets = pd.read_csv(f'{ex}/detections.csv.gz')
    P = pd.read_csv(f'{GT}/pairs/heldout_table_pairs.csv')
    F_ = pd.read_csv(f'{GT}/pairs/heldout_followup_pairs.csv')
    o = score(lab[['frame', 'raw_id', 'id', 'cx', 'cy']], P, dets)
    blind_fail = o[P.kind == 'blind'][o[P.kind == 'blind'] != 'correct'].value_counts().to_dict()
    allp = pd.concat([P, F_], ignore_index=True)
    d = dets[(dets.cls == 1) & (dets.raw_id >= 0)]
    by = {f: g for f, g in d.groupby('frame')}

    def track(f, x, y):
        g = by.get(int(f))
        if g is None:
            return None
        k = np.hypot(g.cx.values - x, g.cy.values - y)
        return g.iloc[int(k.argmin())] if k.min() <= 40 else None
    neg = {}
    for p in allp[(~allp.same) & (allp.kind != 'blind')].itertuples():
        a, b = track(p.frame_a, p.xa, p.ya), track(p.frame_b, p.xb, p.yb)
        if a is not None and b is not None:
            neg.setdefault(int(b.raw_id), []).append((int(a.raw_id), a, b))
    wrong = []
    for e in ev[(ev.kind == 'reid') & (ev.gap > 300)].itertuples():
        earlier = set(lab[(lab.id == e.id) & (lab.frame < e.frame) & (lab.raw_id != e.raw_id)].raw_id)
        hit = [x for x in neg.get(int(e.raw_id), []) if x[0] in earlier]
        if hit:
            _, a, b = hit[0]
            cut = lambda r: bool(r.x1 <= 2 or r.y1 <= 2 or r.x2 >= 1918 or r.y2 >= 1078)  # noqa: E731
            wrong.append({'frame': int(e.frame), 'gap_s': round(e.gap / 29.97, 0), 'peak': round(float(e.resp), 3),
                          'match_dist_px': round(float(e.dist), 1), 'earlier_cut_by_edge': cut(a), 'later_cut_by_edge': cut(b)})
    W_ = pd.DataFrame(wrong)
    return {'blind_failures': blind_fail, 'wrong_links': len(W_),
            'wrong_links_with_a_sighting_cut_by_the_edge': int((W_.earlier_cut_by_edge | W_.later_cut_by_edge).sum()) if len(W_) else 0,
            'wrong_links_peak_median': float(W_.peak.median()) if len(W_) else None,
            'all_links_peak_median': float(ev[(ev.kind == 'reid') & (ev.gap > 300)].resp.median()),
            'wrong_link_list': wrong}


def main():
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    F = yaml.safe_load(open('flights.yaml'))
    N = {'flights': flight_table(F), 'agreement': agreement(), 'census': census(F), 'heldout_failures': heldout_failures()}
    json.dump(N, open(f'{OUT}/numbers.json', 'w'), indent=1, default=str)
    print(json.dumps(N, indent=1, default=str))


if __name__ == '__main__':
    main()
