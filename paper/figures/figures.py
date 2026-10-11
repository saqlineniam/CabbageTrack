"""
All figures of the paper, made from the published runs (data/runs/) and the outputs of paper/reproduce.py (house palette,
minimal panels, PNG 300 dpi + PDF). Output: outputs/figures/ (main) and outputs/figures/supplementary/.

  Figure01_concept            problem sketch (original design)
  Figure02_study_site         site figure (restructured original)
  Figure03_pipeline           six stages of the final model (original design, content updated)
  Figure04_detections         one frame per flight with the detector output (Fields A, B, C, held-out)
  Figure05_development        human-checked accuracy after every development step (Fields A, B, C)
  Figure06_identity_maps      every plant identity on the camera map, revisited or not (Fields A, B, C)
  Figure07_mosaics            development mosaics from the video with camera odometry and GNSS (A, B, C)
  Figure08_odometry_vs_gnss   camera odometry vs drone GNSS, true scale, with a 6 m zoom at the end of the row (A, B, C)
  Figure09_fieldA_reidentified  Field A row A: full frame on the first and the last pass, all classes segmented, plant numbers
  Figure10_heldout            held-out: identity map (camera odometry) and mosaic with odometry and GNSS, side by side
  Figure11_heldout_reidentified  held-out: same part of the field on the earlier and later pass, segmentation + identities
  Figure12_sensitivity        settings check on Fields A, B, C
  FigureS4/S5                 as Figure 9 for Field B and Field C
  FigureS1a/S1b               annotation tool (copied)
  FigureS2_sharp_turn         Field B: frame-to-frame rotation, fast turns
  FigureS3_fieldB_before_after Field B identity map, Field A method vs final model
Usage: python paper/figures/figures.py [01 02 ... S5]   (run paper/reproduce.py first for mosaics, GNSS, steps)
"""
import json
import os
import shutil
import sys

import cv2
import matplotlib
import matplotlib.patheffects
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
import matplotlib.patheffects  # noqa: E402,F811

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'groundtruth'))
from palette import C, use  # noqa: E402

OUT = 'outputs/figures'
SUP = os.path.join(OUT, 'supplementary')
CAB = 'data/weights/cabbage.pt'
F = {
    'A': dict(name='Field A', key='fieldA', video='data/videos/fieldA.mp4',
              extract='data/runs/fieldA/extract', run='data/runs/fieldA', weights=CAB,
              gsd=4.15, frame=1200),
    'B': dict(name='Field B', key='fieldB', video='data/videos/fieldB.mp4',
              extract='data/runs/fieldB/extract', run='data/runs/fieldB',
              weights='data/weights/young_cabbage.pt', gsd=4.07, frame=3600),
    'C': dict(name='Field C', key='fieldC', video='data/videos/fieldC.mp4',
              extract='data/runs/fieldC/extract', run='data/runs/fieldC',
              weights='data/weights/cauliflower.pt', gsd=2.74, frame=600),
    'H': dict(name='Held-out field', key='heldout', video='data/videos/heldout.mp4',
              extract='data/runs/heldout/extract', run='data/runs/heldout', weights=CAB,
              gsd=2.96, frame=1500),
}
CLASS_COL = {1: C['leaf_green'], 0: C['steel_blue'], 2: C['orange'], 3: C['aqua_marine']}
CLASS_NAME = {1: 'Plant', 0: 'Aisle', 2: 'Divider', 3: 'Drain'}
FPS = 29.97
GNSS_COL, GNSS_ALPHA = C['slate_blue'], 0.9          # bolder GNSS line (user, 2026-10-10)


def save(fig, name, sup=False):
    d = SUP if sup else OUT
    os.makedirs(d, exist_ok=True)
    for ext, kw in (('png', {'dpi': 300}), ('pdf', {})):
        fig.savefig(os.path.join(d, f'{name}.{ext}'), bbox_inches='tight', facecolor='white', **kw)
    plt.close(fig)
    print('->', os.path.join(d, name))


def read_frame(video, f):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, img = cap.read()
    cap.release()
    return img


def revisited(lab):
    return lab.groupby('id').frame.apply(lambda f: 1 + int((np.diff(np.sort(f.unique())) > 300).sum())) >= 2


# ---------------------------------------------------------------- 1-3: concept, site, pipeline
def figure01():
    import orig_style_figures as O
    O.OUT = OUT
    O.fig1_concept()
    for ext in ('png', 'pdf'):
        os.replace(os.path.join(OUT, f'fig1_concept.{ext}'), os.path.join(OUT, f'Figure01_concept.{ext}'))


def figure02():
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        shutil.copy(f'data/results/figures/Figure02_study_site.{ext}', os.path.join(OUT, f'Figure02_study_site.{ext}'))
    print('->', os.path.join(OUT, 'Figure02_study_site'))


def figure03():
    import orig_style_figures as O
    from matplotlib.patches import Rectangle as R
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(-3.2, 30.5); ax.axis('off')
    ax.text(50, 29.3, 'CabbageTrack Pipeline: Six Processing Stages', ha='center', va='center', fontsize=15,
            fontweight='bold', color=O.NAVY, family='serif')
    ax.plot([0.5, 99.5], [27.6, 27.6], color=O.NAVY, lw=2.2)
    w, gap = 15.4, 1.2
    x0s = [1 + i * (w + gap) for i in range(6)]
    O.box(ax, x0s[0], 22.9, x0s[1] + w - x0s[0], 3.4, O.NAVY, r=0.4)
    ax.text((x0s[0] + x0s[1] + w) / 2, 24.6, 'Every frame: detection, map, rotation', ha='center',
            va='center', color='white', fontsize=9.5, fontweight='bold', family='serif')
    O.box(ax, x0s[2], 22.9, x0s[5] + w - x0s[2], 3.4, O.CRIMSON, r=0.4)
    ax.text((x0s[2] + x0s[5] + w) / 2, 24.6, 'Every new plant track: re-identification', ha='center', va='center',
            color='white', fontsize=9.5, fontweight='bold', family='serif')
    stages = [
        (O.NAVY, '#E6E9EC', 'Detection &\nTracking', ['YOLOv9c-seg, conf 0.5', 'BoT-SORT within a pass', 'plant positions per frame']),
        (O.SLATE, '#E8EEF2', 'Odometry &\nRotation', ['phase correlation: 2-D map', 'ORB: rotation per frame', 'fast turns > 0.3°/frame']),
        (O.AQUA, '#E3F1F0', 'Revisit\nQuery', ['earlier frames sharing', '≥ 25 % of the image', 'one link per visit, no turns']),
        (O.CRIMSON, '#F7E1E5', 'Link\nVerification', ['rotate earlier view (≥ 1°)', 'correlate overlap only, ≥ 0.08', 'within 400 px of prediction*']),
        (O.DARKRED, '#F3E0E1', 'Cross-Pass\nRegistration', ['move remembered plants', 'by rotation and shift', 'skip visible identities']),
        (O.OCEAN, '#E5ECF0', 'Assignment', ['Hungarian, gate 60 px', 'patience 60 frames', 'else new identity']),
    ]
    for i, (col, fill, title, lines) in enumerate(stages):
        x0 = x0s[i]
        O.box(ax, x0, 2.4, w, 19.6, fill, col, lw=1.4, r=0.5)
        ax.add_patch(R((x0 + 0.7, 17.6), w - 1.4, 3.6, fc=col, ec='none'))
        ax.text(x0 + w / 2, 19.4, f'Stage {i + 1}', ha='center', va='center', color='white', fontsize=11.5,
                fontweight='bold', family='serif')
        ax.text(x0 + w / 2, 14.3, title, ha='center', va='center', color=col, fontsize=11, fontweight='bold',
                linespacing=1.05, family='serif')
        ax.plot([x0 + 1.2, x0 + w - 1.2], [10.9, 10.9], color=col, lw=0.6, alpha=0.6)
        for k, t in enumerate(lines):
            ax.text(x0 + w / 2, 8.9 - 2.35 * k, t, ha='center', va='center', color='#1B1B1B', fontsize=7.0, family='serif')
        if i < 5:
            O.arrow(ax, (x0 + w + 0.05, 12.2), (x0s[i + 1] - 0.05, 12.2), O.AQUA, lw=1.2, ms=9)
    ax.plot([x0s[2] - gap / 2] * 2, [2.0, 22.4], ls=(0, (3, 2)), color=O.AQUA, lw=1.0)
    O.box(ax, 6, -3.0, 88, 3.6, 'white', O.AQUA, lw=1.0, r=0.4)
    for x, c, t in ((8, O.NAVY, 'Map-building stages'), (27, O.CRIMSON, 'Re-identification stages'), (49, O.AQUA, 'Data flow')):
        ax.add_patch(R((x, -2.05), 2.0, 1.7, fc=c, ec='none'))
        ax.text(x + 2.8, -1.2, t, ha='left', va='center', fontsize=9, color=O.NAVY, family='serif')
    ax.text(62, -1.2, '* prediction corrected by recent clear revisits (drift)', ha='left', va='center', fontsize=8,
            color=O.NAVY, family='serif')
    save(fig, 'Figure03_pipeline')


# ---------------------------------------------------------------- 4: detections
def figure04():
    from ultralytics import YOLO
    use()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6))
    for ax, (k, letter) in zip(axes.ravel(), zip('ABCH', 'abcd')):
        f = F[k]
        img = read_frame(f['video'], f['frame'])
        res = YOLO(f['weights']).predict(img, imgsz=640, conf=0.5, retina_masks=True, verbose=False)[0]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255
        over = rgb.copy()
        if res.masks is not None:
            for m, c in zip(res.masks.data.cpu().numpy() > 0.5, res.boxes.cls.cpu().numpy().astype(int)):
                col = np.array(matplotlib.colors.to_rgb(CLASS_COL[c]))
                over[m] = 0.5 * over[m] + 0.5 * col
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], '-', color=CLASS_COL[c], lw=0.45)
        ax.imshow(over)
        px = 1000.0 / f['gsd']                                   # pixels per metre
        x0, y0 = 1920 - 110 - px, 1080 - 70
        ax.add_patch(Rectangle((x0, y0), px, 20, fc='white', ec=C['midnight_navy'], lw=0.6))
        ax.text(x0 + px / 2, y0 - 14, '1 m', ha='center', va='bottom', color='white', fontsize=7, fontweight='bold')
        ax.set_title(f"({letter}) {f['name']}", loc='left', fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1920); ax.set_ylim(1080, 0)
    handles = [Patch(color=CLASS_COL[c], label=CLASS_NAME[c]) for c in (1, 0, 2, 3)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.6)
    save(fig, 'Figure04_detections')


# ---------------------------------------------------------------- 5: development path
def figure05():
    use()
    names = {'fieldA': 'Field A', 'fieldB': 'Field B', 'fieldC': 'Field C'}
    cols = {'fieldA': C['ocean_blue'], 'fieldB': C['orange'], 'fieldC': C['leaf_green']}
    short = ['Field A\nmethod', '+ overlap\nsearch', '+ drift\ncorrection', '+ rotation', '+ skip fast\nturns',
             '+ overlap-only\nalignment', '+ drift from\nclear links', 'patience 60\n(frozen)']
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for f, name in names.items():
        d = pd.read_csv(f'outputs/development/dev_steps_{f}.csv')
        y = 100 * (d.blind_ok + d.links_ok) / (d.blind_n + d.links_n)
        ax.plot(range(len(d)), y, '-o', color=cols[f], ms=3.5, lw=1.4, label=name)
        dy = {'fieldA': 3.5, 'fieldB': -3.5}.get(f, 0)
        ax.text(len(d) - 1 + 0.12, y.iloc[-1] + dy, f'{y.iloc[-1]:.0f} %', va='center', fontsize=7, color=cols[f])
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, fontsize=6.6)
    ax.set_ylabel('human-confirmed pairs\nre-identified (%)', fontsize=8)
    ax.set_ylim(0, 104); ax.set_xlim(-0.3, len(short) - 0.3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.tight_layout()
    save(fig, 'Figure05_development')


# ---------------------------------------------------------------- 6: identity maps
def identity_map(ax, run, gsd_mm, title):
    lab = pd.read_csv(os.path.join(run, 'labels.csv'))
    mp = pd.read_csv(os.path.join(run, 'map.csv')).set_index('frame')
    g = gsd_mm / 1000
    P = pd.DataFrame({'id': lab.id, 'x': (mp.cam_x.values[lab.frame] + lab.cx - 960) * g,
                      'y': -(mp.cam_y.values[lab.frame] + lab.cy - 540) * g})
    pos = P.groupby('id')[['x', 'y']].median()
    rv = revisited(lab)
    cx, cy = mp.cam_x.values * g, -mp.cam_y.values * g
    ax.plot(cx, cy, '-', color=C['charcoal_gray'], lw=0.5, alpha=0.8)
    ax.plot(pos.x[~rv], pos.y[~rv], 'o', ms=1.5, color=C['orange'])
    ax.plot(pos.x[rv], pos.y[rv], 'o', ms=1.5, color=C['ocean_blue'])
    ax.plot(cx[0], cy[0], 'o', ms=4.5, color=C['leaf_green'])
    ax.set_aspect('equal'); ax.set_xlabel('m', fontsize=7); ax.set_ylabel('m', fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title(f'{title}\n{len(pos)} plants, {int(rv.sum())} revisited', loc='left', fontsize=7.2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def map_legend(fig, y=-0.01):
    handles = [Line2D([], [], color=C['charcoal_gray'], lw=0.8, label='camera path'),
               Line2D([], [], color=C['ocean_blue'], marker='o', ls='', ms=4, label='revisited plant (seen again > 10 s later)'),
               Line2D([], [], color=C['orange'], marker='o', ls='', ms=4, label='plant not revisited'),
               Line2D([], [], color=C['leaf_green'], marker='o', ls='', ms=4.5, label='start')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, y))


def figure06():
    use()
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.6), gridspec_kw={'width_ratios': [1, 1, 0.8]})
    for ax, (k, letter) in zip(axes, zip('ABC', 'abc')):
        identity_map(ax, F[k]['run'], F[k]['gsd'], f"({letter}) {F[k]['name']}")
    map_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'Figure06_identity_maps')


# ---------------------------------------------------------------- 7: development mosaics
def figure07():
    import mosaic_panel as M
    use()
    keys = [('fieldA', '(a) Field A'), ('fieldB', '(b) Field B'), ('fieldC', '(c) Field C')]
    shapes = [cv2.imread(f'outputs/mosaic/{k}/mosaic.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_8).shape for k, _ in keys]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 6.4), gridspec_kw={'width_ratios': [s_[1] / s_[0] for s_ in shapes],
                                                                     'wspace': 0.42})
    for ax, (k, title) in zip(axs, keys):
        M.panel(ax, k, title)
        for t in ax.texts:
            t.set_fontsize(6.5)
        ax.title.set_fontsize(7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 0.06))
    save(fig, 'Figure07_mosaics')


# ---------------------------------------------------------------- 8: odometry vs GNSS, full track + zoom at the turn
def figure08():
    use()
    keys = [('fieldA', '(a) Field A'), ('fieldB', '(b) Field B'), ('fieldC', '(c) Field C')]
    D = {k: pd.read_csv(f'outputs/gnss/{k}/gnss_per_frame.csv') for k, _ in keys}
    S = {k: json.load(open(f'outputs/gnss/{k}/gnss_summary.json')) for k, _ in keys}
    wr = [np.ptp(D[k].gps_E) / np.ptp(D[k].gps_N) + 0.25 for k, _ in keys]
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1], width_ratios=wr, wspace=0.55, hspace=0.12)
    half = 3.0                                                                 # zoom window: 6 m x 6 m
    for j, (k, title) in enumerate(keys):
        d = D[k]
        ax = fig.add_subplot(gs[0, j])
        az = fig.add_subplot(gs[1, j])
        i = int(np.argmax(np.hypot(d.gps_E - d.gps_E[0], d.gps_N - d.gps_N[0])))   # far end of the loop (the turn)
        zx, zy = d.gps_E[i], d.gps_N[i]
        for a in (ax, az):
            a.plot(d.gps_E, d.gps_N, '-', color=GNSS_COL, lw=4.0 if a is ax else 7.0, alpha=GNSS_ALPHA,
                   solid_capstyle='round', label='drone GNSS (one similarity fit)')
            a.plot(d.odo_E, d.odo_N, '-', color=C['crimson'], lw=0.9 if a is ax else 1.3, label='camera odometry (method)')
            a.set_aspect('equal')
            a.tick_params(labelsize=6)
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)
        ax.plot(d.gps_E[0], d.gps_N[0], 'o', ms=4, color=C['leaf_green'], label='start')
        ax.add_patch(Rectangle((zx - half, zy - half), 2 * half, 2 * half, fill=False, ec=C['midnight_navy'], lw=0.7))
        az.set_xlim(zx - half, zx + half); az.set_ylim(zy - half, zy + half)
        ax.set_title(title, loc='left', fontsize=7.5)
        az.set_title('zoom: end of the row (6 m x 6 m)', loc='left', fontsize=6.2)
        ax.text(1.03, 1.0, f"RMSE\n{S[k]['ATE_rmse_m']:.2f} m", transform=ax.transAxes, ha='left', va='top', fontsize=6.5,
                color=C['midnight_navy'])
        ax.set_xlabel('East (m)', fontsize=6.5); az.set_xlabel('East (m)', fontsize=6.5)
        if j == 0:
            ax.set_ylabel('North (m)', fontsize=6.5); az.set_ylabel('North (m)', fontsize=6.5)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h[:2] + h[2:3], l[:2] + l[2:3], loc='lower center', ncol=3, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, 'Figure08_odometry_vs_gnss')


# ---------------------------------------------------------------- 9 (+ S4, S5): a field strip on two distant passes
def pick_pass_pair(k, legs_pair=None, min_gap=1500, min_peak=0.25, max_dx=400, max_dy=250):
    """Earlier / later frame of the same ground on two distant passes: among the frozen run's verified links (gap >
    min_gap, clear peak, views well aligned), the pair that shares the most identified plants."""
    from mosaic_panel import legs_from_map
    f = F[k]
    L = pd.read_csv(os.path.join(f['run'], 'links.csv'))
    mp = pd.read_csv(os.path.join(f['run'], 'map.csv'))
    leg = legs_from_map(mp)
    lab = pd.read_csv(os.path.join(f['run'], 'labels.csv'))
    ids = lab.groupby('frame').id.apply(set)
    L = L[(L.gap > min_gap) & (L.resp >= min_peak) & (L.dx.abs() < max_dx) & (L.dy.abs() < max_dy)].copy()
    if legs_pair:
        L = L[(leg[L.frame.values] == legs_pair[0]) & (leg[L.s.values] == legs_pair[1])]
    L['shared'] = [len(ids.get(int(r.s), set()) & ids.get(int(r.frame), set())) for r in L.itertuples()]
    best = L.sort_values(['shared', 'gap'], ascending=False).iloc[0]
    return int(best.s), int(best.frame), int(best.shared)


def pass_pair_figure(k, name, sup=False, legs_pair=None, ground='over the same ground', **pick):
    from ultralytics import YOLO
    use()
    f = F[k]
    s_fr, t_fr, shared = pick_pass_pair(k, legs_pair, **pick)
    lab = pd.read_csv(os.path.join(f['run'], 'labels.csv'))
    raw2id = lab.drop_duplicates('raw_id').set_index('raw_id').id
    dets = pd.read_csv(os.path.join(f['extract'], 'detections.csv.gz'))
    dd_ = dets[(dets.cls == 1) & (dets.raw_id >= 0)]
    byf = {fr_: g_ for fr_, g_ in dd_.groupby('frame')}

    def ident(fr_, x_, y_):
        g_ = byf.get(int(fr_))
        if g_ is None:
            return None
        k_ = np.hypot(g_.cx.values - x_, g_.cy.values - y_)
        return raw2id.get(int(g_.raw_id.values[k_.argmin()])) if k_.min() < 60 else None

    model = YOLO(f['weights'])
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.6))
    found = {}
    for ax, fr, which in zip(axes, (s_fr, t_fr), ('earlier', 'later')):
        img = read_frame(f['video'], fr)
        res = model.predict(img, imgsz=640, conf=0.5, retina_masks=True, verbose=False)[0]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255
        over = rgb.copy()
        items = []
        if res.masks is not None:
            for mk, c in zip(res.masks.data.cpu().numpy() > 0.5, res.boxes.cls.cpu().numpy().astype(int)):
                ys, xs = np.nonzero(mk)
                if not len(xs):
                    continue
                over[mk] = 0.55 * over[mk] + 0.45 * np.array(matplotlib.colors.to_rgb(CLASS_COL[c]))
                items.append((mk, c, ident(fr, xs.mean(), ys.mean()) if c == 1 else None, xs.mean(), ys.mean()))
        ax.imshow(over)
        found[which] = {pid for _, c, pid, _, _ in items if pid is not None}
        for mk, c, pid, mx, my in items:
            cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], '-', color=CLASS_COL[c] if c != 1 else C['deep_teal'], lw=0.5)
            if pid is not None:
                ax.text(mx, my, str(pid), ha='center', va='center', fontsize=5.2, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.12', ec='none', alpha=0.9, fc=C['ocean_blue']))
        px = 1000.0 / f['gsd']
        x0, y0 = 1920 - 110 - px, 1080 - 60
        ax.add_patch(Rectangle((x0, y0), px, 18, fc='white', ec=C['midnight_navy'], lw=0.6))
        ax.text(x0 + px / 2, y0 - 12, '1 m', ha='center', va='bottom', color='white', fontsize=7, fontweight='bold',
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.6, foreground=C['midnight_navy'])])
        ax.set_xlim(0, 1920); ax.set_ylim(1080, 0); ax.set_xticks([]); ax.set_yticks([])
    same = len(found['earlier'] & found['later'])
    axes[0].set_title(f"(a) {f['name']}, earlier pass: frame {s_fr}, t = {s_fr / FPS:.1f} s", loc='left', fontsize=8)
    axes[1].set_title(f"(b) {f['name']}, later pass {ground}: frame {t_fr}, t = {t_fr / FPS:.1f} s "
                      f"({(t_fr - s_fr) / FPS:.0f} s later) - {same} plants carry the same number as in (a)",
                      loc='left', fontsize=8)
    handles = [Patch(color=CLASS_COL[c], label=CLASS_NAME[c]) for c in (1, 0, 2, 3)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.5)
    save(fig, name, sup=sup)
    print(f'   {k}: frames {s_fr} -> {t_fr}, {same} plants with the same number in both frames')


def figure09():
    pass_pair_figure('A', 'Figure09_fieldA_reidentified', legs_pair=('leg3_down', 'leg0_up'))


def figure_s4():
    pass_pair_figure('B', 'FigureS4_fieldB_reidentified', sup=True)


def figure_s5():
    # passes ~1350 px apart sideways and weak alignments (Field C): the shared plants lie in the overlap strip
    pass_pair_figure('C', 'FigureS5_fieldC_reidentified', sup=True, min_peak=0.08, max_dx=1920, max_dy=1080,
                     ground='partly over the same ground (overlap on the right)')


# ---------------------------------------------------------------- 9-10: held-out
def heldout_examples():
    """Three human-confirmed blind plants the frozen model re-identified on the held-out flight (longest gaps, three parts
    of the flight, plant well inside both frames), and an identity lookup through the tracker tracks."""
    import score as G
    f = F['H']
    P = pd.read_csv('data/groundtruth/pairs/gt_pairs.csv').query("seq == 'f2' and kind == 'blind'").reset_index(drop=True)
    lab = pd.read_csv(os.path.join(f['run'], 'labels.csv'))
    dets = pd.read_csv(os.path.join(f['extract'], 'detections.csv.gz'))
    o = G.score(lab, P, dets).values
    P['gap'] = P.frame_b - P.frame_a
    P['edge'] = np.minimum.reduce([P.xa, P.ya, 1920 - P.xa, 1080 - P.ya, P.xb, P.yb, 1920 - P.xb, 1080 - P.yb])
    ok = P[(o == 'correct') & (P.edge > 300)].sort_values('gap', ascending=False)
    picks, used = [], []
    for i, r in ok.iterrows():
        band = int(r.frame_b // 700)
        if band not in used:
            picks.append(i); used.append(band)
        if len(picks) == 3:
            break
    raw2id = lab.drop_duplicates('raw_id').set_index('raw_id').id
    dd_ = dets[(dets.cls == 1) & (dets.raw_id >= 0)]
    byf = {fr_: g_ for fr_, g_ in dd_.groupby('frame')}

    def ident(fr_, x_, y_, tol=40):
        g_ = byf.get(int(fr_))
        if g_ is None:
            return None
        k_ = np.hypot(g_.cx.values - x_, g_.cy.values - y_)
        return raw2id.get(int(g_.raw_id.values[k_.argmin()])) if k_.min() < tol else None
    return P, picks, ident


def figure10():
    from cabbagetrack.gnss import read_srt, umeyama
    from cabbagetrack.mosaic import refine
    use()
    f = F['H']
    run, s = f['run'], 0.5
    n = len(pd.read_csv(os.path.join(run, 'map.csv')))
    cx, cy = refine(run, f['extract'], n)[:2]
    x0, y0 = cx.min(), cy.min()
    mos = cv2.cvtColor(cv2.imread('outputs/mosaic/heldout/mosaic.jpg'), cv2.COLOR_BGR2RGB)
    ox, oy = (cx + 960 - x0) * s, (cy + 540 - y0) * s
    g = read_srt('data/gnss/heldout.SRT')
    m = min(n, len(g))
    sc, R, t = umeyama(np.c_[cx[:m], -cy[:m]], g[['E', 'N']].values[:m])
    pg = (R.T @ (g[['E', 'N']].values[:m] - t).T).T / sc
    gx, gy = (pg[:, 0] + 960 - x0) * s, (-pg[:, 1] + 540 - y0) * s
    P, picks, _ = heldout_examples()
    fig = plt.figure(figsize=(5.4, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.32], wspace=0.12)
    am = fig.add_subplot(gs[0, 0])
    identity_map(am, run, f['gsd'], '(a) Held-out field: camera odometry')
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mos)
    ax.plot(gx, gy, '-', color=GNSS_COL, lw=3.0, alpha=GNSS_ALPHA, label='drone GNSS (one similarity fit)')
    ax.plot(ox, oy, '-', color=C['crimson'], lw=1.0, label='camera odometry (method)')
    ax.plot(ox[0], oy[0], 'o', ms=5, color=C['leaf_green'], label='start')
    for i, letter in zip(picks, 'abc'):                  # where the plants of Figure 11 are
        r = P.loc[i]
        u, v = (cx[int(r.frame_a)] + r.xa - x0) * s, (cy[int(r.frame_a)] + r.ya - y0) * s
        ax.add_patch(Circle((u, v), 38, fill=False, ec=C['orange'], lw=1.4))
        ax.text(u + 46, v, letter, color='white', fontsize=6.5, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.12', fc=C['orange'], ec='none'))
    gsd = json.load(open('outputs/gnss/heldout/gnss_summary.json'))
    L = 2.0 / (gsd['scale_mm_per_px'] / 1000 / s)
    hgt = mos.shape[0]
    ax.add_patch(Rectangle((60, hgt - 110), L, 22, fc='white', ec=C['midnight_navy'], lw=0.6))
    ax.text(60 + L / 2, hgt - 125, '2 m', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold',
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.6, foreground=C['midnight_navy'])])
    ax.text(1.03, 1.0, f"RMSE\n{gsd['ATE_rmse_m']:.2f} m", transform=ax.transAxes, ha='left', va='top', fontsize=6.5,
            color=C['midnight_navy'])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('(b) Mosaic from the video', loc='left', fontsize=7.2)
    h1 = [Line2D([], [], color=C['charcoal_gray'], lw=0.8, label='camera path'),
          Line2D([], [], color=C['ocean_blue'], marker='o', ls='', ms=4, label='revisited plant (seen again > 10 s later)'),
          Line2D([], [], color=C['orange'], marker='o', ls='', ms=4, label='plant not revisited')]
    h2, _ = ax.get_legend_handles_labels()
    fig.legend(handles=h1 + h2, loc='lower center', ncol=2, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    save(fig, 'Figure10_heldout')


def figure11():
    """Held-out: the same part of the field on the earlier and the later pass, plant segmentation with identity numbers."""
    from ultralytics import YOLO
    use()
    f = F['H']
    P, picks, ident = heldout_examples()
    model = YOLO(f['weights'])
    W, Hc = 1100, 680
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.0))
    for row, (i, letter) in enumerate(zip(picks, 'abc')):
        r = P.loc[i]
        tid = ident(r.frame_b, r.xb, r.yb)
        for col, (fr, x, y) in enumerate([(r.frame_a, r.xa, r.ya), (r.frame_b, r.xb, r.yb)]):
            img = read_frame(f['video'], fr)
            res = model.predict(img, imgsz=640, conf=0.5, retina_masks=True, verbose=False)[0]
            xx = int(np.clip(x - W / 2, 0, 1920 - W)); yy = int(np.clip(y - Hc / 2, 0, 1080 - Hc))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255
            ax = axes[row, col]
            mark = C['crimson'] if col else C['orange']
            over = rgb.copy()
            plants = []
            if res.masks is not None:
                for mk, c in zip(res.masks.data.cpu().numpy() > 0.5, res.boxes.cls.cpu().numpy().astype(int)):
                    if c != 1:
                        continue
                    ys, xs = np.nonzero(mk)
                    if not len(xs):
                        continue
                    pid = ident(fr, xs.mean(), ys.mean(), tol=60)
                    hit = pid is not None and pid == tid
                    col_rgb = np.array(matplotlib.colors.to_rgb(mark if hit else C['leaf_green']))
                    over[mk] = 0.55 * over[mk] + 0.45 * col_rgb
                    plants.append((mk, pid, hit, xs.mean(), ys.mean()))
            ax.imshow(over[yy:yy + Hc, xx:xx + W])
            for mk, pid, hit, mx, my in plants:
                cnts, _ = cv2.findContours(mk[yy:yy + Hc, xx:xx + W].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], '-', color=mark if hit else C['deep_teal'], lw=1.3 if hit else 0.5)
                u, v = mx - xx, my - yy
                if pid is not None and 0 <= u < W and 0 <= v < Hc:
                    ax.text(u, v, str(pid), ha='center', va='center', fontsize=6.4 if hit else 5.0, color='white',
                            fontweight='bold', bbox=dict(boxstyle='round,pad=0.14', ec='none', alpha=0.9,
                                                         fc=mark if hit else C['ocean_blue']))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(mark if col else C['orange']); sp.set_linewidth(0.8)
            if col == 0:
                ax.set_xlabel(f'earlier pass: frame {int(fr)}, t = {fr / FPS:.1f} s', fontsize=6.5, labelpad=2)
                ax.set_title(f'({letter}) plant {tid}', loc='left', fontsize=7.5)
            else:
                ax.set_xlabel(f'later pass: frame {int(fr)}, t = {fr / FPS:.1f} s  ({(r.frame_b - r.frame_a) / FPS:.0f} s later)',
                              fontsize=6.5, labelpad=2)
    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    save(fig, 'Figure11_heldout_reidentified')


# ---------------------------------------------------------------- 11: sensitivity
def figure12():
    use()
    base = {'gate': 60, 'min_overlap': 0.25, 'shift_tol': 400, 'drift_min_peak': 0.2, 'turn_rate': 0.3,
            'rot_min_deg': 1.0, 'patience': 30}
    label = {'gate': 'gate (px)', 'min_overlap': 'image overlap', 'shift_tol': 'shift tolerance (px)',
             'drift_min_peak': 'drift: min. peak', 'turn_rate': 'fast turn (°/frame)', 'rot_min_deg': 'min. rotation (°)',
             'patience': 'patience (frames)'}
    names = {'fieldA': ('Field A', C['ocean_blue']), 'fieldB': ('Field B', C['orange']), 'fieldC': ('Field C', C['leaf_green'])}
    D = {f: pd.read_csv(f'outputs/development/settings_{f}.csv').set_index('setting') for f in names}
    tot = {'fieldA': 671, 'fieldB': 775, 'fieldC': 156}
    fig, axes = plt.subplots(1, 7, figsize=(7.4, 2.0), sharey=True)
    for ax, (p, b) in zip(axes, base.items()):
        key = {float(s_.split('=')[1]): s_ for s_ in D['fieldA'].index if s_.startswith(p + '=')}
        key[float(b)] = 'base'
        vals = sorted(key)
        for f, (name, col) in names.items():
            y = [100 * D[f].loc[key[v], 'correct'] / tot[f] for v in vals]
            ax.plot(vals, y, '-o', color=col, ms=2.5, lw=1.0, label=name)
        if p == 'gate':
            ax.text(80, 33, '80 px:\n+8 wrong\nlinks (C)', ha='right', va='top', fontsize=5.4, color=C['leaf_green'])
        frozen = 60 if p == 'patience' else b
        ax.axvline(frozen, color=C['charcoal_gray'], lw=0.6, ls=':')
        ax.set_xlabel(label[p], fontsize=6.2); ax.tick_params(labelsize=5.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('pairs re-identified (%)', fontsize=6.5)
    axes[0].set_ylim(0, 104)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'Figure12_sensitivity')


# ---------------------------------------------------------------- supplementary
def figure_s1():
    os.makedirs(SUP, exist_ok=True)
    for a, b in (('figS1a_gt_tool_partA', 'FigureS1a_annotation_links'), ('figS1b_gt_tool_partB', 'FigureS1b_annotation_blind')):
        shutil.copy(f'data/results/figures/supplementary/{b}.png', os.path.join(SUP, f'{b}.png'))   # screenshots
    print('->', SUP, 'FigureS1a/S1b')


def figure_s2():
    use()
    rot = pd.read_csv(os.path.join(F['B']['run'], 'rotation.csv'))
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    t = rot.frame / FPS
    ax.plot(t, rot.phi_deg, '-', color=C['slate_blue'], lw=0.5)
    fast = rot.in_fast_turn.values
    ax.fill_between(t, -3, 3, where=fast, color=C['orange'], alpha=0.25, lw=0, label='inside a fast turn (frames not used)')
    ax.axhline(0.3, color=C['charcoal_gray'], lw=0.5, ls=':'); ax.axhline(-0.3, color=C['charcoal_gray'], lw=0.5, ls=':')
    ax.set_ylim(-3, 3); ax.set_xlim(0, t.max())
    ax.set_xlabel('time (s)', fontsize=7); ax.set_ylabel('rotation (°/frame)', fontsize=7); ax.tick_params(labelsize=6.5)
    ax.text(0.99, 0.95, 'last pass vs first pass (right bed): −2.9°\nreturn pass vs outbound pass (left bed): +0.5°',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.3, color=C['midnight_navy'])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left')
    fig.tight_layout()
    save(fig, 'FigureS2_fieldB_sharp_turn', sup=True)


def figure_s3():
    use()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 4.6))
    identity_map(axes[0], 'data/runs/fieldB_v1', F['B']['gsd'], '(a) Field A method')
    identity_map(axes[1], F['B']['run'], F['B']['gsd'], '(b) Final model')
    map_legend(fig, y=-0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'FigureS3_fieldB_before_after', sup=True)


ALL = {'01': figure01, '02': figure02, '03': figure03, '04': figure04, '05': figure05, '06': figure06, '07': figure07,
       '08': figure08, '09': figure09, '10': figure10, '11': figure11, '12': figure12, 'S1': figure_s1, 'S2': figure_s2,
       'S3': figure_s3, 'S4': figure_s4, 'S5': figure_s5}

if __name__ == '__main__':
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
