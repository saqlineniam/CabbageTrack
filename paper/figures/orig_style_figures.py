"""
Figures 1 and 3 of the submission, redrawn in their original design (same layout, serif type and house colours) with
the content corrected for the revised method. The originals (Cabbagetrack_Manuscript_Submission.docx, image1/image3)
are not touched.

  fig1_concept   problem sketch: earlier pass, plants out of view, later pass in the opposite direction, no GPS
                 (was: "loop pass", one crop row, "372 plants matched", a turn)
  fig3_pipeline  six processing stages of the revised method (was: six stages of the 1-D pipeline)

Output: outputs/paper_figures/fig1_concept.{png,pdf}, fig3_pipeline_stages.{png,pdf}
Used by paper/figures/figures.py (Figures 1 and 3)
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse  # noqa: E402

OUT = 'outputs/figures'
NAVY, CRIMSON, AQUA, TEAL = '#02203A', '#C01331', '#4EA9A2', '#1C694B'
SLATE, DARKRED, OCEAN, GREY = '#44728C', '#8F020A', '#1A465E', '#5D6161'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix'})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (('png', {'dpi': 300}), ('pdf', {})):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight', facecolor='white', **kw)
    plt.close(fig)
    print('->', name)


def box(ax, x, y, w, h, fc, ec=None, lw=1.2, r=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec or fc, lw=lw))


def arrow(ax, p0, p1, color, lw=2.2, rad=0.0, ms=16):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def fig1_concept():
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.set_xlim(0, 100); ax.set_ylim(-7, 46); ax.axis('off')
    ax.text(50, 44.2, 'GPS-Free Plant Re-Identification Across UAV Passes', ha='center', va='center',
            fontsize=16, fontweight='bold', color=NAVY)
    ax.text(50, 40.6, 'every plant seen on an earlier pass must be recognised on a later pass', ha='center',
            va='center', fontsize=14, fontweight='bold', color=NAVY)
    ax.plot([2, 98], [38.0, 38.0], color=NAVY, lw=2.6)
    # earlier pass
    box(ax, 4, 33.2, 9, 3.3, NAVY, r=0.5)
    ax.text(8.5, 34.85, 'START', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.text(8.5, 31.7, 'UAV', ha='center', va='center', color=NAVY, fontsize=11, fontweight='bold')
    ax.plot(8.5, 29.4, marker='^', ms=11, color=NAVY)
    arrow(ax, (8.5, 29.4), (80, 29.4), NAVY, lw=2.6)
    ax.text(45, 32.2, 'Earlier pass: plants detected, tracked and given identities', ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)
    # crop bed with several plant lines, numbered plants
    ax.add_patch(Rectangle((11, 15.4), 70.5, 11.6, fc='none', ec=AQUA, lw=1.4))
    ax.text(1.0, 21.2, 'Crop bed', ha='left', va='center', fontsize=12.5, style='italic', color=AQUA)
    rng = np.random.default_rng(7)
    xs = np.linspace(14.0, 78.5, 22)
    n = 0
    for yl in (24.6, 21.2, 17.8):
        for x in xs:
            n += 1
            y = yl + rng.uniform(-0.35, 0.35)
            ax.add_patch(Ellipse((x, y), 2.65, 2.75, fc=TEAL, ec='white', lw=0.6))
            ax.text(x, y, str(n), ha='center', va='center', fontsize=4.6, color='white', fontweight='bold')
    # out of view between the passes
    box(ax, 85.0, 15.4, 14, 11.6, '#E3F1F0', AQUA, lw=1.6, r=0.6)
    ax.add_patch(Rectangle((85.8, 25.2), 12.4, 1.0, fc=AQUA, ec='none'))
    ax.text(92, 22.4, 'Out of view', ha='center', va='center', fontsize=12.5, fontweight='bold', color=NAVY)
    ax.text(92, 19.4, '10–200 s', ha='center', va='center', fontsize=11, style='italic', color=AQUA)
    ax.text(92, 17.0, 'no GPS', ha='center', va='center', fontsize=11, style='italic', color=AQUA)
    arrow(ax, (82, 29.4), (91.5, 27.2), AQUA, lw=1.6, rad=-0.35, ms=13)
    arrow(ax, (91.5, 15.2), (82.5, 11.3), AQUA, lw=1.6, rad=-0.35, ms=13)
    # later pass, opposite direction, shifted sideways
    ax.plot(81.0, 11.0, marker='^', ms=11, color=CRIMSON)
    ax.text(81.0, 8.4, 'UAV', ha='center', va='center', color=CRIMSON, fontsize=11, fontweight='bold')
    arrow(ax, (81.0, 11.0), (8.0, 11.0), CRIMSON, lw=2.6)
    ax.text(43, 7.9, 'Later pass (reversed, shifted sideways): re-identify each plant, no GPS',
            ha='center', va='center', fontsize=13, fontweight='bold', color=CRIMSON)
    box(ax, 4, 2.2, 9, 3.3, CRIMSON, r=0.5)
    ax.text(8.5, 3.85, 'END', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    # legend
    box(ax, 5, -6.3, 90, 4.6, 'white', AQUA, lw=1.2, r=0.4)
    for x, c, t in ((7.5, NAVY, 'Earlier pass'), (27.5, CRIMSON, 'Later pass: re-identification (no GPS)'),
                    (66.5, TEAL, 'Cabbage plant (numbered)')):
        ax.add_patch(Rectangle((x, -5.0), 2.2, 2.0, fc=c, ec='none'))
        ax.text(x + 3.2, -4.0, t, ha='left', va='center', fontsize=11, color=NAVY)
    save(fig, 'fig1_concept')


def fig3_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.25))
    ax.set_xlim(0, 100); ax.set_ylim(-3.2, 30.5); ax.axis('off')
    ax.text(50, 29.3, 'CabbageTrack Pipeline: Six Processing Stages', ha='center', va='center', fontsize=15,
            fontweight='bold', color=NAVY)
    ax.plot([0.5, 99.5], [27.6, 27.6], color=NAVY, lw=2.2)
    w, gap = 15.4, 1.2
    x0s = [1 + i * (w + gap) for i in range(6)]
    box(ax, x0s[0], 22.9, x0s[1] + w - x0s[0], 3.4, NAVY, r=0.4)
    ax.text((x0s[0] + x0s[1] + w) / 2, 24.6, 'Every frame: detection, tracking and map', ha='center', va='center',
            color='white', fontsize=9.5, fontweight='bold')
    box(ax, x0s[2], 22.9, x0s[5] + w - x0s[2], 3.4, CRIMSON, r=0.4)
    ax.text((x0s[2] + x0s[5] + w) / 2, 24.6, 'Every new plant track: re-identification', ha='center', va='center',
            color='white', fontsize=9.5, fontweight='bold')
    stages = [
        (NAVY, '#E6E9EC', 'Detection &\nTracking', ['YOLOv9c-seg, conf 0.5', 'BoT-SORT per pass', 'plant positions per frame']),
        (SLATE, '#E8EEF2', 'Phase Corr.\nOdometry', ['cv2.phaseCorrelate()', '2-D map, no projection', 'clamp >25 px → 0']),
        (AQUA, '#E3F1F0', 'Revisit\nQuery', ['memory > 1.5 s old', 'within 600 px on map', '3 candidate frames']),
        (CRIMSON, '#F7E1E5', 'Link\nVerification', ['whole-image phase corr.', 'peak ≥ 0.08', 'shift within 400 px']),
        (DARKRED, '#F3E0E1', 'Cross-Pass\nRegistration', ['move remembered plants', 'by measured shift', 'skip visible identities']),
        (OCEAN, '#E5ECF0', 'Assignment', ['Hungarian, gate 60 px', 'patience 30 frames', 'else new identity']),
    ]
    for i, (col, fill, title, lines) in enumerate(stages):
        x0 = x0s[i]
        box(ax, x0, 2.4, w, 19.6, fill, col, lw=1.4, r=0.5)
        ax.add_patch(Rectangle((x0 + 0.7, 17.6), w - 1.4, 3.6, fc=col, ec='none'))
        ax.text(x0 + w / 2, 19.4, f'Stage {i + 1}', ha='center', va='center', color='white', fontsize=11.5,
                fontweight='bold')
        ax.text(x0 + w / 2, 14.3, title, ha='center', va='center', color=col, fontsize=11, fontweight='bold',
                linespacing=1.05)
        ax.plot([x0 + 1.2, x0 + w - 1.2], [10.9, 10.9], color=col, lw=0.6, alpha=0.6)
        for k, t in enumerate(lines):
            ax.text(x0 + w / 2, 8.9 - 2.35 * k, t, ha='center', va='center', color='#1B1B1B', fontsize=7.4)
        if i < 5:
            arrow(ax, (x0 + w + 0.05, 12.2), (x0s[i + 1] - 0.05, 12.2), AQUA, lw=1.2, ms=9)
    ax.plot([x0s[2] - gap / 2] * 2, [2.0, 22.4], ls=(0, (3, 2)), color=AQUA, lw=1.0)
    box(ax, 18, -3.0, 64, 3.6, 'white', AQUA, lw=1.0, r=0.4)
    for x, c, t in ((20.5, NAVY, 'Map-building stages'), (42.5, CRIMSON, 'Re-identification stages'),
                    (67.0, AQUA, 'Data flow')):
        ax.add_patch(Rectangle((x, -2.05), 2.0, 1.7, fc=c, ec='none'))
        ax.text(x + 2.8, -1.2, t, ha='left', va='center', fontsize=9.5, color=NAVY)
    save(fig, 'fig3_pipeline_stages')


if __name__ == '__main__':
    fig1_concept()
    fig3_pipeline()
