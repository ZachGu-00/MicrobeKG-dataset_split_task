"""Shared figure conventions for the MicrobeKG-Bench paper (NeurIPS text width 5.5 in)."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.clawsgo/skills/figure-style"))
from kernel import apply_figure_style, panel_letter, panel_crops  # noqa: F401

TEXT_W = 5.5
FONT = "Liberation Sans"
SIZES = (7.5, 6.5, 6)

LEARNED = ["transe", "distmult", "complex", "rotate", "tucker", "rgcn", "ultra-zeroshot", "ultra-finetune"]
LABEL = {"popularity": "Degree prior", "cn": "CN", "ra": "RA", "l3": "L3", "transe": "TransE",
         "distmult": "DistMult", "complex": "ComplEx", "rotate": "RotatE", "tucker": "TuckER",
         "rgcn": "R-GCN", "ultra-zeroshot": "ULTRA (0-shot)", "ultra-finetune": "ULTRA (tuned)",
         "random": "Random", "ncn": "NCN", "gbm": "GBM", "nbfnet": "NBFNet", "KGNMDA": "KGNMDA",
         "rotate-rs": "RotatE (RS)", "tucker-rs": "TuckER (RS)", "rgcn-rs": "R-GCN (RS)"}
SHORT = dict(LABEL, **{"ultra-zeroshot": "ULTRA-0", "ultra-finetune": "ULTRA-FT", "popularity": "Degree"})
# One hue per method, threaded through every figure; Okabe-Ito based, CVD-checked.
COLOR = {"popularity": "#111111", "l3": "#E69F00", "cn": "#F0C75E", "ra": "#C9A227",
         "transe": "#56B4E9", "distmult": "#0072B2", "complex": "#7E6BB3", "rotate": "#009E73",
         "tucker": "#D55E00", "rgcn": "#CC79A7", "ultra-zeroshot": "#8C8C2E", "ultra-finetune": "#882255",
         "random": "#BBBBBB",
         "KATZHMDA": "#56B4E9", "LRLSHMDA": "#0072B2", "BiRWHMDA": "#009E73", "Degree": "#111111",
         "ncn": "#44AA99", "gbm": "#332288", "nbfnet": "#661100", "KGNMDA": "#CC79A7",
         "rotate-rs": "#009E73", "tucker-rs": "#D55E00", "rgcn-rs": "#CC79A7"}
MARKER = {"popularity": "D", "l3": "^", "cn": "^", "ra": "^", "rgcn": "s", "ultra-zeroshot": "P",
          "ultra-finetune": "P", "ncn": "s", "nbfnet": "s", "gbm": "X", "rotate-rs": "v", "tucker-rs": "v",
          "rgcn-rs": "v", "KGNMDA": "v"}
GREY = "#8A8A8A"
LIGHT = "#E6E6E6"

REGIME = {"t1_pair": "Pair", "t1_study_reveal": "Study+R", "t1_study": "Study", "t1_study_strict": "Strict",
          "t1_temporal": "Temporal", "t1_external": "Out-of-time", "t1_cold_microbe": "Cold microbe",
          "t1_cold_disease": "Cold disease", "t1_temporal_nodis": "Temporal (no Disbiome)",
          "t1_study_ctx_open": "Study + context", "t1_study_ctx_full": "Study + full context"}
# v3 names: Study is renamed Study-exclusive; Strict is the main transfer regime
REGIME3 = dict(REGIME, **{"t1_study": "Study-excl.", "t1_study_strict": "Strict"})


def style():
    apply_figure_style(frame="open", font=FONT, sizes=SIZES)
    import matplotlib as mpl
    mpl.rcParams.update({"axes.titlepad": 4, "legend.handlelength": 1.0, "legend.handletextpad": 0.4,
                         "legend.borderaxespad": 0.2, "legend.columnspacing": 0.9,
                         "axes.labelpad": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2})


def bbox_check(fig, ignore_rotated=True):
    """figure-style 9.1: overlapping visible text boxes and text outside the figure."""
    import matplotlib as mpl
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    bad = []
    for i, (a, ba) in enumerate(texts):
        if ba.x0 < fb.x0 - 1 or ba.x1 > fb.x1 + 1 or ba.y0 < fb.y0 - 1 or ba.y1 > fb.y1 + 1:
            bad.append(("outside", a.get_text()[:30]))
        for b, bb in texts[i + 1:]:
            if ignore_rotated and (a.get_rotation() % 360 not in (0, 90) or b.get_rotation() % 360 not in (0, 90)):
                continue
            if ba.overlaps(bb):
                bad.append((a.get_text()[:30], b.get_text()[:30]))
    return bad


def crops(fig, png, stem):
    """figure-style 9.2: save per-panel crops for visual review."""
    from PIL import Image
    img = Image.open(png)
    out = []
    for letter, box in panel_crops(fig).items():
        p = f"{stem}_crop_{letter}.png"
        img.crop(box).save(p)
        out.append(p)
    return out


def marker_collisions(fig, pad_pt=1.0):
    """Visible text or legend boxes that cover the centre of a scatter marker or a plotted point.

    bbox_check() only compares text with text; this also catches labels printed on top of data."""
    import matplotlib as mpl
    import numpy as np
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi_scale = fig.dpi / 72.0
    pts = []   # (x_px, y_px, radius_px, label)
    for ax in fig.axes:
        for c in ax.collections:
            if not isinstance(c, mpl.collections.PathCollection) or not c.get_visible():
                continue
            off = c.get_offsets()
            if len(off) == 0:
                continue
            xy = c.get_offset_transform().transform(np.asarray(off))
            sizes = c.get_sizes()
            rad = (np.sqrt(sizes[0] if len(sizes) else 36.0) / 2.0 + pad_pt) * dpi_scale
            pts += [(x, y, rad, "scatter") for x, y in xy]
        for ln in ax.lines:
            if ln.get_visible() and ln.get_marker() not in (None, "", "None") and ln.get_markersize() > 0:
                xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float))
                rad = (ln.get_markersize() / 2.0 + pad_pt) * dpi_scale
                pts += [(x, y, rad, "line-marker") for x, y in xy]
    boxes = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_text().strip() and t.get_visible() and t.get_clip_on() is False or (
                t.get_text().strip() and t.get_visible() and not isinstance(t.axes, type(None))):
            boxes.append((t.get_text()[:28], t.get_window_extent(r)))
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            boxes.append(("legend", lg.get_window_extent(r)))
    own = {id(t) for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()}
    bad = []
    for name, bb in boxes:
        for x, y, rad, kind in pts:
            if bb.x0 - rad < x < bb.x1 + rad and bb.y0 - rad < y < bb.y1 + rad:
                bad.append((name, kind, round(x), round(y)))
                break
    return bad
