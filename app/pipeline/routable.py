"""Routable-% measurement — the P0 regression metric. Read-only: measures the
pipeline's current output, changes no behavior.

The honest question for navigation: how many shops can actually REACH the
hallway network? A shop counts as routable iff
  1. its BOUNDARY (the shop's wall, sampled along the polygon) comes within
     NEAR px of the MAIN (largest 8-connected) component of the corridor
     MASK, after the same preprocessing graphbuild applies (close 7, erode 3).
     Boundary-to-mask, not centroid-to-skeleton: a shop's door opens onto the
     hallway EDGE — a wide hallway's centerline is legitimately far from the
     wall, and a big room's centre is far from its own door; and
  2. the straight link from that boundary point to the nearest mask point
     does not cut through another shop (a route through a wall is no route).
Skeleton piece count / main share are still reported (fragmentation signal),
but adjacency is judged against the connected mask.

"Shops" come from classify.shop_cids (size band + isolation cut) so the
metric and the pipeline agree on what a shop is.
NEAR = max(25 px, 0.7% of image width): 25 px is the threshold the
2026-11-05 cross-plan adj% study used at 3600 px width (WTC 94 / FF 100 /
Palassio-SVG 31 / Palassio-PDF 7), kept so baselines stay comparable.
"""
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import labels as L
from . import segment as S
from .classify import shop_cids
from .ingest import ingest
from .run import _bind

NEAR_FRAC = 25.0 / 3600.0
NEAR_MIN_PX = 25.0
BOUNDARY_STEP_PX = 10.0


def winning_segmentation(path: str):
    """Re-run the pipeline's own mode contest (mirrors run.extract's candidate
    loop) and return the winning Segmentation. Duplicated here so measurement
    needs no change to the pipeline itself; keep in sync with run.extract."""
    ing = ingest(path)
    H, W = ing.raster.shape[:2]
    dets = []
    if ing.source_kind == "svg" and ing.svg_root is not None:
        dets = [dict(d, conf=1.0, rot=None) for d in L.svg_labels(ing.svg_root, ing.scale)]
    elif ing.pdf_words:
        dets = [dict(d, conf=1.0, rot=None) for d in L.pdf_labels(ing.pdf_words, ing.scale)]
    if not dets:
        dets = L.ocr_labels(ing.raster)
    label_pts = [(int(d["x"]), int(d["y"])) for d in dets]

    candidates = []
    for builder in (lambda: S.segment_stroke(ing.raster_notext, label_pts),
                    lambda: S.segment_color(ing.raster_notext)):
        try:
            sg = builder()
        except Exception:
            continue
        if sg.corridor is None or not sg.room_polys:
            continue
        bound = _bind(sg, dets, W, H)
        categorized = sum(1 for cid in sg.room_polys if S.category_index(sg, cid) is not None)
        score = (len(bound[0]) + categorized) * 1000 + len(sg.room_polys)
        candidates.append((score, sg))
    if not candidates:
        raise RuntimeError("no corridor region found")
    return max(candidates, key=lambda t: t[0])[1]


def corridor_main(corridor: np.ndarray):
    """Preprocess the corridor exactly as graphbuild does (close 7, erode 3),
    keep the MAIN (largest 8-connected) mask component, and also skeletonize
    for the fragmentation stats. Returns (main-mask points Nx2 [x,y] strided
    for tractability, skeleton piece count, main skeleton piece share)."""
    cor = cv2.morphologyEx(corridor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cor = cv2.erode(cor, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cor, connectivity=8)
    if n <= 1:
        return np.empty((0, 2), np.float32), 0, 0.0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(lab == main)
    pts = np.column_stack([xs, ys]).astype(np.float32)[::3]   # stride: <=~2px error vs 25px NEAR

    skel = skeletonize(cor > 0).astype(np.uint8)
    sn, _, sstats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    if sn <= 1:
        return pts, 0, 0.0
    sareas = sstats[1:, cv2.CC_STAT_AREA]
    return pts, sn - 1, float(sareas.max() / sareas.sum())


def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)


def _seg_x(ax, ay, bx, by, cx, cy, dx, dy):
    d1 = _ccw(cx, cy, dx, dy, ax, ay)
    d2 = _ccw(cx, cy, dx, dy, bx, by)
    d3 = _ccw(ax, ay, bx, by, cx, cy)
    d4 = _ccw(ax, ay, bx, by, dx, dy)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _crosses(ax, ay, bx, by, poly):
    n = len(poly)
    for i in range(n):
        c, d = poly[i], poly[(i + 1) % n]
        if _seg_x(ax, ay, bx, by, float(c[0]), float(c[1]), float(d[0]), float(d[1])):
            return True
    return False


def _boundary_points(poly, step=BOUNDARY_STEP_PX):
    """Sample the polygon perimeter every `step` px (vertices always kept)."""
    pts = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i].astype(np.float64), poly[(i + 1) % n].astype(np.float64)
        pts.append(a)
        L_ = math.dist(a, b)
        for t in range(1, int(L_ // step) + 1):
            pts.append(a + (b - a) * (t * step / L_))
    return np.array(pts, np.float32)


def routable_stats(seg, shape_hw) -> dict:
    """The metric. seg = Segmentation (corridor + room_polys); shape_hw = (H, W)."""
    H, W = shape_hw
    near = max(NEAR_MIN_PX, NEAR_FRAC * W)

    keep = shop_cids(seg.room_polys, shape_hw)
    shops = [{"cid": cid, "poly": seg.room_polys[cid],
              "bnd": _boundary_points(seg.room_polys[cid])} for cid in sorted(keep)]

    out = {"mode": seg.mode, "shops": len(shops), "ok": 0, "far": 0, "cross": 0,
           "pct": 0.0, "pieces": 0, "main_frac": 0.0}
    if seg.corridor is None or not shops:
        return out

    pts, pieces, main_frac = corridor_main(seg.corridor)
    out["pieces"], out["main_frac"] = pieces, round(main_frac, 3)
    if not len(pts):
        out["far"] = len(shops)
        return out

    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    for s in shops:
        dists, idxs = tree.query(s["bnd"])
        j = int(np.argmin(dists))
        dist = float(dists[j])
        bx, by = float(s["bnd"][j, 0]), float(s["bnd"][j, 1])
        nx, ny = float(pts[idxs[j], 0]), float(pts[idxs[j], 1])
        crossed = any(_crosses(bx, by, nx, ny, o["poly"]) for o in shops if o is not s)
        if dist <= near and not crossed:
            out["ok"] += 1
        elif dist > near:
            out["far"] += 1
        else:
            out["cross"] += 1
    out["pct"] = round(100.0 * out["ok"] / len(shops), 1)
    return out


def measure(path: str) -> dict:
    """One call: contest-winning segmentation -> routable stats."""
    seg = winning_segmentation(path)
    return routable_stats(seg, seg.comp.shape)
