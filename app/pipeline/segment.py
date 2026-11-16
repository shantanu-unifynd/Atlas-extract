"""Stage 2+4 — understand the drawing and polygonize rooms.

Two segmentation modes, chosen automatically per plan:
- "stroke": walls are drawn lines; rooms are enclosed empty cells
  (WTC/Phoenix class). Per-plan auto-calibration sweeps ink threshold x
  gap-closing, scored by how many labels land inside rooms (difficulty #3).
- "color": shops are solid colour blocks and the corridor is an enclosed
  dark ribbon (directory-map class, difficulty #10). The legend panel is
  detected, excluded from rooms, and read as a colour->category key (#11).

Corridor is always the walkable region adjacent to the most rooms, never
simply the largest (difficulty #4).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Segmentation:
    mode: str
    comp: np.ndarray                      # component id per pixel (rooms)
    room_ids: list[int] = field(default_factory=list)
    room_polys: dict = field(default_factory=dict)   # id -> Nx2 px polygon
    room_color: dict = field(default_factory=dict)   # id -> mean BGR (color mode)
    corridor: np.ndarray | None = None               # bool mask
    stats: np.ndarray | None = None
    cents: np.ndarray | None = None
    legend: list = field(default_factory=list)       # [(y, BGR)] rows, top->bottom
    calibration: dict = field(default_factory=dict)


def choose_mode(raster_notext: np.ndarray) -> str:
    hsv = cv2.cvtColor(raster_notext, cv2.COLOR_BGR2HSV)
    colored = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 70)).mean()
    return "color" if colored > 0.12 else "stroke"


def _polys(comp, room_ids, eps_px):
    out = {}
    for cid in room_ids:
        mask = (comp == cid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        out[cid] = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
    return out


def _corridor_by_adjacency(walk, comp, room_ids, room_mask):
    wn, wcomp, wstats, _ = cv2.connectedComponentsWithStats(walk.astype(np.uint8), connectivity=4)
    if wn <= 1:
        return None
    room_dil = cv2.dilate(room_mask.astype(np.uint8), np.ones((17, 17), np.uint8))
    cand = np.argsort(wstats[1:, cv2.CC_STAT_AREA])[::-1][:6] + 1
    best, best_adj = None, -1
    rset = set(room_ids)
    for ci in cand:
        md = cv2.dilate((wcomp == ci).astype(np.uint8), np.ones((17, 17), np.uint8))
        touched = set(np.unique(comp[(md > 0) & (room_dil > 0)])) & rset
        if len(touched) > best_adj:
            best, best_adj = ci, len(touched)
    return wcomp == best


def segment_stroke(raster_notext: np.ndarray, label_pts: list[tuple[int, int]]) -> Segmentation:
    gray = cv2.cvtColor(raster_notext, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    min_a, max_a = int(H * W * 0.00005), int(H * W * 0.05)

    best = None
    for thr in (120, 150, 190, 230):
        for k in (3, 9, 15):
            ink = (gray < thr).astype(np.uint8)
            closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) if k > 3 \
                else cv2.dilate(ink, np.ones((3, 3), np.uint8))
            free = (closed == 0).astype(np.uint8)
            n, comp, stats, cents = cv2.connectedComponentsWithStats(free, connectivity=4)
            border = set(comp[0, :]) | set(comp[-1, :]) | set(comp[:, 0]) | set(comp[:, -1])
            rids = [i for i in range(1, n) if i not in border and min_a <= stats[i, 4] <= max_a]
            rset = set(rids)
            hits = sum(1 for x, y in label_pts if 0 <= x < W and 0 <= y < H and comp[y, x] in rset)
            score = hits * 1000 + len(rids)
            if best is None or score > best[0]:
                best = (score, thr, k, comp, stats, cents, rids, closed)

    _, thr, k, comp, stats, cents, rids, closed = best
    seg = Segmentation("stroke", comp, rids, stats=stats, cents=cents,
                       calibration={"ink_threshold": thr, "close_kernel": k})
    seg.room_polys = _polys(comp, rids, eps_px=6)
    room_mask = np.isin(comp, rids)
    walk = (closed == 0) & (~room_mask)
    walk = cv2.morphologyEx(walk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    seg.corridor = _corridor_by_adjacency(walk, comp, rids, room_mask)
    return seg


def segment_color(raster_notext: np.ndarray) -> Segmentation:
    img = raster_notext
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    leg_x, leg_y = int(W * 0.17), int(H * 0.47)

    colored = ((S > 70) & (V > 70)).astype(np.uint8)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, comp, stats, cents = cv2.connectedComponentsWithStats(colored, connectivity=4)
    min_a, max_a = int(H * W * 0.00008), int(H * W * 0.03)
    rids = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (min_a <= a <= max_a):
            continue
        if x + w < leg_x and y + h < leg_y:   # legend chips (difficulty #11)
            continue
        rids.append(i)

    seg = Segmentation("color", comp, rids, stats=stats, cents=cents)
    seg.room_polys = _polys(comp, rids, eps_px=6)
    for cid in rids:
        seg.room_color[cid] = img[comp == cid].mean(axis=0)

    dark = ((V < 110) | ((S < 50) & (V < 140))).astype(np.uint8)
    dn, dcomp, dstats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    border = set(dcomp[0, :]) | set(dcomp[-1, :]) | set(dcomp[:, 0]) | set(dcomp[:, -1])
    cands = [i for i in range(1, dn) if i not in border and dstats[i, 4] > H * W * 0.01]
    if cands:
        seg.corridor = dcomp == max(cands, key=lambda i: dstats[i, 4])

    leg = img[0:leg_y, 0:leg_x]
    lh = cv2.cvtColor(leg, cv2.COLOR_BGR2HSV)
    lmask = ((lh[:, :, 1] > 70) & (lh[:, :, 2] > 70)).astype(np.uint8)
    ln, lcomp, lstats, _ = cv2.connectedComponentsWithStats(lmask, connectivity=4)
    for i in range(1, ln):
        x, y, w, h, a = lstats[i]
        if a < leg.size * 0.0004 or h > leg.shape[0] * 0.08:
            continue
        seg.legend.append((int(y), leg[lcomp == i].mean(axis=0)))
    seg.legend.sort(key=lambda r: r[0])
    return seg


def _bridge_components(mask: np.ndarray, max_gap: int, width: int = 7, rounds: int = 6):
    """Join disconnected mask pieces: repeatedly draw a line from each minor
    component to the nearest main-component point when the gap is small
    (<= max_gap px). Distant pieces stay separate — no fantasy hallways."""
    from scipy.spatial import cKDTree
    out = mask.copy()
    for _ in range(rounds):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n <= 2:
            break
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mys, mxs = np.where(lab == main)
        tree = cKDTree(np.column_stack([mxs, mys])[::5])
        bridged = False
        for ci in range(1, n):
            if ci == main:
                continue
            ys, xs = np.where(lab == ci)
            pts = np.column_stack([xs, ys])[::5]
            d, j = tree.query(pts)
            k = int(np.argmin(d))
            if d[k] > max_gap:
                continue
            mp = tree.data[j[k]].astype(int)
            cv2.line(out, (int(pts[k][0]), int(pts[k][1])), (int(mp[0]), int(mp[1])),
                     1, width)
            bridged = True
        if not bridged:
            break
    return out


def segment_shopgap(stroke_seg: Segmentation, shape_hw) -> Segmentation | None:
    """Reader 3 — the shop-gap walkway (atrium-mall class). The stroke
    reader's CELLS are reliable even when its corridor pick is not: the empty
    gaps BETWEEN classified shops ARE the hallways. Walkway = free space
    within a tight footprint around the shops, minus the shops, with nearby
    pieces auto-bridged. Rooms are reused from the stroke segmentation; only
    the corridor differs. Returns None when too few shops to carve gaps."""
    from .classify import shop_cids
    H, W = shape_hw
    shops = shop_cids(stroke_seg.room_polys, (H, W))
    if len(shops) < 8:
        return None
    shopmask = np.zeros((H, W), np.uint8)
    for cid in shops:
        cv2.fillPoly(shopmask, [stroke_seg.room_polys[cid].astype(np.int32)], 1)
    foot_k = max(9, int(0.05 * W) | 1)
    foot = cv2.dilate(shopmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (foot_k, foot_k)))
    free = ((foot > 0) & (cv2.dilate(shopmask, np.ones((5, 5), np.uint8)) == 0)).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    free = _bridge_components(free, max_gap=int(0.015 * W))
    if not free.any():
        return None
    seg = Segmentation("shopgap", stroke_seg.comp, list(stroke_seg.room_ids),
                       stats=stroke_seg.stats, cents=stroke_seg.cents,
                       calibration=dict(stroke_seg.calibration))
    seg.room_polys = stroke_seg.room_polys
    seg.corridor = free > 0
    return seg


def category_index(seg: Segmentation, cid: int, max_dist: float = 60.0):
    if not seg.legend or cid not in seg.room_color:
        return None
    d, idx = min((float(np.linalg.norm(seg.room_color[cid] - c)), k)
                 for k, (_, c) in enumerate(seg.legend))
    return idx if d < max_dist else None
