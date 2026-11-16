"""Shop classification — P1. Separates real shop cells from plan furniture
(title block, legend box, dimension notes, hairline slivers, giant voids).
Classifies only; nothing is deleted — every room is still emitted, non-shops
just carry kind="other" so downstream (walkway building, routable metric,
admin UI) can ignore them.

Two signals, both measured across the reference plans (2026-11-16):
  1. size band: a shop is 0.06%..3% of the image. Below = dimension/column
     noise (Palassio: 195 of 243 cells), above = voids/atrium frames.
  2. isolation: shops sit wall-to-wall with other shops; plan furniture sits
     alone. Nearest-neighbour (bbox edge-to-edge) distance, as % of the image
     diagonal: Palassio's title/legend box = 18.2%, worst real shop = 3.5%,
     WTC worst = 1.7%. Cells beyond ISOLATION_FRAC of the diagonal are junk.
"""
import cv2
import numpy as np

SHOP_MIN_FRAC = 0.0006
SHOP_MAX_FRAC = 0.03
ISOLATION_FRAC = 0.05


def _rect_dist(r1, r2):
    dx = max(r1[0] - (r2[0] + r2[2]), r2[0] - (r1[0] + r1[2]), 0)
    dy = max(r1[1] - (r2[1] + r2[3]), r2[1] - (r1[1] + r1[3]), 0)
    return (dx * dx + dy * dy) ** 0.5


def shop_cids(room_polys: dict, shape_hw) -> set:
    """Return the cids classified as real shops."""
    H, W = shape_hw
    A = float(H * W)
    diag = (W * W + H * H) ** 0.5

    band = []
    for cid, poly in room_polys.items():
        a = cv2.contourArea(poly.astype(np.float32))
        if SHOP_MIN_FRAC * A <= a <= SHOP_MAX_FRAC * A:
            band.append((cid, cv2.boundingRect(poly.astype(np.int32))))
    if len(band) <= 1:
        return {cid for cid, _ in band}

    keep = set()
    for i, (cid, r) in enumerate(band):
        nn = min(_rect_dist(r, r2) for j, (_, r2) in enumerate(band) if j != i)
        if nn <= ISOLATION_FRAC * diag:
            keep.add(cid)
    return keep
