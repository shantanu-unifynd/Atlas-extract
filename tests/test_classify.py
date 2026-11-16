"""P1 shop-classification tests — synthetic geometry plus the live Palassio
isolation case (title/legend box must classify as non-shop)."""
import pathlib

import numpy as np
import pytest

from app.pipeline.classify import ISOLATION_FRAC, shop_cids

SHAPE = (1000, 1000)   # H, W -> A=1e6, diag~1414


def rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def test_size_band():
    polys = {
        1: rect(100, 100, 50, 50),    # 2500 px = 0.25% -> shop-size
        2: rect(100, 160, 50, 50),    # neighbour, keeps #1 un-isolated
        3: rect(300, 100, 10, 10),    # 100 px = 0.01% -> too small
        4: rect(400, 100, 600, 600),  # 36% -> too big
    }
    assert shop_cids(polys, SHAPE) == {1, 2}


def test_isolated_cell_is_junk():
    # a wall-to-wall row of shops + one same-size box far away (legend/title)
    polys = {i: rect(100 + i * 52, 100, 50, 50) for i in range(5)}
    polys[99] = rect(900, 900, 50, 50)   # ~790 px away > 5% of diag (~71)
    keep = shop_cids(polys, SHAPE)
    assert 99 not in keep
    assert all(i in keep for i in range(5))


def test_near_neighbour_survives():
    polys = {1: rect(100, 100, 50, 50), 2: rect(100, 180, 50, 50)}
    # gap 30 px < ISOLATION_FRAC * diag -> both stay
    assert shop_cids(polys, SHAPE) == {1, 2}


def test_single_band_cell_kept():
    assert shop_cids({1: rect(100, 100, 50, 50)}, SHAPE) == {1}


PALASSIO_PDF = pathlib.Path(
    "/Users/unifynd-pro/Desktop/Atlas-Platform/assets/PHOENIX PALASSIO TRADING FLOOR PLAN - 23-01-26.pdf")


@pytest.mark.slow
@pytest.mark.skipif(not PALASSIO_PDF.exists(), reason="asset not present")
def test_palassio_classification():
    from app.pipeline.routable import winning_segmentation
    seg = winning_segmentation(str(PALASSIO_PDF))
    keep = shop_cids(seg.room_polys, seg.comp.shape)
    # 243 cells -> ~42 shops: the isolated bottom-left title/legend box is out
    assert 38 <= len(keep) <= 43
    assert len(keep) < len(seg.room_polys)
