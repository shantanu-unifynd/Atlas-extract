"""P0 routable-% regression bars (measured 2026-11-16, mask-adjacency metric).

Baselines at lock time: WTC 98.6%, FF 100%, Palassio-SVG-0001 26.9%,
0002 31.5%, Palassio-PDF 32.6%. Floors sit just under the measured values:
a drop below a floor on a previously-good plan is a regression; rises are
improvements and never fail.
"""
import pathlib

import numpy as np
import pytest

from app.pipeline.routable import _crosses, _seg_x, measure

FIX = pathlib.Path(__file__).parent / "fixtures"
ASSETS = pathlib.Path("/Users/unifynd-pro/Desktop/Atlas-Platform/assets")


def test_seg_x_geometry():
    assert _seg_x(0, 0, 10, 10, 0, 10, 10, 0)          # crossing diagonals
    assert not _seg_x(0, 0, 10, 0, 0, 5, 10, 5)        # parallel
    assert not _seg_x(0, 0, 1, 1, 5, 5, 6, 6)          # collinear, disjoint


def test_crosses_polygon():
    square = np.array([[2, 2], [6, 2], [6, 6], [2, 6]])
    assert _crosses(0, 4, 8, 4, square)                # straight through
    assert not _crosses(0, 8, 8, 8, square)            # passes below


def test_wtc_routable_floor():
    r = measure(str(FIX / "wtc-mall.svg"))
    assert r["pct"] >= 95.0, f"WTC routable dropped: {r}"
    assert r["pieces"] == 1


@pytest.mark.slow
def test_ff_routable_floor():
    r = measure(str(FIX / "FF.pdf"))
    assert r["pct"] >= 95.0, f"FF routable dropped: {r}"


@pytest.mark.slow
@pytest.mark.skipif(not (ASSETS / "0001.svg").exists(), reason="asset not present")
def test_palassio_svg_baselines():
    # P2 shop-gap reader gains locked in (measured 2026-11-16: 80.8 / 86.3,
    # both mode=shopgap). Floors sit just under the measured values.
    assert measure(str(ASSETS / "0001.svg"))["pct"] >= 75.0
    assert measure(str(ASSETS / "0002.svg"))["pct"] >= 80.0


PALASSIO_PDF = ASSETS / "PHOENIX PALASSIO TRADING FLOOR PLAN - 23-01-26.pdf"


@pytest.mark.slow
@pytest.mark.skipif(not PALASSIO_PDF.exists(), reason="asset not present")
def test_palassio_pdf_baseline():
    # P2 measured 69.0 (mode=shopgap, 2 pieces main 92%).
    r = measure(str(PALASSIO_PDF))
    assert r["pct"] >= 60.0, f"Palassio-PDF routable dropped below baseline: {r}"
