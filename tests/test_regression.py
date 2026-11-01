"""Regression bars from the P0 evidence runs — the three proven plans are the
fixtures. WTC must stay a full pass; FF holds its color-mode + OCR bars."""
import pathlib

import pytest

from app.pipeline.run import extract

FIX = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def wtc():
    return extract(str(FIX / "wtc-mall.svg"))


def test_wtc_rooms_and_names(wtc):
    r = wtc["report"]
    assert r["mode"] == "stroke"
    assert r["labels_source"] == "vector"
    assert r["rooms"] >= 80
    assert r["rooms_named"] >= 55
    assert r["labels_junk"] <= 30


def test_wtc_graph_connected(wtc):
    r = wtc["report"]
    assert r["graph_components"] == 1
    assert r["graph_nodes"] >= 100


def test_wtc_atlas_shapes(wtc):
    atlas = wtc["atlas"]
    assert atlas["coordinateSystem"]["bounds"]["maxX"] > 0
    assert all(("polygon" in room) or room.get("point") for room in atlas["rooms"])
    node_ids = {n["id"] for n in atlas["graph"]["nodes"]}
    assert all(n["position"].keys() == {"x", "y"} for n in atlas["graph"]["nodes"])
    assert all(e["sourceNodeId"] in node_ids and e["targetNodeId"] in node_ids
               for e in atlas["graph"]["edges"])
    assert all(e["traversalCost"] > 0 for e in atlas["graph"]["edges"])


@pytest.mark.slow
def test_ff_color_mode_and_ocr():
    r = extract(str(FIX / "FF.pdf"))["report"]
    assert r["mode"] == "color"
    assert r["labels_source"] == "ocr"
    assert r["rooms"] >= 70
    assert r["rooms_named"] >= 60
    assert r["rooms_categorized"] >= 70
    assert r["graph_components"] == 1
