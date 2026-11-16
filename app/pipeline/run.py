"""Orchestrator: ingest -> segment (auto mode) -> read text -> bind ->
graph -> Atlas-shaped output + verification report. One call, no questions."""
import numpy as np

from .ingest import ingest
from . import labels as L
from . import segment as S
from .graphbuild import build_graph
from .adapter import to_atlas


def _bind(seg, dets, W, H, radius=30):
    per_room, corridor_labels, junk = {}, [], []
    rset = set(seg.room_ids)
    for d in dets:
        xi, yi = int(d["x"]), int(d["y"])
        if not (0 <= xi < W and 0 <= yi < H):
            junk.append(d); continue
        cid = seg.comp[yi, xi]
        if cid not in rset:
            y0, y1 = max(0, yi - radius), min(H, yi + radius)
            x0, x1 = max(0, xi - radius), min(W, xi + radius)
            cand = [c for c in np.unique(seg.comp[y0:y1, x0:x1]) if c in rset]
            if cand:
                cid = min(cand, key=lambda c: (seg.cents[c][0] - xi) ** 2 + (seg.cents[c][1] - yi) ** 2)
            elif seg.corridor is not None and seg.corridor[yi, xi]:
                corridor_labels.append(d); continue
            else:
                junk.append(d); continue
        per_room.setdefault(int(cid), []).append(d)
    return per_room, corridor_labels, junk


def detect_labels(ing):
    """Vector labels first (free + exact); OCR only when the file has none."""
    dets = []
    if ing.source_kind == "svg" and ing.svg_root is not None:
        dets = [dict(d, conf=1.0, rot=None) for d in L.svg_labels(ing.svg_root, ing.scale)]
    elif ing.pdf_words:
        dets = [dict(d, conf=1.0, rot=None) for d in L.pdf_labels(ing.pdf_words, ing.scale)]
    if dets:
        return dets, False
    return L.ocr_labels(ing.raster), True


def select_segmentation(ing, dets):
    """The mode contest — the ONE place a segmentation is chosen (extract and
    the routable harness both call this; never duplicate it).

    Mode is chosen by evidence, not heuristics: run the segmenters, bind the
    labels with the SAME radius binding used downstream, and keep whichever
    yields more distinct labeled rooms (raw hit counts reward giant blobs —
    Phoenix; raw centre hits punish colour rooms with text holes — FF).
    Label-score ties (stroke vs shopgap share rooms) are broken by hallway
    quality: the routable-shop count — the judge's second question.
    Returns (seg, (per_room, corridor_labels, junk))."""
    from .routable import routable_stats
    H, W = ing.raster.shape[:2]
    label_pts = [(int(d["x"]), int(d["y"])) for d in dets]

    def _score(sg):
        bound = _bind(sg, dets, W, H)
        # A detected legend whose colours match many rooms is decisive evidence
        # of the colour convention (FF). Absent a legend this term is 0 and the
        # labeled-room count decides (Phoenix stroke vs colour).
        categorized = sum(1 for cid in sg.room_polys if S.category_index(sg, cid) is not None)
        labels_score = (len(bound[0]) + categorized) * 1000 + len(sg.room_polys)
        return (labels_score, routable_stats(sg, (H, W))["ok"]), bound

    candidates = []
    stroke_sg = None
    for builder in (lambda: S.segment_stroke(ing.raster_notext, label_pts),
                    lambda: S.segment_color(ing.raster_notext)):
        try:
            sg = builder()
        except Exception:
            continue
        if sg.mode == "stroke" and sg.room_polys:
            stroke_sg = sg
        if sg.corridor is None or not sg.room_polys:
            continue
        candidates.append((*_score(sg), sg))
    if not candidates:
        raise RuntimeError("no corridor region found")
    (win_labels, win_ok), bound, seg = max(candidates, key=lambda t: t[0])
    # Reader 3 — shop-gap walkway derived from the stroke cells (atrium malls
    # where the enclosed-void corridor pick fails). A working incumbent is
    # only displaced on CLEAR evidence: at least as good on labels AND a
    # routable-shop win by a real margin, never a marginal tie-break.
    if stroke_sg is not None:
        try:
            sg3 = S.segment_shopgap(stroke_sg, (H, W))
        except Exception:
            sg3 = None
        if sg3 is not None and sg3.corridor is not None:
            (l3, ok3), b3 = _score(sg3)
            from .classify import shop_cids
            margin = max(2, len(shop_cids(sg3.room_polys, (H, W))) // 10)
            if l3 >= win_labels and ok3 >= win_ok + margin:
                seg, bound = sg3, b3
    return seg, bound


def extract(path: str) -> dict:
    ing = ingest(path)
    H, W = ing.raster.shape[:2]

    dets, used_ocr = detect_labels(ing)
    seg, (per_room, corridor_labels, junk) = select_segmentation(ing, dets)
    mode = seg.mode

    names, categories = {}, {}
    for cid in seg.room_polys:
        room_dets = per_room.get(cid, [])
        if room_dets:
            names[cid] = L.join_room_detections(room_dets) if used_ocr \
                else max((d["text"] for d in room_dets), key=len)
        if seg.mode == "color":
            k = S.category_index(seg, cid)
            if k is not None:
                categories[cid] = k

    G, entries = build_graph(seg.corridor, seg.room_polys)

    from .confidence import build_verification
    # No scale-bar detection yet — always flagged for the admin to set once.
    room_conf, flags, verification = build_verification(
        seg, names, G, entries, ing.scale, has_scale_bar=False)

    import networkx as nx
    report = {
        "input": path.split("/")[-1],
        "mode": seg.mode,
        "calibration": seg.calibration,
        "labels_total": len(dets),
        "labels_source": "ocr" if used_ocr else "vector",
        "rooms": len(seg.room_polys),
        "rooms_named": len(names),
        "rooms_categorized": len(categories),
        "labels_in_corridor": len(corridor_labels),
        "labels_junk": len(junk),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "graph_components": nx.number_connected_components(G),
        "legend_categories": len(seg.legend),
        "verification": verification,
    }
    atlas = to_atlas(seg, names, categories, G, entries, corridor_labels,
                     ing.scale, (W, H), room_conf)
    return {"report": report, "flags": flags, "atlas": atlas}
