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


def extract(path: str) -> dict:
    ing = ingest(path)
    H, W = ing.raster.shape[:2]

    # vector labels first (free + exact); OCR only when the file has none
    dets = []
    if ing.source_kind == "svg" and ing.svg_root is not None:
        dets = [dict(d, conf=1.0, rot=None) for d in L.svg_labels(ing.svg_root, ing.scale)]
    elif ing.pdf_words:
        dets = [dict(d, conf=1.0, rot=None) for d in L.pdf_labels(ing.pdf_words, ing.scale)]
    used_ocr = False
    if not dets:
        dets = L.ocr_labels(ing.raster)
        used_ocr = True

    label_pts = [(int(d["x"]), int(d["y"])) for d in dets]

    # Mode is chosen by evidence, not heuristics: run both segmenters, bind the
    # labels with the SAME radius binding used downstream, and keep whichever
    # yields more distinct labeled rooms (raw hit counts reward giant blobs —
    # Phoenix; raw centre hits punish colour rooms with text holes — FF).
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
        # A detected legend whose colours match many rooms is decisive evidence
        # of the colour convention (FF). Absent a legend this term is 0 and the
        # labeled-room count decides (Phoenix stroke vs colour).
        categorized = sum(1 for cid in sg.room_polys if S.category_index(sg, cid) is not None)
        score = (len(bound[0]) + categorized) * 1000 + len(sg.room_polys)
        candidates.append((score, sg, bound))
    if not candidates:
        raise RuntimeError("no corridor region found")
    _, seg, (per_room, corridor_labels, junk) = max(candidates, key=lambda t: t[0])
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
