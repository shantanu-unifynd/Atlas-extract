"""Confidence + flags — the verification layer. Every room and the graph get a
0..1 confidence from cheap geometric/semantic signals, and anything below bar
becomes a flagged item the admin reviews. The whole point of the service: the
admin verifies the flags, never authors from scratch.
"""
import cv2
import numpy as np


def room_confidence(poly, named: bool) -> tuple[float, list[str]]:
    reasons = []
    c = 1.0
    verts = len(poly)
    if verts > 12:                       # jagged outline = likely overlapping strokes
        c -= 0.3
        reasons.append("outline is jagged (overlapping strokes?)")
    if verts < 4:
        c -= 0.2
        reasons.append("degenerate outline")
    if not named:
        c -= 0.25
        reasons.append("no name detected")
    rect = cv2.minAreaRect(poly.astype(np.int32))
    (w, h) = rect[1]
    if w > 0 and h > 0:
        ar = max(w, h) / min(w, h)
        if ar > 8:
            c -= 0.3
            reasons.append("extreme aspect ratio")
    return max(0.0, round(c, 2)), reasons


def build_verification(seg, names, G, entries, scale, has_scale_bar):
    """Return (per_room_conf dict, flags list, summary dict)."""
    import networkx as nx

    inv = 1.0 / scale
    per_room, flags = {}, []
    low = 0
    for cid, poly in seg.room_polys.items():
        conf, reasons = room_confidence(poly, cid in names)
        per_room[cid] = conf
        if conf < 0.6:
            low += 1
            cx, cy = entries[cid]["pos"] if cid in entries else poly.mean(axis=0)
            flags.append({
                "kind": "room",
                "id": f"room-{cid}",
                "name": names.get(cid),
                "confidence": conf,
                "reason": "; ".join(reasons) or "low confidence",
                "at": {"x": round(float(cx) * inv, 2), "y": round(float(cy) * inv, 2)},
            })

    components = nx.number_connected_components(G)
    isolated = [cid for cid in seg.room_polys
                if ("entry", cid) in G and G.degree(("entry", cid)) == 0]
    if components > 1:
        flags.append({"kind": "graph", "id": "graph",
                      "reason": f"graph is split into {components} disconnected components",
                      "confidence": 0.4})
    for cid in isolated:
        flags.append({"kind": "room", "id": f"room-{cid}", "name": names.get(cid),
                      "reason": "room not connected to the walkway graph", "confidence": 0.3})
    if not has_scale_bar:
        flags.append({"kind": "scale", "id": "scale",
                      "reason": "no scale bar found — distances are relative units",
                      "confidence": 0.5})

    summary = {
        "rooms_low_confidence": low,
        "rooms_isolated": len(isolated),
        "graph_components": components,
        "flags": len(flags),
        "has_metric_scale": bool(has_scale_bar),
    }
    return per_room, flags, summary
