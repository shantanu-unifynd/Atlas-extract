"""Output adapter — emit the exact shapes the Atlas client renders
(rooms / labels / graph with positioned nodes and distance-weighted edges),
plus corridor kiosks as point rooms. Coordinates are returned in source
units so they overlay the original plan 1:1."""
import itertools

from .classify import shop_cids


def to_atlas(seg, names, categories, G, entries, corridor_labels, scale, size,
             room_conf=None):
    room_conf = room_conf or {}
    inv = 1.0 / scale
    shops = shop_cids(seg.room_polys, seg.comp.shape)
    rooms, labels = [], []
    for cid, poly in seg.room_polys.items():
        rid = f"room-{cid}"
        rooms.append({
            "id": rid,
            "name": names.get(cid),
            "category": f"legend-{categories[cid]}" if cid in categories else None,
            "kind": "shop" if cid in shops else "other",
            "confidence": room_conf.get(cid, 1.0),
            "polygon": [{"x": round(float(x) * inv, 2), "y": round(float(y) * inv, 2)}
                        for x, y in poly],
        })
        if cid in names:
            # Non-shop rooms carry no routing entry; anchor their label at the
            # polygon centre instead.
            cx, cy = entries[cid]["pos"] if cid in entries else poly.mean(axis=0)
            labels.append({"id": f"label-{cid}", "text": names[cid], "roomId": rid,
                           "position": {"x": round(cx * inv, 2), "y": round(cy * inv, 2)}})
    for i, d in enumerate(corridor_labels):
        rooms.append({"id": f"kiosk-{i}", "name": d["text"], "category": "KIOSK",
                      "polygon": None,
                      "point": {"x": round(d["x"] * inv, 2), "y": round(d["y"] * inv, 2)}})

    node_id = {}
    counter = itertools.count(1)
    nodes, edges = [], []
    for p in G.nodes:
        nid = f"n{next(counter)}"
        node_id[p] = nid
        if isinstance(p, tuple) and p[0] == "entry":
            cid = p[1]
            x, y = entries[cid]["pos"]
            nodes.append({"id": nid, "nodeType": "ROOM_ENTRY",
                          "label": names.get(cid),
                          "position": {"x": round(x * inv, 2), "y": round(y * inv, 2)}})
        else:
            nodes.append({"id": nid, "nodeType": "CORRIDOR_INTERSECTION", "label": None,
                          "position": {"x": round(p[0] * inv, 2), "y": round(p[1] * inv, 2)}})
    for a, b, d in G.edges(data=True):
        edges.append({"id": f"e{len(edges)+1}", "sourceNodeId": node_id[a],
                      "targetNodeId": node_id[b],
                      "traversalCost": round(d["weight"] * inv, 2),
                      "edgeType": "WALKWAY"})

    W, H = size
    return {
        "coordinateSystem": {"bounds": {"minX": 0, "minY": 0,
                                        "maxX": round(W * inv, 2), "maxY": round(H * inv, 2)}},
        "rooms": rooms,
        "labels": labels,
        "graph": {"nodes": nodes, "edges": edges},
    }
