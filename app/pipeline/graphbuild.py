"""Stage 5 — the graph from empty space. Skeletonize the corridor (medial
axis), contract degree-2 chains into distance-weighted edges, prune spurs,
and attach every room at its nearest corridor point."""
import math

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


def build_graph(corridor: np.ndarray, room_polys: dict, spur_px: float = 24.0):
    cor = cv2.morphologyEx(corridor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cor = cv2.erode(cor, np.ones((3, 3), np.uint8))
    skel = skeletonize(cor > 0)
    ys, xs = np.nonzero(skel)
    px = set(zip(xs.tolist(), ys.tolist()))

    def nbrs(p):
        x, y = p
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx or dy) and (x + dx, y + dy) in px:
                    yield (x + dx, y + dy)

    deg = {p: sum(1 for _ in nbrs(p)) for p in px}
    anchors = [p for p, d in deg.items() if d >= 3 or d == 1]
    G = nx.Graph()
    seen = set()
    for s in anchors:
        for nb in nbrs(s):
            prev, cur, length = s, nb, 1.0
            while deg.get(cur, 0) == 2:
                nxt = [q for q in nbrs(cur) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                length += math.dist(prev, cur)
            ek = tuple(sorted((s, cur)))
            if ek in seen or s == cur:
                continue
            seen.add(ek)
            G.add_edge(s, cur, weight=length)

    for _ in range(4):
        spurs = [p for p in list(G.nodes) if G.degree(p) == 1 and
                 min((d["weight"] for *_, d in G.edges(p, data=True)), default=1e9) < spur_px]
        if not spurs:
            break
        G.remove_nodes_from(spurs)
    if G.number_of_nodes():
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    skn = list(G.nodes)
    cor_pts = np.column_stack(np.nonzero(corridor))
    entries = {}
    for cid, poly in room_polys.items():
        cx, cy = poly.mean(axis=0)
        j = int(np.argmin((cor_pts[:, 1] - cx) ** 2 + (cor_pts[:, 0] - cy) ** 2))
        door = (int(cor_pts[j, 1]), int(cor_pts[j, 0]))
        best = min(skn, key=lambda p: (p[0] - door[0]) ** 2 + (p[1] - door[1]) ** 2)
        G.add_edge(("entry", cid), best,
                   weight=math.dist(door, best) + math.dist((cx, cy), door))
        entries[cid] = {"pos": (int(cx), int(cy)), "door": door, "attach": best}
    return G, entries
