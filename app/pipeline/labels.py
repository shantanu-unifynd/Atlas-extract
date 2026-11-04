"""Stage 3 — read the text. Vector labels when the file has them (exact, free);
OCR when it does not.

Difficulty log #2: SVG label x/y are pre-transform — the full nested
transform chain is composed. #9: no-text-layer PDFs go through OCR.
#12: OCR fragments are joined in reading order, not confidence order.
"""
import math
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

SVG_NS = "{http://www.w3.org/2000/svg}"


def _parse_transform(s):
    m = [1, 0, 0, 1, 0, 0]
    if not s:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix" and len(v) == 6:
            t = v
        elif name == "translate":
            t = [1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0]
        elif name == "scale":
            t = [v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0]
        elif name == "rotate":
            a = math.radians(v[0]); ca, sa = math.cos(a), math.sin(a)
            t = [ca, sa, -sa, ca, 0, 0]
            if len(v) == 3:
                cx, cy = v[1], v[2]
                t = [ca, sa, -sa, ca, cx - ca * cx + sa * cy, cy - sa * cx - ca * cy]
        else:
            t = [1, 0, 0, 1, 0, 0]
        a, b, c, d, e, f = m
        A, B, C, D, E, F = t
        m = [a*A + c*B, b*A + d*B, a*C + c*D, b*C + d*D, a*E + c*F + e, b*E + d*F + f]
    return m


def _compose(m, t):
    a, b, c, d, e, f = m
    A, B, C, D, E, F = t
    return [a*A + c*B, b*A + d*B, a*C + c*D, b*C + d*D, a*E + c*F + e, b*E + d*F + f]


def svg_labels(root: ET.Element, scale: float) -> list[dict]:
    """[{text, x, y}] in raster pixels, duplicates (shadow copies) removed."""
    out = []

    def walk(el, m):
        for child in el:
            cm = _compose(m, _parse_transform(child.get("transform"))) if child.get("transform") else m
            if child.tag == f"{SVG_NS}text":
                x, y = child.get("x"), child.get("y")
                txt = "".join(child.itertext()).strip()
                if x is None or y is None:
                    for ts in child:
                        if ts.get("x") and ts.get("y"):
                            x, y = ts.get("x"), ts.get("y")
                            break
                if txt and x and y:
                    try:
                        X = float(x.split()[0]) if " " in x else float(x)
                        Y = float(y.split()[0]) if " " in y else float(y)
                    except ValueError:
                        continue
                    a, b, c, d, e, f = cm
                    out.append({"text": re.sub(r"\s+", " ", txt),
                                "x": (a*X + c*Y + e) * scale, "y": (b*X + d*Y + f) * scale})
            else:
                walk(child, cm)

    walk(root, [1, 0, 0, 1, 0, 0])
    seen, uniq = set(), []
    for l in out:
        k = (l["text"], round(l["x"] / 6), round(l["y"] / 6))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(l)
    return uniq


def pdf_labels(words: list, scale: float) -> list[dict]:
    """[{text, x, y}] in raster pixels. PyMuPDF `get_text("words")` returns one
    row per word tagged with its (block, line, word_no); regroup them into the
    original text lines so a multi-word shop name ("SHOPPERS STOP") arrives as
    one label instead of several fragments — mirroring how a whole SVG <text>
    element arrives (otherwise room naming, which keeps the longest single
    detection, would truncate names to their longest word)."""
    lines = {}
    for w in words:
        if len(w) < 8 or not str(w[4]).strip():
            continue
        lines.setdefault((w[5], w[6]), []).append(w)
    out = []
    for members in lines.values():
        members.sort(key=lambda m: m[7])  # word order within the line
        text = re.sub(r"\s+", " ", " ".join(str(m[4]) for m in members)).strip()
        if not text:
            continue
        x0 = min(m[0] for m in members); y0 = min(m[1] for m in members)
        x1 = max(m[2] for m in members); y1 = max(m[3] for m in members)
        out.append({"text": text, "x": (x0 + x1) / 2 * scale, "y": (y0 + y1) / 2 * scale})
    return out


def ocr_labels(raster: np.ndarray, min_conf: float = 0.5) -> list[dict]:
    """Whole-plan OCR plus two rotated passes for vertical names.
    Returns [{text, x, y, conf}] in raster pixels."""
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    H, W = raster.shape[:2]

    def run(im, rot):
        res, _ = engine(im)
        dets = []
        for box, txt, conf in (res or []):
            if conf < min_conf or sum(ch.isalpha() for ch in txt) < 2:
                continue
            cx = sum(p[0] for p in box) / 4.0
            cy = sum(p[1] for p in box) / 4.0
            if rot == "cw":       # rotated image coords -> original
                cx, cy = cy, im.shape[1] - 1 - cx
            elif rot == "ccw":
                cx, cy = im.shape[0] - 1 - cy, cx
            dets.append({"text": txt.strip(), "x": cx, "y": cy, "conf": float(conf), "rot": rot})
        return dets

    dets = run(raster, None)
    dets += run(cv2.rotate(raster, cv2.ROTATE_90_CLOCKWISE), "cw")
    dets += run(cv2.rotate(raster, cv2.ROTATE_90_COUNTERCLOCKWISE), "ccw")
    return dets


def join_room_detections(dets: list[dict]) -> str:
    """Difficulty #12: join multi-fragment names in reading order."""
    if not dets:
        return ""
    best_rot = max({d.get("rot") for d in dets},
                   key=lambda r: sum(d["conf"] for d in dets if d.get("rot") == r))
    group = [d for d in dets if d.get("rot") == best_rot]
    if best_rot in ("cw", "ccw"):
        group.sort(key=lambda d: (round(d["x"] / 40), d["y"] if best_rot == "ccw" else -d["y"]))
    else:
        group.sort(key=lambda d: (round(d["y"] / 40), d["x"]))
    seen, parts = set(), []
    for d in group:
        t = d["text"]
        if any(t.lower() in u.lower() or u.lower() in t.lower() for u in seen):
            continue
        seen.add(t)
        parts.append(t)
        if len(parts) == 3:
            break
    return " ".join(parts)[:40]
