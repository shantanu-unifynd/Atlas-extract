"""Stage 1 — ingest. Any supported plan becomes two rasters (with text /
geometry-only) plus page metadata. SVGs additionally keep their XML tree so
labels can be read exactly instead of OCR'd.

Difficulty log #1: SVGs without a viewBox do not scale when resized — one is
always injected before rendering.
"""
from dataclasses import dataclass, field
import re
import xml.etree.ElementTree as ET

import cv2
import fitz  # PyMuPDF
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"
TARGET_WIDTH = 3600  # raster width the CV stages work at


@dataclass
class IngestResult:
    raster: np.ndarray          # BGR, with text
    raster_notext: np.ndarray   # BGR, text stripped (SVG) or identical (no text layer)
    scale: float                # raster px per source unit
    source_kind: str            # "svg" | "pdf"
    svg_root: ET.Element | None = None
    pdf_words: list = field(default_factory=list)  # (x0,y0,x1,y1,word) in source units


def _render_svg(svg_bytes: bytes) -> np.ndarray:
    pix = fitz.open(stream=svg_bytes, filetype="svg").load_page(0).get_pixmap(alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _svg_dims(root: ET.Element) -> tuple[float, float]:
    vb = root.get("viewBox")
    if vb:
        parts = [float(v) for v in re.split(r"[\s,]+", vb.strip())]
        return parts[2], parts[3]
    def num(s):
        m = re.match(r"([0-9.]+)", s or "")
        return float(m.group(1)) if m else None
    w, h = num(root.get("width")), num(root.get("height"))
    if not w or not h:
        raise ValueError("SVG has neither viewBox nor numeric width/height")
    return w, h


def _prep_svg(root: ET.Element, width_px: int, strip_text: bool) -> bytes:
    clone = ET.fromstring(ET.tostring(root))
    w, h = _svg_dims(clone)
    if not clone.get("viewBox"):
        clone.set("viewBox", f"0 0 {w} {h}")
    clone.set("width", str(width_px))
    clone.set("height", str(int(h * width_px / w)))
    if strip_text:
        def rm(parent):
            for child in list(parent):
                if child.tag == f"{{{SVG_NS}}}text":
                    parent.remove(child)
                else:
                    rm(child)
        rm(clone)
    return ET.tostring(clone)


def ingest(path: str) -> IngestResult:
    if path.lower().endswith(".svg"):
        ET.register_namespace("", SVG_NS)
        root = ET.parse(path).getroot()
        w, _ = _svg_dims(root)
        raster = _render_svg(_prep_svg(root, TARGET_WIDTH, strip_text=False))
        notext = _render_svg(_prep_svg(root, TARGET_WIDTH, strip_text=True))
        return IngestResult(raster, notext, TARGET_WIDTH / w, "svg", svg_root=root)

    if path.lower().endswith(".pdf"):
        page = fitz.open(path).load_page(0)
        zoom = TARGET_WIDTH / page.rect.width
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        raster = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        words = [tuple(wd[:5]) for wd in page.get_text("words")]
        # No-text raster for PDFs with a real text layer is a later refinement
        # (redaction render); plans with outline text are identical anyway.
        return IngestResult(raster, raster.copy(), zoom, "pdf", pdf_words=words)

    raise ValueError(f"unsupported input: {path}")
