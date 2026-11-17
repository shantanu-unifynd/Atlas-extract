"""Raster-image ingest (JPEG/PNG class) — format-coverage tests. Fast: ingest
only, no OCR/extract."""
import pathlib

import numpy as np
import pytest

from app.pipeline.ingest import TARGET_WIDTH, ingest

HOSPITAL = pathlib.Path(
    "/Users/unifynd-pro/Desktop/Atlas-Platform/assets/Hospital floor plan.jpeg")


@pytest.mark.skipif(not HOSPITAL.exists(), reason="asset not present")
def test_jpeg_ingest_shapes_and_scale():
    ing = ingest(str(HOSPITAL))
    assert ing.source_kind == "image"
    assert ing.raster.shape[1] == TARGET_WIDTH
    assert ing.raster.shape == ing.raster_notext.shape
    assert ing.raster.dtype == np.uint8
    # No vector text layer: identical rasters, empty word list -> OCR fallback.
    assert ing.pdf_words == []
    assert ing.svg_root is None
    assert ing.scale > 0


def test_unsupported_still_rejected(tmp_path):
    bogus = tmp_path / "plan.dxf"
    bogus.write_text("0\nSECTION\n")
    with pytest.raises(ValueError, match="unsupported input"):
        ingest(str(bogus))
