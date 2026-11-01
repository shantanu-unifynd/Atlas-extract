# Atlas Extract

Automatic floor-plan extraction service for the Atlas indoor-navigation platform.

Input: a floor plan (SVG / PDF; DXF/IFC/raster planned). Output: rooms with
names and categories, plus a walkable, distance-weighted navigation graph —
in the exact JSON shapes the Atlas backend publishes to the visitor client.
The admin's job is verification, not authoring.

Pipeline (all automatic): ingest → understand the drawing → read the text →
polygonize rooms → build the graph from empty space. Design, P0 evidence on
three real plans, and the growing difficulty log live in
`../Docs/roadmap/extraction-automation-plan.md`.

## Run

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m app.cli tests/fixtures/wtc-mall.svg out/
    .venv/bin/uvicorn app.main:app --port 5600   # POST /extract

## Tests

    .venv/bin/python -m pytest tests/ -x -q
