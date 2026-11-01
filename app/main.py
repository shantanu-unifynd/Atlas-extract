"""Atlas Extract service. POST a floor plan, get rooms + labels + graph in
Atlas's published-map shapes plus a verification report."""
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile

from app.pipeline.run import extract

app = FastAPI(title="Atlas Extract", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract")
async def extract_plan(file: UploadFile):
    suffix = "." + (file.filename or "plan.svg").rsplit(".", 1)[-1].lower()
    if suffix not in (".svg", ".pdf"):
        raise HTTPException(422, "unsupported format — send SVG or PDF")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        return extract(path)
    except Exception as exc:  # surfaced to the caller; nothing partial is written
        raise HTTPException(500, f"extraction failed: {exc}") from exc
