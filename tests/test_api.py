"""Service-surface tests — the real FastAPI app via TestClient, so the HTTP
contract (not just the pipeline function) is verified before any integration."""
import pathlib

from fastapi.testclient import TestClient

from app.main import app

FIX = pathlib.Path(__file__).parent / "fixtures"
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"ok": True}


def test_extract_rejects_unsupported_format():
    r = client.post("/extract", files={"file": ("plan.txt", b"nope", "text/plain")})
    assert r.status_code == 422


def test_extract_svg_returns_report_flags_atlas():
    with open(FIX / "wtc-mall.svg", "rb") as f:
        r = client.post("/extract", files={"file": ("wtc-mall.svg", f, "image/svg+xml")})
    assert r.status_code == 200
    body = r.json()
    assert {"report", "flags", "atlas"} <= set(body)
    assert body["report"]["graph_components"] == 1
    assert body["atlas"]["graph"]["nodes"]
