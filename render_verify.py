"""Dev-only: render the service's output as a verification overlay (rooms tinted
by confidence, flags marked). Not part of the service — just evidence."""
import json, sys
import cv2, numpy as np
from app.pipeline.run import extract
from app.pipeline.ingest import ingest

src, out = sys.argv[1], sys.argv[2]
res = extract(src)
atlas, flags = res["atlas"], res["flags"]
ing = ingest(src)
img = ing.raster.copy(); s = ing.scale
flag_rooms = {f["id"] for f in flags if f["kind"] == "room"}
for room in atlas["rooms"]:
    if not room.get("polygon"): continue
    pts = np.array([[int(p["x"]*s), int(p["y"]*s)] for p in room["polygon"]], np.int32)
    c = room.get("confidence", 1.0)
    flagged = room["id"] in flag_rooms
    col = (60,120,240) if flagged else (int(90+120*(1-c)), int(200*c)+40, 120)
    tint = img.copy(); cv2.fillPoly(tint, [pts], col); img = cv2.addWeighted(img, 0.6, tint, 0.4, 0)
    cv2.polylines(img, [pts], True, (255,255,255) if not flagged else (0,80,255), 2 if not flagged else 4)
for n in atlas["graph"]["nodes"]:
    x,y = int(n["position"]["x"]*s), int(n["position"]["y"]*s)
    cv2.circle(img, (x,y), 6 if n["nodeType"]=="ROOM_ENTRY" else 4,
               (40,40,220) if n["nodeType"]=="ROOM_ENTRY" else (200,120,40), -1)
for e in atlas["graph"]["edges"]:
    pass
for f in flags:
    if "at" in f:
        x,y = int(f["at"]["x"]*s), int(f["at"]["y"]*s)
        cv2.drawMarker(img, (x,y), (0,0,255), cv2.MARKER_TRIANGLE_UP, 40, 4)
cv2.imwrite(out, img)
print(json.dumps(res["report"]["verification"]))
print("flags:", [f"{f['kind']}:{f.get('name') or f['id']}" for f in flags])
