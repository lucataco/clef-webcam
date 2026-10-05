import base64
import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from clef import Clef
from systemone_api import add_systemone_routes

MAX_SIDE = int(os.environ.get("CLEF_MAX_SIDE", "336"))

clef = Clef()
print(f"clef-flash ready on {clef.hardware} ({clef.device})", flush=True)

warmup = Image.new("RGB", (MAX_SIDE, MAX_SIDE * 3 // 4))
for _ in range(2):
    clef.decide({"model": "clef-flash", "state": "warmup", "images": [warmup], "questions": {"dark": {"type": "noul"}}})

app = FastAPI()
add_systemone_routes(app, clef, MAX_SIDE)


class DecideRequest(BaseModel):
    image: str
    questions: dict
    state: str | dict = "A live webcam frame from a laptop."


@app.post("/decide")
def decide(body: DecideRequest):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(body.image.split(",", 1)[-1]))).convert("RGB")
    except Exception as error:
        raise HTTPException(400, f"invalid image: {error}")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    try:
        response, latency_ms = clef.decide(
            {"model": "clef-flash", "state": body.state, "images": [image], "questions": body.questions}
        )
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise HTTPException(400, f"invalid questions: {error}")
    return {**response, "latency_ms": round(latency_ms, 1), "hardware": clef.hardware, "device": clef.device}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")
