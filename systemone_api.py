"""System One HTTP endpoints sharing the webcam's loaded Clef model."""

import base64
import io
import json
from math import isfinite
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

Description = str | dict[str, Any] | list[Any]


class Question(BaseModel):
    instructions: Description | None = None


class NoulQuestion(Question):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], Description | None] | None = None


class ChoiceQuestion(Question):
    type: Literal["choice"]
    criteria: dict[str, Description | None] = Field(min_length=1)


class ScoreQuestion(Question):
    type: Literal["score"]
    criteria: list[Description] = Field(min_length=1)


NamedQuestion = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]
InlineImage = Annotated[str, Field(min_length=1, max_length=8 * 1024 * 1024)]


class TwoFrameVideo(BaseModel):
    frames: list[InlineImage] = Field(min_length=2, max_length=2)
    fps: float = Field(default=2.0, gt=0, le=120, allow_inf_nan=False, strict=True)


class SystemOneRequest(BaseModel):
    model: Literal["clef-flash"]
    state: Description
    questions: dict[str, NamedQuestion] = Field(min_length=1, max_length=64)
    images: list[InlineImage] = Field(default_factory=list, max_length=4)
    videos: list[TwoFrameVideo] = Field(default_factory=list, max_length=1)


def decode_image(encoded: str, max_side: int) -> Image.Image:
    with Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[-1], validate=True))) as source:
        # Pillow reports some JPEGs as MPO. Use their first frame, like /decide.
        if source.format not in ("JPEG", "MPO", "PNG", "WEBP"):
            raise ValueError("images must be JPEG, PNG, or WebP")
        image = source.convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def add_systemone_routes(app: FastAPI, clef, max_side: int):
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Rejected inputs can contain NaN/Infinity, which JSONResponse cannot
        # serialize. Preserve the validation details with those values as text.
        detail = jsonable_encoder(error.errors(), custom_encoder={
            float: lambda value: value if isfinite(value) else str(value)
        })
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.post("/v1/systemone")
    def systemone(body: SystemOneRequest, response: Response):
        try:
            # Preserve omitted fields and explicit nulls for the upstream helper.
            record = body.model_dump(exclude_unset=True)
            json.dumps(record, allow_nan=False)
            if body.images:
                record["images"] = [decode_image(image, max_side) for image in body.images]
            if body.videos:
                clip = body.videos[0]
                frames = [decode_image(frame, max_side) for frame in clip.frames]
                if frames[0].size != frames[1].size:
                    raise ValueError("video frames must have matching dimensions after resizing")
                record["videos"] = [np.stack([np.asarray(frame) for frame in frames])]
                # These are already selected chronological frames. Do not resample
                # or duplicate them, and provide their real spacing for timestamps.
                record["media_kwargs"] = {"do_sample_frames": False, "cap_pixels_per_frame": True,
                    "video_metadata": [{"total_num_frames": 2, "fps": clip.fps,
                                        "frames_indices": [0, 1]}]}
            result, latency_ms = clef.decide(record)
        except (ValueError, KeyError, TypeError, AttributeError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(422, detail=f"invalid System One request: {error}") from error
        response.headers["Server-Timing"] = f"inference;dur={latency_ms:.1f}"
        response.headers["X-Clef-Hardware"] = clef.hardware
        response.headers["X-Clef-Device"] = clef.device
        return result

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": "clef-flash",
                            "description": "Cloudflare Clef-flash 9B decision model, running locally.",
                            "release_date": "2026-10-01"}]}
