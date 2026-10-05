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


class SystemOneRequest(BaseModel):
    model: Literal["clef-flash"]
    state: Description
    questions: dict[str, NamedQuestion] = Field(min_length=1, max_length=64)
    images: list[InlineImage] = Field(default_factory=list, max_length=4)


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
