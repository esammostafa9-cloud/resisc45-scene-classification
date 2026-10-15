"""REST API for the scene classifier.

Run locally:
    RESISC45_MODEL_PATH=runs/<run>/model.keras python -m resisc45.api
Then open http://localhost:8000/docs for interactive API docs.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from . import __version__
from .inference import Classifier

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class ClassScore(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top_k: list[ClassScore]


def create_app(classifier: Classifier | None = None) -> FastAPI:
    """Build the app. Pass ``classifier`` in tests; otherwise it is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or Classifier.from_env()
        yield

    app = FastAPI(
        title="Aerial Scene Classifier API",
        description="Classify aerial / satellite images into 45 NWPU-RESISC45 scene categories.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/classes")
    def classes() -> list[str]:
        return app.state.classifier.class_names

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        file: Annotated[UploadFile, File(description="JPEG, PNG, BMP or GIF image")],
        top_k: Annotated[int, Query(ge=1, le=45)] = 5,
    ) -> dict:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(400, "Empty file.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image larger than 10 MB.")
        try:
            return app.state.classifier.predict_bytes(data, top_k).to_dict()
        except (tf.errors.InvalidArgumentError, ValueError) as err:
            raise HTTPException(400, "Could not decode the file as an image.") from err

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
