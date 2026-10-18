"""Load a trained model once and classify images from paths or raw bytes.

Shared by the CLI (``predict.py``), the REST API (``api.py``) and the Gradio demo, so all three
preprocess images exactly like training: decode, resize to the model's input size, raw 0-255
pixels. ResNet-50's ``preprocess_input`` is part of the saved model, so it is NOT applied here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .utils import load_json


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "top_k": [{"label": c, "probability": round(p, 4)} for c, p in self.top_k],
        }


class Classifier:
    """A trained model plus its class names."""

    def __init__(self, model: keras.Model, class_names: list[str]):
        n_out = model.output_shape[-1]
        if n_out != len(class_names):
            raise ValueError(f"Model has {n_out} outputs but {len(class_names)} class names.")
        self.model = model
        self.class_names = list(class_names)
        self.img_size = int(model.input_shape[1])

    @classmethod
    def from_path(cls, model_path: str | Path, class_names_path: str | Path | None = None):
        """Load ``model.keras``; class names default to ``class_names.json`` beside it."""
        model_path = Path(model_path)
        names = load_json(class_names_path or model_path.parent / "class_names.json")
        return cls(keras.models.load_model(model_path), names)

    @classmethod
    def from_env(cls):
        """Load the model named by the ``RESISC45_MODEL_PATH`` environment variable."""
        path = os.environ.get("RESISC45_MODEL_PATH")
        if not path:
            raise RuntimeError(
                "Set RESISC45_MODEL_PATH to a trained model, e.g. runs/<timestamp>/model.keras"
            )
        return cls.from_path(path)

    def _decode(self, data: bytes | tf.Tensor) -> tf.Tensor:
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image = tf.image.resize(image, [self.img_size, self.img_size])
        return tf.cast(image, tf.float32)

    def predict_proba(self, images: list[bytes]) -> np.ndarray:
        """Softmax probabilities, shape ``(len(images), num_classes)``."""
        batch = tf.stack([self._decode(b) for b in images])
        return self.model.predict(batch, verbose=0)

    def predict_bytes(self, data: bytes, top_k: int = 5) -> Prediction:
        """Classify one encoded image (JPEG / PNG / BMP / GIF bytes)."""
        return self.to_prediction(self.predict_proba([data])[0], top_k)

    def predict_path(self, path: str | Path, top_k: int = 5) -> Prediction:
        return self.predict_bytes(Path(path).read_bytes(), top_k)

    def to_prediction(self, prob: np.ndarray, top_k: int) -> Prediction:
        order = np.argsort(prob)[::-1][: max(1, min(top_k, len(prob)))]
        top = [(self.class_names[i], float(prob[i])) for i in order]
        return Prediction(label=top[0][0], confidence=top[0][1], top_k=top)
