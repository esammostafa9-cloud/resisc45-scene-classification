"""Inference and the Gradio demo, using a small untrained model."""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from resisc45.config import Config
from resisc45.inference import Classifier
from resisc45.model import build_model

CLASSES = ["airport", "beach", "forest"]
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    cfg = Config(img_size=32, pretrained_weights=None, dense_units=8)
    model, _ = build_model(len(CLASSES), cfg)
    out = tmp_path_factory.mktemp("model")
    model.save(out / "model.keras")
    (out / "class_names.json").write_text(json.dumps(CLASSES))
    return out


@pytest.fixture(scope="module")
def classifier(model_dir):
    return Classifier.from_path(model_dir / "model.keras")


def image_bytes(fmt="JPEG", size=(48, 40)) -> bytes:
    buf = io.BytesIO()
    pixels = np.random.default_rng(0).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buf, format=fmt)
    return buf.getvalue()


# --- inference -------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_predict_bytes_returns_sorted_top_k(classifier, fmt):
    pred = classifier.predict_bytes(image_bytes(fmt), top_k=2)
    assert pred.label in CLASSES
    assert len(pred.top_k) == 2
    probs = [p for _, p in pred.top_k]
    assert probs == sorted(probs, reverse=True)
    assert pred.confidence == probs[0]


def test_top_k_is_clamped_to_number_of_classes(classifier):
    pred = classifier.predict_bytes(image_bytes(), top_k=99)
    assert len(pred.top_k) == len(CLASSES)
    assert sum(p for _, p in pred.top_k) == pytest.approx(1.0, abs=1e-4)


def test_predict_bytes_rejects_image_over_pixel_limit(classifier):
    import struct
    import zlib

    from resisc45.inference import MAX_IMAGE_PIXELS

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", MAX_IMAGE_PIXELS + 1, 1, 8, 2, 0, 0, 0)
    oversized = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="pixel limit"):
        classifier.predict_bytes(oversized)


def test_class_name_mismatch_is_rejected(classifier):
    with pytest.raises(ValueError, match="outputs"):
        Classifier(classifier.model, [*CLASSES, "extra"])


def test_from_env_loads_model_path(model_dir, monkeypatch):
    monkeypatch.setenv("RESISC45_MODEL_PATH", str(model_dir / "model.keras"))
    assert Classifier.from_env().class_names == CLASSES


def test_from_env_requires_model_path(monkeypatch):
    monkeypatch.delenv("RESISC45_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="RESISC45_MODEL_PATH"):
        Classifier.from_env()


# --- Gradio demo -----------------------------------------------------------------------


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_app", REPO_ROOT / "demo" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_builds_and_has_examples(classifier):
    import gradio as gr

    demo_app = _load_demo_module()
    assert isinstance(demo_app.build_demo(classifier), gr.Blocks)
    assert len(list((REPO_ROOT / "demo" / "examples").glob("*.jpg"))) >= 4


def test_result_card_flags_low_confidence():
    demo_app = _load_demo_module()
    assert "unsure" not in demo_app.result_card("harbor", 0.97, "ship")
    unsure = demo_app.result_card("palace", 0.41, "church")
    assert "unsure" in unsure
    assert "Church" in unsure
