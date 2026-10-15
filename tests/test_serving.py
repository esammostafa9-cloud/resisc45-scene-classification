"""Inference, REST API, Gradio demo and Hub upload, using a small untrained model."""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from resisc45 import hub
from resisc45.api import create_app
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


def test_class_name_mismatch_is_rejected(classifier):
    with pytest.raises(ValueError, match="outputs"):
        Classifier(classifier.model, CLASSES + ["extra"])


def test_from_env_prefers_local_model(model_dir, monkeypatch):
    monkeypatch.setenv("RESISC45_MODEL_PATH", str(model_dir / "model.keras"))
    assert Classifier.from_env().class_names == CLASSES


# --- REST API ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(classifier):
    with TestClient(create_app(classifier)) as c:
        yield c


def test_health_and_classes(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/classes").json() == CLASSES


def test_predict_endpoint(client):
    r = client.post("/predict?top_k=3", files={"file": ("x.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in CLASSES
    assert [s["label"] for s in body["top_k"]][0] == body["label"]
    assert len(body["top_k"]) == 3


@pytest.mark.parametrize(
    "payload, status",
    [(b"", 400), (b"definitely not an image", 400)],
)
def test_predict_rejects_bad_uploads(client, payload, status):
    r = client.post("/predict", files={"file": ("x.jpg", payload, "image/jpeg")})
    assert r.status_code == status


def test_predict_validates_top_k(client):
    r = client.post("/predict?top_k=0", files={"file": ("x.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 422


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
    assert "unsure" in unsure and "Church" in unsure


# --- Hugging Face Hub upload (no network) -----------------------------------------------


class FakeHfApi:
    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def record(*args, **kwargs):
            FakeHfApi.calls.append((name, args, kwargs))

        return record


def test_push_model_uploads_model_names_and_card(model_dir, monkeypatch):
    import huggingface_hub

    FakeHfApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    hub.push_model(
        model_dir / "model.keras",
        "user/repo",
        None,
        REPO_ROOT / "results" / "test_metrics.json",
        private=False,
    )

    uploaded = {kw["path_in_repo"] for name, _, kw in FakeHfApi.calls if name == "upload_file"}
    assert uploaded == {"model.keras", "class_names.json", "README.md"}


def test_model_card_includes_real_metrics():
    rows = hub._metrics_rows(REPO_ROOT / "results" / "test_metrics.json")
    assert "| Accuracy | 0.9441 |" in rows
    card = hub.MODEL_CARD.format(github="g", repo="r", metrics_rows=rows)
    assert card.startswith("---\nlicense: mit")
