"""Shared fixtures: a tiny synthetic image dataset so tests never need the real data."""

import numpy as np
import pytest
from PIL import Image

from resisc45.config import Config

CLASSES = ["airport", "beach", "forest"]
PER_CLASS = 10


@pytest.fixture
def tiny_dataset(tmp_path):
    """data/<wrapper>/<class>/*.jpg, mimicking an archive with an extra parent folder."""
    root = tmp_path / "data" / "NWPU-RESISC45"
    rng = np.random.default_rng(0)
    for c, name in enumerate(CLASSES):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(PER_CLASS):
            # Each class has a distinct mean colour so a model can actually learn something.
            pixels = rng.integers(0, 60, size=(40, 40, 3)) + c * 90
            Image.fromarray(pixels.astype(np.uint8)).save(folder / f"{name}_{i:03d}.jpg")
        (folder / "notes.txt").write_text("not an image")
    return tmp_path / "data"


@pytest.fixture
def tiny_config():
    return Config(
        img_size=32,  # smallest input ResNet-50 accepts
        batch_size=8,
        val_size=0.2,
        test_size=0.2,
        expected_classes=len(CLASSES),
        expected_per_class=PER_CLASS,
        pretrained_weights=None,  # no download in tests
        dense_units=16,
        head_epochs=1,
        fine_tune_epochs=1,
    )
