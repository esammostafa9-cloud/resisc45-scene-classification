import zipfile

import numpy as np
import pytest
from conftest import CLASSES, PER_CLASS

from resisc45.data import (
    extract_archive,
    find_class_root,
    load_dataset,
    make_dataset,
    split_arrays,
    split_dataset,
    verify_dataset,
)


def test_find_class_root_skips_wrapper_folder(tiny_dataset):
    assert find_class_root(tiny_dataset).name == "NWPU-RESISC45"


def test_load_dataset_collects_images_only(tiny_dataset):
    ds = load_dataset(tiny_dataset)
    assert ds.class_names == CLASSES
    assert len(ds.paths) == len(CLASSES) * PER_CLASS
    assert np.bincount(ds.labels).tolist() == [PER_CLASS] * len(CLASSES)


def test_load_dataset_from_zip(tiny_dataset, tmp_path):
    archive = tmp_path / "NWPU.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in tiny_dataset.rglob("*.jpg"):
            zf.write(f, f.relative_to(tiny_dataset))
    ds = load_dataset(archive)
    assert ds.num_classes == len(CLASSES)
    assert (tmp_path / "NWPU" / ".extracted").exists()


def test_extract_archive_rejects_zip_slip(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(ValueError, match="Unsafe path"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_verify_dataset(tiny_dataset):
    ds = load_dataset(tiny_dataset)
    verify_dataset(ds, len(CLASSES), PER_CLASS)
    verify_dataset(ds, None, None)
    with pytest.raises(ValueError, match="classes"):
        verify_dataset(ds, 45, None)
    with pytest.raises(ValueError, match="per class"):
        verify_dataset(ds, None, 700)


def test_split_is_stratified_disjoint_and_deterministic(tiny_dataset):
    ds = load_dataset(tiny_dataset)
    split = split_dataset(ds, val_size=0.2, test_size=0.2, seed=42)

    assert len(split) == len(ds.paths)
    assert split["path"].is_unique
    counts = split.groupby(["split", "label"]).size().unstack()
    assert (counts.loc["train"] == 6).all()
    assert (counts.loc["val"] == 2).all()
    assert (counts.loc["test"] == 2).all()

    again = split_dataset(ds, val_size=0.2, test_size=0.2, seed=42)
    assert split.equals(again)


def test_pipeline_yields_raw_pixel_batches(tiny_dataset):
    ds = load_dataset(tiny_dataset)
    split = split_dataset(ds, 0.2, 0.2, seed=42)
    paths, labels = split_arrays(split, ds.root, "train")

    images, y = next(iter(make_dataset(paths, labels, img_size=32, batch_size=4, training=True)))
    assert tuple(images.shape) == (4, 32, 32, 3)
    assert images.dtype.name == "float32"
    assert float(images.numpy().max()) > 1.0  # still 0-255, not normalised
    assert y.shape == (4,)
