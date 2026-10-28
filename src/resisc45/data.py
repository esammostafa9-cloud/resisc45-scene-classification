"""Dataset discovery, verification, splitting and the tf.data input pipeline."""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")
MAX_ARCHIVE_MEMBERS = 50_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024**3


@dataclass
class Dataset:
    """Every image in the dataset, with integer labels aligned to ``class_names``."""

    root: Path
    paths: np.ndarray
    labels: np.ndarray
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def _count_images(folder: Path) -> int:
    try:
        return sum(1 for f in os.listdir(folder) if _is_image(f))
    except (NotADirectoryError, FileNotFoundError):
        return 0


def extract_archive(zip_path: str | Path, dest: str | Path) -> Path:
    """Extract ``zip_path`` into ``dest`` once; later calls reuse the extracted copy."""
    zip_path, dest = Path(zip_path), Path(dest)
    marker = dest / ".extracted"
    if marker.exists():
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"Archive contains more than {MAX_ARCHIVE_MEMBERS} entries.")
        total_size = sum(member.file_size for member in members)
        if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("Archive expands beyond the 2 GiB safety limit.")
        for member in members:
            # Refuse entries that would escape the destination folder ("zip slip").
            if not (dest_resolved / member.filename).resolve().is_relative_to(dest_resolved):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(dest, members=members)
    marker.touch()
    return dest


def find_class_root(base: str | Path, min_classes: int = 2) -> Path:
    """Return the folder that directly holds the class sub-folders.

    Archives often wrap the data in one or more extra parent folders, so walk the tree and
    pick the first folder with at least ``min_classes`` sub-folders that contain images.
    """
    base = Path(base)
    for root, dirs, _ in os.walk(base):
        if sum(_count_images(Path(root) / d) > 0 for d in dirs) >= min_classes:
            return Path(root)
    raise FileNotFoundError(f"No folder with at least {min_classes} image class folders in {base}")


def load_dataset(data_dir: str | Path, min_classes: int = 2) -> Dataset:
    """Collect (path, label) pairs for every image under ``data_dir``.

    ``data_dir`` may be the extracted dataset folder or the ``NWPU.zip`` archive itself, which
    is extracted next to it on first use.
    """
    data_dir = Path(data_dir)
    if data_dir.suffix.lower() == ".zip":
        data_dir = extract_archive(data_dir, data_dir.with_suffix(""))

    root = find_class_root(data_dir, min_classes=min_classes)
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir() and _count_images(d) > 0)

    paths, labels = [], []
    for label, name in enumerate(class_names):
        for fname in sorted(os.listdir(root / name)):
            if _is_image(fname):
                paths.append(str(root / name / fname))
                labels.append(label)

    return Dataset(
        root=root,
        paths=np.array(paths),
        labels=np.array(labels, dtype=np.int64),
        class_names=class_names,
    )


def verify_dataset(
    ds: Dataset, expected_classes: int | None, expected_per_class: int | None
) -> None:
    """Fail loudly if the dataset is not the one we expect (e.g. a partial extraction)."""
    if expected_classes is not None and ds.num_classes != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {ds.num_classes}.")
    if expected_per_class is not None:
        counts = set(np.bincount(ds.labels, minlength=ds.num_classes).tolist())
        if counts != {expected_per_class}:
            raise ValueError(
                f"Expected {expected_per_class} images per class, found counts {sorted(counts)}."
            )


def split_dataset(ds: Dataset, val_size: float, test_size: float, seed: int) -> pd.DataFrame:
    """Stratified train / val / test split.

    Returns a table with one row per image (``path`` relative to the dataset root, ``label``,
    ``class_name``, ``split``) so the exact split can be saved and reused for evaluation.
    """
    holdout = val_size + test_size
    train_p, temp_p, train_l, temp_l = train_test_split(
        ds.paths, ds.labels, test_size=holdout, stratify=ds.labels, random_state=seed
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=test_size / holdout, stratify=temp_l, random_state=seed
    )

    frames = []
    for split, p, y in (
        ("train", train_p, train_l),
        ("val", val_p, val_l),
        ("test", test_p, test_l),
    ):
        rel = [Path(x).relative_to(ds.root).as_posix() for x in p]
        frames.append(
            pd.DataFrame(
                {
                    "path": rel,
                    "label": y,
                    "class_name": [ds.class_names[i] for i in y],
                    "split": split,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def split_arrays(
    split_df: pd.DataFrame, root: str | Path, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute image paths and labels for one split of a split table."""
    part = split_df[split_df["split"] == split]
    paths = np.array([str(Path(root) / p) for p in part["path"]])
    return paths, part["label"].to_numpy(dtype=np.int64)


def load_image(path: tf.Tensor | str, img_size: int) -> tf.Tensor:
    """Read, decode and resize one image. Pixels stay in the raw 0-255 range.

    ResNet-50's ``preprocess_input`` runs inside the model, after augmentation, so that
    brightness and contrast augmentation operate on real pixel values.
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size])
    image.set_shape([img_size, img_size, 3])
    return tf.cast(image, tf.float32)


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    img_size: int,
    batch_size: int,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Batched, prefetched pipeline. Only the training pipeline is shuffled."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
