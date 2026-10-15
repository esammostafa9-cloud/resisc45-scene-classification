"""Classify one or more aerial images with a trained model.

Usage:
    python -m resisc45.predict --model runs/<run>/model.keras image.jpg [more images or folders]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .data import IMAGE_EXTS, load_image
from .utils import load_json


def collect_images(inputs: list[Path]) -> list[Path]:
    """Expand folders into the image files they contain (non-recursive), keep files as-is."""
    images = []
    for p in inputs:
        if p.is_dir():
            images.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            images.append(p)
        else:
            raise FileNotFoundError(p)
    return images


def predict(
    model: keras.Model, images: list[Path], class_names: list[str], top_k: int = 3
) -> pd.DataFrame:
    img_size = model.input_shape[1]
    ds = (
        tf.data.Dataset.from_tensor_slices([str(p) for p in images])
        .map(lambda p: load_image(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
    )
    probs = model.predict(ds, verbose=0)

    rows = []
    for path, prob in zip(images, probs, strict=True):
        top = np.argsort(prob)[::-1][:top_k]
        row = {"image": str(path)}
        for rank, idx in enumerate(top, start=1):
            row[f"top{rank}"] = class_names[idx]
            row[f"top{rank}_prob"] = round(float(prob[idx]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files and/or folders")
    parser.add_argument("--model", required=True, type=Path, help="Trained .keras model")
    parser.add_argument(
        "--class-names", type=Path, help="class_names.json (default: next to the model)"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional CSV file for the predictions")
    args = parser.parse_args(argv)

    class_names = load_json(args.class_names or args.model.parent / "class_names.json")
    model = keras.models.load_model(args.model)
    images = collect_images(args.inputs)
    if not images:
        parser.error("no images found")

    results = predict(model, images, class_names, args.top_k)
    print(results.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
