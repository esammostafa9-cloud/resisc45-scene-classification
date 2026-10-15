"""Classify one or more aerial images with a trained model.

Usage:
    python -m resisc45.predict --model runs/<run>/model.keras image.jpg [more images or folders]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import IMAGE_EXTS
from .inference import Classifier

BATCH_SIZE = 32


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


def predict(classifier: Classifier, images: list[Path], top_k: int = 3) -> pd.DataFrame:
    rows = []
    for start in range(0, len(images), BATCH_SIZE):
        chunk = images[start : start + BATCH_SIZE]
        probs = classifier.predict_proba([p.read_bytes() for p in chunk])
        for path, prob in zip(chunk, probs, strict=True):
            pred = classifier.to_prediction(prob, top_k)
            row = {"image": str(path)}
            for rank, (label, p) in enumerate(pred.top_k, start=1):
                row[f"top{rank}"] = label
                row[f"top{rank}_prob"] = round(p, 4)
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

    images = collect_images(args.inputs)
    if not images:
        parser.error("no images found")
    classifier = Classifier.from_path(args.model, args.class_names)

    results = predict(classifier, images, args.top_k)
    print(results.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
