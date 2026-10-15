"""Evaluate a trained model on the held-out test split.

Usage:
    python -m resisc45.evaluate --run-dir runs/<run> --data-dir data/NWPU-RESISC45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import Config
from .data import load_dataset, make_dataset, split_arrays
from .utils import get_logger, load_json, save_json
from .visualize import confusion_subset, plot_confusion_matrix, plot_sample_predictions

log = get_logger()


def evaluate_model(
    model: keras.Model,
    paths: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    cfg: Config,
    out_dir: str | Path,
    n_samples: int = 6,
) -> dict[str, float]:
    """Score ``model`` once on the given (test) images and write metrics, tables and figures."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_ds = make_dataset(paths, labels, cfg.img_size, cfg.batch_size)  # unshuffled
    test_loss, _ = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = labels

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "test_loss": float(test_loss),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    save_json(metrics, out_dir / "test_metrics.json")

    all_labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().round(4).to_csv(out_dir / "classification_report.csv")
    log.info(
        "\n%s",
        classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=class_names,
            digits=3,
            zero_division=0,
        ),
    )

    pd.DataFrame(
        {
            "path": paths,
            "true": [class_names[i] for i in y_true],
            "pred": [class_names[i] for i in y_pred],
            "confidence": probs.max(axis=1).round(4),
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plot_confusion_matrix(cm, class_names, fig_dir / "confusion_matrix.png")
    if len(class_names) > 10:
        for worst, cmap, tag in ((False, "Blues", "best"), (True, "Reds", "worst")):
            idx = confusion_subset(cm, 10, worst=worst)
            plot_confusion_matrix(
                cm[np.ix_(idx, idx)],
                [class_names[i] for i in idx],
                fig_dir / f"confusion_matrix_{tag}10.png",
                title=f"Confusion matrix - 10 {tag.upper()} classes",
                cmap=cmap,
            )

    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(len(paths), size=min(n_samples, len(paths)), replace=False)
    plot_sample_predictions(
        paths[sample],
        y_true[sample],
        probs[sample],
        class_names,
        fig_dir / "sample_predictions.png",
    )

    log.info("Test metrics: %s", metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path, help="Output folder of a training run"
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset folder or NWPU.zip")
    parser.add_argument("--model", type=Path, help="Model file (default: <run-dir>/model.keras)")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.run_dir / "config.yaml")
    class_names = load_json(args.run_dir / "class_names.json")
    split_df = pd.read_csv(args.run_dir / "split.csv")
    root = load_dataset(args.data_dir).root
    paths, labels = split_arrays(split_df, root, "test")

    model = keras.models.load_model(args.model or args.run_dir / "model.keras")
    evaluate_model(model, paths, labels, class_names, cfg, args.run_dir)


if __name__ == "__main__":
    main()
