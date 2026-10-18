"""Plotting helpers. Every function writes a PNG and returns the figure.

Figures are built with Matplotlib's object-oriented API (no pyplot state), so they render
the same from the command line, in CI, or on a headless server.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay


def _save(fig: Figure, path: str | Path) -> Figure:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_class_distribution(counts: np.ndarray, class_names: list[str], path: str | Path) -> Figure:
    fig = Figure(figsize=(14, 4))
    ax = fig.add_subplot()
    ax.bar(class_names, counts, color="#4C72B0")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution")
    return _save(fig, path)


def plot_history(histories: dict[str, dict[str, list[float]]], path: str | Path) -> Figure:
    """Training vs validation accuracy and loss, with all phases on one continuous x-axis."""
    fig = Figure(figsize=(14, 5))
    axes = fig.subplots(1, 2)

    # Concatenate the phases so each curve is one continuous line.
    series: dict[str, list[float]] = {
        k: [] for k in ("accuracy", "val_accuracy", "loss", "val_loss")
    }
    starts: list[tuple[int, str]] = []  # (first epoch of phase, phase name)
    for name, h in histories.items():
        starts.append((len(series["accuracy"]) + 1, name))
        for key, values in series.items():
            values.extend(h[key])
    epochs = np.arange(1, len(series["accuracy"]) + 1)

    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(epochs, series[key], "-o", ms=3, label=f"train {key}")
        ax.plot(epochs, series[f"val_{key}"], "-o", ms=3, label=f"val {key}")
        for start, name in starts[1:]:
            ax.axvline(start - 0.5, color="grey", ls="--", lw=1)
            ax.text(start - 0.3, 0.02, name, transform=ax.get_xaxis_transform(), color="grey")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    path: str | Path,
    title: str = "Confusion matrix",
    cmap: str = "Blues",
) -> Figure:
    n = len(class_names)
    size = max(8, n * 0.45)
    fig = Figure(figsize=(size, size * 0.9))
    ax = fig.add_subplot()
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax,
        cmap=cmap,
        xticks_rotation=90,
        values_format="d",
        colorbar=True,
        text_kw={"fontsize": 6 if n > 20 else 9},
    )
    ax.set_title(title, fontsize=14, pad=12)
    return _save(fig, path)


def confusion_subset(cm: np.ndarray, n: int = 10, worst: bool = False) -> np.ndarray:
    """Indices of the ``n`` classes with the fewest (or most) misclassified images."""
    errors = cm.sum(axis=1) - np.diag(cm)
    order = np.argsort(errors, kind="stable")
    chosen = order[::-1][:n] if worst else order[:n]
    return np.sort(chosen)


def plot_sample_predictions(
    image_paths: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    path: str | Path,
    top_k: int = 5,
) -> Figure:
    """One row per image: the image, then a bar chart of its top-k softmax probabilities."""
    n = len(image_paths)
    fig = Figure(figsize=(9, 2.8 * n))
    axes = np.atleast_2d(fig.subplots(n, 2, gridspec_kw={"width_ratios": [1, 1.6]}))
    top_k = min(top_k, probs.shape[1])

    for row, (img_path, true, prob) in enumerate(zip(image_paths, y_true, probs, strict=True)):
        top = np.argsort(prob)[::-1][:top_k]
        pred = int(top[0])
        colour = "green" if pred == int(true) else "red"

        axes[row, 0].imshow(mpimg.imread(img_path))
        axes[row, 0].set_title(f"True: {class_names[int(true)]}", fontsize=10)
        axes[row, 0].axis("off")

        ypos = np.arange(top_k)[::-1]
        colours = ["#4C72B0"] * top_k
        colours[0] = colour
        axes[row, 1].barh(ypos, prob[top], color=colours)
        axes[row, 1].set_yticks(ypos, [class_names[j] for j in top], fontsize=9)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_xlabel("Softmax probability")
        axes[row, 1].set_title(
            f"Pred: {class_names[pred]} ({prob[pred]:.2%})", fontsize=10, color=colour
        )
    return _save(fig, path)
