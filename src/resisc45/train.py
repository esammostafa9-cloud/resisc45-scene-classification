"""Two-phase transfer-learning training on NWPU-RESISC45.

Phase 1 trains only the classification head on a frozen ResNet-50. Phase 2 unfreezes the top
residual stages and fine-tunes them at a much lower learning rate. The final model is the
phase checkpoint with the best *validation* accuracy; the test split is used only once, at
the end.

Usage:
    python -m resisc45.train --data-dir data/NWPU-RESISC45 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import keras
import numpy as np

from .config import Config
from .data import load_dataset, make_dataset, split_arrays, split_dataset, verify_dataset
from .evaluate import evaluate_model
from .model import build_model, compile_model, unfreeze_top_stages
from .utils import get_logger, save_json, set_seed
from .visualize import plot_class_distribution, plot_history

log = get_logger()


def _callbacks(ckpt: Path, csv_log: Path, cfg: Config, lr_factor: float, min_lr: float) -> list:
    return [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lr_factor, patience=cfg.lr_patience, min_lr=min_lr, verbose=1
        ),
        keras.callbacks.CSVLogger(csv_log),
    ]


def train(data_dir: Path, cfg: Config, out_dir: Path, run_eval: bool = True) -> Path:
    """Run the full pipeline and return the path of the selected model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    cfg.to_yaml(out_dir / "config.yaml")

    # Data
    ds = load_dataset(data_dir)
    verify_dataset(ds, cfg.expected_classes, cfg.expected_per_class)
    log.info("Dataset: %d images, %d classes (%s)", len(ds.paths), ds.num_classes, ds.root)
    save_json(ds.class_names, out_dir / "class_names.json")
    plot_class_distribution(
        np.bincount(ds.labels, minlength=ds.num_classes),
        ds.class_names,
        out_dir / "figures" / "class_distribution.png",
    )

    split_df = split_dataset(ds, cfg.val_size, cfg.test_size, cfg.seed)
    split_df.to_csv(out_dir / "split.csv", index=False)
    log.info("Split sizes: %s", split_df["split"].value_counts().to_dict())

    pipelines = {}
    for split in ("train", "val"):
        paths, labels = split_arrays(split_df, ds.root, split)
        pipelines[split] = make_dataset(
            paths, labels, cfg.img_size, cfg.batch_size, training=split == "train", seed=cfg.seed
        )

    # Model
    model, base_model = build_model(ds.num_classes, cfg)
    phases = [
        ("phase1_head", cfg.head_epochs, cfg.head_lr, cfg.head_lr_factor, cfg.head_min_lr),
        (
            "phase2_finetune",
            cfg.fine_tune_epochs,
            cfg.fine_tune_lr,
            cfg.fine_tune_lr_factor,
            cfg.fine_tune_min_lr,
        ),
    ]

    histories, candidates = {}, []
    for name, epochs, lr, lr_factor, min_lr in phases:
        if epochs <= 0:
            log.info("Skipping %s (0 epochs)", name)
            continue
        if name == "phase2_finetune":
            unfreeze_top_stages(base_model, cfg.fine_tune_stages)
        compile_model(model, lr)
        log.info("%s: %d trainable weight tensors, lr=%g", name, len(model.trainable_weights), lr)

        ckpt = out_dir / f"{name}_best.keras"
        history = model.fit(
            pipelines["train"],
            validation_data=pipelines["val"],
            epochs=epochs,
            callbacks=_callbacks(ckpt, out_dir / f"history_{name}.csv", cfg, lr_factor, min_lr),
        )
        h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        histories[name] = h
        best_val = max(h["val_accuracy"])
        candidates.append((best_val, name, ckpt))
        log.info("%s best val accuracy: %.4f", name, best_val)

    if not candidates:
        raise ValueError("Both phases have 0 epochs; nothing to train.")

    save_json(histories, out_dir / "history.json")
    plot_history(histories, out_dir / "figures" / "training_curves.png")

    # Model selection on validation accuracy only.
    best_val, best_name, best_ckpt = max(candidates)
    final_path = out_dir / "model.keras"
    shutil.copyfile(best_ckpt, final_path)
    save_json(
        {"selected_phase": best_name, "val_accuracy": round(best_val, 4)},
        out_dir / "selection.json",
    )
    log.info("Selected %s (val accuracy %.4f) -> %s", best_name, best_val, final_path)

    if run_eval:
        test_paths, test_labels = split_arrays(split_df, ds.root, "test")
        final_model = keras.models.load_model(final_path)
        evaluate_model(final_model, test_paths, test_labels, ds.class_names, cfg, out_dir)

    return final_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset folder or NWPU.zip")
    parser.add_argument("--config", type=Path, help="YAML config (default: built-in defaults)")
    parser.add_argument("--output-dir", type=Path, help="Run folder (default: runs/<timestamp>)")
    parser.add_argument("--head-epochs", type=int, help="Override phase 1 epoch budget")
    parser.add_argument("--fine-tune-epochs", type=int, help="Override phase 2 epoch budget")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the 45-class / 700-per-class dataset checks (e.g. for a subset)",
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Do not evaluate on the test split"
    )
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(
        args.config,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
    )
    if args.no_verify:
        cfg.expected_classes = cfg.expected_per_class = None

    out_dir = args.output_dir or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    train(args.data_dir, cfg, out_dir, run_eval=not args.skip_eval)


if __name__ == "__main__":
    main()
