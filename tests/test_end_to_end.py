"""Train -> evaluate -> predict on the tiny synthetic dataset, through the real CLIs."""

import json

import pandas as pd
import pytest

from resisc45 import evaluate, predict, train


@pytest.mark.slow
def test_train_evaluate_predict(tiny_dataset, tiny_config, tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    tiny_config.to_yaml(cfg_path)
    run = tmp_path / "run"

    train.main(
        ["--data-dir", str(tiny_dataset), "--config", str(cfg_path), "--output-dir", str(run)]
    )

    for name in [
        "model.keras",
        "config.yaml",
        "class_names.json",
        "split.csv",
        "history.json",
        "selection.json",
        "test_metrics.json",
        "classification_report.csv",
        "test_predictions.csv",
        "figures/training_curves.png",
        "figures/confusion_matrix.png",
        "figures/sample_predictions.png",
        "figures/class_distribution.png",
    ]:
        assert (run / name).exists(), name

    metrics = json.loads((run / "test_metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(pd.read_csv(run / "test_predictions.csv")) == 6  # 20% of 30 images

    # Standalone evaluation reuses the saved split.
    evaluate.main(["--run-dir", str(run), "--data-dir", str(tiny_dataset)])

    image_dir = tiny_dataset / "NWPU-RESISC45" / "beach"
    out_csv = tmp_path / "pred.csv"
    predict.main(
        [
            "--model",
            str(run / "model.keras"),
            str(image_dir),
            "--top-k",
            "2",
            "--output",
            str(out_csv),
        ]
    )
    preds = pd.read_csv(out_csv)
    assert len(preds) == 10
    assert set(preds.columns) >= {"image", "top1", "top1_prob", "top2", "top2_prob"}
