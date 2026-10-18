"""Publish the trained model and the Gradio demo to the Hugging Face Hub.

Log in once with `hf auth login`, then:

    # 1. Model: uploads model.keras, class_names.json and a generated model card
    python -m resisc45.hub model --model runs/<run>/model.keras --repo <hf-user>/resisc45-resnet50

    # 2. Demo: deploys demo/ as a Gradio Space that loads the model from step 1
    python -m resisc45.hub space --repo <hf-user>/aerial-scene-classifier \
        --model-repo <hf-user>/resisc45-resnet50
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .inference import Classifier
from .utils import load_json

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_URL = "https://github.com/esammostafa9-cloud/resisc45-scene-classification"

MODEL_CARD = """---
license: other
license_name: research-and-educational-use
library_name: keras
pipeline_tag: image-classification
tags:
  - remote-sensing
  - scene-classification
  - resnet50
  - transfer-learning
  - tensorflow
metrics:
  - accuracy
  - f1
---

# ResNet-50 fine-tuned on NWPU-RESISC45

Classifies aerial / satellite images into **45 land-use scene categories**.
Training code, evaluation and full results: {github}

## Results (held-out test set, 4,725 images)

| Metric | Score |
|---|---|
{metrics_rows}

## Usage

```python
# pip install "resisc45[api] @ git+{github}.git"
from resisc45.inference import Classifier

clf = Classifier.from_hub("{repo}")
pred = clf.predict_path("aerial.jpg", top_k=5)
print(pred.label, pred.confidence)
```

Input: any JPEG/PNG image. It is resized to 224×224 and fed as raw 0–255 RGB pixels, because
ResNet-50 preprocessing is built into the model.

## Training

- ImageNet-pretrained ResNet-50 + GAP → Dropout → Dense(256, L2) → BN → ReLU → Dropout → softmax
- Phase 1: frozen backbone, Adam 1e-3. Phase 2: `conv4` + `conv5` unfrozen (BatchNorm frozen),
  Adam 1e-5
- Augmentation: flips, rotation, zoom, translation, contrast, brightness
- Stratified 70 / 15 / 15 split with seed 42; the model was selected on validation accuracy

## Limitations

Look-alike classes are the main error source: palace vs church, dense vs medium residential,
railway vs railway station. The model is trained on top-down imagery only.

## License and intended use

The training code is MIT-licensed. These weights were trained on NWPU-RESISC45, whose authors
have not published a license, so the model is shared for **non-commercial research and
educational use** only. It is provided as is, without warranty of any kind.

## Dataset

G. Cheng, J. Han and X. Lu, "Remote Sensing Image Scene Classification: Benchmark and State of
the Art," *Proceedings of the IEEE*, 105(10), 2017. doi:10.1109/JPROC.2017.2675998
"""


def _metrics_rows(metrics_path: Path | None) -> str:
    if metrics_path is None or not metrics_path.exists():
        return "| Accuracy | see repository |"
    metrics = load_json(metrics_path)
    names = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro precision",
        "macro_recall": "Macro recall",
        "macro_f1": "Macro F1",
        "test_loss": "Test loss",
    }
    return "\n".join(f"| {label} | {metrics[k]} |" for k, label in names.items() if k in metrics)


def push_model(
    model_path: Path, repo: str, class_names: Path | None, metrics: Path | None, private: bool
) -> str:
    from huggingface_hub import HfApi

    names_path = class_names or model_path.parent / "class_names.json"
    # Fail before uploading anything if the model and class names don't belong together.
    clf = Classifier.from_path(model_path, names_path)
    print(f"Model OK: {len(clf.class_names)} classes, input {clf.img_size}x{clf.img_size}")

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(
            MODEL_CARD.format(github=GITHUB_URL, repo=repo, metrics_rows=_metrics_rows(metrics)),
            encoding="utf-8",
        )
        for local, remote in (
            (model_path, "model.keras"),
            (names_path, "class_names.json"),
            (card, "README.md"),
        ):
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=repo,
                commit_message=f"Upload {remote}",
            )
    url = f"https://huggingface.co/{repo}"
    print(f"Model published: {url}")
    return url


def push_space(repo: str, model_repo: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
    api.add_space_variable(repo, "RESISC45_HUB_REPO", model_repo)
    api.upload_folder(
        folder_path=str(REPO_ROOT / "demo"),
        repo_id=repo,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc", "flagged/*"],
        commit_message="Deploy demo",
    )
    url = f"https://huggingface.co/spaces/{repo}"
    print(f"Space deployed (it takes a few minutes to build): {url}")
    return url


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("model", help="Upload a trained model to a Hub model repo")
    m.add_argument("--model", required=True, type=Path, help="Trained model.keras")
    m.add_argument("--repo", required=True, help="e.g. <hf-user>/resisc45-resnet50")
    m.add_argument("--class-names", type=Path, help="Default: class_names.json beside the model")
    m.add_argument("--metrics", type=Path, default=REPO_ROOT / "results" / "test_metrics.json")
    m.add_argument("--private", action="store_true")

    s = sub.add_parser("space", help="Deploy demo/ as a Gradio Space")
    s.add_argument("--repo", required=True, help="e.g. <hf-user>/aerial-scene-classifier")
    s.add_argument("--model-repo", required=True, help="Model repo the Space should load")
    s.add_argument("--private", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "model":
        push_model(args.model, args.repo, args.class_names, args.metrics, args.private)
    else:
        push_space(args.repo, args.model_repo, args.private)


if __name__ == "__main__":
    main()
