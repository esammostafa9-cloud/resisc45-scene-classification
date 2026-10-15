"""Experiment configuration: one dataclass, loadable from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    seed: int = 42

    # Data
    img_size: int = 224
    batch_size: int = 32
    val_size: float = 0.15
    test_size: float = 0.15
    expected_classes: int | None = 45
    expected_per_class: int | None = 700

    # Model
    pretrained_weights: str | None = "imagenet"
    dense_units: int = 256
    dropout_pool: float = 0.30
    dropout_head: float = 0.20
    l2: float = 1e-4

    # Phase 1: frozen backbone
    head_epochs: int = 25
    head_lr: float = 1e-3
    head_lr_factor: float = 0.5
    head_min_lr: float = 1e-6

    # Phase 2: fine-tuning
    fine_tune_epochs: int = 15
    fine_tune_lr: float = 1e-5
    fine_tune_lr_factor: float = 0.3
    fine_tune_min_lr: float = 1e-7
    fine_tune_stages: list[str] = field(default_factory=lambda: ["conv4", "conv5"])

    # Callbacks
    early_stopping_patience: int = 5
    lr_patience: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides) -> Config:
        """Load a config file (if given), then apply keyword overrides that are not None."""
        values = {}
        if path is not None:
            with open(path, encoding="utf-8") as f:
                values = yaml.safe_load(f) or {}
        values.update({k: v for k, v in overrides.items() if v is not None})

        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
