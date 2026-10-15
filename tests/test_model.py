import numpy as np
from keras import layers

from resisc45.config import Config
from resisc45.model import build_model, unfreeze_top_stages


def test_model_output_is_a_probability_distribution(tiny_config):
    model, base = build_model(num_classes=5, cfg=tiny_config)
    assert not base.trainable

    out = model.predict(np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32"), verbose=0)
    assert out.shape == (2, 5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)


def test_unfreeze_only_requested_stages_and_keeps_batchnorm_frozen(tiny_config):
    _, base = build_model(num_classes=5, cfg=tiny_config)
    unfreeze_top_stages(base, ["conv4", "conv5"])

    trainable = [layer for layer in base.layers if layer.trainable]
    assert trainable, "expected some layers to be unfrozen"
    for layer in trainable:
        assert layer.name.startswith(("conv4", "conv5"))
        assert not isinstance(layer, layers.BatchNormalization)
    early = [layer for layer in base.layers if layer.name.startswith(("conv1", "conv2", "conv3"))]
    assert not any(layer.trainable for layer in early)


def test_config_yaml_round_trip(tmp_path):
    cfg = Config(head_epochs=3, fine_tune_stages=["conv5"])
    cfg.to_yaml(tmp_path / "c.yaml")
    assert Config.from_yaml(tmp_path / "c.yaml") == cfg
    assert Config.from_yaml(tmp_path / "c.yaml", head_epochs=7, batch_size=None).head_epochs == 7


def test_default_yaml_matches_dataclass_defaults():
    from pathlib import Path

    default_yaml = Path(__file__).parents[1] / "configs" / "default.yaml"
    assert Config.from_yaml(default_yaml) == Config()
