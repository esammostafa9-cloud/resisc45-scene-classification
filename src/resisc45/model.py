"""ResNet-50 transfer-learning classifier."""

from __future__ import annotations

import keras
from keras import layers
from keras.regularizers import l2

from .config import Config


def build_augmentation(seed: int) -> keras.Sequential:
    """Train-time augmentation. Overhead imagery has no fixed orientation, so flips and
    rotations create realistic new views without changing the label. Inactive at inference."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.15, seed=seed),  # +/- ~54 degrees
            layers.RandomZoom(0.15, seed=seed),
            layers.RandomTranslation(0.10, 0.10, seed=seed),
            layers.RandomContrast(0.15, seed=seed),  # sensor / lighting variation
            layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
        ],
        name="data_augmentation",
    )


def build_model(num_classes: int, cfg: Config) -> tuple[keras.Model, keras.Model]:
    """Frozen ResNet-50 backbone + regularised classification head.

    Returns ``(model, base_model)``; the backbone is returned separately so it can be
    partially unfrozen for fine-tuning.
    """
    base_model = keras.applications.ResNet50(
        weights=cfg.pretrained_weights,
        include_top=False,
        input_shape=(cfg.img_size, cfg.img_size, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(cfg.img_size, cfg.img_size, 3), name="image")
    x = build_augmentation(cfg.seed)(inputs)  # raw 0-255 pixels, training only
    x = keras.applications.resnet50.preprocess_input(x)
    # training=False keeps BatchNorm in inference mode, even after unfreezing.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.dropout_pool)(x)
    x = layers.Dense(cfg.dense_units, use_bias=False, kernel_regularizer=l2(cfg.l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(cfg.dropout_head)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="resnet50_resisc45")
    return model, base_model


def unfreeze_top_stages(base_model: keras.Model, stages: list[str]) -> None:
    """Make the given residual stages (e.g. conv4, conv5) trainable.

    BatchNormalization layers stay frozen so their ImageNet statistics are not disturbed
    by the comparatively small dataset.
    """
    base_model.trainable = True
    prefixes = tuple(stages)
    for layer in base_model.layers:
        layer.trainable = layer.name.startswith(prefixes) and not isinstance(
            layer, layers.BatchNormalization
        )


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",  # integer labels + softmax output
        metrics=["accuracy"],
    )
