"""Shared model definition for FER2013 facial expression recognition.

Defining the architecture in code (instead of relying on model_from_json)
keeps the model loadable across Keras versions: the original JSON file was
written by Keras 2.12 and cannot be parsed by Keras 3 (TensorFlow >= 2.16).
"""
import json
import os

import numpy as np
import keras
from keras import layers

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
IMG_SIZE = 48


def build_model(num_classes: int = len(EMOTIONS)) -> keras.Model:
    """Baseline CNN, layer-for-layer identical to the original repository.

    5 conv blocks (Conv -> BatchNorm -> ReLU) with mixed 'valid'/'same'
    padding, Dense(250) + Dropout(0.5), softmax over 7 emotions.
    Layer names match the original so legacy fer.h5 weights still load.
    """
    def conv_block(x, filters, padding, idx):
        sfx = "" if idx == 0 else f"_{idx}"
        x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding=padding,
                          name=f"conv2d{sfx}")(x)
        x = layers.BatchNormalization(axis=3, name=f"batch_normalization{sfx}")(x)
        return layers.Activation("relu", name=f"activation{sfx}")(x)

    inp = keras.Input((IMG_SIZE, IMG_SIZE, 1), name="input_1")
    x = conv_block(inp, 32, "valid", 0)
    x = layers.Dropout(0.25, name="dropout")(x)
    x = conv_block(x, 64, "same", 1)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)
    x = conv_block(x, 64, "valid", 2)
    x = layers.Dropout(0.25, name="dropout_1")(x)
    x = conv_block(x, 128, "same", 3)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)
    x = conv_block(x, 128, "valid", 4)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d_2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(250, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout_2")(x)
    out = layers.Dense(num_classes, activation="softmax", name="dense_1")(x)
    return keras.Model(inp, out, name="fer_cnn")


def load_trained_model(path: str | None = None) -> tuple[keras.Model, str]:
    """Load the best available model.

    Order: explicit path -> models/v1/fer_model.keras (retrained, leak-free)
    -> models/v1/fer.weights.h5 -> models/baseline/fer.h5 (original weights; leaky protocol).
    Weight files that don't match the baseline architecture are skipped.
    Returns (model, path).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    names = ["models/v1/fer_model.keras", "models/v1/fer.weights.h5", "models/baseline/fer.h5"]
    candidates = [path] if path else [os.path.join(here, n) for n in names]
    errors = []
    for p in candidates:
        if not p or not os.path.exists(p):
            continue
        try:
            if p.endswith(".keras"):
                return keras.models.load_model(p, compile=False), p
            model = build_model()
            model.load_weights(p)
            return model, p
        except Exception as e:  # incompatible / corrupt file: try the next one
            errors.append(f"{p}: {type(e).__name__}: {e}")
            if path:
                raise
    raise FileNotFoundError(
        "No usable model file found. Tried: " + ", ".join(c for c in candidates if c)
        + ("\n" + "\n".join(errors) if errors else "")
    )


def load_prior_correction(model_path: str) -> np.ndarray | None:
    """Per-class multipliers that undo the class-weighted training bias.

    A model trained with class weights w_c over-predicts rare classes (e.g. disgust).
    Multiplying its probabilities by w_c^-a and renormalising restores realistic
    class frequencies. The vector (with `a` chosen on the validation set) is stored
    as "inference_prior_correction" in metrics.json next to the model.
    """
    if not model_path.endswith(".keras"):
        return None  # original weights were not trained with class weights
    mpath = os.path.join(os.path.dirname(os.path.abspath(model_path)), "metrics.json")
    try:
        with open(mpath) as f:
            corr = json.load(f).get("inference_prior_correction")
    except (OSError, ValueError):
        return None
    if corr is None or len(corr) != len(EMOTIONS):
        return None
    return np.asarray(corr, dtype="float32")


def apply_correction(probs: np.ndarray, corr: np.ndarray | None) -> np.ndarray:
    if corr is None or len(probs) == 0:
        return probs
    q = probs * corr
    return q / q.sum(axis=1, keepdims=True)
