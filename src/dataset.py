"""Enumerate and load the NWPU-RESISC45 subset defined in config."""

import random

import numpy as np
from PIL import Image

import config


def list_samples():
    """Return (paths, labels) for a seeded sample of the configured classes.

    Each class contributes at most config.images_per_class images, chosen
    deterministically so every run sees the same subset.
    """
    rng = random.Random(config.random_seed)
    paths, labels = [], []
    for label in config.selected_classes:
        class_dir = config.dataset_root / label
        files = sorted(class_dir.glob(f"{label}_*.jpg"))
        if not files:
            raise FileNotFoundError(f"no images found for class '{label}' in {class_dir}")
        chosen = files if len(files) <= config.images_per_class else rng.sample(
            files, k=config.images_per_class
        )
        for path in sorted(chosen):
            paths.append(str(path))
            labels.append(label)
    return paths, labels


def load_gray(path):
    """Load an image as an 8-bit grayscale numpy array."""
    with Image.open(path) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.uint8)


def load_rgb(path):
    """Load an image as an RGB numpy array (for figure montages)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
