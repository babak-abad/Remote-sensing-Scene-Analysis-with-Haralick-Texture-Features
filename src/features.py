"""Haralick / GLCM texture-feature extraction with an on-disk cache.

The grey-level co-occurrence matrix is built at the four canonical angles
(0, 45, 90, 135 degrees); the full Haralick descriptor set is derived from each
angle and then aggregated into one vector per image (see config.feature_aggregation).
"""

import hashlib
import json

import numpy as np
from tqdm import tqdm

import config
from src import dataset

# The 13 stable Haralick descriptors, in mahotas' order.
haralick_names = [
    "Angular Second Moment",
    "Contrast",
    "Correlation",
    "Variance",
    "Inverse Difference Moment",
    "Sum Average",
    "Sum Variance",
    "Sum Entropy",
    "Entropy",
    "Difference Variance",
    "Difference Entropy",
    "Info. Measure of Correlation 1",
    "Info. Measure of Correlation 2",
]

# The 6 GLCM properties skimage exposes (fallback backend).
skimage_prop_names = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]


def _per_angle_matrix(gray):
    """Return a (n_angles, n_descriptors) texture matrix for one grayscale image."""
    if config.feature_backend == "mahotas":
        import mahotas.features as mahotas_features

        return mahotas_features.haralick(gray, distance=config.glcm_distance)

    from skimage.feature import graycomatrix, graycoprops

    glcm = graycomatrix(
        gray,
        distances=[config.glcm_distance],
        angles=config.glcm_angles,
        levels=config.glcm_levels,
        symmetric=True,
        normed=True,
    )
    props = [graycoprops(glcm, name)[0] for name in skimage_prop_names]
    return np.asarray(props, dtype=np.float64).T  # (n_angles, n_props)


def _aggregate(per_angle):
    """Collapse the per-angle matrix into one feature vector per config.feature_aggregation."""
    if config.feature_aggregation == "mean":
        return per_angle.mean(axis=0)
    if config.feature_aggregation == "all_angles":
        return per_angle.ravel()
    if config.feature_aggregation == "mean_ptp":
        return np.concatenate([per_angle.mean(axis=0), np.ptp(per_angle, axis=0)])
    raise ValueError(f"unknown feature_aggregation: {config.feature_aggregation!r}")


def extract_features(gray):
    """Extract one aggregated Haralick feature vector from a grayscale image."""
    return _aggregate(_per_angle_matrix(gray))


def feature_labels():
    """Human-readable names for each dimension of the aggregated feature vector."""
    base = haralick_names if config.feature_backend == "mahotas" else skimage_prop_names
    if config.feature_aggregation == "mean":
        return list(base)
    if config.feature_aggregation == "all_angles":
        return [f"{n} @{int(np.degrees(a))}deg" for a in config.glcm_angles for n in base]
    return [f"{n} (mean)" for n in base] + [f"{n} (range)" for n in base]


def _signature():
    """A hash of every parameter that affects the extracted matrix, for cache keying."""
    payload = dict(
        classes=config.selected_classes,
        per_class=config.images_per_class,
        seed=config.random_seed,
        backend=config.feature_backend,
        distance=config.glcm_distance,
        angles=[round(a, 6) for a in config.glcm_angles],
        aggregation=config.feature_aggregation,
    )
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return digest, payload


def build_feature_matrix(force=False):
    """Return (X, y, paths, names), computing and caching the matrix on first run.

    The cache short-circuits the expensive extraction step: a run whose parameters
    match an existing .npz reloads it instead of recomputing.
    """
    digest, payload = _signature()
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.cache_dir / f"features_{digest}.npz"

    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["paths"], feature_labels()

    paths, labels = dataset.list_samples()
    features = [
        extract_features(dataset.load_gray(path))
        for path in tqdm(paths, desc="Haralick features", unit="img")
    ]
    feature_matrix = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels)
    path_array = np.asarray(paths)

    np.savez_compressed(
        cache_path,
        X=feature_matrix,
        y=label_array,
        paths=path_array,
        signature=json.dumps(payload),
    )
    return feature_matrix, label_array, path_array, feature_labels()
