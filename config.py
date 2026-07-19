"""Central settings for the NWPU-RESISC45 Haralick texture pipeline.

Every tunable value the project uses lives here so behaviour changes in one place.
No credentials are stored; this is a plain-script/library project, so a settings
module is used rather than a .env file.
"""

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
project_root = Path(__file__).resolve().parent
dataset_root = project_root / "NWPU-RESISC45"

assets_dir = project_root / "assets"
cache_dir = assets_dir / "cache"
outputs_dir = assets_dir / "outputs"
diagrams_dir = assets_dir / "diagrams"

# Joomla media folder the exported web images map to (used only by the article build).
article_media_slug = "remote-sensing-haralick"

# --------------------------------------------------------------------------- #
# Dataset subset (curated, texturally-distinct classes)
# --------------------------------------------------------------------------- #
selected_classes = [
    "beach",
    "chaparral",
    "dense_residential",
    "forest",
    "freeway",
    "harbor",
    "mountain",
    "parking_lot",
]
images_per_class = 150
random_seed = 42

# --------------------------------------------------------------------------- #
# Haralick / GLCM feature extraction
# --------------------------------------------------------------------------- #
# "mahotas" = full 13 Haralick descriptors per direction (canonical).
# "skimage" = graycomatrix + graycoprops (6-property GLCM subset) fallback.
feature_backend = "mahotas"

# Co-occurrence offset in pixels and the four canonical directions (radians).
glcm_distance = 1
glcm_angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
glcm_levels = 256

# How the per-angle descriptors collapse into one vector:
#   "mean_ptp"  -> mean across angles ++ per-angle range   (rotation-invariant + directional)
#   "all_angles"-> every angle kept, concatenated
#   "mean"      -> mean across angles only
feature_aggregation = "mean_ptp"

# --------------------------------------------------------------------------- #
# Retrieval (best-match search)
# --------------------------------------------------------------------------- #
retrieval_metric = "euclidean"  # one of: euclidean, cityblock, cosine
retrieval_top_k = 5

# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
test_size = 0.3

svm_params = dict(kernel="rbf", C=10.0, gamma="scale")
mlp_params = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    alpha=1e-3,
    max_iter=800,
    early_stopping=True,
)

# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
kmeans_k = len(selected_classes)
kmeans_n_init = 10

# OPTICS builds a density-based reachability ordering; clusters are then extracted
# from it. The dbscan method cuts the ordering at a reachability threshold (eps);
# points below no core stay unlabelled as noise (-1).
optics_min_samples = 8
optics_cluster_method = "dbscan"
optics_eps = 1.6

# --------------------------------------------------------------------------- #
# Figure export (PNG master -> web .jpg). Width in px, quality per rule 5.8/5.9.
# --------------------------------------------------------------------------- #
figure_export = {
    "class_montage": dict(width=900, quality=25),
    "glcm_angles": dict(width=900, quality=25),
    "glcm_construction": dict(width=1000, quality=25),
    "pipeline": dict(width=1000, quality=25),
    "retrieval": dict(width=1000, quality=25),
    "feature_signature": dict(width=800, quality=25),
    "confusion_svm": dict(width=700, quality=25),
    "confusion_mlp": dict(width=700, quality=25),
    "learning_curve": dict(width=700, quality=25),
    "cluster_kmeans": dict(width=700, quality=25),
    "cluster_optics": dict(width=700, quality=25),
    "glcm_skimage_demo": dict(width=800, quality=25),
}
