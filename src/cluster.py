"""Unsupervised grouping of the texture features: K-means and OPTICS."""

import numpy as np
from sklearn.cluster import OPTICS, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import config


def _scores(standardized, true_labels, cluster_labels):
    """External agreement (ARI, NMI) and internal cohesion (silhouette)."""
    valid = cluster_labels != -1  # OPTICS marks noise as -1
    result = dict(
        ari=float(adjusted_rand_score(true_labels, cluster_labels)),
        nmi=float(normalized_mutual_info_score(true_labels, cluster_labels)),
        n_clusters=int(len(set(cluster_labels[valid]))),
        noise_fraction=float(np.mean(cluster_labels == -1)),
    )
    if valid.sum() > len(set(cluster_labels[valid])) > 1:
        result["silhouette"] = float(silhouette_score(standardized[valid], cluster_labels[valid]))
    else:
        result["silhouette"] = None
    return result


def run_clustering(feature_matrix, labels):
    """Cluster with K-means and OPTICS; return labels, scores and a 2-D PCA embedding."""
    standardized = StandardScaler().fit_transform(feature_matrix)
    true_labels = np.asarray(labels)

    kmeans_labels = KMeans(
        n_clusters=config.kmeans_k,
        n_init=config.kmeans_n_init,
        random_state=config.random_seed,
    ).fit_predict(standardized)

    optics = OPTICS(
        min_samples=config.optics_min_samples,
        cluster_method=config.optics_cluster_method,
        eps=config.optics_eps,
    ).fit(standardized)
    optics_labels = optics.labels_

    embedding = PCA(n_components=2, random_state=config.random_seed).fit_transform(standardized)

    return {
        "embedding": embedding,
        "true_labels": true_labels,
        "reachability": optics.reachability_[optics.ordering_],
        "kmeans": {"labels": kmeans_labels, "scores": _scores(standardized, true_labels, kmeans_labels)},
        "optics": {"labels": optics_labels, "scores": _scores(standardized, true_labels, optics_labels)},
    }
