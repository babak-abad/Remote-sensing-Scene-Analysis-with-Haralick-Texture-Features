"""Content-based retrieval: find the best-matched images by feature distance."""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import config


def standardize(feature_matrix):
    """Z-score the feature matrix so no single descriptor dominates the distance."""
    return StandardScaler().fit_transform(feature_matrix)


def find_best_match(query_index, feature_matrix, metric=None, top_k=None):
    """Return the top-k nearest neighbours of one image in feature space.

    The query itself is excluded. Results are (index, distance) pairs sorted by
    increasing distance under the chosen metric.
    """
    metric = metric or config.retrieval_metric
    top_k = top_k or config.retrieval_top_k

    standardized = standardize(feature_matrix)
    query = standardized[query_index : query_index + 1]
    distances = cdist(query, standardized, metric=metric)[0]
    distances[query_index] = np.inf

    order = np.argsort(distances)[:top_k]
    return [(int(i), float(distances[i])) for i in order]


def retrieval_precision_at_k(feature_matrix, labels, metric=None, top_k=None):
    """Mean fraction of the top-k neighbours that share the query's label."""
    metric = metric or config.retrieval_metric
    top_k = top_k or config.retrieval_top_k

    standardized = standardize(feature_matrix)
    distance_matrix = cdist(standardized, standardized, metric=metric)
    np.fill_diagonal(distance_matrix, np.inf)

    labels = np.asarray(labels)
    hits = []
    for i in range(len(labels)):
        neighbours = np.argsort(distance_matrix[i])[:top_k]
        hits.append(np.mean(labels[neighbours] == labels[i]))
    return float(np.mean(hits))
