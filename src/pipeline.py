"""End-to-end driver: build features, then retrieve, classify and cluster.

Run from the project root inside the virtual environment:

    python -m src.pipeline
"""

import json

import numpy as np

import config
from src import classify, cluster, features, retrieval


def main():
    feature_matrix, labels, paths, names = features.build_feature_matrix()
    print(f"Features: {feature_matrix.shape[0]} images x {feature_matrix.shape[1]} dims "
          f"({config.feature_backend}/{config.feature_aggregation})")

    precision = retrieval.retrieval_precision_at_k(feature_matrix, labels)
    print(f"Retrieval precision@{config.retrieval_top_k} ({config.retrieval_metric}): {precision:.3f}")

    classification = classify.run_classifiers(feature_matrix, labels)
    print(f"SVM  accuracy={classification['svm']['accuracy']:.3f} "
          f"macro-F1={classification['svm']['macro_f1']:.3f}")
    print(f"MLP  accuracy={classification['mlp']['accuracy']:.3f} "
          f"macro-F1={classification['mlp']['macro_f1']:.3f}")

    clustering = cluster.run_clustering(feature_matrix, labels)
    for name in ("kmeans", "optics"):
        scores = clustering[name]["scores"]
        print(f"{name.upper():7s} ARI={scores['ari']:.3f} NMI={scores['nmi']:.3f} "
              f"silhouette={scores['silhouette']} clusters={scores['n_clusters']} "
              f"noise={scores['noise_fraction']:.2f}")

    summary = {
        "config": {
            "classes": config.selected_classes,
            "images_per_class": config.images_per_class,
            "feature_backend": config.feature_backend,
            "feature_aggregation": config.feature_aggregation,
            "feature_dim": int(feature_matrix.shape[1]),
            "n_images": int(feature_matrix.shape[0]),
            "retrieval_metric": config.retrieval_metric,
            "top_k": config.retrieval_top_k,
        },
        "retrieval_precision_at_k": precision,
        "classification": {
            "class_order": classification["class_order"],
            "svm": {k: classification["svm"][k] for k in ("accuracy", "macro_f1", "confusion")},
            "mlp": {k: classification["mlp"][k] for k in ("accuracy", "macro_f1", "confusion")},
        },
        "clustering": {
            "kmeans": clustering["kmeans"]["scores"],
            "optics": clustering["optics"]["scores"],
        },
    }

    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.outputs_dir / "results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
