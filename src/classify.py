"""Supervised classification of the texture features: RBF-SVM and a simple ANN."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def split(feature_matrix, labels):
    """Stratified train/test split with the configured seed and test size."""
    return train_test_split(
        feature_matrix,
        labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )


def _evaluate(model, x_train, x_test, y_train, y_test, class_order):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return dict(
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        confusion=confusion_matrix(y_test, predictions, labels=class_order).tolist(),
    )


def run_classifiers(feature_matrix, labels):
    """Train the SVM and the MLP; return per-model metrics and the class order."""
    x_train, x_test, y_train, y_test = split(feature_matrix, labels)
    class_order = sorted(np.unique(labels).tolist())

    svm = make_pipeline(StandardScaler(), SVC(**config.svm_params))
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(random_state=config.random_seed, **config.mlp_params),
    )

    return {
        "class_order": class_order,
        "svm": _evaluate(svm, x_train, x_test, y_train, y_test, class_order),
        "mlp": _evaluate(mlp, x_train, x_test, y_train, y_test, class_order),
    }
