"""Линейные модели для лабораторной работы № 3."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _standardize_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Масштабирует признаки по среднему и отклонению train-части."""
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale == 0] = 1

    train_scaled = (train_features - mean) / scale
    test_scaled = (test_features - mean) / scale
    return train_scaled, test_scaled, mean, scale


def _restore_weights(
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Возвращает коэффициенты к исходному масштабу признаков."""
    original_weights = weights / scale
    original_bias = bias - float(np.dot(mean, original_weights))
    return original_weights, original_bias


def _classification_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> ClassificationMetrics:
    """Считает метрики бинарной классификации."""
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучает линейную регрессию пакетным градиентным спуском."""
    train_features = np.asarray(data.train_features, dtype=float)
    test_features = np.asarray(data.test_features, dtype=float)
    train_target = np.asarray(data.train_target, dtype=float).reshape(-1)

    train_scaled, test_scaled, feature_mean, feature_scale = (
        _standardize_features(train_features, test_features)
    )
    target_mean = train_target.mean()
    target_scale = train_target.std()
    if target_scale == 0:
        target_scale = 1
    target_scaled = (train_target - target_mean) / target_scale

    weights = np.zeros(train_scaled.shape[1], dtype=float)
    bias = 0.0
    loss_history = []
    row_count = len(train_target)

    for _ in range(data.epochs):
        predictions = train_scaled @ weights + bias
        errors = predictions - target_scaled
        weights -= data.learning_rate * (2 / row_count) * (train_scaled.T @ errors)
        bias -= data.learning_rate * 2 * errors.mean()

        scaled_predictions = train_scaled @ weights + bias
        real_predictions = scaled_predictions * target_scale + target_mean
        real_errors = real_predictions - train_target
        loss_history.append(float(np.mean(real_errors ** 2)))

    weights *= target_scale
    bias = bias * target_scale + target_mean
    weights, bias = _restore_weights(
        weights, bias, feature_mean, feature_scale
    )
    predictions = test_scaled @ (weights * feature_scale) + (
        bias + np.dot(feature_mean, weights)
    )
    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучает бинарную логистическую регрессию градиентным спуском."""
    train_features = np.asarray(data.train_features, dtype=float)
    test_features = np.asarray(data.test_features, dtype=float)
    train_target = np.asarray(data.train_target, dtype=float).reshape(-1)

    train_scaled, test_scaled, feature_mean, feature_scale = (
        _standardize_features(train_features, test_features)
    )
    weights = np.zeros(train_scaled.shape[1], dtype=float)
    bias = 0.0
    loss_history = []
    row_count = len(train_target)

    for _ in range(data.epochs):
        scores = train_scaled @ weights + bias
        probabilities = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
        errors = probabilities - train_target
        weights -= data.learning_rate * (train_scaled.T @ errors) / row_count
        bias -= data.learning_rate * errors.mean()

        updated_scores = train_scaled @ weights + bias
        updated_probabilities = 1 / (
            1 + np.exp(-np.clip(updated_scores, -500, 500))
        )
        clipped = np.clip(updated_probabilities, 1e-15, 1 - 1e-15)
        loss = -np.mean(
            train_target * np.log(clipped)
            + (1 - train_target) * np.log(1 - clipped)
        )
        loss_history.append(float(loss))

    test_scores = test_scaled @ weights + bias
    test_probabilities = 1 / (1 + np.exp(-np.clip(test_scores, -500, 500)))
    predictions = (test_probabilities >= 0.5).astype(int)
    weights, bias = _restore_weights(
        weights, bias, feature_mean, feature_scale
    )
    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=predictions,
    )


def compare_classical_models(
    data: ClassificationInput,
) -> BaselineComparisonResult:
    """Сравнивает три модели на переданном train/test-разбиении."""
    train_features = np.asarray(data.train_features, dtype=float)
    test_features = np.asarray(data.test_features, dtype=float)
    train_target = np.asarray(data.train_target, dtype=int).reshape(-1)
    test_target = np.asarray(data.test_target, dtype=int).reshape(-1)

    own_result = train_logistic_classifier(data)
    own_metrics = _classification_metrics(
        test_target, own_result.probabilities, own_result.predictions
    )

    train_scaled, test_scaled, _, _ = _standardize_features(
        train_features, test_features
    )
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(train_scaled, train_target)
    logistic_probabilities = logistic_model.predict_proba(test_scaled)[:, 1]
    logistic_predictions = logistic_model.predict(test_scaled)

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(train_features, train_target)
    tree_probabilities = tree_model.predict_proba(test_features)[:, 1]
    tree_predictions = tree_model.predict(test_features)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(train_scaled, train_target)
    knn_probabilities = knn_model.predict_proba(test_scaled)[:, 1]
    knn_predictions = knn_model.predict(test_scaled)

    return BaselineComparisonResult(
        logistic_regression=_classification_metrics(
            test_target, logistic_probabilities, logistic_predictions
        ),
        decision_tree=_classification_metrics(
            test_target, tree_probabilities, tree_predictions
        ),
        knn=_classification_metrics(
            test_target, knn_probabilities, knn_predictions
        ),
    )
