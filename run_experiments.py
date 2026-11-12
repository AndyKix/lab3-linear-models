"""Подготовка данных и повторный расчёт результатов лабораторной."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import compare_classical_models, train_linear_regression
from linear_models import train_logistic_classifier


DATA_DIR = Path(__file__).parent
RANDOM_STATE = 42


def get_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Собирает основные метрики бинарной классификации."""
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def main() -> None:
    """Обучает модели и сохраняет измеренные метрики в JSON."""
    student_data = pd.read_csv(DATA_DIR / "Student_Performance.csv")
    student_data["Extracurricular Activities"] = (
        student_data["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    )

    regression_features = student_data.drop(columns="Performance Index").to_numpy()
    regression_target = student_data["Performance Index"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        regression_features,
        regression_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    regression_input = RegressionInput(
        train_features=x_train,
        train_target=y_train,
        test_features=x_test,
        test_target=y_test,
        learning_rate=0.05,
        epochs=800,
    )
    regression_result = train_linear_regression(regression_input)
    regression_mse = mean_squared_error(y_test, regression_result.predictions)

    credit_data = pd.read_csv(DATA_DIR / "german.csv", sep=";")
    classification_target = credit_data.iloc[:, 0].to_numpy()
    classification_features = credit_data.iloc[:, 1:].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        classification_features,
        classification_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=classification_target,
    )
    classification_input = ClassificationInput(
        train_features=x_train,
        train_target=y_train,
        test_features=x_test,
        test_target=y_test,
        learning_rate=0.1,
        epochs=1000,
    )

    own_result = train_logistic_classifier(classification_input)
    own_metrics = get_metrics(
        y_test, own_result.probabilities, own_result.predictions
    )
    comparison = compare_classical_models(classification_input)

    results = {
        "linear_regression": {"test_mse": float(regression_mse)},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": comparison.logistic_regression.__dict__,
            "decision_tree": comparison.decision_tree.__dict__,
            "knn": comparison.knn.__dict__,
        },
    }

    output_path = DATA_DIR / "experiment_results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
