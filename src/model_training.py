"""Treinamento inicial e comparação de classificadores no conjunto de validação."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data_preparation import (
    create_preprocessing_pipeline,
    load_dataset,
    prepare_data,
    split_data,
)


def create_models() -> dict[str, Pipeline]:
    """Crie classificadores com pré-processamentos independentes e sem ajuste."""
    classifiers = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    return {
        name: Pipeline([
            ("preprocessing", create_preprocessing_pipeline()),
            ("classifier", classifier),
        ])
        for name, classifier in classifiers.items()
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Ajuste cada pipeline exclusivamente com os dados de treinamento."""
    models = create_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calcule métricas binárias com maligno=1 e matriz na ordem [0, 1].

    A matriz segue [[VN, FP], [FN, VP]]: linhas reais, colunas previstas.
    Métricas sem denominador válido recebem zero.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def evaluate_models(
    models: dict[str, Pipeline], X_validation: pd.DataFrame, y_validation: pd.Series,
) -> dict[str, dict]:
    """Preveja e avalie na validação, sem reajustar os pipelines."""
    return {
        name: calculate_metrics(y_validation, model.predict(X_validation))
        for name, model in models.items()
    }


def display_results(results: dict[str, dict]) -> None:
    """Apresente a comparação de validação e as matrizes de confusão."""
    comparison = pd.DataFrame.from_dict({
        name: {metric: value for metric, value in scores.items()
               if metric != "confusion_matrix"}
        for name, scores in results.items()
    }, orient="index")
    comparison.index.name = "Modelo"
    print("\nComparação no conjunto de validação:")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nPrecision, Recall e F1-score consideram maligno (1) como classe positiva.")
    print("O Recall indica a proporção de casos malignos identificados.")
    print("Matrizes: linhas = classe real; colunas = classe prevista; ordem [0, 1].")
    print("0 = benigno; 1 = maligno. Estrutura: [[VN, FP], [FN, VP]].")
    for name, scores in results.items():
        matrix = pd.DataFrame(
            scores["confusion_matrix"],
            index=["Real 0 (B)", "Real 1 (M)"],
            columns=["Previsto 0 (B)", "Previsto 1 (M)"],
        )
        print(f"\n{name}:\n{matrix.to_string()}")


def main() -> None:
    features, target = prepare_data(load_dataset())
    X_train, X_validation, _, y_train, y_validation, _ = split_data(features, target)
    print(f"Treinamento: {X_train.shape} | Validação: {X_validation.shape}")
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_validation, y_validation)
    display_results(results)
    print("\nO conjunto de teste permanece reservado para a avaliação final.")


if __name__ == "__main__":
    main()
