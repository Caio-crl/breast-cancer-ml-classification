"""Avaliação final da Regressão Logística e interpretação na validação."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, recall_score
from sklearn.pipeline import Pipeline

from src.data_preparation import load_dataset, prepare_data, split_data
from src.model_training import calculate_metrics, create_models


SELECTED_MODEL = "Regressão Logística"


def train_selected_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Ajuste somente o modelo previamente selecionado, na configuração existente."""
    model = create_models()[SELECTED_MODEL]
    model.fit(X_train, y_train)
    return model


def evaluate_final_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    """Calcule as métricas finais exclusivamente no teste, sem novo ajuste."""
    return calculate_metrics(y_test, model.predict(X_test))


def coefficient_importance(model: Pipeline) -> pd.DataFrame:
    """Ordene os coeficientes padronizados por magnitude, preservando seus sinais."""
    names = model.feature_names_in_
    transformed_names = model.named_steps["preprocessing"].get_feature_names_out()
    if not np.array_equal(names, transformed_names):
        raise ValueError("O pré-processamento deve preservar as características originais.")
    coefficients = model.named_steps["classifier"].coef_[0]
    table = pd.DataFrame({
        "Coeficiente": coefficients,
        "Valor absoluto": np.abs(coefficients),
        "Sinal": np.where(coefficients > 0, "positivo",
                         np.where(coefficients < 0, "negativo", "zero")),
    }, index=pd.Index(names, name="Característica"))
    return table.sort_values("Valor absoluto", ascending=False, kind="stable")


def validation_permutation_importance(
    model: Pipeline, X_validation: pd.DataFrame, y_validation: pd.Series,
) -> pd.DataFrame:
    """Meça a queda de Recall por permutação na validação, sem reajustar o modelo."""
    scorer = make_scorer(recall_score, pos_label=1, zero_division=0)
    result = permutation_importance(
        model, X_validation, y_validation,
        scoring=scorer, random_state=42, n_repeats=10,
    )
    table = pd.DataFrame({
        "Importância média": result.importances_mean,
        "Desvio padrão": result.importances_std,
    }, index=pd.Index(X_validation.columns, name="Característica"))
    return table.sort_values("Importância média", ascending=False, kind="stable")


def display_results(
    metrics: dict, coefficients: pd.DataFrame, permutations: pd.DataFrame,
) -> None:
    """Apresente métricas de teste e dez características por técnica de interpretação."""
    print("\nAvaliação final no conjunto de teste (classe positiva: maligno=1):")
    for metric in ("Accuracy", "Precision", "Recall", "F1-score"):
        print(f"{metric}: {metrics[metric]:.4f}")
    print("\nMatriz de confusão: [[VN, FP], [FN, VP]]; ordem das classes [0, 1].")
    print("0 = benigno; 1 = maligno. Linhas reais e colunas previstas.")
    print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Real 0 (B)", "Real 1 (M)"],
        columns=["Previsto 0 (B)", "Previsto 1 (M)"],
    ).to_string())
    print("\nDez maiores coeficientes absolutos após padronização:")
    print(coefficients.head(10).to_string(float_format=lambda value: f"{value:.4f}"))
    print("Coeficientes positivos aumentam a pontuação (log-odds) para maligno;")
    print("negativos a reduzem, mantidas as demais características constantes.")
    print("\nDez maiores importâncias médias por permutação na validação:")
    print(permutations.head(10).to_string(float_format=lambda value: f"{value:.4f}"))
    print("Importância = queda média de Recall ao permutar a característica (10 repetições).")
    print("Valores negativos indicam aumento de Recall após a permutação.")
    print("\nLimitações: características correlacionadas dificultam a interpretação individual")
    print("dos coeficientes e podem reduzir a importância por permutação.")
    print("As associações não estabelecem causalidade. Uma única divisão de uma base pública")
    print("não demonstra generalização clínica. A seleção e os parâmetros permanecem fixos.")


def main() -> None:
    features, target = prepare_data(load_dataset())
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(
        features, target,
    )
    print(f"Modelo selecionado previamente na validação: {SELECTED_MODEL}")
    print(f"Treinamento: {len(X_train)} | Validação (permutação): {len(X_validation)}")
    print(f"Teste (avaliação final): {len(X_test)} | Características: {features.shape[1]}")
    model = train_selected_model(X_train, y_train)
    coefficients = coefficient_importance(model)
    permutations = validation_permutation_importance(model, X_validation, y_validation)
    metrics = evaluate_final_model(model, X_test, y_test)
    display_results(metrics, coefficients, permutations)


if __name__ == "__main__":
    main()
