"""Geração reproduzível dos gráficos de validação, teste e interpretabilidade."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_preparation import load_dataset, prepare_data, split_data
from src.model_evaluation import (
    SELECTED_MODEL,
    coefficient_importance,
    evaluate_final_model,
    validation_permutation_importance,
)
from src.model_training import display_results, evaluate_models, train_models


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def save_figure(figure: Figure, path: Path) -> None:
    """Salve em 300 dpi com renderização Agg, sem depender de interface gráfica."""
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Gráfico salvo: {path}")


def plot_validation_comparison(results: dict[str, dict]) -> Figure:
    """Compare as quatro métricas calculadas exclusivamente na validação."""
    metrics = ("Accuracy", "Precision", "Recall", "F1-score")
    figure = Figure(figsize=(10, 6), layout="constrained")
    axis = figure.subplots()
    positions = np.arange(len(metrics))
    width = 0.35
    for i, (name, scores) in enumerate(results.items()):
        values = [scores[metric] for metric in metrics]
        bars = axis.bar(
            positions + (i - 0.5) * width, values, width,
            label=name, color=("#32648e", "#d48436")[i],
        )
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=9)
    axis.set_xticks(positions, metrics)
    axis.set_ylim(0, 1.14)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.set_ylabel("Valor da métrica")
    axis.set_xlabel("Métrica (classe positiva: maligno = 1)")
    axis.set_title("Comparação dos modelos no conjunto de validação", pad=12)
    axis.legend(loc="upper center", ncols=2, frameon=False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    return figure


def plot_confusion_matrices(results: dict[str, dict], title: str) -> Figure:
    """Mostre contagens na ordem [benigno=0, maligno=1] com escala comum."""
    figure = Figure(figsize=(6 * len(results), 5), layout="constrained")
    axes = figure.subplots(1, len(results), squeeze=False)[0]
    maximum = max(scores["confusion_matrix"].max() for scores in results.values())
    labels = ["Benigno (0)", "Maligno (1)"]
    for axis, (name, scores) in zip(axes, results.items()):
        sns.heatmap(
            scores["confusion_matrix"], ax=axis, annot=True, fmt="d",
            cmap="Blues", vmin=0, vmax=maximum, cbar=False, square=True,
            xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 16},
        )
        axis.set_title(name, pad=12)
        axis.set_xlabel("Diagnóstico previsto")
        axis.set_ylabel("Diagnóstico real")
        axis.tick_params(axis="both", rotation=0)
    figure.suptitle(title, fontsize=14)
    return figure


def plot_logistic_coefficients(coefficients: pd.DataFrame) -> Figure:
    """Exiba os dez maiores coeficientes absolutos, mantendo o sinal original."""
    top = coefficients.head(10)
    values = top["Coeficiente"]
    figure = Figure(figsize=(11, 7), layout="constrained")
    axis = figure.subplots()
    bars = axis.barh(
        top.index, values, color=["#32648e" if value >= 0 else "#d48436" for value in values],
    )
    axis.bar_label(bars, labels=[f"{value:+.4f}" for value in values], padding=4, fontsize=9)
    axis.invert_yaxis()
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.margins(x=0.18)
    axis.set_xlabel("Coeficiente após padronização (sinal preservado)")
    axis.set_ylabel("Característica")
    axis.set_title("Regressão Logística: dez maiores coeficientes absolutos", pad=12)
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=0.2)
    return figure


def plot_permutation_importance(permutations: pd.DataFrame) -> Figure:
    """Exiba a queda média de Recall e seu desvio padrão nas dez repetições."""
    top = permutations.head(10)
    figure = Figure(figsize=(11, 7), layout="constrained")
    axis = figure.subplots()
    axis.barh(
        top.index, top["Importância média"], xerr=top["Desvio padrão"],
        color="#32648e", capsize=3, error_kw={"elinewidth": 1, "ecolor": "#555555"},
    )
    axis.invert_yaxis()
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Queda média de Recall da classe maligna (± desvio padrão)")
    axis.set_ylabel("Característica")
    axis.set_title("Regressão Logística: importância por permutação na validação\n"
                   "Dez maiores médias; dez repetições", pad=12)
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=0.2)
    return figure


def main() -> None:
    features, target = prepare_data(load_dataset())
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(
        features, target,
    )
    models = train_models(X_train, y_train)
    validation_results = evaluate_models(models, X_validation, y_validation)
    selected_model = models[SELECTED_MODEL]
    test_metrics = evaluate_final_model(selected_model, X_test, y_test)
    coefficients = coefficient_importance(selected_model)
    permutations = validation_permutation_importance(selected_model, X_validation, y_validation)

    display_results(validation_results)
    print(f"\nTeste — {SELECTED_MODEL}:")
    for metric in ("Accuracy", "Precision", "Recall", "F1-score"):
        print(f"{metric}: {test_metrics[metric]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(plot_validation_comparison(validation_results),
                RESULTS_DIR / "validation_model_comparison.png")
    save_figure(plot_confusion_matrices(validation_results, "Matrizes de confusão — validação"),
                RESULTS_DIR / "validation_confusion_matrices.png")
    save_figure(plot_confusion_matrices({SELECTED_MODEL: test_metrics}, "Matriz de confusão — teste"),
                RESULTS_DIR / "test_confusion_matrix.png")
    save_figure(plot_logistic_coefficients(coefficients), RESULTS_DIR / "logistic_coefficients.png")
    save_figure(plot_permutation_importance(permutations), RESULTS_DIR / "permutation_importance.png")


if __name__ == "__main__":
    main()
