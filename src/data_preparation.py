"""Carregamento, divisão e pré-processamento dos dados Breast Cancer Wisconsin."""

from itertools import combinations
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "data.csv"
FEATURE_COLUMNS = tuple(
    f"{measure}_{suffix}"
    for suffix in ("mean", "se", "worst")
    for measure in (
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave points", "symmetry",
        "fractal_dimension",
    )
)


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Leia o CSV sem modificar o arquivo original."""
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Valide o esquema e retorne as 30 características e o alvo B=0, M=1.

    Valores numéricos ausentes são mantidos para imputação posterior.
    O DataFrame recebido e seus índices são preservados.
    """
    required = {*FEATURE_COLUMNS, "id", "Unnamed: 32", "diagnosis"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas necessárias ausentes: {sorted(missing)}")
    if not df.columns.is_unique:
        raise ValueError("O dataset possui nomes de colunas duplicados.")

    diagnosis = df["diagnosis"]
    invalid = ~diagnosis.isin(["B", "M"])
    if invalid.any():
        values = diagnosis.loc[invalid].unique().tolist()
        raise ValueError(f"Diagnósticos inválidos: {values}. Esperados: B ou M.")

    features = df.loc[:, list(FEATURE_COLUMNS)].copy()
    non_numeric = [name for name in features if not is_numeric_dtype(features[name])]
    if non_numeric:
        raise ValueError(f"Características não numéricas: {non_numeric}")

    target = diagnosis.map({"B": 0, "M": 1}).astype("int64")
    return features, target


def indices_are_disjoint(*datasets: pd.DataFrame) -> bool:
    """Verifique se os conjuntos não compartilham índices."""
    return all(
        left.index.intersection(right.index).empty
        for left, right in combinations(datasets, 2)
    )


def split_data(
    features: pd.DataFrame, target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Retorne X_treino, X_validação, X_teste, y_treino, y_validação e y_teste.

    As proporções 60/20/20 estão sujeitas ao arredondamento para registros
    inteiros. As duas divisões são estratificadas e usam random_state=42.
    Nenhuma transformação é ajustada nesta etapa.
    """
    if not features.index.is_unique or not target.index.is_unique:
        raise ValueError("Os índices originais devem ser únicos para a divisão.")
    if not features.index.equals(target.index):
        raise ValueError("Características e alvo devem ter os mesmos índices e ordem.")

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        features, target, test_size=0.4, random_state=42, stratify=target,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.5, random_state=42,
        stratify=y_remaining,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def create_preprocessing_pipeline() -> Pipeline:
    """Retorne um pipeline sem ajuste; use fit somente no conjunto de treino.

    Validação e teste devem usar transform no mesmo objeto já ajustado.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def main() -> None:
    features, target = prepare_data(load_dataset())
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(
        features, target,
    )
    print(f"Registros: {len(features)} | Características: {features.shape[1]}")
    for name, X, y in (
        ("Treinamento", X_train, y_train),
        ("Validação", X_validation, y_validation),
        ("Teste", X_test, y_test),
    ):
        print(f"\n{name}: X={X.shape}, y={y.shape}")
        distribution = pd.DataFrame({
            "registros": y.value_counts().sort_index(),
            "percentual": y.value_counts(normalize=True).sort_index().mul(100).round(2),
        })
        distribution.index = distribution.index.map({0: "B (0)", 1: "M (1)"})
        print(distribution.to_string())
    if not indices_are_disjoint(X_train, X_validation, X_test):
        raise ValueError("Há sobreposição de índices entre os conjuntos.")
    print("\nSem sobreposição de índices entre treinamento, validação e teste.")
    print("Pipeline disponível sem ajuste prévio em create_preprocessing_pipeline().")


if __name__ == "__main__":
    main()
