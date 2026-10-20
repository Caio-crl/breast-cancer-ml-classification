"""Verifica se o ambiente possui as dependências básicas do projeto."""

import sys

import matplotlib
import numpy
import pandas
import seaborn
import sklearn


def main() -> None:
    print("Tech Challenge - Fase 1 | Ambiente configurado")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Pandas: {pandas.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Seaborn: {seaborn.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")


if __name__ == "__main__":
    main()
