# Classificação de Câncer de Mama com Machine Learning

Projeto acadêmico desenvolvido para o Tech Challenge da FIAP. O objetivo é analisar características de amostras de tumores de mama e comparar modelos de classificação para diferenciar casos benignos e malignos.

> **Uso exclusivamente educacional:** os modelos e resultados deste repositório não constituem diagnóstico clínico nem substituem a avaliação de profissionais de saúde.

## Objetivos

- Explorar e visualizar dados médicos tabulares.
- Preparar os dados com um pipeline de pré-processamento.
- Comparar pelo menos dois algoritmos de classificação.
- Avaliar os resultados com accuracy, recall, F1-score e matriz de confusão.
- Interpretar os fatores que influenciam as previsões e discutir limitações.

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**

Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

Os arquivos originais do dataset são armazenados localmente em `data/raw/` e não são versionados neste repositório. O arquivo pode ser obtido a partir da fonte acima.

## Requisitos

- Python 3.11
- Git (para versionamento)
- Docker (opcional, para execução em contêiner)

## Instalação no Windows (PowerShell)

Na pasta raiz do projeto, execute:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.check_environment
```

Se o PowerShell impedir a ativação do ambiente virtual, execute `Set-ExecutionPolicy -Scope Process Bypass` e tente ativá-lo novamente. Confira se `python --version` retorna uma versão 3.11.x antes de criar o ambiente.

## Instalação no Linux ou macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.check_environment
```

## JupyterLab

Com o ambiente virtual ativo, execute:

```bash
jupyter lab
```

Os notebooks de análise ficam na pasta `notebooks/`.

## Docker

Na pasta raiz do projeto, execute:

```bash
docker build -t tech-challenge-fase1 .
docker run --rm tech-challenge-fase1
```

O comando atual do contêiner verifica as versões das dependências. O treinamento dos modelos será disponibilizado no código do projeto conforme sua implementação.

## Organização

```text
tech-challenge-fase1/
├── data/raw/             # Dados originais, fora do Git
├── notebooks/            # Análises exploratórias e experimentos
├── results/              # Gráficos e métricas
├── src/                  # Código Python
│   ├── __init__.py
│   └── check_environment.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Limitações

A análise usa uma base pública de dados para fins de estudo e demonstração de técnicas de aprendizado de máquina. Seu desempenho em dados de teste não garante segurança, validade clínica ou generalização para outros contextos hospitalares.
