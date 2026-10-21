# Classificação de câncer de mama com Machine Learning

Projeto acadêmico do Tech Challenge da FIAP que utiliza o dataset Breast Cancer Wisconsin Diagnostic para classificar amostras de tumores de mama como benignas ou malignas. O trabalho explora Machine Learning como apoio à análise de dados médicos, abrangendo análise exploratória, preparação dos dados, comparação de classificadores, avaliação final e interpretabilidade.

> **Uso acadêmico:** o projeto não é uma ferramenta de diagnóstico clínico. Seus resultados não substituem a avaliação médica.

## Tecnologias

- **Python 3.11:** linguagem do projeto; execução local conferida com Python 3.11.9.
- **Pandas e NumPy:** organização e manipulação dos dados.
- **Scikit-learn:** divisão estratificada, pré-processamento, classificadores, métricas e importância por permutação.
- **Matplotlib e Seaborn:** visualização dos dados e dos resultados.
- **JupyterLab:** execução do notebook de análise exploratória.
- **unittest:** testes automatizados da biblioteca padrão do Python.
- **Docker:** imagem para instalação das dependências e verificação do ambiente.

As versões das dependências estão definidas em [requirements.txt](requirements.txt).

## Dataset

Fonte: [Breast Cancer Wisconsin (Diagnostic) Data Set — Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data).

A base contém **569 registros**. São utilizadas **30 características numéricas** para prever o diagnóstico, codificado como **benigno: B = 0** e **maligno: M = 1**. As colunas `id` e `Unnamed: 32` não participam das características preditoras.

Para obter os dados:

1. Acesse a página do dataset no Kaggle e faça o download; o acesso pode exigir uma conta.
2. Extraia o arquivo `data.csv` do download.
3. Coloque-o em `data/raw/data.csv`, preservando os nomes das colunas e o conteúdo original.

O CSV é ignorado pelo Git e deve ser disponibilizado localmente antes de executar as análises. Os módulos leem os dados sem alterar o arquivo original.

## Estrutura do projeto

```text
.
├── data/
│   └── raw/
│       └── data.csv                    # Dataset local, fora do Git
├── notebooks/
│   └── exploratory_analysis.ipynb       # Análise exploratória
├── results/
│   ├── validation_model_comparison.png
│   ├── validation_confusion_matrices.png
│   ├── test_confusion_matrix.png
│   ├── logistic_coefficients.png
│   └── permutation_importance.png
├── src/
│   ├── __init__.py
│   ├── check_environment.py
│   ├── data_preparation.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── results_reporting.py
├── tests/
│   ├── test_data_preparation.py
│   ├── test_model_training.py
│   └── test_model_evaluation.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Instalação

Requisitos: **Python 3.11**, **Git** e acesso ao dataset. Docker é opcional.

No Windows, abra o PowerShell. Substitua `URL_DO_REPOSITORIO` pela URL de clonagem deste repositório:

```powershell
git clone "URL_DO_REPOSITORIO" breast-cancer-wisconsin
cd breast-cancer-wisconsin
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Confira se `python --version` indica **3.11.x** antes de criar o ambiente virtual. Se a política do PowerShell impedir a ativação, execute `Set-ExecutionPolicy -Scope Process Bypass` e tente ativá-lo novamente; a alteração vale para a sessão atual.

Por fim, baixe e posicione o CSV conforme a seção [Dataset](#dataset). Execute os comandos seguintes na raiz do projeto, com o ambiente virtual ativo.

## Execução

| Comando | Função |
|---|---|
| `python -m src.check_environment` | Verifica a importação das dependências e exibe suas versões. |
| `python -m src.data_preparation` | Valida os dados e apresenta dimensões, distribuição das classes e ausência de sobreposição entre os conjuntos. |
| `python -m src.model_training` | Treina os dois classificadores e compara métricas e matrizes de confusão na validação. |
| `python -m src.model_evaluation` | Treina a Regressão Logística selecionada, avalia no teste e apresenta coeficientes e importância por permutação na validação. |
| `python -m src.results_reporting` | Reproduz os resultados a partir do CSV e salva os cinco gráficos em `results/`, a 300 dpi e sem interface gráfica. |
| `python -m unittest discover -s tests -v` | Executa os testes de preparação, treinamento, avaliação e interpretabilidade. |

Cada módulo de análise executa seu próprio fluxo a partir do CSV. Os modelos permanecem em memória; apenas o módulo de gráficos grava os PNGs, substituindo os arquivos de mesmo nome.

Para abrir a análise exploratória no JupyterLab:

```powershell
jupyter lab notebooks/exploratory_analysis.ipynb
```

Selecione o kernel Python do ambiente virtual para executar as células.

## Modelos e metodologia

A **Regressão Logística** fornece uma referência linear para classificação binária e permite examinar os coeficientes das características. A **Random Forest**, formada por múltiplas árvores de decisão, permite comparar essa abordagem com um modelo capaz de representar relações não lineares.

| Modelo | Configuração explícita |
|---|---|
| Regressão Logística | `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=200`, `random_state=42` |

Os demais parâmetros permanecem nos valores padrão. Não foi realizada otimização de hiperparâmetros.

A divisão é estratificada pelo diagnóstico, com `random_state=42` nas duas etapas de separação:

| Conjunto | Proporção prevista | Registros |
|---|---:|---:|
| Treinamento | 60% | 341 |
| Validação | 20% | 114 |
| Teste | 20% | 114 |

As quantidades consideram o arredondamento para registros inteiros. Os índices originais são preservados e não há sobreposição entre os conjuntos.

Cada classificador possui seu próprio pipeline com imputação de valores ausentes pela mediana (`SimpleImputer`) e padronização (`StandardScaler`). O pré-processamento e o classificador são ajustados exclusivamente nos 341 registros de treinamento. A mesma preparação é aplicada à Random Forest para manter um fluxo consistente.

Os modelos são comparados somente na validação. A **Regressão Logística foi selecionada antes da avaliação final**, considerando especialmente seu maior Recall para a classe maligna. Somente esse modelo é avaliado no teste, mantendo o ajuste realizado no treinamento.

## Resultados

Accuracy representa a proporção de acertos totais. Precision, Recall e F1-score consideram **maligno (1)** como classe positiva. O Recall mede a proporção de casos malignos identificados.

### Comparação na validação

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regressão Logística | 0,9737 | 1,0000 | 0,9302 | 0,9639 |
| Random Forest | 0,9561 | 0,9750 | 0,9070 | 0,9398 |

### Avaliação final da Regressão Logística

| Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0,9649 | 1,0000 | 0,9048 | 0,9500 |

Matriz de confusão: `[[72, 0], [4, 38]]`, na ordem `[[VN, FP], [FN, VP]]`. As linhas representam a classe real e as colunas a classe prevista, com ordem **[benigno (0), maligno (1)]**.

Foram identificados corretamente 72 casos benignos e 38 malignos. Houve **quatro falsos negativos**: casos malignos classificados como benignos. Não houve falsos positivos nessa amostra. A Precision de 1,0000 deve ser interpretada junto ao Recall e aos casos malignos não identificados; esses resultados não demonstram adequação clínica.

### Interpretabilidade

A análise utiliza os coeficientes da Regressão Logística após padronização e a importância por permutação, calculada **somente na validação**, com Recall da classe maligna, `random_state=42` e `n_repeats=10`.

Coeficientes positivos aumentam a pontuação estimada para maligno, e negativos a reduzem, mantidas as demais características constantes. A importância por permutação mede a queda média de Recall ao embaralhar cada característica. Essas medidas descrevem o comportamento do modelo e não estabelecem relações causais.

## Visualizações

Os gráficos podem ser reproduzidos com `python -m src.results_reporting`. Clique em cada imagem para consultar a versão completa.

**Comparação na validação:** quatro métricas dos dois classificadores.

<a href="results/validation_model_comparison.png"><img src="results/validation_model_comparison.png" alt="Comparação de Accuracy, Precision, Recall e F1-score na validação" width="680"></a>

**Matrizes de validação:** contagens por classe real e prevista para os dois modelos.

<a href="results/validation_confusion_matrices.png"><img src="results/validation_confusion_matrices.png" alt="Matrizes de confusão dos dois classificadores na validação" width="680"></a>

**Avaliação final:** matriz da Regressão Logística no teste.

<a href="results/test_confusion_matrix.png"><img src="results/test_confusion_matrix.png" alt="Matriz de confusão da Regressão Logística no teste" width="420"></a>

**Coeficientes:** dez maiores valores absolutos, preservando os sinais.

<a href="results/logistic_coefficients.png"><img src="results/logistic_coefficients.png" alt="Dez maiores coeficientes absolutos da Regressão Logística" width="680"></a>

**Permutação na validação:** dez maiores médias de queda de Recall; barras de erro representam o desvio padrão das repetições.

<a href="results/permutation_importance.png"><img src="results/permutation_importance.png" alt="Importância por permutação medida pelo Recall na validação" width="680"></a>

## Docker

O [Dockerfile](Dockerfile) utiliza `python:3.11-slim` e o diretório de trabalho `/app`. Primeiro copia `requirements.txt` e instala as dependências; depois copia o código do projeto. O arquivo `.dockerignore` exclui `data/raw/*`, mantendo o dataset fora da imagem.

O comando padrão é `python -m src.check_environment`. Os comandos abaixo sobrescrevem esse padrão para executar cada módulo, sem alterar a imagem.

### Construção e verificação do ambiente

No Windows, abra o PowerShell na raiz do projeto, com Docker Desktop em execução e configurado para contêineres Linux. Não é necessário ativar o ambiente virtual Python local.

```powershell
docker build -t breast-cancer-wisconsin .
docker run --rm breast-cancer-wisconsin python -m src.check_environment
```

### Dataset e execução dos módulos

Disponibilize `data/raw/data.csv` conforme a seção [Dataset](#dataset). Na mesma sessão do PowerShell, resolva os diretórios locais:

```powershell
$dataPath = (Resolve-Path -LiteralPath '.\data\raw').Path
$resultsPath = (Resolve-Path -LiteralPath '.\results').Path
```

O bind mount conecta `data/raw/` a `/app/data/raw` com `readonly`: o contêiner pode ler o CSV, mas não alterá-lo. As aspas preservam caminhos do Windows que contêm espaços.

**Preparação dos dados:**

```powershell
docker run --rm --mount "type=bind,source=$dataPath,target=/app/data/raw,readonly" breast-cancer-wisconsin python -m src.data_preparation
```

**Treinamento e comparação na validação:**

```powershell
docker run --rm --mount "type=bind,source=$dataPath,target=/app/data/raw,readonly" breast-cancer-wisconsin python -m src.model_training
```

**Avaliação final da Regressão Logística e interpretabilidade:**

```powershell
docker run --rm --mount "type=bind,source=$dataPath,target=/app/data/raw,readonly" breast-cancer-wisconsin python -m src.model_evaluation
```

**Geração dos cinco gráficos:**

```powershell
docker run --rm --mount "type=bind,source=$dataPath,target=/app/data/raw,readonly" --mount "type=bind,source=$resultsPath,target=/app/results" breast-cancer-wisconsin python -m src.results_reporting
```

O segundo bind mount permite gravar em `/app/results`. Os PNGs ficam disponíveis na pasta local `results/`, inclusive após a remoção do contêiner, e substituem os arquivos de mesmo nome. A renderização não exige interface gráfica. A Random Forest é avaliada somente na validação; o teste é utilizado apenas pela Regressão Logística selecionada.

### Testes automatizados

```powershell
docker run --rm --mount "type=bind,source=$dataPath,target=/app/data/raw,readonly" breast-cancer-wisconsin python -m unittest discover -s tests -v
```

A execução foi validada no Docker Desktop com Python 3.11: os 26 testes passaram, as métricas reproduziram os resultados documentados e os cinco gráficos foram gerados no diretório local. O dataset permaneceu fora da imagem e montado somente para leitura. Os testes emitiram um aviso de depreciação do SciPy sobre opções do otimizador, sem falhas.

## Limitações

- A base possui relativamente poucos registros: 569 amostras.
- A avaliação utiliza uma única divisão de treinamento, validação e teste, sem validação cruzada ou avaliação externa.
- Os quatro falsos negativos no teste evidenciam casos malignos não identificados pelo modelo.
- Características correlacionadas dificultam a interpretação individual dos coeficientes e podem reduzir a importância por permutação.
- Os resultados descrevem esta amostra e não representam validação clínica nem garantem generalização para outros contextos.
- O modelo não substitui a avaliação médica.
