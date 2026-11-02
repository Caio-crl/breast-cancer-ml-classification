# Computer Vision — classificação de massas mamárias

Entrega extra do Tech Challenge — Fase 1 da FIAP: classificação binária de recortes ROI de massas mamárias com MobileNetV3 Small. O módulo é independente dos modelos tabulares. Seu desempenho descreve este experimento acadêmico e **não autoriza aplicação clínica**.

## Dataset e rótulos

Fonte utilizada: [CBIS-DDSM: Breast Cancer Image Dataset — Kaggle, awsaf49](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset). O download deve ser feito pela página pública, respeitando suas condições de acesso. Imagens e metadados não são distribuídos neste repositório.

São utilizados exclusivamente recortes de massas identificados como `cropped images` nos metadados de conversão. Máscaras, mamografias completas e imagens de calcificações não integram a entrada. A associação entre CSV e JPG valida UID, paciente, caso, existência e unicidade; não usa apenas o nome final nem a ordem dos arquivos.

| Diagnóstico original | Classe |
|---|---:|
| BENIGN | 0 |
| BENIGN_WITHOUT_CALLBACK | 0 |
| MALIGNANT | 1 |

Os 1.696 recortes selecionados foram decodificados com Pillow, com dimensões válidas e modo original `L` (escala de cinza). A auditoria SHA-256 não encontrou duplicatas exatas. Essa checagem de bytes não detecta similaridade perceptual nem comprova correção diagnóstica dos rótulos.

## Partições por paciente

O teste oficial foi preservado. A partir do treino oficial, aproximadamente 80% dos pacientes ficaram no treino e 20% na validação, com seed 42. A estratégia agrupa pacientes por diagnósticos benignos, malignos ou mistos e busca equilíbrio de classes usando somente o treino oficial. Todas as lesões do mesmo paciente permanecem juntas; não se forçam proporções exatas de imagens.

| Partição | Imagens | Pacientes | Benignas (0) | Malignas (1) |
|---|---:|---:|---:|---:|
| Treino | 1.049 | 552 | 542 | 507 |
| Validação | 269 | 139 | 139 | 130 |
| Teste oficial | 378 | 201 | 231 | 147 |
| Total | 1.696 | 892 | 912 | 784 |

Não há pacientes compartilhados entre partições. Os manifestos preservam os campos `patient_id, case_id, image_path, pathology, label, split, official_split, width, height, mode, sha256`. Os caminhos são relativos à raiz do CBIS-DDSM, permitindo mudar de Windows para Colab sem editar os CSVs.

## Imagens, arquitetura e treinamento

O Dataset abre imagens sob demanda, converte escala de cinza em RGB e redimensiona proporcionalmente para um canvas 224 × 224 com padding preto. Não há corte aleatório da lesão. Apenas no treino é aplicada rotação leve de até ±5°, com canvas expandido. Validação e teste usam processamento determinístico. A normalização RGB utiliza média `[0.485, 0.456, 0.406]` e desvio `[0.229, 0.224, 0.225]` dos pesos ImageNet; a geometria de letterbox é uma escolha do projeto, diferente do center crop padrão da biblioteca.

A [MobileNetV3 Small do torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) utiliza pesos `IMAGENET1K_V1`, extrator e estatísticas BatchNorm congelados, e cabeça classificadora treinável com dois logits. A obtenção dos pesos oficiais é explícita, com verificação de hash antes do treinamento; não há fallback para features aleatórias congeladas.

Configuração do procedimento: CrossEntropyLoss, AdamW, learning rate 0,001, weight decay 0,0001, lote 8, seed 42, até 10 épocas, early stopping com paciência 3 e melhora mínima de F1 de validação de 0,0001. O maior F1 da classe maligna na validação seleciona o checkpoint; empates mantêm o anterior. Treino e validação não acessam o teste.

O treinamento `run_001` selecionou a **época 4**, checkpoint `best_epoch_004.pt`, e parou na época 7. As métricas de validação registradas foram Accuracy 0,6283, Precision 0,5852, Recall 0,7923 e F1 0,6732. Não foi realizado fine-tuning. O procedimento está no [notebook de treinamento](../notebooks/vision_training_colab.ipynb) e no [guia de treinamento](vision_training.md).

## Avaliação final independente

O [notebook de avaliação](../notebooks/vision_evaluation_colab.ipynb) restaurou o estado completo de `best_epoch_004.pt` com carregamento estrito. A avaliação foi executada no Google Colab, em **GPU Tesla T4**, exclusivamente nas 378 imagens do teste oficial: 231 benignas e 147 malignas. Foi utilizado `model.eval()`, sem gradientes, com decisão pelo maior logit. Não houve novo treinamento, ajuste de limiar ou seleção de hiperparâmetros pelo teste.

Os resultados abaixo foram obtidos nessa execução no Colab; não são métricas de validação nem de uma rede aleatória.

| Métrica por recorte | Valor |
|---|---:|
| Accuracy | 0.6005291005291006 |
| Precision da classe maligna | 0.491869918699187 |
| Recall da classe maligna | 0.8231292517006803 |
| F1-score da classe maligna | 0.6157760814249363 |

Matriz de confusão, com linhas = classe real e colunas = classe prevista:

| Real / Previsto | Benigno (0) | Maligno (1) |
|---|---:|---:|
| Benigno (0) | TN = 106 | FP = 125 |
| Maligno (1) | FN = 26 | TP = 121 |

Accuracy = `(106 + 121) / 378`; Precision = `121 / (121 + 125)`; Recall = `121 / (121 + 26)`; F1 = `2 × 121 / (2 × 121 + 125 + 26)`. A matriz é `[[106, 125], [26, 121]]`.

## Interpretação e limitações

- **26 falsos negativos:** 26 dos 147 recortes malignos foram classificados como benignos (17,69%). Isso impede interpretar o Recall de 82,31% como garantia de identificação de malignidade. O experimento não pode ser utilizado para descartar câncer ou orientar conduta clínica.
- **125 falsos positivos:** 125 dos 231 recortes benignos foram classificados como malignos (54,11%). A Precision de 49,19% significa que menos da metade dos 246 recortes previstos como malignos eram de fato malignos. Se transposto sem validação para um fluxo assistencial, esse tipo de erro poderia levar a alarmes e investigações desnecessárias; esse impacto não foi medido no experimento.
- Os resultados são **por recorte ROI**, não por paciente ou exame. O modelo recebe uma lesão já localizada; não detecta lesões em mamografias completas nem representa um sistema de rastreamento.
- Uma única divisão e um único treinamento não demonstram robustez entre diferentes amostras, serviços, populações ou equipamentos. Não foram realizados validação externa, avaliação clínica, estudo de calibração ou estimativa de intervalos de confiança.
- A conversão de cinza para RGB permite utilizar ImageNet, mas não acrescenta informação radiológica. Redimensionamento e padding podem limitar detalhes disponíveis ao modelo; não foi medida a contribuição de cada escolha.
- Não foi realizada revisão individual dos erros para identificar causas dos falsos negativos ou positivos. Não se deve atribuí-los a características específicas das lesões sem análise adicional.
- As tarefas e bases tabular e de imagens são distintas. Comparar diretamente suas métricas não permite concluir superioridade de uma abordagem.

**Não há suporte para aplicação clínica. O sistema não substitui avaliação médica e seus resultados não devem orientar diagnóstico, exclusão de malignidade ou tratamento.**

## Reprodução da avaliação existente

Não é necessário repetir o treinamento. Use o checkpoint selecionado e os manifestos preservados. O Google Colab deve ter PyTorch/torchvision compatíveis, pandas, Pillow e matplotlib; a primeira célula verifica versões, operação do torchvision e CPU/CUDA. Não reinstale uma pilha funcional. CPU é alternativa quando CUDA não estiver disponível.

1. Abra [Google Colab](https://colab.research.google.com/) e envie [vision_evaluation_colab.ipynb](../notebooks/vision_evaluation_colab.ipynb) pelo menu **Arquivo → Fazer upload de notebook**.
2. Disponibilize um ZIP de código correspondente ao notebook, conforme a seção abaixo. A configuração é `CODE_SOURCE="upload_zip"`; a célula confere nomes e SHA-256 dos fontes.
3. Execute os testes sintéticos de avaliação. Não prossiga com erros ou skips; testes ignorados não comprovam a inferência.
4. Monte o Drive e ajuste `BASE` se necessário. A execução documentada usa a estrutura abaixo, que precisa ser disponibilizada separadamente; não é obtida por clone do Git.
5. Execute a extração, que seleciona somente o manifesto de teste e seus 378 JPGs, preservando caminhos e verificando SHA-256. Treino e validação não são extraídos.
6. Execute a avaliação. O notebook restaura o checkpoint, calcula as métricas por argmax e exibe a matriz e o relatório. A GPU depende da disponibilidade do Colab.

```text
MyDrive/FIAP-Tech-Challenge-Computer-Vision/
├── cbis_ddsm_cv7.zip
└── artifacts/vision/
    ├── run_001/best_epoch_004.pt
    └── evaluation_run_001_<identificador>/
        ├── metrics.json
        ├── confusion_matrix.png
        └── evaluation_report.md
```

O ZIP de dados contém `cbis_ddsm/jpeg/<UID>/<arquivo>.jpg` e `processed/manifest_{train,val,test}.csv`. Ele é um pacote local de dados selecionados, não um arquivo distribuído pelo repositório. Quem parte do download público deve disponibilizar os metadados em `data/vision/raw/cbis_ddsm/csv/` e os JPGs em `data/vision/raw/cbis_ddsm/jpeg/`, executar o inspetor e preparar manifestos em uma pasta nova:

```text
python -m src.vision.inspect_dataset data/vision/raw/cbis_ddsm
python -m src.vision.prepare_manifest data/vision/raw/cbis_ddsm --output-dir data/vision/processed_reproduction --seed 42 --val-fraction 0.2
```

A pasta de saída da preparação deve ser nova. Para reproduzir exatamente a avaliação registrada, use os manifestos já validados e o checkpoint original; não sobrescreva nenhum deles. A regeneração permite verificar o procedimento, mas não é necessária para avaliar o modelo existente.

Também é possível avaliar uma cópia já extraída, a partir da raiz do projeto:

```text
python -m src.vision.evaluation --checkpoint /drive/artifacts/vision/run_001/best_epoch_004.pt --dataset-root /dados/cbis_ddsm --manifest-dir /dados/processed --output-dir /drive/artifacts/vision/evaluation_run_001_nova --device auto --batch-size 8
```

O diretório de saída deve ser novo e separado dos dados e do `run_001`. O JSON registra hashes do checkpoint e manifesto, dispositivo, versões e contagens; o PNG e o Markdown contêm somente resultados agregados. Não são salvas imagens individuais nem identificadores de pacientes. O checkpoint e os resultados completos permanecem no Drive; esta página registra os números da execução concluída.

## Pacote de código para o notebook de avaliação

Na raiz do repositório, execute o trecho abaixo com Python. Ele inclui somente os arquivos esperados pelo notebook, confere os hashes e grava o ZIP em Downloads. Não inclui imagens, manifestos ou pesos. Se os fontes diferirem dos hashes do notebook, use o conjunto correspondente à execução; não remova a validação para contornar a divergência.

```python
import ast
import hashlib
import json
from pathlib import Path
import zipfile

root = Path.cwd()
notebook = json.loads((root / "notebooks/vision_evaluation_colab.ipynb").read_text(encoding="utf-8"))
hashes = None
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_HASHES" for target in node.targets
        ):
            hashes = ast.literal_eval(node.value)
assert hashes, "Lista de fontes não encontrada no notebook."
for name, expected in hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
output = Path.home() / "Downloads/tech-challenge-cv-evaluation-code.zip"
assert not output.exists(), "ZIP já existe; preserve-o ou escolha outro nome."
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in hashes:
        archive.write(root / name, name)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
print(output)
```

## Testes e arquivos do módulo

```text
python -m unittest discover -s tests/vision -p "test_evaluation.py" -v
python -m unittest discover -s tests/vision -v
python -m unittest discover -s tests -v
```

Os testes usam fixtures sintéticas, sem dataset real, GPU ou acesso à internet. Os dependentes de PyTorch/torchvision são ignorados explicitamente quando essas bibliotecas não estão disponíveis; isso não valida o comportamento do modelo. Os testes de treinamento existentes são independentes da inferência final e usam apenas fixtures, não o experimento real.

- [Infraestrutura de imagens e modelo](vision_infrastructure.md).
- [Smoke test no Colab](vision_colab_validation.md).
- [Treinamento e checkpoints](vision_training.md).
- [Implementação da avaliação](../src/vision/evaluation.py).

Dados em `data/vision/`, artefatos em `artifacts/vision/`, arquivos ZIP, checkpoints, pesos e ambientes virtuais específicos são excluídos pelo Git e pelo contexto Docker. Não incorpore esses arquivos ao versionamento. Preserve o ambiente tabular separado.
