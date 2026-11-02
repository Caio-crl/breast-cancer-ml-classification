# Treinamento supervisionado da MobileNetV3 Small

O [notebook de treinamento](../notebooks/vision_training_colab.ipynb) documenta o procedimento que gerou `run_001/best_epoch_004.pt`. O treinamento foi concluído; não é necessário repeti-lo para reproduzir a inferência. Utilize o [notebook de avaliação](../notebooks/vision_evaluation_colab.ipynb) com o checkpoint existente. [Metodologia, resultados de teste e limitações](computer_vision.md) estão na documentação central.

## Configuração

- MobileNetV3 Small, pesos ImageNet `IMAGENET1K_V1`, duas saídas em logits.
- Extrator e estatísticas BatchNorm congelados; cabeça classificadora treinável; sem fine-tuning no experimento.
- CrossEntropyLoss sem pesos de classe; AdamW com learning rate 0,001 e weight decay 0,0001.
- Batch size 8, `num_workers=0`, seed 42, máximo de 10 épocas.
- Early stopping com paciência 3 e `min_delta=0,0001`, baseado no maior F1 maligno de validação. Empates mantêm a época anterior.

As métricas são por imagem, com MALIGNANT=1 e predição pelo maior logit. A loss de cada época é ponderada pelo número de imagens dos lotes. Somente treino e validação são acessados: `splits=("train", "val")`. O teste não participa de treinamento, early stopping ou seleção de hiperparâmetros.

O checkpoint selecionado foi o da **época 4**; o early stopping ocorreu na **época 7**. Métricas de validação registradas: Accuracy 0,6283, Precision 0,5852, Recall 0,7923, F1 0,6732. Esses valores não são os resultados do teste independente.

## Pesos ImageNet e checkpoint treinado

A inicialização usa o arquivo oficial indicado por `MobileNet_V3_Small_Weights.IMAGENET1K_V1.url`, conforme a [documentação do torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html). O treinamento verifica o prefixo SHA-256 publicado no nome (`047dcff4`), carrega estritamente o state_dict de 1000 classes antes de substituir a saída e confere os tensores preservados. Pesos ausentes ou inconsistentes bloqueiam a preparação; não se treinam features aleatórias congeladas.

No notebook, `ALLOW_WEIGHT_DOWNLOAD=False` impede download implícito. O arquivo oficial pode ser disponibilizado manualmente em `WEIGHTS_PATH`; a célula de download só deve ser habilitada quando necessário em uma reprodução do procedimento. **A avaliação do checkpoint treinado não exige baixar esses pesos novamente.**

O checkpoint binário contém `format_version`, arquitetura, época, `model_state_dict`, estado do AdamW, metadados, histórico e métricas de validação. A avaliação restaura todos os parâmetros estritamente e rejeita formatos incompatíveis; não continua com uma rede aleatória.

## Notebook e dados

Abra o notebook de treinamento no Colab e confira a configuração antes de qualquer execução. `CODE_SOURCE="upload_zip"` exige a lista exata `CODE_FILES`, incluindo os nomes adicionados por `CODE_FILES.update`. Alternativamente, use uma branch publicada via `github` ou uma cópia com `existing`. Não presuma que mudanças locais estão disponíveis no GitHub.

Para criar um ZIP de código compatível a partir da raiz do repositório, execute em Python:

```python
import ast
import json
from pathlib import Path
import zipfile

root = Path.cwd()
notebook = json.loads((root / "notebooks/vision_training_colab.ipynb").read_text(encoding="utf-8"))
files = set()
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CODE_FILES" for target in node.targets
        ):
            files = set(ast.literal_eval(node.value))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "update"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "CODE_FILES"):
                files.update(ast.literal_eval(call.args[0]))
assert files
output = Path.home() / "Downloads/vision-training-code.zip"
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(files):
        archive.write(root / name, name)
print(output)
```

O ZIP contém apenas os arquivos de código, testes e documentação esperados, sem dataset, manifestos ou pesos; o modo `x` impede sobrescrita. Envie o notebook separadamente.

Os diretórios ignorados precisam ser disponibilizados manualmente no Drive. Ajuste `DATA_PACKAGE`, `DATA_BASE`, `WEIGHTS_PATH` e `OUTPUT_DIR`; os caminhos padrão do notebook são exemplos e podem diferir da pasta utilizada na execução concluída. O pacote de dados utilizado está em `MyDrive/FIAP-Tech-Challenge-Computer-Vision/cbis_ddsm_cv7.zip`. A extração do notebook de treinamento seleciona somente os 1.318 JPGs e os manifestos de treino/validação, preserva os caminhos relativos e confere hashes.

`prepare_training(config)` monta modelo, AdamW e loaders sem executar épocas. `RUN_TRAINING=False` é o padrão do notebook; mantenha-o assim para inspecionar a preparação sem repetir o experimento. A CLI também só executa épocas com `--run`. Dispositivo `auto` utiliza CUDA quando disponível e CPU como alternativa. Seeds e operações determinísticas são configuradas, mas resultados idênticos entre versões e hardwares diferentes não são garantidos.

## Artefatos e preservação

O diretório de cada treinamento deve ser novo, fora dos dados brutos e manifestos. Dentro do repositório, use somente `artifacts/vision/`, ignorado por Git e Docker. No Drive, preserve `run_001` e avalie em uma pasta separada.

- `config.json`: configuração, seed, arquitetura, versões, origem e hashes dos pesos e manifestos, contagens e critério de seleção.
- `history.csv`: loss, Accuracy, Precision, Recall e F1 por época de treino e validação.
- `best_epoch_NNN.pt`: checkpoint salvo em cada melhora do critério de validação.
- `summary.json`: checkpoint e época selecionados, métricas de validação e parada.
- `failure.json`: diagnóstico se uma execução iniciada falhar.

Diretórios existentes são rejeitados; não há retomada automática ou sobrescrita silenciosa. Não versione checkpoints, resultados gerados, imagens ou manifestos. Não use o teste para decidir novos ajustes. Os [resultados finais](computer_vision.md#avaliação-final-independente) são de uma avaliação independente, sem novo treinamento ou ajuste de limiar.
