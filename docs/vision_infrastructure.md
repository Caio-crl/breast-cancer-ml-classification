# Infraestrutura de imagens e modelo

O módulo `src/vision/` é independente dos pipelines tabulares. A [documentação de Computer Vision](computer_vision.md) reúne dataset, partições, metodologia, resultados finais e limitações.

## Ambiente

Use um ambiente separado com PyTorch, torchvision, pandas e Pillow. [requirements-vision.txt](../requirements-vision.txt) registra o par de referência torch 2.7.1 / torchvision 0.22.1; preserve um par compatível já disponível no Colab em vez de reinstalar sem necessidade. Consulte a [matriz oficial de versões](https://pytorch.org/get-started/previous-versions/#v2-7-1) se precisar preparar outro ambiente. NumPy e scikit-learn são usados pela preparação e sua suíte de testes; matplotlib produz gráficos de avaliação. Não altere o ambiente tabular para executar visão computacional.

CPU é suficiente para testes e inferência; CUDA é usada somente quando disponível. O [notebook de smoke test](../notebooks/vision_smoke_colab.ipynb) verifica importações, versões e compatibilidade antes da integração.

## Dataset e transformações

`MammographyDataset(manifest_path, dataset_root)` valida os campos essenciais do manifesto, os rótulos, a origem oficial e os caminhos relativos `jpeg/<UID>/<arquivo>.jpg`. Abre cada imagem sob demanda com Pillow e converte cinza para RGB. Retorna imagem transformada e rótulo inteiro, benigno=0 e maligno=1. Não modifica os arquivos originais.

O manifesto declara dimensões, modo e SHA-256. O Dataset valida a declaração; a auditoria completa de conteúdo pertence à preparação dos manifestos. A avaliação final volta a conferir os hashes dos JPGs de teste.

O processamento preserva a proporção por redimensionamento e padding preto até 224 × 224, sem cortar a lesão. Apenas treino recebe rotação de até ±5°, antes do redimensionamento e com canvas expandido. Validação e teste são determinísticos. Não há flips, ajustes de contraste ou cortes aleatórios. Usa-se a normalização ImageNet: média `[0.485, 0.456, 0.406]`, desvio `[0.229, 0.224, 0.225]`.

## DataLoaders

`build_dataloaders` preserva os manifestos existentes, sem refazer divisões. Os padrões são batch size 8, `num_workers=0`, seed 42 e `drop_last=False`. Apenas treino usa shuffle e augmentation. A função aceita `splits` para carregar exclusivamente as partições necessárias; compara pacientes, casos, caminhos e hashes entre as partições solicitadas.

```python
from src.vision.dataset import build_dataloaders

loaders = build_dataloaders(
    "/dados/processed", "/dados/cbis_ddsm",
    batch_size=8, num_workers=0, splits=("test",),
)
```

A independência global das três partições é verificada na preparação. Carregar somente teste não refaz essa auditoria contra os demais manifestos. Para multiprocessing no Windows, use scripts protegidos por `if __name__ == "__main__":`.

## Rede e pesos

`build_model` constrói MobileNetV3 Small com dois logits. Entrada: `(batch, 3, 224, 224)`; saída: `(batch, 2)`. `pretrained=True` exige pesos ImageNet locais ou em cache, carrega o state_dict original e substitui a última camada por duas saídas. Nenhum download é implícito. O treinamento congela o extrator e suas estatísticas BatchNorm, mantendo a cabeça treinável.

`pretrained=False` é adequado para smoke tests técnicos; **não produz o modelo treinado**. Na avaliação final, a arquitetura é construída sem download e todos os parâmetros são substituídos estritamente pelo state_dict do checkpoint treinado antes da inferência. Arquivo ausente, formato incompatível ou parâmetros inconsistentes impedem a avaliação. Os metadados de proveniência não constituem, por si só, autenticação externa do checkpoint.

## Verificação técnica

```text
python -m unittest discover -s tests/vision -v
python -m src.vision.smoke_test --dataset-root /dados/cbis_ddsm --manifest-dir /dados/processed --device cpu
```

O smoke test usa somente um lote, pesos aleatórios, `eval()` e `inference_mode()`, sem métricas preditivas ou treinamento. Seus logits não são resultados do experimento. A [avaliação final](computer_vision.md#avaliação-final-independente) exige o checkpoint selecionado e o teste oficial. Dependências ausentes geram skips explícitos nos testes correspondentes; skips não demonstram funcionamento da rede.
