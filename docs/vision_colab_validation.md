# Verificação da infraestrutura no Google Colab

O [notebook de smoke test](../notebooks/vision_smoke_colab.ipynb) verifica Dataset, DataLoaders e MobileNetV3 Small sem treinamento. Ele complementa o [notebook de avaliação final](../notebooks/vision_evaluation_colab.ipynb), que exige o checkpoint treinado. Consulte a [documentação central](computer_vision.md) para resultados e reprodução da avaliação independente.

## Execução

1. Abra [Google Colab](https://colab.research.google.com/) e envie `vision_smoke_colab.ipynb` por **Arquivo → Fazer upload de notebook**.
2. Execute a inspeção de versões, importações e compatibilidade de PyTorch/torchvision. CPU é suficiente para o smoke test; GPU não é garantida.
3. Configure `CODE_SOURCE`. Para ZIP, envie exatamente os arquivos listados em `CODE_FILES` no notebook. Para GitHub, informe uma branch publicada que contenha esses arquivos. Uma cópia extraída pode ser usada com `existing` e `EXISTING_ROOT`.
4. Mantenha `INSTALL_IF_NEEDED=False` quando o ambiente estiver funcional. Caso contrário, corrija apenas as dependências indicadas, no runtime separado do Colab, e reinicie-o se necessário.
5. Execute os testes e a integração sintética. Verifique formatos `(batch, 3, 224, 224)` e `(batch, 2)`, finitude dos logits e congelamento do extrator. Skips não contam como aprovação efetiva.
6. Para usar imagens reais, disponibilize os três manifestos e os recortes referenciados no Drive, ajuste `REAL_DATA_ROOT` e `REAL_MANIFEST_DIR` e habilite `USE_REAL_DATA=True`. Os diretórios ignorados pelo Git não são obtidos por clone.

O upload de código não deve conter dados, manifestos, pesos, ambientes virtuais ou saídas de notebooks com informações de pacientes. O ZIP deve conter somente a lista exata esperada pelo notebook. Para gerar um pacote a partir da raiz do projeto, execute em Python:

```python
import ast
import json
from pathlib import Path
import zipfile

root = Path.cwd()
notebook = json.loads((root / "notebooks/vision_smoke_colab.ipynb").read_text(encoding="utf-8"))
files = None
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CODE_FILES" for target in node.targets
        ):
            files = ast.literal_eval(node.value)
assert files
output = Path.home() / "Downloads/vision-smoke-code.zip"
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(files):
        archive.write(root / name, name)
print(output)
```

O modo exclusivo `x` evita sobrescrever pacotes existentes. O notebook valida os nomes, o tamanho e a ausência de links simbólicos antes de extrair o código. O notebook é enviado separadamente.

O smoke test com dados reais usa caminhos portáveis e não modifica imagens ou manifestos. A auditoria de hashes e decodificação completa é responsabilidade da preparação dos dados; o smoke test não a substitui. Pesos aleatórios são usados somente para validação técnica, sem interpretação clínica. Não utilize esse notebook para obter métricas finais: execute a avaliação independente com `best_epoch_004.pt`.
