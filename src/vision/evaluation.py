"""Avaliação independente: somente teste, argmax, sem treino ou downloads."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from src.vision.training import ARCHITECTURE, classification_metrics, require_torch, select_device


def final_metrics(actual, predicted):
    actual, predicted = list(actual), list(predicted)
    result = classification_metrics(actual, predicted)
    tn = sum(a == p == 0 for a, p in zip(actual, predicted))
    tp = sum(a == p == 1 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    return {**result, "images": len(actual), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "matrix_rows": "true labels [benign, malignant]",
            "matrix_columns": "predicted labels [benign, malignant]"}


def prepare_test_package(package, destination):
    """Extraia apenas o manifesto oficial de teste e seus JPGs, verificando SHA-256.

    O destino deve ser novo: uma extração incompleta nunca é publicada.
    Não lê manifestos nem imagens de treino/validação.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Destino já existe; reutilize a cópia manualmente ou escolha outro diretório.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Entradas duplicadas no ZIP.")
        manifest = archive.read("processed/manifest_test.csv")
        rows = list(csv.DictReader(io.StringIO(manifest.decode("utf-8-sig"))))
        if not rows or any(not {"split", "official_split", "image_path", "sha256"} <= row.keys() for row in rows):
            raise ValueError("Manifesto de teste vazio ou incompleto.")
        seen = set()
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            stage = Path(temporary) / "data"
            for row in rows:
                relative = PurePosixPath(row["image_path"])
                if (row["split"] != "test" or row["official_split"] != "test"
                        or relative.is_absolute() or ".." in relative.parts
                        or "\\" in row["image_path"] or len(relative.parts) != 3
                        or relative.parts[0] != "jpeg" or relative.suffix.lower() != ".jpg"
                        or relative.as_posix() != row["image_path"] or row["image_path"] in seen):
                    raise ValueError("Partição/caminho inválido ou repetido no teste.")
                seen.add(row["image_path"])
                target = stage / "cbis_ddsm" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("cbis_ddsm/" + relative.as_posix()) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                with target.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest != row["sha256"]:
                    raise ValueError("SHA-256 divergente em imagem de teste.")
            (stage / "processed").mkdir()
            (stage / "processed/manifest_test.csv").write_bytes(manifest)
            stage.rename(destination)
    return destination


def load_evaluation_model(checkpoint_path, device="cpu"):
    """Restaure TODO o state_dict treinado estritamente; nunca use fallback aleatório."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint treinado ausente: {path}")
    torch = require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint deve ser um dicionário CV-8.")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Metadados do checkpoint incompatíveis.")
    provenance = metadata.get("pretrained_weights", {})
    if not isinstance(provenance, dict):
        raise ValueError("Origem dos pesos ausente ou incompatível.")
    if (checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != ARCHITECTURE
            or metadata.get("architecture") != ARCHITECTURE
            or type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] < 1
            or metadata.get("test_used") is not False
            or not provenance.get("hash_verified") or not provenance.get("backbone_load_verified")
            or not isinstance(checkpoint.get("validation_metrics"), dict)
            or not checkpoint.get("history")):
        raise ValueError("Checkpoint incompatível ou sem evidência de treinamento CV-8/ImageNet verificado.")
    from src.vision.model import build_model
    model = build_model(pretrained=False, freeze_features=True)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError("Parâmetros do checkpoint incompatíveis com MobileNetV3 Small binária.") from exc
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Checkpoint contém parâmetros não finitos.")
    model.requires_grad_(False)
    model.to(device).eval()
    return model, checkpoint


def infer(model, loader, device):
    """Somente inferência; sem otimizador, backward ou atualização de BatchNorm."""
    torch = require_torch()
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if tuple(logits.shape) != (len(labels), 2) or not torch.isfinite(logits).all():
                raise ValueError("Logits inválidos ou não finitos.")
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return final_metrics(actual, predicted)


def evaluate(checkpoint_path, dataset_root, manifest_dir, output_dir, device="auto", batch_size=8):
    """Avalie todo o teste uma vez e grave apenas resultados agregados em pasta nova."""
    output = Path(output_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    for protected in (Path(dataset_root).resolve(), Path(manifest_dir).resolve(), checkpoint_path.parent):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("Resultados devem ficar separados dos dados e do diretório do checkpoint.")
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo) and not output.is_relative_to(repo / "artifacts/vision"):
        raise ValueError("Use artifacts/vision/ para resultados dentro do repositório.")
    if output.exists():
        raise FileExistsError("Resultados já existem; escolha outro diretório.")
    selected_device = select_device(device)
    model, checkpoint = load_evaluation_model(checkpoint_path, selected_device)
    if checkpoint["epoch"] != 4:
        raise ValueError("Esta avaliação final exige o checkpoint selecionado best_epoch_004.pt (época 4).")
    from src.vision.dataset import build_dataloaders
    loaders = build_dataloaders(manifest_dir, dataset_root, batch_size=batch_size,
                                num_workers=0, splits=("test",))
    records = loaders["test"].dataset.records
    counts = records.label.astype(int).value_counts().to_dict()
    if len(records) != 378 or counts != {0: 231, 1: 147}:
        raise ValueError(f"Teste oficial esperado: 378 imagens, 231 benignas/147 malignas; encontrado {counts}.")
    for path, expected in zip(loaders["test"].dataset.paths, records.sha256):
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError("Hash de imagem diverge do manifesto de teste.")
    metrics = infer(model, loaders["test"], selected_device)
    import torch
    import torchvision
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with checkpoint_path.open("rb") as source:
        checkpoint_hash = hashlib.file_digest(source, "sha256").hexdigest()
    result = {**metrics, "checkpoint_sha256": checkpoint_hash,
              "checkpoint_epoch": checkpoint["epoch"], "architecture": ARCHITECTURE,
              "manifest_sha256": hashlib.sha256((Path(manifest_dir) / "manifest_test.csv").read_bytes()).hexdigest(),
              "device": str(selected_device), "torch_version": str(torch.__version__),
              "torchvision_version": str(torchvision.__version__),
              "decision": "argmax of two logits; no threshold selection", "positive_class": 1}
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    matrix = metrics["confusion_matrix"]
    axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=16,
                      color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benigno", "Maligno"],
             yticklabels=["Benigno", "Maligno"], xlabel="Predição", ylabel="Diagnóstico real",
             title="Teste oficial — MobileNetV3 Small")
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=False)
    try:
        figure.savefig(output / "confusion_matrix.png", dpi=180)
    finally:
        plt.close(figure)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Avaliação independente — Computer Vision", "",
             f"Checkpoint selecionado pela validação: época {checkpoint['epoch']}.",
             "Teste oficial: 378 recortes (231 benignos, 147 malignos).", "",
             *[f"- {name}: {metrics[name]:.4f}" for name in ("accuracy", "precision", "recall", "f1")],
             f"- TN={metrics['tn']}; FP={metrics['fp']}; FN={metrics['fn']}; TP={metrics['tp']}", "",
             "## Falsos negativos", "",
             f"{metrics['fn']} de 147 recortes malignos foram classificados como benignos "
             f"({metrics['fn'] / 147:.2%}). O recall maligno é {metrics['recall']:.2%}.",
             "Esses erros representam lesões malignas não identificadas neste experimento. "
             "As métricas são por recorte, não por paciente. Não permitem concluir segurança clínica "
             "nem eficácia de rastreamento. O modelo é um experimento acadêmico e não substitui avaliação médica.",
             "Sem inspeção individual dos casos, não é possível atribuir causas aos falsos negativos.", "",
             "Predição por maior logit; nenhum limiar ou hiperparâmetro foi ajustado no teste. "
             "Nenhum treinamento foi realizado nesta avaliação."]
    (output / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset-root", "manifest-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    result = evaluate(args.checkpoint, args.dataset_root, args.manifest_dir,
                      args.output_dir, args.device, args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
