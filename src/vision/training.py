"""Treinamento CV-8, somente treino/validação, com pesos ImageNet obrigatórios.

Importação e métricas não exigem PyTorch. A CLI só inicia épocas com --run;
sem essa opção prepara e apresenta a configuração, sem criar artefatos.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
from urllib.parse import urlparse


ARCHITECTURE = "mobilenet_v3_small_binary"
METRICS = ("loss", "accuracy", "precision", "recall", "f1")


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: str
    manifest_dir: str
    output_dir: str = "artifacts/vision/run_001"
    weights_path: str | None = None
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 10
    patience: int = 3
    min_delta: float = 1e-4
    device: str = "auto"
    trainable_feature_blocks: int = 0

    def validate(self):
        for name in ("seed", "num_workers", "trainable_feature_blocks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} deve ser inteiro não negativo.")
        for name in ("batch_size", "max_epochs", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} deve ser inteiro positivo.")
        for name in ("learning_rate", "weight_decay", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"{name} inválido.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device deve ser auto, cpu ou cuda.")


def require_torch():
    if any(importlib.util.find_spec(name) is None for name in ("torch", "torchvision")):
        raise ModuleNotFoundError("Treinamento não iniciado: PyTorch/torchvision ausentes. "
                                  "Use o ambiente separado do Colab; nada foi instalado.")
    import torch
    return torch


def classification_metrics(y_true, y_pred):
    """Métricas por imagem, classe positiva MALIGNANT=1; denominador zero -> 0."""
    actual, predicted = list(y_true), list(y_pred)
    if not actual or len(actual) != len(predicted):
        raise ValueError("Vetores de rótulos vazios ou com tamanhos diferentes.")
    if any(value not in (0, 1) for value in actual + predicted):
        raise ValueError("Rótulos devem ser 0 ou 1.")
    tp = sum(a == p == 1 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 1e-4
    best_score: float | None = None
    bad_epochs: int = 0

    def update(self, score):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("F1 de validação inválido.")
        improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_score, self.bad_epochs = float(score), 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def verify_weight_file(path, official_url):
    """Verifique o prefixo SHA-256 publicado no nome oficial; registre o hash completo."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Pesos ImageNet obrigatórios ausentes; features aleatórias não serão utilizadas.")
    url = urlparse(official_url)
    match = re.search(r"-([0-9a-f]{8,64})\.pth$", url.path)
    if url.scheme != "https" or url.netloc != "download.pytorch.org" or not match:
        raise ValueError("Origem/prefixo de pesos oficiais inválido.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if not digest.startswith(match[1]):
        raise ValueError("SHA-256 dos pesos não corresponde ao prefixo oficial; treinamento bloqueado.")
    return {"path": str(path), "url": official_url, "sha256": digest,
            "published_hash_prefix": match[1], "hash_verified": True}


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    import numpy as np
    np.random.seed(seed % (2**32))
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def select_device(request="auto"):
    torch = require_torch()
    if request not in {"auto", "cpu", "cuda"}:
        raise ValueError("Dispositivo inválido.")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponível.")
    return torch.device("cuda" if request != "cpu" and torch.cuda.is_available() else "cpu")


def validate_output_dir(config):
    output = Path(config.output_dir).expanduser().resolve()
    for protected in (config.dataset_root, config.manifest_dir):
        path = Path(protected).expanduser().resolve()
        if output == path or output.is_relative_to(path):
            raise ValueError("Artefatos devem ficar fora dos dados brutos e dos manifestos.")
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo) and not output.is_relative_to(repo / "artifacts/vision"):
        raise ValueError("Dentro do repositório, utilize artifacts/vision/ para proteger resultados.")
    if output.exists():
        raise FileExistsError("Diretório de execução já existe; escolha outro para não sobrescrever artefatos.")
    return output


def build_training_model(config):
    """Cheque hash oficial, carga estrita e igualdade dos tensores antes de treinar."""
    torch = require_torch()
    from torchvision.models import MobileNet_V3_Small_Weights
    from src.vision.model import build_model, resolve_pretrained_weights

    path = resolve_pretrained_weights(config.weights_path)
    provenance = verify_weight_file(path, MobileNet_V3_Small_Weights.IMAGENET1K_V1.url)
    model = build_model(pretrained=True, weights_path=path, freeze_features=True,
                        trainable_feature_blocks=config.trainable_feature_blocks)
    state = torch.load(path, map_location="cpu", weights_only=True)
    head_prefix = f"classifier.{len(model.classifier) - 1}."
    for name, tensor in model.state_dict().items():
        if not name.startswith(head_prefix) and (name not in state or not torch.equal(tensor.cpu(), state[name])):
            raise ValueError(f"Peso pré-treinado não foi preservado na inicialização: {name}")
    if model.classifier[-1].out_features != 2:
        raise ValueError("A cabeça deve produzir dois logits.")
    provenance["backbone_load_verified"] = True
    return model, provenance


@dataclass
class PreparedTraining:
    config: TrainingConfig
    model: object
    loaders: dict
    optimizer: object
    device: object
    metadata: dict


def prepare_training(config):
    """Monte a sessão sem épocas, forward, download ou criação do diretório de saída."""
    config.validate()
    validate_output_dir(config)
    seed_everything(config.seed)
    torch = require_torch()
    from src.vision.dataset import build_dataloaders
    from src.vision.model import parameter_counts

    model, weights = build_training_model(config)
    loaders = build_dataloaders(config.manifest_dir, config.dataset_root,
                                batch_size=config.batch_size, num_workers=config.num_workers,
                                seed=config.seed, splits=("train", "val"))
    for split, loader in loaders.items():
        if set(loader.dataset.records.label.astype(int)) != {0, 1}:
            raise ValueError(f"{split} deve conter ambas as classes.")
    device = select_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    manifest_hashes = {}
    for split in ("train", "val"):
        path = Path(config.manifest_dir) / f"manifest_{split}.csv"
        manifest_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    import torchvision
    metadata = {"config": asdict(config), "architecture": ARCHITECTURE, "seed": config.seed,
                "pretrained_weights": weights, "device": str(device),
                "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                "cuda_build": torch.version.cuda, "manifest_sha256": manifest_hashes,
                "parameters": parameter_counts(model),
                "partitions": {s: {"images": len(loader.dataset),
                                    "patients": int(loader.dataset.records.patient_id.nunique())}
                               for s, loader in loaders.items()},
                "selection": "max val_f1, improvement > min_delta; ties keep earlier epoch",
                "positive_class": 1, "test_used": False}
    return PreparedTraining(config, model, loaders, optimizer, device, metadata)


def run_epoch(model, loader, device, optimizer=None):
    """Uma passagem de treino ou validação; loss média ponderada por imagens."""
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, samples = 0.0, 0
    actual, predicted = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if labels.dtype != torch.int64 or labels.ndim != 1 or not torch.all((labels == 0) | (labels == 1)):
                raise ValueError("Alvos devem ser inteiros 0/1 em um vetor.")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if tuple(logits.shape) != (len(labels), 2) or not torch.isfinite(logits).all():
                raise ValueError("Logits inválidos ou não finitos.")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Loss não finita.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
            actual.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if not samples:
        raise ValueError("DataLoader vazio.")
    return {"loss": total_loss / samples, **classification_metrics(actual, predicted)}


def load_checkpoint(path, model, device="cpu"):
    torch = require_torch()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("format_version") != 1:
        raise ValueError("Checkpoint de arquitetura/formato incompatível.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return checkpoint


def fit(prepared):
    """Execute somente quando solicitado explicitamente; nunca recebe loader de teste."""
    config = prepared.config
    config.validate()
    if set(prepared.loaders) != {"train", "val"}:
        raise ValueError("Treinamento exige exclusivamente os loaders train e val.")
    provenance = prepared.metadata.get("pretrained_weights", {})
    if not provenance.get("hash_verified") or not provenance.get("backbone_load_verified"):
        raise ValueError("Inicialização ImageNet não verificada; treinamento bloqueado.")
    if config.trainable_feature_blocks == 0 and any(p.requires_grad for p in prepared.model.features.parameters()):
        raise ValueError("Features deveriam estar congeladas.")
    output = validate_output_dir(config)
    torch = require_torch()
    seed_everything(config.seed)
    for loader in prepared.loaders.values():
        generator = getattr(loader, "generator", None)
        if generator is not None:
            generator.manual_seed(config.seed)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(prepared.metadata, stream, ensure_ascii=False, indent=2)
    stopper = EarlyStopping(config.patience, config.min_delta)
    history, best_epoch, best_path = [], None, None
    try:
        for epoch in range(1, config.max_epochs + 1):
            train = run_epoch(prepared.model, prepared.loaders["train"], prepared.device, prepared.optimizer)
            val = run_epoch(prepared.model, prepared.loaders["val"], prepared.device)
            row = {"epoch": epoch, **{f"train_{m}": train[m] for m in METRICS},
                   **{f"val_{m}": val[m] for m in METRICS}}
            history.append(row)
            improved, stop = stopper.update(val["f1"])
            if improved:
                candidate = output / f"best_epoch_{epoch:03d}.pt"
                checkpoint = {"format_version": 1, "architecture": ARCHITECTURE, "epoch": epoch,
                              "model_state_dict": prepared.model.state_dict(),
                              "optimizer_state_dict": prepared.optimizer.state_dict(),
                              "metadata": prepared.metadata, "history": list(history),
                              "validation_metrics": val}
                with candidate.open("xb") as stream:
                    torch.save(checkpoint, stream)
                best_epoch, best_path = epoch, candidate
            history_temp = output / "history.csv.tmp"
            with history_temp.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            history_temp.replace(output / "history.csv")
            print(json.dumps(row), flush=True)
            if stop:
                break
        selected = load_checkpoint(best_path, prepared.model, str(prepared.device))
        summary = {"best_epoch": best_epoch, "best_checkpoint": best_path.name,
                   "validation_metrics": selected["validation_metrics"], "epochs_executed": len(history),
                   "stopped_early": len(history) < config.max_epochs, "test_used": False}
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        return {**summary, "history": history, "output_dir": str(output)}
    except Exception as exc:
        with (output / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"error": repr(exc), "completed_epochs": len(history)}, stream, indent=2)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/vision/run_001")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run", action="store_true", help="Autoriza iniciar épocas; sem esta flag só prepara.")
    args = parser.parse_args(argv)
    config = TrainingConfig(args.dataset_root, args.manifest_dir, args.output_dir, args.weights,
                            device=args.device, max_epochs=args.epochs,
                            learning_rate=args.learning_rate, batch_size=args.batch_size)
    try:
        prepared = prepare_training(config)
        print(json.dumps(prepared.metadata, ensure_ascii=False, indent=2))
        if args.run:
            fit(prepared)
        else:
            print("Configuração preparada. Nenhuma época executada; use --run somente após revisão.")
    except ModuleNotFoundError as exc:
        print(str(exc))
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Treinamento não concluído: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
