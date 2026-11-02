"""Dataset e DataLoaders da CV-5; nenhuma imagem é alterada ou pré-carregada.

Letterbox preserva toda a lesão: redimensiona proporcionalmente e centraliza em
224x224 com padding preto. Só o treino recebe rotação de até 5 graus, com canvas
expandido antes do letterbox para evitar corte. Não há crop, flip ou contraste.
A normalização é a de MobileNet_V3_Small_Weights.IMAGENET1K_V1.
"""

import csv
from itertools import combinations
from pathlib import Path, PurePosixPath
import random
import re

import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.transforms import InterpolationMode

from src.vision.inspect_dataset import LABELS, MASS_CASE, PATIENT


REQUIRED_FIELDS = {"patient_id", "case_id", "image_path", "pathology", "label",
                   "split", "official_split", "width", "height", "mode", "sha256"}
SPLITS = ("train", "val", "test")


class Letterbox:
    """Redimensionamento proporcional com padding, sem distorção ou recorte."""

    def __init__(self, size=224):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("size deve ser um inteiro positivo.")
        self.size = size

    def __call__(self, image):
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Imagem com dimensões inválidas.")
        scale = self.size / max(image.size)
        dimensions = tuple(max(1, min(self.size, round(value * scale))) for value in image.size)
        resized = image.resize(dimensions, Image.Resampling.BILINEAR)
        left = (self.size - resized.width) // 2
        top = (self.size - resized.height) // 2
        return ImageOps.expand(resized, (left, top, self.size - resized.width - left,
                                        self.size - resized.height - top), fill=0)


def build_transforms(training=False):
    normalization = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    steps = []
    if training:
        steps.append(transforms.RandomRotation(
            5, interpolation=InterpolationMode.BILINEAR, expand=True, fill=0))
    steps.extend([Letterbox(224), transforms.ToTensor(),
                  transforms.Normalize(mean=normalization.mean, std=normalization.std)])
    return transforms.Compose(steps)


class MammographyDataset(Dataset):
    """Leia metadados uma vez e abra um único JPG por acesso.

    A auditoria SHA-256/decodificação completa pertence à CV-4. Aqui validamos o
    manifesto e os caminhos; não recalculamos todos os hashes em cada época.
    Sem transform explícito, aplica-se o processamento determinístico de avaliação.
    """

    def __init__(self, manifest_path, dataset_root, transform=None, expected_split=None):
        self.root = Path(dataset_root).resolve()
        if not self.root.is_dir():
            raise ValueError("Raiz do dataset inexistente.")
        manifest_path = Path(manifest_path)
        with manifest_path.open(encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), [])
        if len(header) != len(set(header)):
            raise ValueError("Colunas duplicadas no manifesto.")
        self.records = pd.read_csv(manifest_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        missing = REQUIRED_FIELDS.difference(self.records.columns)
        if missing or self.records.empty:
            raise ValueError(f"Manifesto vazio ou campos essenciais ausentes: {sorted(missing)}")
        if self.records[list(REQUIRED_FIELDS)].apply(lambda c: c.str.strip().eq("")).any().any():
            raise ValueError("Campos essenciais vazios.")
        if self.records.case_id.duplicated().any() or self.records.image_path.duplicated().any():
            raise ValueError("Casos ou caminhos repetidos no manifesto.")
        if self.records.split.nunique() != 1 or self.records.split.iloc[0] not in SPLITS:
            raise ValueError("O manifesto deve conter uma única partição válida.")
        self.split = self.records.split.iloc[0]
        if expected_split is not None and self.split != expected_split:
            raise ValueError(f"Partição esperada: {expected_split}; encontrada: {self.split}.")
        self.paths = []
        for row in self.records.itertuples(index=False):
            if row.label not in {"0", "1"} or LABELS.get(row.pathology) != int(row.label):
                raise ValueError(f"Rótulo/patologia inválido: {row.case_id}")
            case = MASS_CASE.fullmatch(row.case_id)
            origin = "Test" if self.split == "test" else "Training"
            if (not PATIENT.fullmatch(row.patient_id) or case is None or case[2] != row.patient_id
                    or case[1] != origin or row.official_split != ("test" if self.split == "test" else "train")):
                raise ValueError(f"Paciente/caso/origem oficial inconsistente: {row.case_id}")
            relative = PurePosixPath(row.image_path)
            if (relative.is_absolute() or "\\" in row.image_path or ".." in relative.parts
                    or relative.as_posix() != row.image_path or len(relative.parts) != 3
                    or relative.parts[0] != "jpeg" or relative.suffix.lower() != ".jpg"):
                raise ValueError(f"Caminho relativo inválido: {row.image_path}")
            path = self.root.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Caminho fora da raiz ou imagem ausente: {row.image_path}")
            if not re.fullmatch(r"[0-9a-f]{64}", row.sha256):
                raise ValueError(f"SHA-256 inválido: {row.case_id}")
            if any(not str(value).isdigit() or int(value) <= 0 for value in (row.width, row.height)):
                raise ValueError(f"Dimensões inválidas no manifesto: {row.case_id}")
            self.paths.append(path)
        self.transform = transform if transform is not None else build_transforms()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = self.paths[index]
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Imagem passou a apontar para fora da raiz.")
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise ValueError(f"Conteúdo não JPEG: {path}")
            rgb = image.convert("RGB")
        with rgb:
            transformed = self.transform(rgb)
        return transformed, int(self.records.iloc[index].label)


def seed_worker(worker_id):
    """Função no topo do módulo para multiprocessing com spawn no Windows."""
    random.seed(torch.initial_seed() % (2**32))


def build_dataloaders(manifest_dir, dataset_root, batch_size=8, num_workers=0, seed=42,
                      splits=SPLITS):
    """Preserve as partições CV-4; embaralhe apenas o treino, sem novo split.

    num_workers=0 evita processos extras e facilita Windows/Colab com pouca RAM.
    Cada loader possui gerador próprio; avaliação não altera a ordem do treino.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size deve ser positivo.")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers deve ser inteiro não negativo.")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or not set(splits).issubset(SPLITS):
        raise ValueError("Partições solicitadas inválidas.")
    datasets = {split: MammographyDataset(
        Path(manifest_dir) / f"manifest_{split}.csv", dataset_root,
        transform=build_transforms(training=split == "train"), expected_split=split) for split in splits}
    for left, right in combinations(splits, 2):
        a, b = datasets[left].records, datasets[right].records
        for field in ("patient_id", "case_id", "image_path", "sha256"):
            if set(a[field]) & set(b[field]):
                raise ValueError(f"Compartilhamento de {field} entre {left} e {right}.")
    return {split: DataLoader(
        dataset, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers,
        drop_last=False, pin_memory=False, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed)) for split, dataset in datasets.items()}
