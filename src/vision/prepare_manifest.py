"""Prepare manifestos de recortes de massas, sem alterar imagens ou treinar.

python -m src.vision.prepare_manifest data/vision/raw/cbis_ddsm
Somente grava saídas se metadados, decodificação e partições forem válidos.
"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path, PurePosixPath
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.vision.inspect_dataset import inspect_dataset


SPLITS = ("train", "val", "test")
COLUMNS = ["patient_id", "case_id", "image_path", "pathology", "label",
           "split", "official_split", "width", "height", "mode", "sha256"]
SPLIT_TRIALS = 256


def split_by_patient(records, seed=42, val_fraction=0.2):
    """Divida somente o treino oficial; pacientes mistos formam um terceiro estrato.

    Compare até 256 candidatos com seed fixa, usando apenas o treino oficial.
    Priorize a proporção de classes por imagem, depois a fração de imagens.
    Estratos raros usam busca agrupada sem estratificação. Nenhuma classe pode
    faltar. A seleção nunca consulta os rótulos do teste oficial.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("A fração de validação deve estar entre zero e um.")
    if records.empty or records["case_id"].duplicated().any():
        raise ValueError("Registros vazios ou identificadores de caso repetidos.")
    if records["image_path"].duplicated().any():
        raise ValueError("Caminhos de imagem repetidos.")
    if not records["official_split"].isin(["train", "test"]).all():
        raise ValueError("Partição oficial desconhecida.")
    if not records["label"].isin([0, 1]).all():
        raise ValueError("Classe desconhecida.")
    origin_counts = records.groupby("patient_id")["official_split"].nunique()
    if origin_counts.gt(1).any():
        raise ValueError("Paciente compartilhado entre treino e teste oficiais.")
    training = records[records.official_split.eq("train")]
    counts = pd.crosstab(training.patient_id, training.label).reindex(columns=[0, 1], fill_value=0)
    counts = counts.sort_index()
    n_patients = len(counts)
    n_val = math.ceil(n_patients * val_fraction)
    if n_val < 1 or n_val >= n_patients:
        raise ValueError("Pacientes insuficientes para separar treino e validação.")
    strata = counts.apply(lambda row: "mixed" if (row > 0).all()
                          else "benign" if row[0] > 0 else "malignant", axis=1)
    frequencies = strata.value_counts()
    stratified = bool(frequencies.min() >= 2 and len(frequencies) <= min(n_val, n_patients - n_val))
    total_counts = counts.sum().to_numpy()
    if (total_counts == 0).any():
        raise ValueError("O treino oficial precisa conter ambas as classes.")
    baseline_ratio = total_counts[1] / total_counts.sum()
    best = None
    generator = np.random.default_rng(seed)
    for _ in range(SPLIT_TRIALS):
        trial_seed = int(generator.integers(0, 2**31 - 1))
        _, validation_ids = train_test_split(
            counts.index.to_numpy(), test_size=n_val, random_state=trial_seed,
            stratify=strata.to_numpy() if stratified else None)
        val_counts = counts.loc[validation_ids].sum().to_numpy()
        train_counts = total_counts - val_counts
        if (val_counts == 0).any() or (train_counts == 0).any():
            continue
        score = (max(abs(val_counts[1] / val_counts.sum() - baseline_ratio),
                     abs(train_counts[1] / train_counts.sum() - baseline_ratio)),
                 abs(val_counts.sum() / total_counts.sum() - val_fraction),
                 tuple(sorted(validation_ids)))
        if best is None or score < best[0]:
            best = (score, set(validation_ids))
    if best is None:
        raise ValueError("Não foi possível manter ambas as classes sem dividir pacientes; "
                         "nenhum manifesto será gravado.")
    result = records.copy()
    result["split"] = "train"
    result.loc[result.patient_id.isin(best[1]), "split"] = "val"
    result.loc[result.official_split.eq("test"), "split"] = "test"
    return result.sort_values(["split", "patient_id", "case_id"]).reset_index(drop=True), {
        "seed": seed, "val_patient_fraction": val_fraction, "candidates": SPLIT_TRIALS,
        "strategy": "patient_phenotype_stratified_search" if stratified else "patient_group_search",
        "selection": "train-only class-ratio deviation, then image-fraction deviation",
        "patient_phenotypes": {str(k): int(v) for k, v in frequencies.items()},
    }


def audit_images(records, dataset_root, progress=None):
    """Decodifique e calcule hashes incrementalmente, com uma imagem por vez."""
    root = Path(dataset_root).resolve()
    result = records.copy()
    for column in ("width", "height", "mode", "sha256"):
        result[column] = None
    issues = []
    for position, (index, row) in enumerate(result.iterrows(), 1):
        relative = PurePosixPath(row.image_path)
        path = root.joinpath(*relative.parts).resolve()
        try:
            if relative.is_absolute() or "\\" in row.image_path or not path.is_relative_to(root):
                raise ValueError("Caminho não relativo/portável ou fora da raiz do dataset.")
            if not path.is_file():
                raise ValueError("Arquivo ausente.")
            # Um único handle e comparação de stat detectam alterações durante a leitura.
            with path.open("rb") as source:
                before = path.stat()
                digest = hashlib.file_digest(source, "sha256").hexdigest()
                source.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(source) as image:
                        if image.format != "JPEG":
                            raise ValueError("O arquivo não contém uma imagem JPEG.")
                        image.load()
                        width, height = image.size
                        mode = image.mode
                        if width <= 0 or height <= 0:
                            raise ValueError("Dimensões inválidas.")
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Arquivo alterado durante a auditoria.")
            result.loc[index, ["width", "height", "mode", "sha256"]] = [width, height, mode, digest]
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            issues.append(f"{row.case_id}: {row.image_path}: {type(exc).__name__}: {exc}")
        if progress is not None and (position % 100 == 0 or position == len(result)):
            progress(position, len(result))
    return result, issues


def validate_partitions(records):
    """Verifique grupos, rótulos, origem oficial e duplicatas exatas entre partições."""
    issues = []
    if records.case_id.duplicated().any():
        issues.append("Identificadores de caso repetidos.")
    if records.image_path.duplicated().any():
        issues.append("Caminhos de imagem repetidos.")
    for left, right in combinations(SPLITS, 2):
        a = set(records.loc[records.split.eq(left), "patient_id"])
        b = set(records.loc[records.split.eq(right), "patient_id"])
        if a & b:
            issues.append(f"Vazamento de pacientes entre {left} e {right}: {sorted(a & b)}")
    if not records.loc[records.official_split.eq("test"), "split"].eq("test").all():
        issues.append("Registros do teste oficial foram retirados do teste.")
    if not records.loc[records.split.eq("test"), "official_split"].eq("test").all():
        issues.append("O teste contém registros de outra origem oficial.")
    for split in SPLITS:
        if set(records.loc[records.split.eq(split), "label"]) != {0, 1}:
            issues.append(f"A partição {split} não contém ambas as classes.")
    duplicates = []
    for digest, group in records.dropna(subset=["sha256"]).groupby("sha256", sort=True):
        if len(group) < 2:
            continue
        duplicate = {"sha256": digest, "cases": group.case_id.tolist(),
                     "splits": sorted(group.split.unique().tolist()),
                     "patients": sorted(group.patient_id.unique().tolist())}
        duplicates.append(duplicate)
        if group.split.nunique() > 1:
            issues.append(f"Duplicata SHA-256 entre partições: {duplicate}")
        if group.label.nunique() > 1:
            issues.append(f"Duplicata SHA-256 com diagnósticos conflitantes: {duplicate}")
    return duplicates, issues


def prepare_manifest(dataset_root, output_dir=None, seed=42, val_fraction=0.2, progress=None):
    """Audite tudo antes de criar saídas; em falha retorne diagnóstico sem CSVs novos."""
    inspection = inspect_dataset(dataset_root)
    summary = {"valid": False, "issues": list(inspection["issues"]),
               "association_records": len(inspection["records"])}
    if summary["issues"]:
        return None, summary
    records = pd.DataFrame(inspection["records"])
    try:
        records, split_info = split_by_patient(records, seed, val_fraction)
    except ValueError as exc:
        summary["issues"].append(str(exc))
        return None, summary
    if progress is not None:
        progress(0, len(records))
    records, image_issues = audit_images(records, dataset_root, progress)
    duplicates, partition_issues = validate_partitions(records)
    summary["issues"].extend(image_issues + partition_issues)
    decoded = records.dropna(subset=["width", "height", "mode", "sha256"])
    summary.update({
        "split_criteria": split_info, "images_audited": len(records),
        "images_decoded": len(decoded), "image_errors": len(image_issues),
        "unique_hashes": int(decoded.sha256.nunique()), "duplicate_groups": duplicates,
        "patient_overlap": any(records.groupby("patient_id")["split"].nunique().gt(1)),
        "image_modes": dict(Counter(decoded["mode"])),
        "dimensions": {"width_min": int(decoded.width.min()), "width_max": int(decoded.width.max()),
                       "height_min": int(decoded.height.min()), "height_max": int(decoded.height.max()),
                       "unique_sizes": len(decoded[["width", "height"]].drop_duplicates())} if len(decoded) else {},
        "partitions": {split: {
            "images": len(part), "patients": int(part.patient_id.nunique()),
            "classes": {str(k): int(v) for k, v in part.label.value_counts().sort_index().items()},
            "pathology": {str(k): int(v) for k, v in part.pathology.value_counts().sort_index().items()},
        } for split in SPLITS for part in [records[records.split.eq(split)]]},
        "hash_scope": "SHA-256 dos bytes do arquivo; sem similaridade perceptual",
    })
    summary["valid"] = not summary["issues"]
    if summary["valid"] and output_dir is not None:
        output = Path(output_dir).resolve()
        root = Path(dataset_root).resolve()
        if output == root or output.is_relative_to(root):
            raise ValueError("As saídas devem ficar fora do dataset bruto.")
        # Não sobrescreva um conjunto anterior; evita misturar gerações ou ocultar falhas.
        targets = [output / f"manifest_{split}.csv" for split in SPLITS] + [output / "preparation_summary.json"]
        if any(path.exists() for path in targets):
            raise FileExistsError("Saídas já existem; escolha outro diretório de saída.")
        output.mkdir(parents=True, exist_ok=True)
        for split, target in zip(SPLITS, targets):
            records.loc[records.split.eq(split), COLUMNS].to_csv(target, index=False, encoding="utf-8")
        targets[-1].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/vision/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        _, summary = prepare_manifest(
            args.dataset_root, args.output_dir, args.seed, args.val_fraction,
            progress=lambda n, total: print(f"Auditoria JPEG: {n}/{total}", flush=True))
    except (OSError, ValueError) as exc:
        print(f"Preparação não concluída: {exc}")
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        print("Preparação inválida: nenhum manifesto novo foi gravado; saídas anteriores não foram alteradas.")
    else:
        print(f"Manifestos gravados em: {args.output_dir.resolve()}")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
