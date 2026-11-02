"""Inspecione CSVs de massas e associações JPEG sem modificar o dataset.

Execute: python -m src.vision.inspect_dataset CAMINHO_DO_DATASET
Somente dicom_info.csv com SeriesInstanceUID, SeriesDescription, PatientID e image_path
permite confirmar associações. Nenhum pareamento por basename é realizado.
"""

import argparse
from collections import Counter
import csv
from pathlib import Path, PurePosixPath
import re

import pandas as pd


PATH_FIELDS = ("image file path", "cropped image file path", "ROI mask file path")
REQUIRED_COLUMNS = {"patient_id", "pathology", *PATH_FIELDS}
LABELS = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}
CONVERSION_COLUMNS = {"SeriesInstanceUID", "SeriesDescription", "PatientID", "image_path"}
UID = re.compile(r"[0-9]+(?:\.[0-9]+)+")
PATIENT = re.compile(r"P_\d+")
MASS_CASE = re.compile(
    r"Mass-(Training|Test)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)_(\d+)"
)


def _read_csv(path, issues):
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), [])
        if len(header) != len(set(header)):
            issues.append(f"{path.name}: colunas duplicadas.")
            return None
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
        issues.append(f"Não foi possível ler {path.name}: {exc}")
        return None


def _jpeg_path(root, value):
    """Rebaseie somente pelo componente explícito jpeg, sem buscar por nome."""
    parts = PurePosixPath(value.replace("\\", "/")).parts
    positions = [i for i, part in enumerate(parts) if part.lower() == "jpeg"]
    if len(positions) > 1:
        return None
    if positions:
        parts = parts[positions[0]:]
    candidate = root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root) or candidate.suffix.lower() != ".jpg":
        return None
    return candidate


def _conversion_index(root, jpgs, issues):
    path = root / "csv" / "dicom_info.csv"
    if not path.is_file():
        issues.append("dicom_info.csv ausente: associação DICOM/JPEG não verificável.")
        return {}
    frame = _read_csv(path, issues)
    if frame is None:
        return {}
    missing = CONVERSION_COLUMNS.difference(frame.columns)
    if missing:
        issues.append(f"dicom_info.csv: colunas necessárias ausentes: {sorted(missing)}.")
        return {}
    index = {}
    image_identities = {}
    for number, row in enumerate(frame.to_dict("records"), 2):
        series = row["SeriesInstanceUID"].strip()
        image = _jpeg_path(root, row["image_path"].strip())
        if not UID.fullmatch(series) or image is None or image.parent.name != series:
            issues.append(f"dicom_info.csv linha {number}: UID/caminho JPEG inconsistente.")
            continue
        if image not in jpgs:
            issues.append(f"dicom_info.csv linha {number}: JPEG ausente: {image.relative_to(root)}.")
        description = " ".join(row["SeriesDescription"].casefold().split())
        patient_case = row["PatientID"].strip()
        identity = (series, description, patient_case)
        if image in image_identities and image_identities[image] != identity:
            issues.append(f"dicom_info.csv linha {number}: metadados conflitantes para o mesmo JPEG.")
        image_identities[image] = identity
        index.setdefault(series, set()).add((image, description, patient_case))
    return index


def inspect_dataset(dataset_root):
    """Retorne um relatório estruturado; inconsistências nunca corrigem os dados."""
    root = Path(dataset_root).resolve()
    report = {"root": str(root), "splits": {}, "overlap": None, "jpg_count": None,
              "records": [],
              "issues": []}
    issues = report["issues"]
    if not root.is_dir():
        issues.append(f"Dataset inexistente ou não é diretório: {root}")
        return report
    jpgs = {p.resolve() for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() == ".jpg" and p.resolve().is_relative_to(root)}
    report["jpg_count"] = len(jpgs)
    if not jpgs:
        issues.append("Nenhuma imagem JPG encontrada.")
    conversion = _conversion_index(root, jpgs, issues)
    used_images = {}
    patient_sets = {}
    for split in ("train", "test"):
        path = root / "csv" / f"mass_case_description_{split}_set.csv"
        if not path.is_file():
            issues.append(f"CSV obrigatório ausente: {path.name}")
            continue
        frame = _read_csv(path, issues)
        if frame is None:
            continue
        summary = {"records": len(frame), "columns": list(frame.columns),
                   "pathology": {}, "classes": {}, "patients": 0,
                   "usable_classes": {0: 0, 1: 0},
                   "path_fields": {}, "crop_mapping": dict.fromkeys(
                       ("matched", "unresolved", "ambiguous", "missing", "inconsistent", "reused"), 0)}
        report["splits"][split] = summary
        if frame.empty:
            issues.append(f"{path.name}: nenhum registro disponível.")
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            issues.append(f"{path.name}: colunas obrigatórias ausentes: {sorted(missing)}.")
        if "pathology" in frame:
            counts = Counter(frame["pathology"].str.strip())
            summary["pathology"] = dict(counts)
            summary["classes"] = {label: sum(n for name, n in counts.items()
                                             if LABELS.get(name) == label) for label in (0, 1)}
            unknown = set(counts).difference(LABELS)
            if unknown:
                issues.append(f"{path.name}: patologia desconhecida/ausente: {sorted(unknown)}.")
        if "patient_id" in frame:
            patients = frame["patient_id"].str.strip()
            valid = {p for p in patients if PATIENT.fullmatch(p)}
            patient_sets[split] = valid
            summary["patients"] = len(valid)
            invalid = set(patients).difference(valid)
            if invalid:
                issues.append(f"{path.name}: patient_id inválido/ausente: {sorted(invalid)}.")
        for field in PATH_FIELDS:
            if field in frame:
                values = frame[field].str.strip()
                summary["path_fields"][field] = {
                    "filled": int(values.ne("").sum()),
                    "unique": int(values[values.ne("")].nunique()),
                    "extensions": dict(Counter(PurePosixPath(v.replace("\\", "/")).suffix.lower()
                                               for v in values if v))}
                if values.eq("").any():
                    issues.append(f"{path.name}: {int(values.eq('').sum())} caminhos vazios em {field}.")
        if missing:
            continue
        for number, row in enumerate(frame.to_dict("records"), 2):
            context = f"{path.name} linha {number}"
            stats = summary["crop_mapping"]
            valid_record = True
            for field in PATH_FIELDS:
                found = set(PATIENT.findall(row[field]))
                if found and found != {row["patient_id"].strip()}:
                    issues.append(f"{context}: paciente incompatível no campo {field}.")
                    valid_record = False
            crop = PurePosixPath(row["cropped image file path"].strip().replace("\\", "/"))
            case_name = crop.parts[0] if crop.parts else ""
            case = MASS_CASE.fullmatch(case_name)
            if (case is None or case[2] != row["patient_id"].strip()
                    or case[1] != {"train": "Training", "test": "Test"}[split]):
                issues.append(f"{context}: identificador de caso/paciente/partição inconsistente.")
                valid_record = False
            if case is not None:
                for column, expected in (("left or right breast", case[3]),
                                         ("image view", case[4]), ("abnormality id", case[5])):
                    if column in row and row[column].strip() != expected:
                        issues.append(f"{context}: identificador de caso incompatível com {column}.")
                        valid_record = False
                for field in PATH_FIELDS:
                    reference = PurePosixPath(row[field].strip().replace("\\", "/"))
                    expected = case_name.rsplit("_", 1)[0] if field == "image file path" else case_name
                    if (len(reference.parts) != 4 or reference.parts[0] != expected
                            or not UID.fullmatch(reference.parent.name)
                            or not UID.fullmatch(reference.parent.parent.name)):
                        issues.append(f"{context}: caminho/UID/caso inconsistente em {field}.")
                        valid_record = False
            if row["pathology"].strip() not in LABELS:
                valid_record = False
            if not valid_record:
                stats["inconsistent"] += 1
                continue
            series = crop.parent.name
            candidates = {item for item in conversion.get(series, set())
                          if item[1] == "cropped images"}
            if not candidates:
                stats["unresolved"] += 1
                issues.append(f"{context}: correspondência não confirmada como cropped images por metadados.")
            elif len(candidates) > 1:
                stats["ambiguous"] += 1
                issues.append(f"{context}: caminhos de imagem ambíguos para a série {series}.")
            else:
                image, _, patient_case = next(iter(candidates))
                if patient_case != case_name:
                    stats["inconsistent"] += 1
                    issues.append(f"{context}: PatientID do JPEG incompatível com paciente/caso do CSV.")
                elif image not in jpgs:
                    stats["missing"] += 1
                    issues.append(f"{context}: JPEG do recorte ausente.")
                else:
                    identity = (split, row["patient_id"].strip(), row["pathology"].strip())
                    if image in used_images:
                        stats["reused"] += 1
                        issues.append(f"{context}: JPEG reutilizado; associação anterior {used_images[image]}.")
                    else:
                        stats["matched"] += 1
                        summary["usable_classes"][LABELS[identity[2]]] += 1
                        used_images[image] = identity
                        report["records"].append({
                            "patient_id": identity[1], "case_id": case_name,
                            "image_path": image.relative_to(root).as_posix(),
                            "pathology": identity[2], "label": LABELS[identity[2]],
                            "official_split": split,
                        })
    if {"train", "test"}.issubset(patient_sets):
        report["overlap"] = sorted(patient_sets["train"] & patient_sets["test"])
        if report["overlap"]:
            issues.append(f"Sobreposição de pacientes entre treino e teste: {report['overlap']}.")
    else:
        issues.append("Interseção de pacientes não verificável: CSV/coluna de paciente indisponível.")
    return report


def format_report(report):
    """Formate contagens verificadas e preserve todos os problemas encontrados."""
    jpg_count = report["jpg_count"]
    lines = [f"CBIS-DDSM — inspeção somente leitura: {report['root']}",
             f"Arquivos JPG disponíveis: {jpg_count if jpg_count is not None else 'não verificável'}"]
    for split, info in report["splits"].items():
        lines.extend([f"\n{split}: {info['records']} registros; {info['patients']} pacientes válidos únicos",
                      f"Colunas: {', '.join(info['columns'])}",
                      f"Pathology: {info['pathology']}", f"Classes (0=benigno, 1=maligno): {info['classes']}",
                      f"Recortes com associação validada por classe: {info['usable_classes']}",
                      f"Campos de caminho: {info['path_fields']}",
                      f"Recortes/JPEG: {info['crop_mapping']}"])
    overlap = report["overlap"]
    lines.extend([f"\nTotal de recortes com associação validada: {sum(s['crop_mapping']['matched'] for s in report['splits'].values())}",
                  f"Pacientes válidos em comum: {len(overlap) if overlap is not None else 'não verificável'}",
                  f"Problemas encontrados: {len(report['issues'])}"])
    lines.extend(f"- {issue}" for issue in report["issues"])
    lines.append("Contagem de JPGs não comprova imagens legíveis nem ausência de duplicatas de conteúdo.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Raiz contendo csv/ e jpeg/.")
    args = parser.parse_args(argv)
    report = inspect_dataset(args.dataset_root)
    print(format_report(report))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
