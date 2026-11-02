"""Preparação reproduzível com pequenos JPEGs sintéticos e pacientes agrupados."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image

from src.vision.inspect_dataset import inspect_dataset
from src.vision.prepare_manifest import (
    COLUMNS, SPLITS, main, prepare_manifest, split_by_patient,
)


class PrepareManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / "raw"
        (self.root / "csv").mkdir(parents=True)
        self.rows = {"train": [], "test": []}
        self.conversions = []
        self.images = []
        # Seis pacientes de cada fenótipo: benigno, maligno e misto.
        for patient in range(1, 19):
            labels = ["BENIGN"] if patient <= 6 else ["MALIGNANT"] if patient <= 12 else [
                "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"]
            for lesion, pathology in enumerate(labels, 1):
                self.add_record("train", patient, lesion, pathology)
        for patient, pathology in ((101, "BENIGN"), (102, "BENIGN_WITHOUT_CALLBACK"),
                                   (103, "MALIGNANT"), (104, "MALIGNANT")):
            self.add_record("test", patient, 1, pathology)
        self.write_csvs()

    def add_record(self, split, patient, lesion, pathology):
        number = len(self.images) + 1
        uid = f"1.2.{number}"
        prefix = "Training" if split == "train" else "Test"
        case = f"Mass-{prefix}_P_{patient:05d}_LEFT_CC_{lesion}"
        relative = f"jpeg/{uid}/crop.jpg"
        path = self.root / relative
        path.parent.mkdir(parents=True)
        pixels = np.random.default_rng(number).integers(0, 256, (12, 10, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path, format="JPEG")
        self.images.append(path)
        self.rows[split].append({
            "patient_id": f"P_{patient:05d}", "pathology": pathology,
            "image file path": f"{case.rsplit('_', 1)[0]}/1.5.1/1.5.2/full.dcm",
            "cropped image file path": f"{case}/1.5.1/{uid}/crop.dcm",
            "ROI mask file path": f"{case}/1.5.1/{uid}/mask.dcm",
        })
        self.conversions.append({"SeriesInstanceUID": uid, "SeriesDescription": "cropped images",
                                 "PatientID": case, "image_path": relative})
        mask_relative = f"jpeg/{uid}/mask.jpg"
        Image.new("L", (10, 12), 255).save(self.root / mask_relative, format="JPEG")
        self.conversions.append({**self.conversions[-1], "SeriesDescription": "ROI mask images",
                                 "image_path": mask_relative})

    def write_csvs(self):
        for split, rows in self.rows.items():
            pd.DataFrame(rows).to_csv(self.root / "csv" / f"mass_case_description_{split}_set.csv", index=False)
        pd.DataFrame(self.conversions).to_csv(self.root / "csv/dicom_info.csv", index=False)

    def prepare(self, **kwargs):
        return prepare_manifest(self.root, **kwargs)

    def test_labels_integrity_modes_and_dimensions(self):
        records, summary = self.prepare()
        self.assertTrue(summary["valid"], summary["issues"])
        self.assertEqual(summary["images_decoded"], 28)
        self.assertEqual(summary["image_errors"], 0)
        self.assertEqual(summary["image_modes"], {"RGB": 28})
        self.assertEqual(summary["dimensions"], {"width_min": 10, "width_max": 10,
                                                "height_min": 12, "height_max": 12, "unique_sizes": 1})
        self.assertTrue(records[records.pathology.isin(["BENIGN", "BENIGN_WITHOUT_CALLBACK"])].label.eq(0).all())
        self.assertTrue(records[records.pathology.eq("MALIGNANT")].label.eq(1).all())
        self.assertEqual(summary["unique_hashes"], 28)
        self.assertEqual(summary["duplicate_groups"], [])

    def test_official_test_preserved_and_no_patient_leakage(self):
        records, summary = self.prepare()
        self.assertTrue(summary["valid"])
        self.assertEqual(len(records[records.split.eq("test")]), 4)
        self.assertTrue(records[records.official_split.eq("test")].split.eq("test").all())
        self.assertFalse(records.groupby("patient_id").split.nunique().gt(1).any())
        self.assertEqual(summary["partitions"]["val"]["patients"], 4)
        for split in SPLITS:
            self.assertEqual(set(records.loc[records.split.eq(split), "label"]), {0, 1})

    def test_reproducible_even_if_input_order_changes_and_test_labels_change(self):
        source = pd.DataFrame(inspect_dataset(self.root)["records"])
        a, _ = split_by_patient(source)
        b, _ = split_by_patient(source.sample(frac=1, random_state=7))
        assert_frame_equal(a, b)
        changed = source.copy()
        changed.loc[changed.official_split.eq("test"), "label"] = 1 - changed.loc[
            changed.official_split.eq("test"), "label"]
        c, _ = split_by_patient(changed)
        self.assertEqual(a[a.official_split.eq("train")].split.tolist(),
                         c[c.official_split.eq("train")].split.tolist())

    def test_mixed_diagnoses_and_multiple_lesions_stay_together(self):
        records, summary = self.prepare()
        mixed = records[records.patient_id.eq("P_00013")]
        self.assertEqual(len(mixed), 2)
        self.assertEqual(set(mixed.label), {0, 1})
        self.assertEqual(mixed.split.nunique(), 1)
        self.assertEqual(summary["split_criteria"]["patient_phenotypes"]["mixed"], 6)

    def test_rare_mixed_patient_uses_deterministic_group_fallback(self):
        source = pd.DataFrame(inspect_dataset(self.root)["records"])
        source = source[~source.patient_id.isin([f"P_{p:05d}" for p in range(14, 19)])]
        a, criteria = split_by_patient(source)
        b, _ = split_by_patient(source)
        assert_frame_equal(a, b)
        self.assertEqual(criteria["strategy"], "patient_group_search")
        self.assertEqual(a[a.patient_id.eq("P_00013")].split.nunique(), 1)

    def test_impossible_class_coverage_fails_without_splitting_patients(self):
        source = pd.DataFrame(inspect_dataset(self.root)["records"])
        source = source[source.patient_id.isin(["P_00001", "P_00007", "P_00101", "P_00103"])]
        with self.assertRaisesRegex(ValueError, "Não foi possível"):
            split_by_patient(source)

    def test_missing_image_stops_before_output(self):
        self.images[0].unlink()
        records, summary = self.prepare(output_dir=self.base / "out")
        self.assertIsNone(records)
        self.assertFalse(summary["valid"])
        self.assertTrue(any("ausente" in issue for issue in summary["issues"]))
        self.assertFalse((self.base / "out").exists())

    def test_corrupt_and_truncated_images_are_reported_and_no_output(self):
        self.images[0].write_bytes(b"not a JPEG")
        self.images[1].write_bytes(self.images[1].read_bytes()[:-30])
        records, summary = self.prepare(output_dir=self.base / "out")
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["image_errors"], 2)
        self.assertEqual(summary["images_decoded"], 26)
        self.assertEqual(len(records), 28)
        self.assertFalse((self.base / "out").exists())

    def test_exact_duplicate_between_partitions_blocks_output(self):
        self.images[-4].write_bytes(self.images[0].read_bytes())
        records, summary = self.prepare(output_dir=self.base / "out")
        self.assertEqual(len(records), 28)
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["unique_hashes"], 27)
        self.assertEqual(len(summary["duplicate_groups"]), 1)
        self.assertTrue(any("entre partições" in issue for issue in summary["issues"]))
        self.assertFalse((self.base / "out").exists())

    def test_same_patient_distinct_images_are_not_content_duplicates(self):
        records, summary = self.prepare()
        self.assertTrue(summary["valid"])
        self.assertTrue(records.patient_id.duplicated().any())
        self.assertFalse(records.image_path.duplicated().any())
        self.assertFalse(records.sha256.duplicated().any())

    def test_repeated_paths_and_case_ids_rejected(self):
        source = pd.DataFrame(inspect_dataset(self.root)["records"])
        with self.assertRaisesRegex(ValueError, "caso repetidos"):
            split_by_patient(pd.concat([source, source.iloc[[0]]]))
        source.loc[1, "image_path"] = source.loc[0, "image_path"]
        with self.assertRaisesRegex(ValueError, "Caminhos de imagem repetidos"):
            split_by_patient(source)

    def test_portable_outputs_no_masks_and_raw_unchanged(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = self.base / "out"
        records, summary = self.prepare(output_dir=output)
        self.assertTrue(summary["valid"])
        restored = []
        for split in SPLITS:
            part = pd.read_csv(output / f"manifest_{split}.csv")
            self.assertEqual(list(part.columns), COLUMNS)
            restored.append(part)
            for value in part.image_path:
                relative = PurePosixPath(value)
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("\\", value)
                self.assertEqual(relative.name, "crop.jpg")
                self.assertTrue(self.root.joinpath(*relative.parts).is_file())
        self.assertEqual(sum(map(len, restored)), len(records))
        self.assertTrue((output / "preparation_summary.json").is_file())
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes()
                                  for p in self.root.rglob("*") if p.is_file()})

    def test_output_cannot_be_written_into_raw_dataset_or_overwrite_existing(self):
        with self.assertRaisesRegex(ValueError, "fora do dataset bruto"):
            self.prepare(output_dir=self.root / "out")
        output = self.base / "out"
        output.mkdir()
        previous = output / "manifest_train.csv"
        previous.write_text("preserve me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare(output_dir=output)
        self.assertEqual(previous.read_text(encoding="utf-8"), "preserve me")

    def test_official_patient_overlap_blocks_preparation(self):
        self.rows["test"][0]["patient_id"] = "P_00001"
        self.write_csvs()
        _, summary = self.prepare(output_dir=self.base / "out")
        self.assertFalse(summary["valid"])
        self.assertFalse((self.base / "out").exists())

    def test_cli_error_and_success_codes(self):
        output = self.base / "out"
        with redirect_stdout(StringIO()):
            self.assertEqual(main([str(self.root), "--output-dir", str(output)]), 0)
            self.assertEqual(main([str(self.root / "absent"), "--output-dir", str(output)]), 1)


if __name__ == "__main__":
    unittest.main()
