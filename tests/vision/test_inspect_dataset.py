"""Inspeção de metadados sintéticos, sem imagens médicas ou dependências DL."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.vision.inspect_dataset import inspect_dataset, main


class InspectDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "csv").mkdir()
        self.rows = {
            "train": [self.row("P_00001", "BENIGN", "1.2.10"),
                      self.row("P_00001", "BENIGN_WITHOUT_CALLBACK", "1.2.11"),
                      self.row("P_00002", "MALIGNANT", "1.2.12")],
            "test": [self.row("P_00003", "MALIGNANT", "1.2.13", "Test")],
        }
        self.conversions = []
        for rows in self.rows.values():
            for row in rows:
                uid = Path(row["cropped image file path"]).parent.name
                relative = f"jpeg/{uid}/image.jpg"
                image = self.root / relative
                image.parent.mkdir(parents=True)
                # O inspetor verifica referências e existência, não decodifica pixels.
                image.write_bytes(b"synthetic placeholder, not an encoded image")
                self.conversions.append({"SeriesInstanceUID": uid,
                                         "SeriesDescription": "cropped images",
                                         "PatientID": row["cropped image file path"].split("/")[0],
                                         "image_path": relative})
        self.write_csvs()

    @staticmethod
    def row(patient, pathology, uid, split="Training"):
        return {"patient_id": patient, "pathology": pathology,
                "image file path": f"Mass-{split}_{patient}_LEFT_CC/1.2.1/1.2.2/full.dcm",
                "cropped image file path": f"Mass-{split}_{patient}_LEFT_CC_1/1.2.3/{uid}/crop.dcm",
                "ROI mask file path": f"Mass-{split}_{patient}_LEFT_CC_1/1.2.3/1.2.4/mask.dcm"}

    def write_csvs(self):
        for split, rows in self.rows.items():
            pd.DataFrame(rows).to_csv(
                self.root / "csv" / f"mass_case_description_{split}_set.csv", index=False)
        pd.DataFrame(self.conversions).to_csv(self.root / "csv/dicom_info.csv", index=False)

    def inspect(self):
        return inspect_dataset(self.root)

    def test_nonexistent_dataset_does_not_create_directory(self):
        path = self.root / "absent"
        report = inspect_dataset(path)
        self.assertIn("Dataset inexistente", report["issues"][0])
        self.assertIsNone(report["jpg_count"])
        self.assertIsNone(report["overlap"])
        self.assertFalse(path.exists())

    def test_required_csv_missing(self):
        (self.root / "csv/mass_case_description_train_set.csv").unlink()
        self.assertTrue(any("CSV obrigatório ausente" in s for s in self.inspect()["issues"]))
        self.assertIsNone(self.inspect()["overlap"])

    def test_required_columns_missing(self):
        del self.rows["train"][0]["pathology"]
        del self.rows["train"][1]["pathology"]
        del self.rows["train"][2]["pathology"]
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["records"], 3)
        self.assertTrue(any("colunas obrigatórias ausentes" in s and "pathology" in s
                            for s in report["issues"]))

    def test_duplicate_headers_are_not_silently_renamed(self):
        (self.root / "csv/mass_case_description_train_set.csv").write_text(
            "patient_id,pathology,pathology\nP_00001,BENIGN,MALIGNANT\n", encoding="utf-8")
        report = self.inspect()
        self.assertTrue(any("colunas duplicadas" in s for s in report["issues"]))
        self.assertNotIn("train", report["splits"])

    def test_counts_classes_paths_and_reliable_mapping(self):
        report = self.inspect()
        train = report["splits"]["train"]
        self.assertEqual(report["issues"], [])
        self.assertEqual((train["records"], train["patients"]), (3, 2))
        self.assertEqual(report["splits"]["test"]["patients"], 1)
        self.assertEqual(train["pathology"],
                         {"BENIGN": 1, "BENIGN_WITHOUT_CALLBACK": 1, "MALIGNANT": 1})
        self.assertEqual(train["classes"], {0: 2, 1: 1})
        self.assertEqual(train["usable_classes"], {0: 2, 1: 1})
        self.assertEqual(report["overlap"], [])
        self.assertEqual(report["jpg_count"], 4)
        self.assertEqual(train["crop_mapping"]["matched"], 3)
        self.assertEqual(train["path_fields"]["cropped image file path"]["unique"], 3)

    def test_patient_overlap_across_views_and_splits(self):
        self.rows["test"] = [self.row("P_00001", "MALIGNANT", "1.2.13", "Test")]
        self.conversions[-1]["PatientID"] = "Mass-Test_P_00001_LEFT_CC_1"
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["overlap"], ["P_00001"])
        self.assertTrue(any("Sobreposição" in s for s in report["issues"]))
        with redirect_stdout(StringIO()):
            self.assertEqual(main([str(self.root)]), 1)

    def test_unknown_and_empty_pathology_are_not_benign(self):
        self.rows["train"][0]["pathology"] = "UNKNOWN"
        self.rows["train"][1]["pathology"] = ""
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["classes"], {0: 0, 1: 1})
        self.assertTrue(any("patologia desconhecida" in s for s in report["issues"]))

    def test_no_images(self):
        for image in self.root.rglob("*.jpg"):
            image.unlink()
        report = self.inspect()
        self.assertEqual(report["jpg_count"], 0)
        self.assertIn("Nenhuma imagem JPG encontrada.", report["issues"])
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["missing"], 3)

    def test_ambiguous_conversion_series(self):
        self.conversions.append({**self.conversions[0], "image_path": "jpeg/1.2.10/other.jpg"})
        (self.root / "jpeg/1.2.10/other.jpg").write_bytes(b"other")
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["ambiguous"], 1)
        self.assertTrue(any("ambíguos" in s for s in report["issues"]))

    def test_basename_is_never_a_fallback(self):
        (self.root / "csv/dicom_info.csv").unlink()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["matched"], 0)
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["unresolved"], 3)
        self.assertTrue(any("dicom_info.csv ausente" in s for s in report["issues"]))

    def test_missing_conversion_columns(self):
        pd.DataFrame({"image_path": ["jpeg/1.2.10/image.jpg"]}).to_csv(
            self.root / "csv/dicom_info.csv", index=False)
        self.assertTrue(any("dicom_info.csv: colunas necessárias ausentes" in s
                            for s in self.inspect()["issues"]))

    def test_masks_are_not_accepted_as_crops(self):
        self.conversions[0]["SeriesDescription"] = "ROI mask images"
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["unresolved"], 1)
        self.assertTrue(any("não confirmada como cropped" in s for s in report["issues"]))

    def test_selects_crop_among_masks_and_full_mammograms_in_same_series(self):
        for description, name in (("ROI mask images", "mask.jpg"),
                                  ("full mammogram images", "full.jpg")):
            relative = f"jpeg/1.2.10/{name}"
            (self.root / relative).write_bytes(b"not a crop")
            self.conversions.append({**self.conversions[0],
                                     "SeriesDescription": description, "image_path": relative})
        self.conversions[0]["SeriesDescription"] = "  CrOpPeD \t IMAGES  "
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["matched"], 3)
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["ambiguous"], 0)
        # Uma máscara presente não pode substituir o recorte ausente.
        (self.root / "jpeg/1.2.10/image.jpg").unlink()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["missing"], 1)
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["matched"], 2)

    def test_conversion_patient_and_case_must_match_without_fallback(self):
        for patient_case in ("Mass-Training_P_99999_LEFT_CC_1",
                             "Mass-Training_P_00001_RIGHT_CC_1",
                             "Mass-Training_P_00001_LEFT_MLO_1",
                             "Mass-Training_P_00001_LEFT_CC_2", ""):
            with self.subTest(patient_case=patient_case):
                self.conversions[0]["PatientID"] = patient_case
                self.write_csvs()
                report = self.inspect()
                self.assertEqual(report["splits"]["train"]["crop_mapping"]["inconsistent"], 1)
                self.assertEqual(report["splits"]["train"]["crop_mapping"]["matched"], 2)
                self.assertTrue(any("PatientID do JPEG incompatível" in s for s in report["issues"]))
                with redirect_stdout(StringIO()):
                    self.assertEqual(main([str(self.root)]), 1)

    def test_multiple_crops_not_resolved_by_selecting_a_matching_patient(self):
        self.conversions.append({**self.conversions[0], "image_path": "jpeg/1.2.10/other.jpg",
                                 "PatientID": "Mass-Training_P_99999_LEFT_CC_1"})
        (self.root / "jpeg/1.2.10/other.jpg").write_bytes(b"other crop")
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["ambiguous"], 1)
        with redirect_stdout(StringIO()):
            self.assertEqual(main([str(self.root)]), 1)

    def test_missing_conversion_patient_column_is_an_error(self):
        for row in self.conversions:
            del row["PatientID"]
        self.write_csvs()
        report = self.inspect()
        self.assertTrue(any("colunas necessárias ausentes" in s and "PatientID" in s
                            for s in report["issues"]))

    def test_conflicting_metadata_for_same_jpeg_is_an_error(self):
        self.conversions.append({**self.conversions[0], "SeriesDescription": "ROI mask images"})
        self.write_csvs()
        report = self.inspect()
        self.assertTrue(any("metadados conflitantes" in s for s in report["issues"]))
        with redirect_stdout(StringIO()):
            self.assertEqual(main([str(self.root)]), 1)

    def test_official_partition_and_case_columns_are_checked(self):
        self.rows["train"][0] = self.row("P_00001", "BENIGN", "1.2.10", "Test")
        for row in self.rows["train"]:
            row["abnormality id"] = "1"
        self.rows["train"][1]["abnormality id"] = "9"
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["inconsistent"], 2)
        self.assertTrue(any("partição inconsistente" in s for s in report["issues"]))

    def test_roi_mask_missing_is_still_reported(self):
        self.conversions.append({**self.conversions[0], "SeriesDescription": "ROI mask images",
                                 "image_path": "jpeg/1.2.10/missing-mask.jpg"})
        self.write_csvs()
        report = self.inspect()
        self.assertTrue(any("JPEG ausente" in s and "missing-mask.jpg" in s for s in report["issues"]))
        with redirect_stdout(StringIO()):
            self.assertEqual(main([str(self.root)]), 1)

    def test_rebases_explicit_jpeg_prefix_and_windows_separators(self):
        self.conversions[0]["image_path"] = r"CBIS-DDSM\jpeg\1.2.10\image.jpg"
        self.write_csvs()
        self.assertEqual(self.inspect()["issues"], [])

    def test_rejects_conversion_uid_disagreement_and_escape(self):
        self.conversions[0]["image_path"] = "jpeg/1.2.11/image.jpg"
        self.conversions[1]["image_path"] = "../outside.jpg"
        self.write_csvs()
        report = self.inspect()
        self.assertEqual(report["splits"]["train"]["crop_mapping"]["unresolved"], 2)
        self.assertEqual(sum("UID/caminho JPEG inconsistente" in s for s in report["issues"]), 2)

    def test_reused_image_is_reported(self):
        self.rows["train"][1] = self.rows["train"][0].copy()
        self.write_csvs()
        self.assertTrue(any("JPEG reutilizado" in s for s in self.inspect()["issues"]))

    def test_invalid_patient_and_conflicting_path_patient(self):
        self.rows["train"][0]["patient_id"] = "Mass-Training_P_00001_LEFT_CC"
        self.rows["train"][1]["image file path"] = "Mass-Training_P_99999_LEFT_CC/full.dcm"
        self.write_csvs()
        report = self.inspect()
        self.assertTrue(any("patient_id inválido" in s for s in report["issues"]))
        self.assertTrue(any("paciente incompatível" in s for s in report["issues"]))

    def test_empty_paths_and_malformed_csv(self):
        self.rows["train"][0]["cropped image file path"] = ""
        self.write_csvs()
        self.assertTrue(any("caminhos vazios" in s for s in self.inspect()["issues"]))
        (self.root / "csv/mass_case_description_test_set.csv").write_text('"unterminated')
        report = self.inspect()
        self.assertTrue(any("Não foi possível ler" in s for s in report["issues"]))

    def test_cli_summary_exit_codes_and_dataset_unchanged(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([str(self.root)]), 0)
        self.assertIn("3 registros; 2 pacientes", output.getvalue())
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        with redirect_stdout(StringIO()):
            self.assertEqual(main([str(self.root / "absent")]), 1)


if __name__ == "__main__":
    unittest.main()
