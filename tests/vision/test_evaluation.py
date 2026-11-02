"""Fixtures sintéticas; sem rede, GPU, dados médicos ou treinamento."""

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from src.vision.evaluation import final_metrics, prepare_test_package, load_evaluation_model, infer, evaluate
from src.vision.training import ARCHITECTURE

HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))
if HAS_TORCH:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.vision.model import build_model


class EvaluationUtilityTests(unittest.TestCase):
    def test_metrics_and_matrix_orientation(self):
        result = final_metrics([0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 3]])
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (2, 1, 1, 3))
        self.assertEqual(result["accuracy"], 5 / 7)
        self.assertEqual(result["recall"], 3 / 4)
        self.assertEqual(result["precision"], 3 / 4)
        self.assertEqual(result["f1"], 3 / 4)

    def test_asymmetric_matrix_and_zero_denominators(self):
        result = final_metrics([0, 0, 1, 1, 1], [0, 1, 0, 0, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [2, 1]])
        self.assertEqual(final_metrics([0, 1], [0, 0])["precision"], 0)
        for actual, predicted in (([], []), ([0], [1, 0]), ([2], [0])):
            with self.assertRaises(ValueError):
                final_metrics(actual, predicted)

    def test_missing_checkpoint(self):
        with TemporaryDirectory() as directory, self.assertRaisesRegex(FileNotFoundError, "ausente"):
            load_evaluation_model(Path(directory) / "absent.pt")

    def make_zip(self, path, relative="jpeg/uid/test.jpg", split="test", digest=None):
        from PIL import Image
        image = io.BytesIO()
        Image.new("L", (11, 17), 70).save(image, format="JPEG")
        payload = image.getvalue()
        row = {"split": split, "official_split": "test", "image_path": relative,
               "sha256": digest or hashlib.sha256(payload).hexdigest()}
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("processed/manifest_test.csv", stream.getvalue())
            archive.writestr("cbis_ddsm/" + relative, payload)
            archive.writestr("processed/manifest_train.csv", "MUST NOT READ")
            archive.writestr("processed/manifest_val.csv", "MUST NOT READ")
            archive.writestr("cbis_ddsm/jpeg/train/a.jpg", "MUST NOT READ")

    def test_extracts_only_test_and_refuses_overwrite(self):
        with TemporaryDirectory() as directory:
            package, target = Path(directory) / "data.zip", Path(directory) / "test"
            self.make_zip(package)
            original_open = zipfile.ZipFile.open
            def guarded(archive, name, *args, **kwargs):
                self.assertNotIn("train", str(name))
                self.assertNotIn("val", str(name))
                return original_open(archive, name, *args, **kwargs)
            with patch.object(zipfile.ZipFile, "open", guarded):
                prepare_test_package(package, target)
            self.assertEqual(len(list(target.rglob("*.jpg"))), 1)
            self.assertEqual(len(list(target.rglob("*.csv"))), 1)
            with self.assertRaises(FileExistsError):
                prepare_test_package(package, target)

    def test_corrupt_hash_traversal_and_non_test_rejected(self):
        for changes in ({"digest": "0" * 64}, {"relative": "jpeg/../test.jpg"}, {"split": "val"}):
            with self.subTest(changes=changes), TemporaryDirectory() as directory:
                package, target = Path(directory) / "data.zip", Path(directory) / "test"
                self.make_zip(package, **changes)
                with self.assertRaises(ValueError):
                    prepare_test_package(package, target)
                self.assertFalse(target.exists())

    def test_output_does_not_overwrite_checkpoint_or_results(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            for output, error in ((base / "run_001/eval", ValueError), (base, FileExistsError)):
                with self.assertRaises(error):
                    evaluate(base / "run_001/best.pt", base / "raw", base / "processed", output)


@unittest.skipUnless(HAS_TORCH, "PyTorch/torchvision ausentes; executar no Colab")
class EvaluationTorchTests(unittest.TestCase):
    def checkpoint(self, model):
        return {"format_version": 1, "architecture": ARCHITECTURE, "epoch": 4,
                "model_state_dict": model.state_dict(), "validation_metrics": {"f1": 0.6},
                "history": [{"epoch": 4}], "metadata": {"architecture": ARCHITECTURE,
                "test_used": False, "pretrained_weights": {"hash_verified": True,
                                                           "backbone_load_verified": True}}}

    def test_checkpoint_roundtrip_all_parameters(self):
        torch.manual_seed(42)
        model = build_model(pretrained=False)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(self.checkpoint(model), path)
            restored, checkpoint = load_evaluation_model(path)
            self.assertEqual(checkpoint["epoch"], 4)
            self.assertFalse(restored.training)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))

    def test_incompatible_format_weights_and_unverified_provenance(self):
        model = build_model(pretrained=False)
        for change in ("format_version", "model_state_dict", "metadata"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                checkpoint = self.checkpoint(model)
                checkpoint[change] = {} if change != "format_version" else 9
                path = Path(directory) / "bad.pt"
                torch.save(checkpoint, path)
                with self.assertRaises(ValueError):
                    load_evaluation_model(path)

    def test_inference_eval_no_grad_no_parameter_or_buffer_updates(self):
        class Fixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(2)
                self.head = nn.Linear(2, 2)
            def forward(self, inputs):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError("Training/gradients forbidden")
                return self.head(self.bn(inputs))
        model = Fixture()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2)
        with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")), \
                patch("src.vision.training.fit", side_effect=AssertionError("training forbidden")):
            result = infer(model, loader, "cpu")
        self.assertEqual(result["images"], 4)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_infer_rejects_nonfinite_logits(self):
        class Bad(nn.Module):
            def forward(self, inputs):
                return torch.full((len(inputs), 2), float("nan"))
        with self.assertRaisesRegex(ValueError, "Logits"):
            infer(Bad(), [(torch.ones(2, 3), torch.tensor([0, 1]))], "cpu")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib ausente")
    def test_evaluation_only_test_loader_and_aggregate_artifacts(self):
        import pandas as pd
        from types import SimpleNamespace
        labels = [0] * 231 + [1] * 147
        model = nn.Linear(2, 2)
        with TemporaryDirectory() as directory:
            base = Path(directory)
            raw, manifests = base / "raw", base / "processed"
            raw.mkdir()
            manifests.mkdir()
            image = raw / "fixture.jpg"
            image.write_bytes(b"synthetic fixture; decoder tested in Dataset tests")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            (manifests / "manifest_test.csv").write_text("synthetic manifest")
            checkpoint = base / "run_001/best_epoch_004.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint fixture")
            records = pd.DataFrame({"label": labels, "sha256": [digest] * 378})
            loader = SimpleNamespace(dataset=SimpleNamespace(records=records, paths=[image] * 378))
            with patch("src.vision.evaluation.select_device", return_value="cpu"), \
                    patch("src.vision.evaluation.load_evaluation_model", return_value=(model, {"epoch": 4})), \
                    patch("src.vision.dataset.build_dataloaders", return_value={"test": loader}) as factory, \
                    patch("src.vision.evaluation.infer", return_value=final_metrics(labels, labels)), \
                    patch("src.vision.training.fit", side_effect=AssertionError("training forbidden")):
                output = base / "evaluation"
                result = evaluate(checkpoint, raw, manifests, output)
            self.assertEqual(factory.call_args.kwargs["splits"], ("test",))
            self.assertEqual(result["confusion_matrix"], [[231, 0], [0, 147]])
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"metrics.json", "confusion_matrix.png", "evaluation_report.md"})
            self.assertEqual(json.loads((output / "metrics.json").read_text())["fn"], 0)
            self.assertTrue((output / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
