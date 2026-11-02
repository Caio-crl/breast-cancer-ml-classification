"""Métricas sem torch e testes CPU sintéticos condicionais, sem rede/dataset real."""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
from io import StringIO
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.vision.training import (
    ARCHITECTURE, EarlyStopping, PreparedTraining, TrainingConfig,
    build_training_model, classification_metrics, fit, load_checkpoint, main,
    prepare_training, require_torch, run_epoch, seed_everything,
    validate_output_dir, verify_weight_file,
)

HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))
if HAS_TORCH:
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import mobilenet_v3_small
    from src.vision.model import MassMobileNet
    from src.vision.dataset import build_dataloaders


class TrainingUtilityTests(unittest.TestCase):
    def test_metrics_positive_malignant_and_zero_denominators(self):
        scores = classification_metrics([0, 0, 1, 1, 1], [0, 1, 0, 1, 1])
        self.assertEqual(scores["accuracy"], 3 / 5)
        self.assertAlmostEqual(scores["precision"], 2 / 3)
        self.assertAlmostEqual(scores["recall"], 2 / 3)
        self.assertAlmostEqual(scores["f1"], 2 / 3)
        self.assertEqual(classification_metrics([0, 1], [0, 0])["f1"], 0)
        self.assertEqual(classification_metrics([0, 0], [0, 0])["recall"], 0)

    def test_invalid_metric_inputs(self):
        for actual, predicted in (([], []), ([0], []), ([2], [1])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                classification_metrics(actual, predicted)

    def test_early_stopping_f1_improvement_ties_and_patience(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual(stopper.update(0.5), (True, False))
        self.assertEqual(stopper.update(0.7), (True, False))
        self.assertEqual(stopper.update(0.705), (False, False))
        self.assertEqual(stopper.update(0.7), (False, True))
        self.assertEqual(stopper.best_score, 0.7)

    def test_nonfinite_early_stopping_rejected(self):
        for score in (float("nan"), float("inf"), -1, 2):
            with self.subTest(score=score), self.assertRaises(ValueError):
                EarlyStopping().update(score)

    def test_invalid_configuration(self):
        config = TrainingConfig("raw", "manifests")
        for changes in ({"max_epochs": 0}, {"batch_size": 0}, {"patience": -1},
                        {"learning_rate": float("nan")}, {"learning_rate": 0},
                        {"device": "gpu"}, {"trainable_feature_blocks": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes).validate()

    def test_weights_hash_and_official_origin_checked_without_torch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pth"
            path.write_bytes(b"synthetic weight-file integrity fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            info = verify_weight_file(path, f"https://download.pytorch.org/models/fixture-{digest[:8]}.pth")
            self.assertEqual(info["sha256"], digest)
            self.assertTrue(info["hash_verified"])
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_weight_file(path, "https://download.pytorch.org/models/fixture-00000000.pth")
            with self.assertRaisesRegex(ValueError, "Origem"):
                verify_weight_file(path, f"https://example.com/fixture-{digest[:8]}.pth")

    def test_missing_weights_are_an_explicit_error(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "obrigatórios ausentes"):
                verify_weight_file(Path(directory) / "missing.pth",
                                   "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth")

    def test_output_guard_preserves_raw_and_existing_artifacts(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            config = TrainingConfig(str(base / "raw"), str(base / "manifests"), str(base / "run"))
            self.assertEqual(validate_output_dir(config), base / "run")
            with self.assertRaises(ValueError):
                validate_output_dir(replace(config, output_dir=str(base / "raw/run")))
            (base / "run").mkdir()
            (base / "run/keep.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                validate_output_dir(config)
            self.assertEqual((base / "run/keep.txt").read_text(encoding="utf-8"), "preserve")

    def test_missing_optional_dependencies_are_explicit_and_no_training_starts(self):
        with patch("src.vision.training.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ModuleNotFoundError, "PyTorch/torchvision ausentes"):
                require_torch()
            with TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
                code = main(["--dataset-root", directory + "/raw", "--manifest-dir", directory + "/manifests",
                             "--output-dir", directory + "/run"])
                self.assertEqual(code, 2)
                self.assertFalse((Path(directory) / "run").exists())
                self.assertIn("nada foi instalado", output.getvalue())


@unittest.skipUnless(HAS_TORCH, "torch/torchvision ausentes; treino CPU sintético e checkpoints pendentes")
class TrainingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.config = TrainingConfig(str(self.base / "raw"), str(self.base / "manifests"),
                                     str(self.base / "run"), device="cpu", max_epochs=1, batch_size=2)

    def session(self, config=None):
        config = config or self.config
        seed_everything(config.seed)
        # Rede mínima que reutiliza a classe/congelamento real; não pesos ImageNet.
        backbone = SimpleNamespace(features=nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2)),
                                   avgpool=nn.AdaptiveAvgPool2d(1), classifier=nn.Sequential(nn.Linear(2, 2)))
        model = MassMobileNet(backbone, freeze_features=True)
        data = TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
        loaders = {split: DataLoader(data, batch_size=2, shuffle=False) for split in ("train", "val")}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
        # Proveniência simulada só neste teste unitário do loop; a API de produção
        # verifica pesos oficiais antes de construir PreparedTraining.
        metadata = {"synthetic_fixture": True, "pretrained_weights": {
            "hash_verified": True, "backbone_load_verified": True}, "architecture": ARCHITECTURE}
        return PreparedTraining(config, model, loaders, optimizer, torch.device("cpu"), metadata)

    def test_trainable_head_updates_and_frozen_features_and_bn_do_not(self):
        prepared = self.session()
        features = {k: v.clone() for k, v in prepared.model.features.state_dict().items()}
        head = [p.clone() for p in prepared.model.classifier.parameters()]
        scores = run_epoch(prepared.model, prepared.loaders["train"], prepared.device, prepared.optimizer)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(head, prepared.model.classifier.parameters())))
        self.assertTrue(all(torch.equal(v, prepared.model.features.state_dict()[k]) for k, v in features.items()))
        self.assertEqual(set(scores), {"loss", "accuracy", "precision", "recall", "f1"})

    def test_validation_does_not_update_parameters_or_calculate_gradients(self):
        prepared = self.session()
        before = {k: v.clone() for k, v in prepared.model.state_dict().items()}
        run_epoch(prepared.model, prepared.loaders["val"], prepared.device)
        self.assertTrue(all(torch.equal(v, prepared.model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in prepared.model.parameters()))

    def test_early_stopping_checkpoint_selection_and_loading(self):
        prepared = self.session(replace(self.config, max_epochs=5, patience=2))
        def scores(f1):
            return {"loss": 0.7, "accuracy": 0.5, "precision": f1, "recall": f1, "f1": f1}
        sequence = [item for score in (0.5, 0.6, 0.6, 0.59) for item in (scores(0.4), scores(score))]
        with patch("src.vision.training.run_epoch", side_effect=sequence), redirect_stdout(StringIO()):
            result = fit(prepared)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["epochs_executed"], 4)
        self.assertTrue(result["stopped_early"])
        self.assertFalse(result["test_used"])
        checkpoint = load_checkpoint(self.base / "run" / result["best_checkpoint"], prepared.model)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["validation_metrics"]["f1"], 0.6)
        self.assertEqual(len(list((self.base / "run").glob("best_epoch_*.pt"))), 2)
        with self.assertRaises(FileExistsError):
            fit(prepared)

    def test_reproducible_cpu_updates_with_same_seed(self):
        a, b = self.session(), self.session()
        scores_a = run_epoch(a.model, a.loaders["train"], a.device, a.optimizer)
        scores_b = run_epoch(b.model, b.loaders["train"], b.device, b.optimizer)
        self.assertEqual(scores_a, scores_b)
        for key, value in a.model.state_dict().items():
            self.assertTrue(torch.equal(value, b.model.state_dict()[key]))

    def test_test_loader_is_refused_before_training(self):
        prepared = self.session()
        prepared.loaders["test"] = object()
        with self.assertRaisesRegex(ValueError, "exclusivamente"):
            fit(prepared)
        self.assertFalse((self.base / "run").exists())

    def test_train_val_loaders_do_not_even_construct_test_dataset(self):
        def dataset_factory(manifest, root, transform, expected_split):
            self.assertIn(expected_split, ("train", "val"))
            data = TensorDataset(torch.zeros(2, 3, 224, 224), torch.tensor([0, 1]))
            prefix = "train" if expected_split == "train" else "val"
            data.records = pd.DataFrame({field: [prefix + "0", prefix + "1"]
                                        for field in ("patient_id", "case_id", "image_path", "sha256")})
            return data
        with patch("src.vision.dataset.MammographyDataset", side_effect=dataset_factory) as factory:
            loaders = build_dataloaders("no-test-manifest", "root", splits=("train", "val"))
        self.assertEqual(set(loaders), {"train", "val"})
        self.assertEqual(factory.call_count, 2)
        prepared_fixture = self.session()
        for loader in loaders.values():
            loader.dataset.records["label"] = ["0", "1"]
        manifests = Path(self.config.manifest_dir)
        manifests.mkdir()
        for split in ("train", "val"):
            (manifests / f"manifest_{split}.csv").write_text("synthetic fixture", encoding="utf-8")
        with patch("src.vision.training.build_training_model", return_value=(
                prepared_fixture.model, prepared_fixture.metadata["pretrained_weights"])), patch(
                "src.vision.dataset.build_dataloaders", return_value=loaders) as loader_factory:
            preview = prepare_training(self.config)
        self.assertEqual(loader_factory.call_args.kwargs["splits"], ("train", "val"))
        self.assertEqual(set(preview.metadata["partitions"]), {"train", "val"})
        self.assertFalse(Path(self.config.output_dir).exists())
        with patch("src.vision.training.prepare_training", return_value=preview), patch(
                "src.vision.training.fit") as trainer, redirect_stdout(StringIO()):
            self.assertEqual(main(["--dataset-root", self.config.dataset_root,
                                   "--manifest-dir", self.config.manifest_dir]), 0)
        trainer.assert_not_called()

    def test_training_model_refuses_missing_imagenet_weights(self):
        config = replace(self.config, weights_path=str(self.base / "absent.pth"))
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("download")):
            with self.assertRaises(FileNotFoundError):
                build_training_model(config)

    def test_frozen_random_model_cannot_bypass_pretrained_provenance(self):
        prepared = self.session()
        prepared.metadata["pretrained_weights"] = {}
        with self.assertRaisesRegex(ValueError, "ImageNet não verificada"):
            fit(prepared)

    def test_backbone_loading_is_verified_with_offline_fixture(self):
        path = self.base / "synthetic.pth"
        backbone = mobilenet_v3_small(weights=None)
        torch.save(backbone.state_dict(), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        synthetic_url = f"https://download.pytorch.org/models/fixture-{digest[:8]}.pth"
        # Hash de fixture sintética, isolando o teste de carga estrita e igualdade.
        with patch("src.vision.training.verify_weight_file", side_effect=lambda p, url: verify_weight_file(p, synthetic_url)):
            model, provenance = build_training_model(replace(self.config, weights_path=str(path)))
        self.assertTrue(provenance["backbone_load_verified"])
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertTrue(torch.equal(next(model.features.parameters()), next(backbone.features.parameters())))


if __name__ == "__main__":
    unittest.main()
