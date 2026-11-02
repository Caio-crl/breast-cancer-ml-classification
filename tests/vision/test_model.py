"""Forward e congelamento offline, sem GPU ou download de pesos."""

from contextlib import redirect_stdout
from io import StringIO
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.vision.smoke_test import main as smoke_main


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))
if HAS_TORCH:
    import torch
    from torchvision.models import mobilenet_v3_small
    from src.vision.model import build_model, parameter_counts


@unittest.skipUnless(HAS_TORCH, "torch/torchvision ausentes; forward e arquitetura não validados neste ambiente")
class ModelTests(unittest.TestCase):
    def test_two_logits_forward_pass_without_weights_or_gradients(self):
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("download")):
            model = build_model(pretrained=False).eval()
            with torch.inference_mode():
                output = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.isfinite(output).all())

    def test_random_initialization_keeps_all_parameters_trainable_by_default(self):
        model = build_model()
        counts = parameter_counts(model)
        self.assertGreater(counts["total"], 0)
        self.assertEqual(counts["total"], counts["trainable"])

    def test_features_frozen_and_batchnorm_state_unchanged(self):
        model = build_model(freeze_features=True).train()
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        self.assertTrue(all(not block.training for block in model.features))
        before = {name: value.clone() for name, value in model.features.named_buffers()}
        with torch.inference_mode():
            model(torch.zeros(2, 3, 224, 224))
        for name, value in model.features.named_buffers():
            self.assertTrue(torch.equal(before[name], value))

    def test_fine_tuning_unfreezes_only_last_requested_blocks(self):
        model = build_model(freeze_features=True)
        model.configure_fine_tuning(1)
        self.assertFalse(any(p.requires_grad for block in list(model.features)[:-1] for p in block.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.features[-1].parameters()))
        self.assertTrue(model.features[-1].training)
        with self.assertRaises(ValueError):
            model.configure_fine_tuning(len(model.features) + 1)

    def test_input_shapes_are_checked(self):
        model = build_model().eval()
        for shape in ((2, 1, 224, 224), (2, 3, 128, 128), (3, 224, 224)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(*shape))

    def test_pretrained_missing_cache_never_downloads(self):
        with TemporaryDirectory() as directory:
            with patch("torch.hub.get_dir", return_value=directory), patch(
                    "torch.hub.download_url_to_file", side_effect=AssertionError("download")):
                with self.assertRaises(FileNotFoundError):
                    build_model(pretrained=True)

    def test_local_backbone_state_is_loaded_before_binary_head_and_freezes_features(self):
        # State dict sintético de 1000 classes, não pesos ImageNet baixados.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-backbone.pth"
            backbone = mobilenet_v3_small(weights=None)
            with torch.no_grad():
                next(backbone.features.parameters()).fill_(0.25)
            torch.save(backbone.state_dict(), path)
            model = build_model(pretrained=True, weights_path=path)
        self.assertTrue(torch.all(next(model.features.parameters()) == 0.25))
        self.assertEqual(model.classifier[-1].out_features, 2)
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))


class SmokeGuardTests(unittest.TestCase):
    def test_missing_dependencies_stop_before_loading_images_or_model(self):
        with patch("src.vision.smoke_test.importlib.util.find_spec", return_value=None):
            with redirect_stdout(StringIO()) as output:
                code = smoke_main(["--dataset-root", "does-not-exist"])
        self.assertEqual(code, 2)
        self.assertIn("não executado", output.getvalue())


if __name__ == "__main__":
    unittest.main()
