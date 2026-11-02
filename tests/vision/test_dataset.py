"""Testes offline de Dataset/transformações; skip explícito sem a pilha PyTorch."""

import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))
if HAS_TORCH:
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchvision import transforms
    from src.vision.dataset import MammographyDataset, Letterbox, build_dataloaders, build_transforms


@unittest.skipUnless(HAS_TORCH, "torch/torchvision ausentes; Dataset e tensores não validados neste ambiente")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / "raw"
        self.manifests = self.base / "manifests"
        self.manifests.mkdir()
        self.frames = {}
        for split_index, split in enumerate(("train", "val", "test")):
            rows = []
            for label in (0, 1):
                number = split_index * 2 + label + 1
                relative = f"jpeg/1.2.{number}/crop.jpg"
                path = self.root / relative
                path.parent.mkdir(parents=True)
                Image.new("L", (80, 40), number * 30).save(path, format="JPEG")
                patient = f"P_{number:05d}"
                rows.append({"patient_id": patient,
                             "case_id": f"Mass-{'Test' if split == 'test' else 'Training'}_{patient}_LEFT_CC_1",
                             "image_path": relative, "pathology": "BENIGN" if label == 0 else "MALIGNANT",
                             "label": label, "split": split, "official_split": "test" if split == "test" else "train",
                             "width": 80, "height": 40, "mode": "L",
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            self.frames[split] = pd.DataFrame(rows)
        self.write_manifests()

    def write_manifests(self):
        for split, frame in self.frames.items():
            frame.to_csv(self.manifests / f"manifest_{split}.csv", index=False)

    def dataset(self, split="train", **kwargs):
        return MammographyDataset(self.manifests / f"manifest_{split}.csv", self.root, **kwargs)

    def test_manifest_labels_and_three_channel_tensor(self):
        dataset = self.dataset()
        image, label = dataset[0]
        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(torch.isfinite(image).all())
        self.assertIsInstance(label, int)
        self.assertEqual([dataset[i][1] for i in range(2)], [0, 1])

    def test_grayscale_is_converted_to_rgb_before_transform(self):
        image, label = self.dataset(transform=lambda image: image.copy())[0]
        with image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((10, 10)), (30, 30, 30))
        self.assertEqual(label, 0)

    def test_lazy_loading_and_original_bytes_preserved(self):
        path = self.root / self.frames["train"].iloc[0].image_path
        before = path.read_bytes()
        with patch("src.vision.dataset.Image.open", side_effect=AssertionError("preload")):
            dataset = self.dataset()
        dataset[0]
        self.assertEqual(path.read_bytes(), before)

    def test_relative_path_resolves_with_another_root(self):
        dataset = self.dataset()
        self.assertEqual(dataset.paths[0], (self.root / self.frames["train"].iloc[0].image_path).resolve())

    def test_invalid_paths_are_rejected(self):
        for value in ("../outside.jpg", "/tmp/image.jpg", r"C:\images\image.jpg",
                      "jpeg/../outside.jpg", "jpeg//1.2.1/crop.jpg", "jpeg/1.2.1/crop.png"):
            with self.subTest(path=value):
                self.frames["train"].loc[0, "image_path"] = value
                self.write_manifests()
                with self.assertRaises(ValueError):
                    self.dataset()

    def test_missing_fields_and_invalid_labels_rejected(self):
        original = self.frames["train"].copy()
        for label in (2, -1, "1.0", "", "not-a-label"):
            self.frames["train"] = original.copy()
            self.frames["train"].loc[0, "label"] = str(label)
            self.write_manifests()
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.dataset()
        self.frames["train"] = original.drop(columns=["image_path"])
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "campos essenciais"):
            self.dataset()

    def test_label_pathology_mismatch_rejected(self):
        self.frames["train"].loc[0, "pathology"] = "MALIGNANT"
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "Rótulo/patologia"):
            self.dataset()

    def test_missing_image_rejected(self):
        (self.root / self.frames["train"].iloc[0].image_path).unlink()
        with self.assertRaisesRegex(ValueError, "imagem ausente"):
            self.dataset()

    def test_letterbox_preserves_ratio_and_entire_image(self):
        result = Letterbox()(Image.new("RGB", (80, 40), (255, 0, 0)))
        with result:
            self.assertEqual(result.size, (224, 224))
            self.assertEqual(result.getchannel("R").getbbox(), (0, 56, 224, 168))
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))

    def test_eval_deterministic_and_imagenet_normalization(self):
        dataset = self.dataset("val")
        first, _ = dataset[0]
        second, _ = dataset[0]
        self.assertTrue(torch.equal(first, second))
        expected = torch.tensor([(90 / 255 - mean) / std for mean, std in zip(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.assertTrue(torch.allclose(first[:, 112, 112], expected, atol=1e-5))

    def test_augmentation_is_light_and_only_for_training(self):
        training, evaluation = build_transforms(True), build_transforms(False)
        rotations = [t for t in training.transforms if isinstance(t, transforms.RandomRotation)]
        self.assertEqual(len(rotations), 1)
        self.assertEqual(rotations[0].degrees, [-5.0, 5.0])
        self.assertTrue(rotations[0].expand)
        self.assertFalse(any(isinstance(t, transforms.RandomRotation) for t in evaluation.transforms))

    def test_loaders_shapes_samplers_and_preserved_partitions(self):
        loaders = build_dataloaders(self.manifests, self.root, batch_size=2)
        self.assertIsInstance(loaders["train"].sampler, RandomSampler)
        for split in ("val", "test"):
            self.assertIsInstance(loaders[split].sampler, SequentialSampler)
        for split, loader in loaders.items():
            images, labels = next(iter(loader))
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertEqual(labels.dtype, torch.int64)
            self.assertEqual(loader.dataset.split, split)
            self.assertEqual(loader.num_workers, 0)
        self.assertTrue(torch.equal(next(iter(loaders["val"]))[0], next(iter(loaders["val"]))[0]))

    def test_loader_train_order_reproducible_with_fixed_seed(self):
        a = build_dataloaders(self.manifests, self.root, batch_size=1)
        b = build_dataloaders(self.manifests, self.root, batch_size=1)
        self.assertEqual([int(labels[0]) for _, labels in a["train"]],
                         [int(labels[0]) for _, labels in b["train"]])

    def test_cross_partition_leakage_rejected(self):
        self.frames["val"].loc[0, "sha256"] = self.frames["train"].loc[0, "sha256"]
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "Compartilhamento de sha256"):
            build_dataloaders(self.manifests, self.root)

    def test_wrong_partition_and_duplicate_manifest_cases_rejected(self):
        with self.assertRaisesRegex(ValueError, "Partição esperada"):
            self.dataset(expected_split="test")
        self.frames["train"] = pd.concat([self.frames["train"], self.frames["train"].iloc[[0]]])
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "repetidos"):
            self.dataset()


if __name__ == "__main__":
    unittest.main()
