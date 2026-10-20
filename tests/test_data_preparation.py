"""Testes da preparação de dados e do isolamento do pré-processamento."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.data_preparation import (
    FEATURE_COLUMNS,
    create_preprocessing_pipeline,
    indices_are_disjoint,
    load_dataset,
    prepare_data,
    split_data,
)


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            column: np.arange(100, dtype=float) + offset
            for offset, column in enumerate(FEATURE_COLUMNS)
        }, index=np.arange(1000, 1100))
        self.data["diagnosis"] = ["B"] * 60 + ["M"] * 40
        self.data["id"] = np.arange(100)
        self.data["Unnamed: 32"] = np.nan

    def test_load_dataset_preserves_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            self.data.to_csv(path, index=False)
            before = path.read_bytes()
            loaded = load_dataset(path)
            self.assertEqual(loaded.shape, (100, 33))
            self.assertEqual(path.read_bytes(), before)

    def test_diagnosis_mapping_and_unused_columns(self):
        original = self.data.copy(deep=True)
        X, y = prepare_data(self.data)
        self.assertEqual(X.shape, (100, 30))
        self.assertEqual(tuple(X.columns), FEATURE_COLUMNS)
        self.assertNotIn("diagnosis", X.columns)
        self.assertNotIn("id", X.columns)
        self.assertNotIn("Unnamed: 32", X.columns)
        self.assertEqual(y.iloc[:60].tolist(), [0] * 60)
        self.assertEqual(y.iloc[60:].tolist(), [1] * 40)
        self.assertTrue(pd.api.types.is_integer_dtype(y))
        self.assertTrue(X.index.equals(self.data.index))
        assert_frame_equal(self.data, original)

    def test_invalid_diagnoses_are_rejected(self):
        for value in ("X", "b", "", 1, None, np.nan):
            with self.subTest(value=value):
                data = self.data.copy()
                data.loc[1000, "diagnosis"] = value
                with self.assertRaisesRegex(ValueError, "Diagnósticos inválidos"):
                    prepare_data(data)

    def test_missing_required_columns_are_rejected(self):
        for column in ("diagnosis", "id", "Unnamed: 32", "radius_mean"):
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "Colunas necessárias ausentes"):
                    prepare_data(self.data.drop(columns=column))

    def test_non_numeric_features_are_rejected(self):
        self.data["radius_mean"] = "texto"
        with self.assertRaisesRegex(ValueError, "Características não numéricas"):
            prepare_data(self.data)

    def test_missing_numeric_values_are_preserved(self):
        self.data.loc[1000, "radius_mean"] = np.nan
        X, _ = prepare_data(self.data)
        self.assertTrue(pd.isna(X.loc[1000, "radius_mean"]))

    def test_split_is_reproducible_and_stratified(self):
        X, y = prepare_data(self.data)
        first = split_data(X, y)
        second = split_data(X, y)
        self.assertEqual([len(part) for part in first], [60, 20, 20, 60, 20, 20])
        for left, right in zip(first[:3], second[:3]):
            assert_frame_equal(left, right)
        for left, right in zip(first[3:], second[3:]):
            assert_series_equal(left, right)
            self.assertAlmostEqual(left.mean(), y.mean())

    def test_split_preserves_indices_and_has_no_overlap(self):
        X, y = prepare_data(self.data)
        splits = split_data(X, y)
        self.assertTrue(indices_are_disjoint(*splits[:3]))
        combined = set().union(*(set(part.index) for part in splits[:3]))
        self.assertEqual(combined, set(X.index))
        for X_part, y_part in zip(splits[:3], splits[3:]):
            assert_frame_equal(X_part, X.loc[X_part.index])
            assert_series_equal(y_part, y.loc[X_part.index])
        self.assertFalse(indices_are_disjoint(X, X.iloc[:1]))

    def test_duplicate_or_misaligned_indices_are_rejected(self):
        X, y = prepare_data(self.data)
        with self.assertRaisesRegex(ValueError, "mesmos índices e ordem"):
            split_data(X, y.iloc[::-1])
        X.index = [0] * len(X)
        y.index = X.index
        with self.assertRaisesRegex(ValueError, "índices originais devem ser únicos"):
            split_data(X, y)

    def test_pipeline_is_unfitted_and_handles_missing_values(self):
        pipeline = create_preprocessing_pipeline()
        with self.assertRaises(NotFittedError):
            check_is_fitted(pipeline)
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
        result = pipeline.fit_transform(X)
        self.assertEqual(result.shape, X.shape)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), [1, 1])

    def test_pipeline_fits_only_training_statistics(self):
        X_train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 9.0], "b": [2., 4., 6., 8.]})
        X_validation = pd.DataFrame({"a": [10000., np.nan], "b": [-10000., 20000.]})
        X_test = pd.DataFrame({"a": [-20000., np.nan], "b": [30000., -40000.]})
        pipeline = create_preprocessing_pipeline()
        pipeline.fit(X_train)
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        np.testing.assert_allclose(imputer.statistics_, [2., 5.])
        expected_train = np.array([[1., 2.], [2., 4.], [2., 6.], [9., 8.]])
        np.testing.assert_allclose(scaler.mean_, expected_train.mean(axis=0))
        np.testing.assert_allclose(scaler.var_, expected_train.var(axis=0))
        statistics = imputer.statistics_.copy()
        mean, variance = scaler.mean_.copy(), scaler.var_.copy()
        for held_out in (X_validation, X_test):
            transformed = pipeline.transform(held_out)
            expected = held_out.fillna({"a": 2., "b": 5.}).to_numpy()
            expected = (expected - mean) / np.sqrt(variance)
            np.testing.assert_allclose(transformed, expected)
        np.testing.assert_array_equal(imputer.statistics_, statistics)
        np.testing.assert_array_equal(scaler.mean_, mean)
        np.testing.assert_array_equal(scaler.var_, variance)
        self.assertEqual(scaler.n_samples_seen_, len(X_train))


if __name__ == "__main__":
    unittest.main()
