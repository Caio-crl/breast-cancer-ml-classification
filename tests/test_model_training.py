"""Testes dos classificadores e do isolamento de treino, validação e teste."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.model_training import (
    calculate_metrics,
    create_models,
    evaluate_models,
    main,
    train_models,
)


class ModelTrainingTests(unittest.TestCase):
    def setUp(self):
        self.X_train = pd.DataFrame({
            "a": [-4., -3., -2., -1., 1., 2., 3., 4.],
            "b": [1., 2., np.nan, 4., 5., 6., 7., 8.],
        }, index=range(10, 18))
        self.y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=self.X_train.index)
        self.X_validation = pd.DataFrame({
            "a": [-10000., 10000., -5., 5.],
            "b": [np.nan, 20000., -30000., 40000.],
        }, index=range(30, 34))
        self.y_validation = pd.Series([0, 1, 0, 1], index=self.X_validation.index)

    def test_model_configurations(self):
        models = create_models()
        self.assertEqual(len(models), 2)
        logistic = models["Regressão Logística"].named_steps["classifier"]
        forest = models["Random Forest"].named_steps["classifier"]
        self.assertIsInstance(logistic, LogisticRegression)
        self.assertIsInstance(forest, RandomForestClassifier)
        self.assertEqual(
            logistic.get_params(),
            LogisticRegression(max_iter=1000, random_state=42).get_params(),
        )
        self.assertEqual(
            forest.get_params(),
            RandomForestClassifier(n_estimators=200, random_state=42).get_params(),
        )

    def test_preprocessing_is_independent_and_unfitted(self):
        models = list(create_models().values())
        for model in models:
            self.assertIsInstance(model, Pipeline)
            preprocessing = model.named_steps["preprocessing"]
            self.assertIsInstance(preprocessing, Pipeline)
            imputer = preprocessing.named_steps["imputer"]
            self.assertIsInstance(imputer, SimpleImputer)
            self.assertEqual(imputer.strategy, "median")
            self.assertIsInstance(preprocessing.named_steps["scaler"], StandardScaler)
            with self.assertRaises(NotFittedError):
                check_is_fitted(model)
        left, right = [m.named_steps["preprocessing"] for m in models]
        self.assertIsNot(left, right)
        for step in ("imputer", "scaler"):
            self.assertIsNot(left.named_steps[step], right.named_steps[step])

    def test_training_predictions_and_reproducibility(self):
        first = train_models(self.X_train, self.y_train)
        second = train_models(self.X_train, self.y_train)
        for name, model in first.items():
            with self.subTest(model=name):
                check_is_fitted(model)
                predicted = model.predict(self.X_validation)
                self.assertEqual(predicted.shape, (len(self.X_validation),))
                self.assertTrue(set(predicted).issubset({0, 1}))
                np.testing.assert_array_equal(
                    predicted, second[name].predict(self.X_validation),
                )

    def test_metric_values_and_class_order(self):
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
        predicted = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        metrics = calculate_metrics(y_true, predicted)
        self.assertAlmostEqual(metrics["Accuracy"], 5 / 8)
        self.assertAlmostEqual(metrics["Precision"], 3 / 4)
        self.assertAlmostEqual(metrics["Recall"], 3 / 5)
        self.assertAlmostEqual(metrics["F1-score"], 2 / 3)
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[2, 1], [2, 3]])

    def test_metrics_without_positive_predictions(self):
        metrics = calculate_metrics(pd.Series([0, 1, 1]), np.array([0, 0, 0]))
        for name in ("Precision", "Recall", "F1-score"):
            self.assertEqual(metrics[name], 0)
        self.assertEqual(metrics["confusion_matrix"].shape, (2, 2))

    def test_confusion_matrix_keeps_both_classes(self):
        metrics = calculate_metrics(pd.Series([0, 0]), np.array([0, 0]))
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[2, 0], [0, 0]])

    def test_evaluation_preserves_training_statistics(self):
        models = train_models(self.X_train, self.y_train)
        snapshots = {}
        for name, model in models.items():
            steps = model.named_steps["preprocessing"].named_steps
            np.testing.assert_allclose(steps["imputer"].statistics_, [0., 5.])
            imputed_train = self.X_train.fillna({"b": 5.}).to_numpy()
            np.testing.assert_allclose(steps["scaler"].mean_, imputed_train.mean(axis=0))
            np.testing.assert_allclose(steps["scaler"].var_, imputed_train.var(axis=0))
            snapshots[name] = (
                steps["imputer"].statistics_.copy(),
                steps["scaler"].mean_.copy(),
                steps["scaler"].var_.copy(),
            )
        results = evaluate_models(models, self.X_validation, self.y_validation)
        self.assertEqual(set(results), set(models))
        for name, model in models.items():
            steps = model.named_steps["preprocessing"].named_steps
            before = snapshots[name]
            np.testing.assert_array_equal(steps["imputer"].statistics_, before[0])
            np.testing.assert_array_equal(steps["scaler"].mean_, before[1])
            np.testing.assert_array_equal(steps["scaler"].var_, before[2])
            self.assertEqual(steps["scaler"].n_samples_seen_, len(self.X_train))
            self.assertEqual(results[name]["confusion_matrix"].shape, (2, 2))
            self.assertEqual(results[name]["confusion_matrix"].sum(), len(self.y_validation))

    def test_main_uses_only_training_and_validation(self):
        raw, features, target = object(), object(), object()
        reserved_X_test, reserved_y_test = object(), object()
        models = {name: Mock(wraps=model) for name, model in create_models().items()}
        splits = (
            self.X_train, self.X_validation, reserved_X_test,
            self.y_train, self.y_validation, reserved_y_test,
        )
        with (
            patch("src.model_training.load_dataset", return_value=raw) as load,
            patch("src.model_training.prepare_data", return_value=(features, target)) as prepare,
            patch("src.model_training.split_data", return_value=splits) as split,
            patch("src.model_training.create_models", return_value=models),
            patch("src.model_training.calculate_metrics", wraps=calculate_metrics) as metrics,
            redirect_stdout(StringIO()) as output,
        ):
            main()
        load.assert_called_once_with()
        prepare.assert_called_once_with(raw)
        split.assert_called_once_with(features, target)
        for model in models.values():
            model.fit.assert_called_once()
            self.assertIs(model.fit.call_args.args[0], self.X_train)
            self.assertIs(model.fit.call_args.args[1], self.y_train)
            model.predict.assert_called_once()
            self.assertIs(model.predict.call_args.args[0], self.X_validation)
        self.assertEqual(metrics.call_count, 2)
        for call in metrics.call_args_list:
            self.assertIs(call.args[0], self.y_validation)
            self.assertEqual(len(call.args[1]), len(self.y_validation))
        for text in ("Accuracy", "Precision", "Recall", "F1-score", "VN, FP", "FN, VP"):
            self.assertIn(text, output.getvalue())


if __name__ == "__main__":
    unittest.main()
